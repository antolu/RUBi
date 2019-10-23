import torch
import torch.nn as nn
from torch.autograd import Variable
from models.mlp import MLP
from models.skip_thoughts import BiSkip
from models.block import Block
from collections import OrderedDict

class BaselineNet(nn.Module):
    def __init__(self, dir_st, vocab, img_emb_size=2048, text_emb_size=4800, mlp_dimensions=[2048, 2048, 3000]):
        super(BaselineNet, self).__init__()

        # also initialise question and image encoder
        self.skip_thought = QuestionEncoder(dir_st, vocab)
        self.fusion_block = Block([text_emb_size, img_emb_size], 2048, chunks=15, rank=15)
        self.mlp = MLP(2048, mlp_dimensions)

    def forward(self, inputs):
        """
        do full foward pass.
        ------
        Parameters
        inputs: item = dict with keys: 'img_embed', 'image', 'quest_vocab_vec', 'quest_size', 'answer_one_hot', 'idx_answer'
        
        """
        b_size = inputs['quest_vocab_vec'].size(0)
        quest_size = inputs['quest_vocab_vec'].size(1)
        n_regions = inputs['img_embed'].size(1)


        # Image embedding (36 regions)
        img_embedding = inputs['img_embed']

        # embedding question
        # question_embedding = inputs['quest_vocab_vec'].expand(b_size, img_embedding.size()[1], inputs['quest_vocab_vec'].size()[1])
        #print("start question embeded...")
        question_embedding = self.skip_thought(inputs).float()
        
        # question_embedding = question_embedding.expand(img_embedding.size()[0], question_embedding[0].size()[0]) 

        expanded_embeddings = question_embedding.unsqueeze(1).expand(b_size, n_regions, question_embedding.shape[1])
        reshaped_q_emb = expanded_embeddings.contiguous().view(b_size*n_regions, -1)
        reshaped_img_emb = img_embedding.contiguous().view(b_size*n_regions, -1)

        # Block fusion 
        #print("start block fusion...")
        block_out = self.fusion_block([reshaped_q_emb, reshaped_img_emb]) 

        block_out = block_out.view(b_size, n_regions, -1)

        # TODO: Max pooling
        #print("start Max pooling...")
        (maxpool, argmax) = torch.max(block_out, dim=1)
        
        # MLP
        #print("start MLP...")
        final_out = self.mlp(maxpool)
        
        out = {
            "max": maxpool,
            "argmax": argmax,
            "q_emb": question_embedding,
            "logits": final_out
        }
        
        return out
    

class QuestionEncoder(nn.Module):
    """
    The question encoder

    Parameters:
    -----------
     - dir_st: str
        The directory containing the skip thought data files
     - vocab: list
        A list of words to initialise the text encoder with
    """
    def __init__(self, dir_st, vocab):
        super().__init__()
        self.text_encoder = BiSkip(dir_st, vocab)

        self.attn_extractor = nn.Sequential(OrderedDict([
            ("lin1", nn.Linear(2400, 512)),
            ("relu", nn.ReLU()),
            ("lin2", nn.Linear(512, 2))
        ]))
        
        self.softmaxMask = SoftMaxMask()

    def forward(self, inputs):

        q_emb = self.text_encoder.embedding(inputs['quest_vocab_vec'])
        (q_emb, _) = self.text_encoder.rnn(q_emb)

        attns = self.attn_extractor(q_emb)
        
        attns = self.softmaxMask(attns, inputs['quest_size'])

        attns_res = []
        for attn in torch.unbind(attns, dim=2):
            attn = attn.unsqueeze(dim=2)
            attn.expand_as(q_emb)

            q_with_attn = attn * q_emb
            attns_res.append(q_with_attn.sum(dim=1))

        q_emb_with_attn = torch.cat(attns_res, dim=1)

        return q_emb_with_attn



class SoftMaxMask(nn.Module):
    """
    Computes the masked Softmax

    Borrowed from Qiao Jin @ https://discuss.pytorch.org/t/apply-mask-softmax/14212/14
    """
    def __init__(self, dim=1, epsilon=1e-5):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon

    def forward(self, inputs, question_lengths):
        
        max_val = torch.max(inputs, dim=self.dim, keepdim=True)[0]  # increases numerical stability
        numerator = torch.exp(inputs - max_val)
    
        mask = torch.zeros_like(inputs).to(device=inputs.device, non_blocking=True)
        t_lengths = question_lengths[:,:,None].expand_as(mask)
        arange_id = torch.arange(mask.size(1)).to(device=inputs.device, non_blocking=True)
        arange_id = arange_id[None,:,None].expand_as(mask)

        mask[arange_id<t_lengths] = 1

        # mask = (inputs != 0).float()
        numerator = numerator * mask  # this step masks

        denominator = torch.sum(numerator, dim=self.dim, keepdim=True) + self.epsilon

        return numerator / denominator
