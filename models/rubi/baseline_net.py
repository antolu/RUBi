import torch
import torch.nn as nn

from torch.autograd import Variable

from models.block import Block
from models.skip_thoughts import BiSkip
from models.mlp import MLP


class BaselineNet(nn.Module):
    def __init__(self, dir_st, vocab, img_emb_size=2048, text_emb_size=4800, mlp_dimensions = [2048, 3000]):
        super(BaselineNet, self).__init__()

        # also initialise question and image encoder
        self.skip_thought = BiSkip(dir_st, vocab)
        self.fusion_block = Block([text_emb_size, img_emb_size], 2048, chunks=15, rank=15)
        self.mlp = MLP(2048, mlp_dimensions)

    def forward(self, inputs):
        """
        do full foward pass.
        ------
        Parameters
        inputs: item = dict with keys: 'img_embed', 'image', 'question', 'answer', 'answer_one_hot'
        
        """

        # Imgage embedding (36 regions)
        img_embedding = inputs['img_embed']
        
        print(inputs['quest_vocab_vec'])

        # embedding question
        question_embedding = inputs['quest_vocab_vec'].expand(img_embedding.size()[0], inputs['quest_vocab_vec'].size(0))
        
        question_embedding = self.skip_thought(Variable(torch.LongTensor(question_embedding)))
        # question_embedding = question_embedding.expand(img_embedding.size()[0], question_embedding[0].size()[0]) 

        # Block fusion        
        block_out = [self.fusion_block([quest_e, img_e]) for quest_e,img_e in zip(question_embedding, img_embedding)] 
        
        # MLP
        final_out = self.mlp(block_out)
        
        #TODO: put answer in the format required by loss.py and rubi.py
        # s ={'lo'}
        
        return final_out
    