import torch
import torch.nn as nn
from torch.autograd import Variable
from san_pytorch.misc.san import Attention
from san_pytorch.misc.ques_emb_net import QuestionEmbedding
from san_pytorch.misc.img_emb_net import ImageEmbedding


class SanBaseline(nn.Module):
    """
    consists of the image encoder, question encoder, and the SAN attention module
    """

    def __init__(self, dir_st, vocab, img_emb_size=2048, text_emb_size=4800, mlp_dimensions = [2048, 3000]):
        super(BaselineNet, self).__init__()

        self.ques_emb_net = QuestionEmbedding(vocab_size, emb_size, hidden_size, rnn_size, num_layers, dropout,
                                                                                            seq_length, use_gpu)
        self.img_emb_net = ImageEmbedding(hidden_size)
        self.attention_mod = Attention()

    def forward(self, inputs):
        """
        do full forward pass.
        ------
        Parameters
        inputs: item = dict with keys: 'img_embed', 'image', 'question', 'answer', 'answer_one_hot'

        """
        # Image embedding (36 regions)
        # TODO check if we dont use the image encoder of SAN
        img_embedding = inputs[img_embed]

         # embedding question
        question_embedding = inputs['quest_vocab_vec'].expand(img_embedding.size()[0], inputs['quest_vocab_vec'].size(0))

        question_embedding = self.skip_thought(Variable(torch.LongTensor(question_embedding)))


        # TODO does SAN image encoder and question encoder also output in 36 regions?
        final_out = [self.attention_mod([quest_e, img_e]) for quest_e, img_e in zip(question_embedding, img_embedding)]

        #TODO: put answer in the format required by loss.py and rubi.py
        # s ={'lo'}

        return final_out

