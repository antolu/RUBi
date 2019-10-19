from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from models.mlp import MLP
from models.skip_thoughts import BiSkip
from collections import OrderedDict

class BaselineNet(Model):
    def __init__(self):
        super().__init__()

        # also initialise question and image encoder
        self.fusion_block = BlockFusion(None)
        

    def call(self, inputs):

        # full forward pass


class QuestionEncoder(nn.Module) :
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

        self.text_encoder = BiSkip(dir_st, vocab)

        self.attn_extractor = nn.Sequential(OrderedDict([
            ("lin1", nn.Linear(2400, 512)),
            ("relu", nn.ReLU()),
            ("lin2", nn.Linear(512, 2)),
            ("mask_softmax", SoftMaxMask())
        ]))

    def forward(self, inputs):

        q_emb = self.question_encoder(inputs)

        attns = self.attn_extractor(q_emb)

        attns_res = []
        for attn in torch.unbind(attns, dim=2):
            attn = attn.unsqueeze(dim=2).expand_as(q_emb)

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
        self.dim = dim
        self.epsilon = epsilon

    def forward(self, inputs):
        max_val = torch.max(inputs, dim=self.dim, keepdim=True)[0] # increases numerical stability
        numerator = torch.exp(inputs - max_val)
        numerator = numerator * (inputs == 0).float()  # this step masks

        denominator = torch.sum(numerator, dim=self.dim, keepdim=True) + self.epsilon

        return numerator / denominator
