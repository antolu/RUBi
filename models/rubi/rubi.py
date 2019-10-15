from __future__ import absolute_import, division, print_function, unicode_literals

from models.mlp import MLP

import torch
import torch.nn as nn


class RUBi(nn.Module):
    """
    Similar to the original implementation the RUBi wraps the original model,
    and requires the model to return a dictionary containing the 'logits' key
    as well as a 'q_emb' key. 
    Returns a dictionary containing:
     - 'logits': the original logits from the base model
     - 'logits_rubi': the updated predictions from the model by the mask
     - 'logits_q': the predictions from the question-only branch
    """
    def __init__(self, base_vqa):
        super().__init__()
        self.model = base_vqa

        dimensions = [2048, 2048, 3000]
        self.mlp = MLP(4800, dimensions)

        self.slp = nn.Linear(3000, 3000)

    def forward(self, inputs):
        text = inputs["q_text"]
        visual_emb = inputs["visual_emb"]
        
        base_out = self.model(text, visual_emb)
        
        x = self.mlp(base_out['q_emb'])

        mask = nn.Sigmoid(x)
        aQM = mask * base_out['logits']

        aQO = self.slp(x)

        out = {}
        out['logits'] = base_out['logits']
        out['logits_q'] = aQO
        out['logits_rubi'] = aQM

        return out
