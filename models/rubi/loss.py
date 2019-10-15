from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn


class RUBiLoss:
    """
    The RUBi loss, superposed question-modality and question-only
    loss. Computes the final loss L_RUBi as
    `L_RUBi = lambda1 * L_QM + lambda2 * L_QO`.

    Args:
      lambda1: The scalar which determines the importance of the
        question-modality loss.
      lambda2: The scalar which determines the importance of the
        question-only loss. 
    """
    
    def __init__(self, lambda1, lambda2):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.loss = nn.CrossEntropyLoss()

    """
    Parameters:
    - predictions: a dict containing keys 'logits', 'logits_q', 'logits'
    - labels: one-hot encodings
    """
    def __call__(self, labels, logits):
        L_QM = self.loss(logits["logits_rubi"], labels)
        L_QO = self.loss(logits["logits_q"], labels)

        return self.lambda1 * L_QM + self.lambda2 * L_QO


class BaselineLoss:
    """
    The loss of the baseline model. 
    """
    
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    """
    Parameters:
    - predictions: a dict containing the key'logits'
    - labels: one-hot encodings
    """
    def __call__(self, labels, logits):
        return nn.CrossEntropyLoss(logits["logits"], labels)
