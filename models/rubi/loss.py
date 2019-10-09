from __future__ import absolute_import, division, print_function unicode_literals

from tensorflow.keras.losses import Loss
from tensorflow.nn import softmax_cross_entropy_with_logits as SoftMax_CE


class RUBi_loss(Loss):
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
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    """
    Parameters:
    - predictions: a dict containing keys 'logits', 'logits_q', 'logits'
    - labels: one-hot encodings
    """
    def call(self, labels, logits):
        L_QM = SoftMax_CE(labels, logits["logits_rubi"])
        L_QO = SoftMax_CE(labels, logits["logits_q"])

        return self.lambda1 * L_QM + self.lambda2 * L_QO


class baseline_loss(Loss):
    """
    The loss of the baseline model. 
    """
    
    def __init__(self, lambda1, lambda2):
        super().__init__()

    """
    Parameters:
    - predictions: a dict containing the key'logits'
    - labels: one-hot encodings
    """
    def call(self, labels, logits):
        return SoftMax_CE(labels, logits["logits"])
