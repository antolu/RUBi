from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, Sequential
from models.mlp import MLP


class RUBi(Model):
    """
    Similar to the PyTorch implementation the RUBi wraps the original model,
    and requires the model to return a dictionary containing the 'logits' key
    as well as a 'q_emb' key. 
    Returns a dictionary containing:
     - 'logits': the original logits from the base model
     - 'logits_rubi': the updated predictions from the model by the mask
     - 'logits_q': the predictions from the question-only branch 
    """
    def __init__(self, base_vqa):
        super(RUBi, self).__init()
        self.model = base_vqa

        dimensions = (2048, 2048, 3000)

        self.mlp = MLP(4800, dimensions)

        self.slp = Dense(3000)

    def call(self, text, image):
        base_out = self.model(text, image)
        
        x = self.mlp(base_out['q_emb'])

        mask = tf.math.sigmoid(x)
        aQM = tf.math.multiply(mask, base_out['logits'])

        aQO = self.slp(x)

        out = {}
        out['logits'] = base_out['logits']
        out['logits_q'] = aQO
        out['logits_rubi'] = aQM

        return out
