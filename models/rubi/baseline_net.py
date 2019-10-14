from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras import Model, Sequential
from models.mlp import MLP

class BaselineNet(Model):
    def __init__(self):
        super().__init__()

        # also initialise question and image encoder
        self.fusion_block = BlockFusion(None)
        

    def call(self, inputs):

        # full forward pass


class BlockFusion(Layer):
    def __init__(self, nv):
        super().__init__()

        mlp_dimensions = (2048, 2048, 3000)
        self.mlp = MLP(2048, mlp_dimensions)

        self.block = Block()

    def call(self, inputs):

        x = self.block(inputs)
        x = self.mlp(x)

        out = {}
        out["q_emb"] = inputs["q_emb"]
        out["logits"] = None

        return out

    
class Block(Layer):
    def __init__(self,
                 input_dims,
                 output_dims,
                 chunks=15,
                 rank=15,
                 projection_size=1000):
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.chunks = chunks
        self.rank = self.rank
        self.projection_size = projection_size

        self.linear1 = Dense(projection_size, activation="none", input_dim=input_dims)

    def call(self, inputs):
