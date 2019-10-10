from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras import Sequential


class MLP(Layer):
    """
    Multi-Layer Perceptron block
    
    Args:
     - input_dim: the input neurons on the first layer.
     - dimensions: a list or tuple of the sizes of the networks.
     - activation: the activation function for each layer.
    """
    def __init__(self, input_dim, dimensions, activation="relu"):
        super().__init__()

        dimensions.insert(0, input_dim)
        self.mlp = Sequential()

        for lyr in range(len(dimensions) - 1):
            self.mlp.add(Dense(dimensions(lyr+1), activation=activation,
                               input_dim=(dimensions(lyr))))

    def call(self, inputs):
        return self.mlp(inputs)
