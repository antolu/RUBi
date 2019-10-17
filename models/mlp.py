from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import torch
import torch.nn as nn


class MLP(nn.Module):
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

        layers = []
        for lyr in range(len(dimensions) - 1):
            layers.append(("linear" + str(lyr + 1),
                           nn.Linear(dimensions[lyr], dimensions[lyr+1])))
            if activation != "none":
                layers.append(("relu" + str(lyr + 1), nn.ReLU()))

        self.mlp = nn.Sequential(OrderedDict(layers))

    def forward(self, inputs):
        return self.mlp(inputs)
