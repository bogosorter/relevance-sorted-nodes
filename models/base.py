import torch
from torch import nn

class BaseNetwork(nn.Module):
    name = 'base_network'

    def __init__(self, layer_sizes):
        super().__init__()

        self.layer_sizes = layer_sizes

        self.flatten = nn.Flatten()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

    def regularization(self, epoch=None, epochs=None):
        return 0

    def normalize(self, weights):
        return torch.abs(weights) / torch.sum(torch.abs(weights))

    def prune(self, amount):
        """
        To be implemented in subclass.
        """
        raise NotImplementedError()
