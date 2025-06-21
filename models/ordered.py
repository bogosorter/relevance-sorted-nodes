import torch
from torch import nn
from models.base import BaseNetwork

class OrderedNetwork(BaseNetwork):
    name = 'ordered_network'

    def __init__(self, layer_sizes, alpha, beta):
        super().__init__(layer_sizes)

        self.alpha = alpha
        self.beta = beta

        self.penalties = []
        for i in range(1, len(layer_sizes) - 1):
            self.penalties.append(
                torch.tensor([self.beta * j ** self.alpha for j in range(layer_sizes[i])])
            )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

    def regularization(self, epoch, epochs):
        total = 0

        for i in range(len(self.penalties)):
            total += torch.sum(
                self.penalties[i][None, :] @ self.normalize(self.layers[2 * (i + 1)].weight).T
            )

        return total / len(self.penalties) * (1 - epoch / epochs)

    def normalize(self, weights):
        return torch.abs(weights) / torch.sum(torch.abs(weights))

    def prune(self, amount):

        # New sizes: prune all except last layer
        new_layer_sizes = [self.layer_sizes[0]]
        for i, size in enumerate(self.layer_sizes[1:], 1):
            if i == len(self.layer_sizes) - 1:
                new_layer_sizes.append(size)  # keep last layer size
            else:
                new_layer_sizes.append(max(1, int(size * amount)))

        pruned_model = BaseNetwork(new_layer_sizes)

        for i in range(len(new_layer_sizes) - 1):
            old_layer = self.layers[2 * i]
            new_layer = pruned_model.layers[2 * i]

            if i == len(new_layer_sizes) - 2:
                # Last linear layer: copy full weights/biases
                new_layer.weight.data = old_layer.weight.data[:, :new_layer_sizes[i]].clone()
                new_layer.bias.data = old_layer.bias.data.clone()
            else:
                # Prune weights/biases
                out_size = new_layer_sizes[i + 1]
                in_size = new_layer_sizes[i]
                new_layer.weight.data = old_layer.weight.data[:out_size, :in_size].clone()
                new_layer.bias.data = old_layer.bias.data[:out_size].clone()

        return pruned_model

