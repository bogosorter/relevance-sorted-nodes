import torch
from torch import nn
from models.base import BaseNetwork

class OrderedNetwork2(BaseNetwork):
    name = 'ordered_network_2'

    def __init__(self, layer_sizes, alpha, beta):
        super().__init__(layer_sizes)

        self.alpha = alpha
        self.beta = beta

        self.penalties = []
        for i in range(1, len(layer_sizes) - 1):
            self.penalties.append(
                torch.tensor([j ** self.alpha for j in range(1, layer_sizes[i] + 1)])
            )
        
        self.scale_weights_by_penalty()

        for penalty in self.penalties: penalty *= self.beta
    
    def scale_weights_by_penalty(self):
        with torch.no_grad():
            for i, penalty in enumerate(self.penalties):
                # Scale incoming weights to layer i
                current_layer = self.layers[2 * i]
                for j in range(current_layer.weight.size(0)):
                    current_layer.weight[j] /= penalty[j]

                # Scale outgoing weights from layer i (i.e., weights in layer i+1)
                next_layer = self.layers[2 * (i + 1)]
                for j in range(next_layer.weight.size(1)):
                    next_layer.weight[:, j] /= penalty[j]
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

    def regularization(self, epoch, epochs):
        total = 0

        for i in range(len(self.penalties)):
            total += torch.sum(
                self.normalize(self.layers[2 * i].weight).T @ self.penalties[i][:, None]
            )
            total += torch.sum(
                self.penalties[i][None, :] @ self.normalize(self.layers[2 * (i + 1)].weight).T
            )

        return total / len(self.penalties)

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

