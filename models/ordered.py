import torch
import torch_pruning as tp
from models.base import BaseNetwork

class OrderedNetwork(BaseNetwork):
    name = 'ordered_network'

    def __init__(self, layer_sizes, alpha, beta):
        super().__init__(layer_sizes, OrderedImportance(), alpha, beta)

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
                self.normalize(self.layers[2 * i].weight).T @ self.penalties[i][:, None]
            )
            total += torch.sum(
                self.penalties[i][None, :] @ self.normalize(self.layers[2 * (i + 1)].weight).T
            )

        return total / len(self.penalties) * (1 - epoch / epochs)

    def normalize(self, weights):
        return torch.abs(weights) / torch.sum(torch.abs(weights))

class OrderedImportance(tp.importance.Importance):
    @torch.no_grad()
    def __call__(self, group, **_):
        _, idxs = group[0]
        return torch.tensor(list(range(len(idxs), 0, -1)), dtype=torch.float32)

