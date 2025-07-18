import torch
from models.base import BaseNetwork
import torch_pruning as tp

class OrderedNetwork(BaseNetwork):
    name = 'ordered_network'

    def __init__(self, layer_sizes, alpha, beta):
        super().__init__(layer_sizes, OrderedImportance(), False, alpha, beta)

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

    def regularization(self, *_):
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

class OrderedImportance(tp.importance.Importance):
    @torch.no_grad()
    def __call__(self, group, **_):
        _, idxs = group[0]
        return torch.tensor(list(range(len(idxs), 0, -1)), dtype=torch.float32)
