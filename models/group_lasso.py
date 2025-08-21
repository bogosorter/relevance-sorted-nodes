import torch
from torch_pruning import importance as imp
from models.base import BaseNetwork

class GroupLassoNetwork(BaseNetwork):
    name = 'group_lasso_network'

    def __init__(self, architecture, alpha, beta):
        super().__init__(architecture, imp.MagnitudeImportance(p=2, group_reduction="gate"), True, alpha, beta)
        self.alpha = alpha
        self.beta = beta

    def regularization(self, *_):
        group_lasso = 0
        l2_weights = 0
        l2_biases = 0

        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() == 2:
                    # group lasso on outgoing weight vectors
                    group_lasso += torch.norm(param, p=2, dim=0).sum()
                # standard L2
                l2_weights += 0.5 * param.norm(p=2).pow(2)
            elif 'bias' in name:
                l2_biases += 0.5 * param.norm(p=2).pow(2)

        return self.alpha * group_lasso + self.beta * (l2_weights + l2_biases)
