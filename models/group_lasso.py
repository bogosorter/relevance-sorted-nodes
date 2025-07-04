import torch
from torch_pruning import importance as imp
from models.base import BaseNetwork

class GroupLassoNetwork(BaseNetwork):
    name = 'group_lasso_network'

    def __init__(self, layer_sizes, lambda_reg):
        super().__init__(layer_sizes, imp.MagnitudeImportance(p=2), lambda_reg)
        self.lambda_reg = lambda_reg

    def regularization(self, *_):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() == 2:
                reg_loss += torch.norm(param, p=2, dim=1).sum()
        return self.lambda_reg * reg_loss
