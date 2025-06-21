import torch
import torch_pruning as tp
from utils.utils import test_loader
from models.base import BaseNetwork

class GroupLassoNetwork(BaseNetwork):
    name = 'group_lasso_network'

    def __init__(self, layer_sizes, lambda_reg):
        super().__init__(layer_sizes)
        self.lambda_reg = lambda_reg

    def regularization(self, epoch=None, epochs=None):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() == 2:
                reg_loss += torch.norm(param, p=2, dim=1).sum()
        return self.lambda_reg * reg_loss

    def prune(self, amount):
        result = GroupLassoNetwork(self.layer_sizes, self.lambda_reg)
        result.load_state_dict(self.state_dict())
        
        example_inputs, _ = next(iter(test_loader))
        example_inputs = example_inputs[:1]
        ignored_layers = [result.layers[-1]]  # Skip final classification layer
        importance = tp.importance.MagnitudeImportance(p=2)

        pruner = tp.pruner.BasePruner(
            result,
            example_inputs,
            importance = importance,
            pruning_ratio = 1 - amount,
            ignored_layers=ignored_layers
        )
        pruner.step()

        return result
