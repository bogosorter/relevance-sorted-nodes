import torch
from torch import nn
import torch_pruning as tp
from utils.utils import test_loader

class GroupLassoNetwork(nn.Module):
    name = 'group_lasso_network'

    def __init__(self, lambda_reg):
        super().__init__()

        self.lambda_reg = lambda_reg

        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

    def regularization(self, epoch, epochs):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() == 2:
                # Group L2 norm across input channels
                reg_loss += torch.norm(param, p=2, dim=1).sum()
        return self.lambda_reg * reg_loss

    def normalize(self, weights):
        return torch.abs(weights) / torch.sum(torch.abs(weights))
    
    def prune(self, amount):
        result = GroupLassoNetwork(self.lambda_reg)
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
