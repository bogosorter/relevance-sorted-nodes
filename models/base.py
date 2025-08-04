import torch
from torch import nn
import torch_pruning as tp
from utils.utils import test_loader

class BaseNetwork(nn.Module):
    name = 'base_network'

    def __init__(self, layer_sizes, importance_criterion, global_pruning, *args):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.importance_criterion = importance_criterion
        self.global_pruning = global_pruning
        self.args = args

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

    def regularization(self, *_):
        return 0

    def normalize(self, weights):
        return torch.abs(weights) / torch.sum(torch.abs(weights))

    def prune(self, amount):
        result = self.__class__(self.layer_sizes, *self.args)
        result.load_state_dict(self.state_dict())
        
        example_inputs, _ = next(iter(test_loader))
        example_inputs = example_inputs[:1]
        ignored_layers = [result.layers[-1]]  # Skip final classification layer

        # At exreme pruning levels, global pruning may remove entire layers
        global_pruning = self.global_pruning and amount > 0.1

        pruner = tp.pruner.BasePruner(
            result,
            example_inputs,
            importance = self.importance_criterion,
            pruning_ratio = 1 - amount,
            ignored_layers = ignored_layers,
            global_pruning = global_pruning
        )
        pruner.step()

        return result
