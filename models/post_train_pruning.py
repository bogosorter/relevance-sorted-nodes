import torch
from torch import nn
import torch_pruning as tp
from utils.utils import test_loader

class PostTrainPruningNetwork(nn.Module):
    name = 'post_train_pruning'

    def __init__(self):
        super().__init__()

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
        return 0

    def normalize(self, weights):
        return torch.abs(weights) / torch.sum(torch.abs(weights))
    
    def prune(self, amount):
        result = PostTrainPruningNetwork()
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
