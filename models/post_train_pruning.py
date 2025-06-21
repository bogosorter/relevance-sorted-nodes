import torch
import torch_pruning as tp
from utils.utils import test_loader
from models.base import BaseNetwork

class PostTrainPruningNetwork(BaseNetwork):
    name = 'post_train_pruning'

    def __init__(self, layer_sizes):
        super().__init__(layer_sizes)
    
    def prune(self, amount):
        result = PostTrainPruningNetwork(self.layer_sizes)
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
