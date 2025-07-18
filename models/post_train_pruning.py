from torch_pruning import importance as imp
from models.base import BaseNetwork

class PostTrainPruningNetwork(BaseNetwork):
    name = 'post_train_pruning'

    def __init__(self, layer_sizes):
        super().__init__(layer_sizes, imp.MagnitudeImportance(p=2), True)
