from torch_pruning import importance as imp
from models.base import BaseNetwork

class PostTrainPruningNetwork(BaseNetwork):
    name = 'post_train_pruning'

    def __init__(self, architecture):
        super().__init__(architecture, imp.MagnitudeImportance(p=2), True)
