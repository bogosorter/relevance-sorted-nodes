from torch_pruning import importance as imp
from models.base import BaseNetwork

class PostTrainPruningNetwork(BaseNetwork):
    name = 'post_train_pruning'

    def __init__(self, architecture, train_loader):
        super().__init__(architecture, train_loader, imp.MagnitudeImportance(p=2), True)
