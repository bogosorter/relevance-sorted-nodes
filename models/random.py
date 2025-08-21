from models.base import BaseNetwork
from torch_pruning import importance as imp

class RandomNetwork(BaseNetwork):
    name = 'random_network'

    def __init__(self, architecture):
        super().__init__(architecture, imp.RandomImportance(), False)
