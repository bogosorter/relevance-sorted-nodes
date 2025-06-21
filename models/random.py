from torch import nn
from models.base import BaseNetwork

class RandomNetwork(BaseNetwork):
    name = 'random_network'

    def __init__(self, layer_sizes):
        super().__init__(layer_sizes)

    def prune(self, amount):

        # New sizes: prune all except last layer
        new_layer_sizes = [self.layer_sizes[0]]
        for i, size in enumerate(self.layer_sizes[1:], 1):
            if i == len(self.layer_sizes) - 1:
                new_layer_sizes.append(size)  # keep last layer size
            else:
                new_layer_sizes.append(max(1, int(size * amount)))

        pruned_model = BaseNetwork(new_layer_sizes)

        for i in range(len(new_layer_sizes) - 1):
            old_layer = self.layers[2 * i]
            new_layer = pruned_model.layers[2 * i]

            if i == len(new_layer_sizes) - 2:
                # Last linear layer: copy full weights/biases
                new_layer.weight.data = old_layer.weight.data[:, :new_layer_sizes[i]].clone()
                new_layer.bias.data = old_layer.bias.data.clone()
            else:
                # Prune weights/biases
                out_size = new_layer_sizes[i + 1]
                in_size = new_layer_sizes[i]
                new_layer.weight.data = old_layer.weight.data[:out_size, :in_size].clone()
                new_layer.bias.data = old_layer.bias.data[:out_size].clone()

        return pruned_model
