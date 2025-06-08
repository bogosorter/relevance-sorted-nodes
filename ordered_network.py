import torch
from torch import nn

class OrderedNetwork(nn.Module):
    name = 'ordered_network'

    def __init__(self, alpha, beta):
        super().__init__()

        self.alpha = alpha
        self.beta = beta

        self.penalties = [
            torch.tensor([self.beta * i ** self.alpha for i in range(128)]),
            torch.tensor([self.beta * i ** self.alpha for i in range(64)])
        ]

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
        total = 0

        # Compute regularization loss for first layer
        a = self.penalties[0][None, :] @ self.normalize(self.layers[2].weight).T
        total += torch.sum(a)


        # Compute regularization loss for second layer
        a = self.penalties[1][None, :] @ self.normalize(self.layers[4].weight).T
        total += torch.sum(a)

        return total / 2 * (1 - epoch / epochs)

    def normalize(self, weights):
        return torch.abs(weights) / torch.sum(torch.abs(weights))

    def prune(self, amount):
        n1 = int(128 * (1 - amount))  # number of neurons to prune in first hidden layer
        n2 = int(64 * (1 - amount))   # number of neurons to prune in second hidden layer

        # grab original layers
        L1, L2, L3 = self.layers[0], self.layers[2], self.layers[4]
        W1, b1 = L1.weight.data, L1.bias.data
        W2, b2 = L2.weight.data, L2.bias.data
        W3, b3 = L3.weight.data, L3.bias.data

        # compute new sizes
        o1 = W1.size(0) - n1      # first hidden out
        o2 = W2.size(0) - n2      # second hidden out
        i2 = o1                   # second hidden in
        i3 = o2                   # third hidden in

        # slice weights & biases
        W1p = W1[:o1, :]
        b1p = b1[:o1]

        W2p = W2[:o2, :i2]
        b2p = b2[:o2]

        W3p = W3[:, :i3]
        b3p = b3

        # build new network
        pruned = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, o1), nn.ReLU(),
            nn.Linear(i2, o2),  nn.ReLU(),
            nn.Linear(i3, 10)
        )

        # load pruned params
        
        pruned_layers = pruned[1::2]  # [Linear, ReLU, Linear, …]
        for lin, Wp, bp in zip(pruned_layers, [W1p, W2p, W3p], [b1p, b2p, b3p]):
            lin.weight.data.copy_(Wp)
            lin.bias.data.copy_(bp)

        # wrap into your NeuralNetwork class if needed:
        result = OrderedNetwork(self.alpha, self.beta)
        result.flatten = pruned[0]
        result.layers = pruned[1:]
        return result
