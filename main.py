import os
import torch

from models.ordered import OrderedNetwork
from models.group_lasso import GroupLassoNetwork
from models.post_train_pruning import PostTrainPruningNetwork
from models.random import RandomNetwork
from utils.utils import train, test, adjust_weights, conf_interval
from utils.datasets import load_mnist, load_fmnist, load_cifar10, load_covertype, load_sdd

# Statistics parameters
repetitions = 5
confidence = 0.95
pruning_steps = 32

# Training parameters

epochs = 5
adjustment_batches = 0
train_loader, test_loader = load_sdd()

optimizer = torch.optim.Adam
criterion = torch.nn.CrossEntropyLoss()

architecture = [48, 64, 32, 11]
models = [
    lambda: OrderedNetwork(architecture, train_loader, 0.9, 5e-3),
    lambda: GroupLassoNetwork(architecture, train_loader, 4e-4,4e-4),
    lambda: PostTrainPruningNetwork(architecture, train_loader),
    lambda: RandomNetwork(architecture, train_loader)
]

# Run the experiment

for model in models:
    results = [[] for _ in range(pruning_steps)]

    for run in range(repetitions):
        instance = model()
        print(f'{instance.name}: {run + 1}/{repetitions}')

        train(instance, epochs, optimizer(instance.parameters()), criterion, train_loader)

        for i in range(pruning_steps):
            model_size = (i + 1) / pruning_steps
            pruned = instance.prune(model_size)
            adjust_weights(pruned, adjustment_batches, optimizer(pruned.parameters()), criterion, train_loader)

            result = test(pruned, test_loader)
            results[i].append(result)

    os.makedirs('output', exist_ok=True)

    with open(f'output/{instance.name}.txt', 'w') as f:
        for i, acc_list in enumerate(results):
            size = (i + 1) / pruning_steps
            mean, lower, upper = conf_interval(acc_list, confidence)
            f.write(f'size: {size * 100:.2f}% - {mean * 100:.2f}% ]{lower * 100:.2f}%, {upper * 100:.2f}%[\n')
