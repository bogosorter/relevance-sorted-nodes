import os
import torch

from models.ordered import OrderedNetwork
from models.group_lasso import GroupLassoNetwork
from models.post_train_pruning import PostTrainPruningNetwork
from models.random import RandomNetwork
from utils.utils import train, test, adjust_weights, conf_interval
from utils.datasets import load_mnist, load_fmnist, load_cifar10, load_covertype, load_sdd

pruning_steps = 32
adjustment_batches = 0
repetitions = 5
confidence = 0.95
train_loader, test_loader = load_covertype()

layers = [54, 64, 32, 7]
models = [
    lambda: OrderedNetwork(layers, train_loader, 0.9, 0.08),
    lambda: GroupLassoNetwork(layers, train_loader, 4e-2 ,4e-2),
    lambda: PostTrainPruningNetwork(layers, train_loader),
    lambda: RandomNetwork(layers, train_loader)
]
model_epochs = [3, 3, 3, 3]

optimizer = torch.optim.Adam
criterion = torch.nn.CrossEntropyLoss()

for model, epochs in zip(models, model_epochs):

    results = [[] for _ in range(pruning_steps)]

    for run in range(repetitions):
        instance = model()
        print(f'{instance.name}: {run + 1}/{repetitions}')

        train(instance, epochs, optimizer(instance.parameters()), criterion, train_loader)

        for i in range(pruning_steps):
            amount = (i + 1) / pruning_steps
            pruned = instance.prune(amount)
            adjust_weights(pruned, adjustment_batches, optimizer(pruned.parameters()), criterion, train_loader)

            acc = test(pruned, test_loader)
            results[i].append(acc)

    os.makedirs('output', exist_ok=True)

    with open(f'output/{instance.name}.txt', 'w') as f:
        for i, acc_list in enumerate(results):
            size = (i + 1) / pruning_steps
            mean, lower, upper = conf_interval(acc_list, confidence)
            f.write(f'size: {size * 100:.2f}% - {mean * 100:.2f}% ]{lower * 100:.2f}%, {upper * 100:.2f}%[\n')
