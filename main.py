import os
import torch

from models.ordered import OrderedNetwork
from models.group_lasso import GroupLassoNetwork
from models.post_train_pruning import PostTrainPruningNetwork
from models.random import RandomNetwork
from utils.utils import train, test, adjust_weights, conf_interval

pruning_steps = 32
adjustment_batches = 40
repetitions = 40
confidence = 0.95

layers = [48, 64, 64, 32, 11]
models = [
    lambda: OrderedNetwork(layers, 0.9, 0.003),
    lambda: GroupLassoNetwork(layers, 6e-4, 6e-4),
    lambda: PostTrainPruningNetwork(layers),
    lambda: RandomNetwork(layers)
]
model_epochs = [10, 10, 10, 10]

optimizer = torch.optim.Adam
criterion = torch.nn.CrossEntropyLoss()

for model, epochs in zip(models, model_epochs):

    results = [[] for _ in range(pruning_steps)]

    for run in range(repetitions):
        instance = model()
        print(f'{instance.name}: {run + 1}/{repetitions}')

        train(instance, epochs, optimizer(instance.parameters()), criterion)

        for i in range(pruning_steps):
            amount = (i + 1) / pruning_steps
            pruned = instance.prune(amount)
            adjust_weights(pruned, adjustment_batches, optimizer(pruned.parameters()), criterion)

            acc = test(pruned)
            results[i].append(acc)

    os.makedirs('output', exist_ok=True)

    with open(f'output/{instance.name}.txt', 'w') as f:
        for i, acc_list in enumerate(results):
            size = (i + 1) / pruning_steps
            mean, lower, upper = conf_interval(acc_list, confidence)
            f.write(f'size: {size * 100:.2f}% - {mean * 100:.2f}% ]{lower * 100:.2f}%, {upper * 100:.2f}%[\n')
