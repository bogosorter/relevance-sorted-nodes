import torch
import numpy as np
from scipy import stats

def train(model, epochs, optimizer, loss_fn, train_loader):
    model.train()
    for epoch in range(epochs):
        for input, target in train_loader:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target) + model.regularization(epoch, epochs)
            loss.backward()
            optimizer.step()

def test(model, test_loader):
    model.eval()

    correct = 0
    size = len(test_loader.dataset)
    with torch.no_grad():
        for input, target in test_loader:
            output = model(input)
            correct += (output.argmax(1) == target).sum().item()

    return correct / size


# Runs the model on a small subset of the training data to adjust the weights
# after pruning
def adjust_weights(model, steps, optimizer, loss_fn, train_loader):
    model.train()

    for _ in range(steps):
        input, target = next(iter(train_loader))
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

def conf_interval(data, conf):
    mean = np.mean(data)
    sem = stats.sem(data)  # standard error of mean
    margin = sem * stats.t.ppf(0.5 + conf / 2, len(data) - 1)
    lower = mean - margin
    upper = mean + margin
    return mean, lower, upper
