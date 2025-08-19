import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats

# Load the Forest CoverType dataset
cov = fetch_covtype()
X = cov.data.astype("float32")
y = cov.target.astype("int64") - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
test_loader = DataLoader(test_data, batch_size=512, shuffle=False)

def train(model, epochs, optimizer, loss_fn):
    model.train()
    for epoch in range(epochs):
        for input, target in train_loader:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target) + model.regularization(epoch, epochs)
            loss.backward()
            optimizer.step()

def test(model):
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
def adjust_weights(model, steps, optimizer, loss_fn):
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
