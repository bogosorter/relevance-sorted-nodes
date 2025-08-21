import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from os import path

def load_mnist():
    train_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_data = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    return train_loader, test_loader

def load_fmnist():
    train_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    return train_loader, test_loader

def load_cifar10():
    train_data = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_data = datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    return train_loader, test_loader

def load_covertype():
    cov = fetch_covtype()
    X = cov.data.astype("float32")
    y = cov.target.astype("int64") - 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False)

    return train_loader, test_loader

def load_sdd():
    if not path.exists('data/sdd/Sensorless_drive_diagnosis.txt'):
        raise FileNotFoundError('Dataset not found. Please download the SDD dataset from https://archive.ics.uci.edu/dataset/325/dataset+for+sensorless+drive+diagnosis and place it under `data/sdd/` directory.')

    data = np.loadtxt("./data/sdd/Sensorless_drive_diagnosis.txt")
    X = data[:, :-1].astype("float32")
    y = data[:, -1].astype("int64") - 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    return train_loader, test_loader
