# Neural Networks with Relevance-Sorted Nodes

This repository contains the code developed during my _Gulbenkian Novos Talentos_ scholarship, under the supervision of professor Alípio Jorge.

We present a method for the training of fully connected feedforward neural networks in which nodes are ranked by significance within each layer through weight manipulation. This ranking enables a structured pruning strategy that removes the least significant nodes first. More details can be found in the [draft paper](./paper.pdf).

This work was later incorporated by Guilherme Manuel into his work [Ordered-LRP: Structured Sparsity Patterns from Unstructured Pruning](https://link.springer.com/chapter/10.1007/978-3-032-41086-3_36), whose [repository](https://github.com/Guilemas/Ordered_LRP) is also available.

## Experiments

This repository contains an implementation of the proposed method, as well as the methods it is compared with, and runs evaluations on different datasets. In each evaluation, the full model is trained and then pruned to a certain fraction on its initial size, and the accuracy is measured.

## Installing

To install, create a Python environment and install the required packages.

> The commands below are intended for a Linux environment, but they should work on other OSs as long as `python3` is replaced with `python` and `pip3` is replaced with `pip`.

```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

Additionally, the [SDD dataset](https://archive.ics.uci.edu/dataset/325/dataset+for+sensorless+drive+diagnosis) should be downloaded and placed under `data/sdd/`.

## Running

`main.py` is the script responsible for running the evaluations and storing the results under `output/`:

```
python3 main.py
```

To create a plot of the results, use `plot.py`:

```
python3 plot.py
```

The dataset can be swapped by changing line 20 on `main.py` from `load_sdd()` on line 20 of `main.py` to `load_covertype()`, `load_mnist()`, `load_fmnist()` or `load_cifar10()`. This will most likely require changing the input layer of the `architecture`, on line 25. Other parameters such as the number of repetitions, epochs and adjustment batches can (and should) also be changed on `main.py`. Each kind of model can also be initialized with its own parameters. Reference Table 1 and Table 2 on the [draft paper](./paper.pdf) for the values used on our experiments.
