import os
import matplotlib.pyplot as plt

directory = "output"

for filename in os.listdir(directory):
    path = os.path.join(directory, filename)
    if not os.path.isfile(path):
        continue

    percentages = []
    accuracies = []

    with open(path, "r") as f:
        for line in f:
            if ':' not in line:
                continue
            pct_str, acc_str = line.strip().split(':')
            pct = float(pct_str.strip().strip('%'))
            acc = float(acc_str.strip().strip('%'))
            percentages.append(pct)
            accuracies.append(acc)

    label = os.path.splitext(filename)[0].replace('_', ' ').title()
    plt.plot(percentages, accuracies, label=label)

plt.xlabel("Model Size")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Model Size")
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
