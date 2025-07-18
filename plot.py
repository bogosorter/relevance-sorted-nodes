import os
import matplotlib.pyplot as plt
import numpy as np

data = "output"

for filename in os.listdir(data):
    path = os.path.join(data, filename)
    if not os.path.isfile(path):
        continue

    percentages = []
    means = []
    lowers = []
    uppers = []

    with open(path, "r") as f:
        for line in f:
            if not line.startswith("size:"):
                continue

            # Format: "size: 15.00% - 98.12% ]97.80%, 98.44%["

            parts = line.strip().split(':', 1)[1].strip()
            size, rest = parts.split('-', 1)
            size = float(size.strip().strip('%'))

            mean, ci = rest.strip().split(']', 1)
            mean = float(mean.strip().strip('%'))

            ci = ci.strip().replace('%','').replace('[','').replace(']','')
            lower, upper = ci.split(',')

            lower = float(lower.strip())
            upper = float(upper.strip())

            percentages.append(size)
            means.append(mean)
            lowers.append(lower)
            uppers.append(upper)

    percentages = np.array(percentages)
    means = np.array(means)
    lowers = np.array(lowers)
    uppers = np.array(uppers)

    label = os.path.splitext(filename)[0].replace('_', ' ').title()

    plt.plot(percentages, means, label=label)
    plt.fill_between(percentages, lowers, uppers, alpha=0.2)

plt.xlabel("Model Size (%)")
plt.ylabel("Accuracy (%)")
plt.title("MNIST")
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
