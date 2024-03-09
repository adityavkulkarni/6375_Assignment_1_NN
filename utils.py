import sys
import time

import numpy as np
import random


# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return max(0, x)


def identity(x):
    return x


# Math
def sigmoid_prime(x):
    return x * (1 - x)


def tanh_prime(x):
    return 1 - (x ** 2)


def relu_prime(x):
    return 1 if x > 0 else 0


# Other
def random_list(length, _range=(-0.1, 0.1)):
    w = [
        -0.99, -0.98, -0.97, -0.96, -0.95, -0.94, -0.93, -0.92, -0.91, -0.90,
        -0.89, -0.88, -0.87, -0.86, -0.85, -0.84, -0.83, -0.82, -0.81, -0.80,
        -0.79, -0.78, -0.77, -0.76, -0.75, -0.74, -0.73, -0.72, -0.71, -0.70,
        -0.69, -0.68, -0.67, -0.66, -0.65, -0.64, -0.63, -0.62, -0.61, -0.60,
        -0.59, -0.58, -0.57, -0.56, -0.55, -0.54, -0.53, -0.52, -0.51, -0.50,
        -0.49, -0.48, -0.47, -0.46, -0.45, -0.44, -0.43, -0.42, -0.41, -0.40,
        -0.39, -0.38, -0.37, -0.36, -0.35, -0.34, -0.33, -0.32, -0.31, -0.30,
        -0.29, -0.28, -0.27, -0.26, -0.25, -0.24, -0.23, -0.22, -0.21, -0.20,
        -0.19, -0.18, -0.17, -0.16, -0.15, -0.14, -0.13, -0.12, -0.11, -0.10,
        0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
        0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30,
        0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40,
        0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50,
        0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60,
        0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70,
        0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80,
        0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90,
        0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99
    ]
    return random.sample(w, length)


# Print
def print_d(message, debug, end="\n"):
    if debug:
            print(f"DEBUG | {message}", end=end)

def show(j, count, size=60, prefix="DEBUG | "):
    x = int(size * j / count)
    out = sys.stdout
    print(f"{prefix}[{u'█' * x}{('.' * (size - x))}] {j}/{count} ", end='\r', file=out,
          flush=True)
def progressbar(it, prefix="", size=60):
    count = len(it)
    start = time.time()
    out = sys.stdout

    """def show(j):
        x = int(size * j / count)
        remaining = ((time.time() - start) / j) * (count - j)

        mins, sec = divmod(remaining, 60)
        time_str = f"{int(mins):02}:{sec:05.2f}"

        print(f"{prefix}[{u'█' * x}{('.' * (size - x))}] {j}/{count} Est wait {time_str}", end='\r', file=out,
              flush=True)"""

    for i, item in enumerate(it):
        yield item
        show(i + 1, count=count, prefix=prefix)
    print("", flush=True, file=out)
