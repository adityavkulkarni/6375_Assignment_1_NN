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


# Weight Initialization
def random_list(length, _range=(-0.1, 0.1)):
        min_val, max_val = _range
        return [random.uniform(min_val, max_val) for _ in range(length)]


def xavier_uniform_init(fan_in, fan_out):
    limit = np.sqrt(3 / float(fan_in + fan_out))
    return list(np.random.uniform(low=-limit, high=limit, size=(1,)))


# Print
def print_d(message, debug, end="\n"):
    if debug:
        print(f"DEBUG | {message}", end=end)
