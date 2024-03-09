import numpy as np
import random


# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return max(0, x)


# Math
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh_prime(x):
    return 1 - (tanh(x) ** 2)


def relu_prime(x):
    return 1 if x > 0 else 0


# Other
def random_list(length, _range=(0, 0.1)):
    return [random.uniform(*_range) for i in range(length)]


# Print
def print_d(message, debug):
    if debug:
        print(f"DEBUG | {message}")
