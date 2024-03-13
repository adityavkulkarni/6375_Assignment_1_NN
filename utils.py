import numpy as np
import random


# Activation Functions
def sigmoid(x):
    """
    Sigmoid activation function.
    :param x: Input value.
    :return: Output value after applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    Hyperbolic tangent activation function.
    :param x: Input value.
    :return: Output value after applying the hyperbolic tangent function.
    """
    return np.tanh(x)


def relu(x):
    """
    Rectified Linear Unit (ReLU) activation function.
    :param x: Input value.
    :return: Output value after applying the ReLU function.
    """
    return max(0, x)


def identity(x):
    """
    Identity activation function.
    :param x: Input value.
    :return: Output value equal to the input value.
    """
    return x


def sigmoid_prime(x):
    """
    Derivative of the sigmoid activation function.
    :param x: Input value (output of sigmoid function).
    :return: Output value after applying the derivative of the sigmoid function.
    """
    return x * (1 - x)


def tanh_prime(x):
    """
    Derivative of the hyperbolic tangent activation function.
    :param x: Input value (output of tanh function).
    :return: Output value after applying the derivative of the tanh function.
    """
    return 1 - (x ** 2)


def relu_prime(x):
    """
    Derivative of the ReLU activation function.
    :param x: Input value.
    :return: Output value after applying the derivative of the ReLU function.
    """
    return 1 if x > 0 else 0


# Weight Initialization
def random_list(length, _range=(-0.1, 0.1)):
    """
    Generate a list of random numbers within a specified range.
    :param length: Length of the list to generate.
    :param _range: Range within which the random numbers should fall (default is (-0.1, 0.1)).
    :return: List of random numbers.
    """
    return [random.uniform(*_range) for _ in range(length)]


def xavier_uniform_init(fan_in, fan_out):
    """
    Xavier (Glorot) uniform weight initialization method.
    :param fan_in: Number of input units.
    :param fan_out: Number of output units.
    :return: Randomly initialized weights within the specified range.
    """
    limit = np.sqrt(6 / float(fan_in + fan_out))
    return list(np.random.uniform(low=-limit, high=limit, size=(10,)))[np.random.randint(0, 10)]


def moving_average(data, alpha):
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return ema
