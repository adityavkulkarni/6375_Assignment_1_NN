import numpy as np


def sigmoid(x):
    pass


def tanh(x):
    pass


def relu(x):
    pass


class Neuron:
    def __init__(self, activation_function, bias):
        self.activation_function = activation_function
        self.weights = []
        self.bias = bias

    def output(self, x):
        pass

    def update_weights(self, dw):
        pass


class NeuralNet:
    def __init__(self, training_data, test_data,
                 learning_rate, epochs):
        self.training_data = training_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.epochs = epochs
