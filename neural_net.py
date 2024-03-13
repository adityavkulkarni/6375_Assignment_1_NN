import sys
import time
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from utils import *


class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.classes_ = None
        self.label_encoder = LabelEncoder()

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return:
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_
        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if str(unique_item) not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x == unique_item else x for x in new_data_list]
        return self.label_encoder.transform(new_data_list)


class Neuron:
    def __init__(self, activation_function, name=None, bias=0):
        """
        Class for a single neuron
        :param activation_function: Activation function: (sigmoid|tanh|relu)
        :param name: Name of neuron (optional)
        """
        # Refers to activation functions from utils.py
        self.activation_function = globals()[activation_function]
        self.name = name
        self.bias = bias  # bias input
        self.net = 0
        self.y = 0

    def output(self, x):
        """
        Function for generating neuron output
        :param x: Input vector - [w.x]
        :return: Output value - activation(sum(wx))
        """
        net = 0
        for i in range(len(x)):
            net += x[i]  # Assumption: input will already be of format: w.x
        self.net = net
        self.y = self.activation_function(net)
        return self.y


class NeuralNet:
    def __init__(self, activation_function="sigmoid", hidden_layer_size=None):
        """
        Initialize the NeuralNet class.
        :param activation_function: Activation function for neurons (default is "sigmoid").
        :param hidden_layer_size: Size of hidden layers (default is [9, 6, 4]).
        """
        self.gradient = None
        self.optimizer = None
        self.learning_rate = None
        if hidden_layer_size is None:
            hidden_layer_size = [9, 6, 4]
        self.activation_function = activation_function
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_count = len(self.hidden_layer_size)
        self.neuron_count = 1
        self.input_layer = []
        self.hidden_layer = []
        self.output_layer = []
        self.W = {}
        self.training_data = None
        self.test_data = None
        self.scaler = None
        self.loss_viz = []
        self.acc_viz = []

    @staticmethod
    def __error(o, t):
        """
        Calculate mean squared error (MSE) between predicted output and actual output.
        :param o: Predicted output.
        :param t: Actual output.
        :return: Mean squared error.
        """
        # Single sample
        if type(o) is float or type(o) is int or type(o) is numpy.float64:
            return 0.5 * (abs(t - o) ** 2)
        # List of samples
        e = 0
        for i in range(len(o)):
            e += abs(t[i] - o[i]) ** 2
        return 0.5 * e / len(t)

    @staticmethod
    def __performance(y_true, y_pred):
        """
        Calculate accuracy between true labels and predicted labels.
        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :return: Accuracy score.
        """
        acc = accuracy_score(y_true, y_pred)
        return acc

    @staticmethod
    def __get_batches(df, batch_size=50):
        """
        Create batches from the given DataFrame.
        :param df: DataFrame to create batches from.
        :param batch_size: Size of each batch (default is 50).
        :return: List of numpy arrays containing batches.
        """
        # Create batches
        num_batches = len(df) // batch_size
        batches = [df.iloc[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
        # If there are remaining samples, add them to the last batch
        if len(df) % batch_size != 0:
            batches.append(df.iloc[num_batches * batch_size:])
        np_batches = []
        for batch in batches:
            data = batch.to_numpy()
            x = np.array([sublist[:-1] for sublist in data])
            y = np.array([sublist[-1] for sublist in data])
            np_batches.append([x, y])
        return np_batches

    def __create_input_layer(self):
        """
        Create the input layer with a specified number of neurons based on the training data.
        :return:
        """
        self.input_layer_size = len(self.training_data.columns) - 1
        for i in range(self.input_layer_size):
            self.input_layer.append(
                Neuron(activation_function="identity",
                       name=self.neuron_count))
            self.W[f"w_{self.neuron_count}"] = 1
            self.neuron_count += 1

    def __create_hidden_layer(self):
        """
        Create the default hidden layer(s) with specified parameters.
        :return:
        """
        prev_layer = self.input_layer
        for hidden_cnt in range(self.hidden_layer_count):
            hidden_layer = []
            for i in range(self.hidden_layer_size[hidden_cnt]):
                neuron = Neuron(activation_function=self.activation_function,
                                name=self.neuron_count, bias=1)
                hidden_layer.append(neuron)
                fan_out = self.hidden_layer_size[hidden_cnt]
                for cnt in range(len(prev_layer)):
                    p = prev_layer[cnt].name
                    # Initialise weight for each connection
                    self.W[f"w_{self.neuron_count}_{p}"] = xavier_uniform_init(fan_in=len(prev_layer),
                                                                               fan_out=fan_out)
                # Initialise bias weight for each neuron
                self.W[f"w_{self.neuron_count}_b"] = 1
                self.neuron_count += 1
            self.hidden_layer.append(hidden_layer)
            prev_layer = hidden_layer

    def __create_output_layer(self):
        """
        create an output layer with single neuron
        :return:
        """
        self.output_layer = [
            Neuron(activation_function="sigmoid",
                   name=self.neuron_count, bias=1)
        ]
        self.W[f"w_{self.neuron_count}_b"] = 1
        for i in range(len(self.hidden_layer[-1])):
            p = self.hidden_layer[-1][i].name
            # Initialise weight
            self.W[f"w_{self.neuron_count}_{p}"] = xavier_uniform_init(fan_in=len(self.hidden_layer[::-1][0]),
                                                                       fan_out=1)
        self.neuron_count += 1

    def train(self, training_data, test_data, learning_rate=0.5, epochs=100,
              optimizer="None", gradient="batch", plot_suffix=""):
        """
        Train the neural network using the provided training data.
        :param training_data: Training data.
        :param test_data: Test data.
        :param learning_rate: Learning rate for training (default is 0.5).
        :param epochs: Number of epochs for training (default is 100).
        :param optimizer: Optimizer for training (default is "None").
        :param gradient: Gradient descent type (default is "batch").
        :param plot_suffix: Suffix for plotting.
        :return: Tuple containing test loss, test accuracy, and training execution time.
        """
        print(f"Training neural network with:\n"
              f"- Learning rate = {learning_rate}\n"
              f"- Activation function = {self.activation_function}\n"
              f"- Gradient descent = {gradient}\n"
              f"- Optimizer = {optimizer}")
        start_time = datetime.now()
        # Init
        v_t = {}
        self.loss_viz = []
        self.acc_viz = []
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.gradient = gradient
        # Refers to derivative of activation function from uutils.py
        activation_function_prime = globals()[f"{self.activation_function}_prime"]
        self.training_data = training_data
        self.test_data = test_data
        # Create Neural network structure
        self.__create_input_layer()
        self.__create_hidden_layer()
        self.__create_output_layer()
        # Batch implementation
        batch_size = len(self.training_data)  # Batch
        if gradient == "minibatch":
            batch_size = 50
        elif gradient == "stochastic":
            batch_size = 1
        batches = self.__get_batches(training_data, batch_size=batch_size)
        # Scaling data
        training_data = self.training_data.to_numpy()
        x = np.array([sublist[:-1] for sublist in training_data])
        self.scaler = StandardScaler()
        self.scaler.fit(x)  # Fit to whole data
        # Initial accuracy of untrained model
        t, t_acc = self.test()
        print(f"Test Accuracy: {t_acc}")
        # Epochs
        for epoch in range(epochs):
            op = []
            # Batch selection
            x, y = batches[np.random.randint(0, len(batches))]
            x = self.scaler.transform(x)
            for i in range(len(x)):
                # Gradient Evaluation
                # Forward Pass
                o = self.__predict(x[i])
                op.append(o)
                # Printing
                q = int(40 * i / len(x))
                print(f"\rEpoch: {epoch + 1}{' '*(3 - len(str(epoch + 1)))} "
                      f"[{u'=' * q}{('.' * (40 - q))}] {i + 1}/{len(x)}",
                      end='', file=sys.stdout, flush=True)
                # Backward Pass
                d = {}
                dw = {}
                # Output layer d
                output_neuron = self.output_layer[0]
                d[f"d_{output_neuron.name}"] = (sigmoid_prime(output_neuron.y) *
                                                (y[i] - output_neuron.y))
                # Output layer -> hidden layer gradient and hidden layer d
                for hidden_neuron in self.hidden_layer[::-1][0]:
                    d[f"d_{hidden_neuron.name}"] = (activation_function_prime(hidden_neuron.y) *
                                                    (self.W[f"w_{output_neuron.name}_{hidden_neuron.name}"] *
                                                     d[f"d_{output_neuron.name}"]))
                    dw[f"w_{output_neuron.name}_{hidden_neuron.name}"] = (learning_rate *
                                                                          d[f"d_{output_neuron.name}"] *
                                                                          hidden_neuron.y)
                    dw[f"w_{hidden_neuron.name}_b"] = (learning_rate * d[f"d_{output_neuron.name}"] *
                                                       output_neuron.bias)
                # Outer hidden layer -> inner hidden layer gradient and inner hidden layer d
                for layer_index in range(self.hidden_layer_count - 1, 0, -1):
                    current_layer = self.hidden_layer[layer_index]
                    next_layer = self.hidden_layer[layer_index - 1]
                    for inner_neuron in next_layer:
                        s = 0
                        for outer_neuron in current_layer:
                            s += (d[f"d_{outer_neuron.name}"] *
                                  self.W[f"w_{outer_neuron.name}_{inner_neuron.name}"])
                            dw[f"w_{outer_neuron.name}_{inner_neuron.name}"] = (learning_rate *
                                                                                d[f"d_{outer_neuron.name}"] *
                                                                                inner_neuron.y)
                            dw[f"w_{inner_neuron.name}_b"] = (learning_rate * d[f"d_{outer_neuron.name}"] *
                                                              inner_neuron.bias)
                        d[f"d_{inner_neuron.name}"] = activation_function_prime(inner_neuron.y) * s
                """for inner_neuron in self.hidden_layer[::-1][1]:
                    s = 0
                    for outer_neuron in self.hidden_layer[::-1][0]:
                        s += (d[f"d_{outer_neuron.name}"] *
                              self.W[f"w_{outer_neuron.name}_{inner_neuron.name}"])
                        dw[f"w_{outer_neuron.name}_{inner_neuron.name}"] = (learning_rate *
                                                                            d[f"d_{outer_neuron.name}"] *
                                                                            inner_neuron.y)
                        dw[f"w_{inner_neuron.name}_b"] = (learning_rate * d[f"d_{outer_neuron.name}"] *
                                                          inner_neuron.bias)
                    d[f"d_{inner_neuron.name}"] = activation_function_prime(inner_neuron.y) * s
                if self.hidden_layer_count == 3:
                    for inner_neuron in self.hidden_layer[::-1][2]:
                        s = 0
                        for outer_neuron in self.hidden_layer[::-1][1]:
                            s += (d[f"d_{outer_neuron.name}"] *
                                  self.W[f"w_{outer_neuron.name}_{inner_neuron.name}"])
                            dw[f"w_{outer_neuron.name}_{inner_neuron.name}"] = (learning_rate *
                                                                                d[f"d_{outer_neuron.name}"] *
                                                                                inner_neuron.y)
                            dw[f"w_{inner_neuron.name}_b"] = (learning_rate * d[f"d_{outer_neuron.name}"] *
                                                              inner_neuron.bias)
                        d[f"d_{inner_neuron.name}"] = activation_function_prime(inner_neuron.y) * s"""
                # Hidden inner layer -> input layer gradient
                for input_neuron in self.input_layer:
                    for hidden_neuron in self.hidden_layer[0]:
                        dw[f"w_{hidden_neuron.name}_{input_neuron.name}"] = (learning_rate *
                                                                             d[f"d_{hidden_neuron.name}"] *
                                                                             input_neuron.y)
                        dw[f"w_{hidden_neuron.name}_b"] = (learning_rate *
                                                           d[f"d_{hidden_neuron.name}"] * hidden_neuron.bias)
                # Gradient calculation over
                # Updating weights
                for key in dw:
                    if optimizer == "momentum":
                        # Store current gradient in v. momentum=0.9
                        v_t[key] = 0.9 * v_t[key] + dw[key] if key in v_t else dw[key]
                        self.W[key] += v_t[key]
                    else:
                        self.W[key] += dw[key]

            # Printing and Plotting
            s = self.__error(op, y)
            t, t_acc = self.test()
            print(f"\rEpoch: {epoch + 1}{' '*(3 - len(str(epoch+1)))} " 
                  f"[{u'=' * 40}] {len(x)}/{len(x)}   "
                  f"Training Loss: {'{:6.5f}'.format(s)}   "
                  f"Val Loss: {'{:6.5f}'.format(t)}\n",
                  end='', file=sys.stdout, flush=True)
            self.loss_viz.append((epoch+1, s, t))
            o1 = []
            for v in op:
                o1.append(1 if v > 0.6 else 0)
            self.acc_viz.append((epoch+1, self.__performance(y, o1), t_acc))
        exec_time = datetime.now() - start_time
        end_time = 'Time elapsed (hh:mm:ss) {}'.format(exec_time)
        print(f"Training Accuracy: {self.acc_viz[-1][1]}\n"
              f"Test Accuracy: {self.acc_viz[-1][2]}\n"
              f"Training Time: {end_time.split('.')[0]}\n")
        self.plot_loss(self.loss_viz, self.acc_viz, f"{self.activation_function}_{optimizer}_{plot_suffix}")
        return t, t_acc, exec_time.total_seconds()

    def plot_loss(self, loss_viz, acc_viz, suffix=""):
        """
        Plot loss and accuracy graphs based on training and validation data.
        :param loss_viz: List of tuples containing epoch, training loss, and validation loss.
        :param acc_viz: List of tuples containing epoch, training accuracy, and validation accuracy.
        :param suffix: Suffix for plot filenames.
        """
        df = pd.DataFrame(loss_viz, columns=['Epochs', 'Training loss', 'Validation loss'])
        fig1 = plt.figure(figsize=(12, 7))
        ax = fig1.add_subplot(1, 1, 1)
        ax.plot(df["Epochs"], df["Training loss"])
        ax.plot(df["Epochs"], df["Validation loss"])
        plt.legend(["Training Loss", "Validation Loss"])
        plt.title("Loss vs Epochs")
        txt = (f"Dataset: {suffix.split('_')[-1]} | Learning Rate: {self.learning_rate} | "
               f"Activation Function: {self.activation_function} | Optimizer: {self.optimizer} | Gradient: {self.gradient}")
        plt.figtext(0.5, 0.01,
                    txt + "\nFinal Loss: Training={:.2f} Validation={:.2f}".format(loss_viz[-1][1], loss_viz[-1][2]),
                    wrap=True, horizontalalignment='center', fontsize=10)
        fig1.savefig(f"./out/{suffix.split('_')[-1]}/loss_{suffix}_{time.time()}.png")

        df = pd.DataFrame(acc_viz, columns=['Epochs', 'Training Accuracy', 'Validation Accuracy'])
        fig2 = plt.figure(figsize=(12, 7))
        ax = fig2.add_subplot(1, 1, 1)
        ax.plot(df["Epochs"], df["Training Accuracy"])
        ax.plot(df["Epochs"], df["Validation Accuracy"])
        plt.legend(["Training Accuracy", "Validation Accuracy"])
        plt.title("Accuracy vs Epochs")
        plt.figtext(0.5, 0.01,
                    txt + "\nFinal Accuracy: Training={:.2f} Validation={:.2f}".format(acc_viz[-1][1], acc_viz[-1][2]),
                    wrap=True, horizontalalignment='center', fontsize=10)
        fig2.savefig(f"./out/{suffix.split('_')[-1]}/acc_{suffix}_{time.time()}.png")
        plt.close()

    def test(self):
        """
        Test the neural network on the test/validation set and calculate loss and accuracy.
        :return: Tuple containing test loss and accuracy.
        """
        test_data = self.test_data.to_numpy()
        x = np.array([sublist[:-1] for sublist in test_data])
        y = np.array([sublist[-1] for sublist in test_data])
        x = self.scaler.transform(x)
        op1 = []
        op = []
        for i in range(len(x)):
            op.append(self.__predict(x[i]))
            op1.append(1 if self.__predict(x[i]) > 0.5 else 0)
        acc = self.__performance(y_true=y, y_pred=op1)
        return self.__error(op, y), acc

    def __predict(self, x):
        """
        Predict the output for a given input.
        :param x: Input data.
        :return: Predicted output.
        """
        if type(x) is pd.Series:  # For individual sample preprocess
            x = x.to_numpy()
            x = self.scaler.transform([x]).reshape((-1, 1))
        ip_o = []
        # input layer -> hidden layer
        for i in range(len(self.input_layer)):
            ip_o.append(self.input_layer[i].output([x[i]]))
        hidden_o = ip_o
        # hidden layer -> output layer For 1 hidden layer
        if self.hidden_layer_size == 1:
            for hidden_cnt in range(self.hidden_layer_count):
                hidden_o = []
                for i in range(self.hidden_layer_size[hidden_cnt]):
                    current_neuron = self.hidden_layer[hidden_cnt][i]
                    hidden_ip = [ip_o[j] * self.W[f"w_{current_neuron.name}_{j+1}"]
                                 for j in range(len(self.input_layer))]
                    hidden_ip.append(current_neuron.bias * self.W[f"w_{current_neuron.name}_b"])
                    hidden_o.append(current_neuron.output(hidden_ip) *
                                    self.W[f"w_{self.output_layer[0].name}_{current_neuron.name}"])
        else:  # hidden layer -> hidden layer
            prev_layer = self.input_layer
            hidden_o = [ip_o, [], [], []]
            for hidden_cnt in range(self.hidden_layer_count):
                for i in range(self.hidden_layer_size[hidden_cnt]):
                    current_neuron = self.hidden_layer[hidden_cnt][i]
                    hidden_ip = [hidden_o[hidden_cnt][j] * self.W[f"w_{current_neuron.name}_{prev_layer[j].name}"]
                                 for j in range(len(prev_layer))]
                    hidden_ip.append(current_neuron.bias * self.W[f"w_{current_neuron.name}_b"])
                    hidden_o[hidden_cnt+1].append(current_neuron.output(hidden_ip))
                    if hidden_cnt == len(self.hidden_layer_size) - 1:
                        hidden_o[hidden_cnt+1].append(current_neuron.output(hidden_ip) *
                                                      self.W[f"w_{self.output_layer[0].name}_{current_neuron.name}"])
                prev_layer = self.hidden_layer[hidden_cnt]
        hidden_o = hidden_o[hidden_cnt + 1]
        # hidden layer -> output layer
        hidden_o.append(self.output_layer[0].bias + self.W[f"w_{self.output_layer[0].name}_b"])
        o = self.output_layer[0].output(hidden_o)
        return o

    def predict(self, x):
        """
        Predict the output label for a given input.
        :param x: Input data.
        :return: Predicted label (binary).
        """
        return 1 if self.__predict(x) > 0.5 else 0


if __name__ == "__main__":
    n1 = Neuron(activation_function="tanh", bias=2,
                name="sample neuron")
    n1.output([2, 3])
