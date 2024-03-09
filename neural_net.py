import numpy
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder

from neural_net_exceptions import *
from utils import *


class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.classes_ = None
        self.label_encoder = LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
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
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x == unique_item else x for x in new_data_list]
        return self.label_encoder.transform(new_data_list)


class Neuron:
    def __init__(self, activation_function, name="neuron", bias=0, debug=False):
        """
        Class for a single neuron
        :param activation_function: Activation function: (sigmoid|tanh|relu)
        :param name: Name of neuron (optional)
        :param debug: True for print_ding debug messages (optional, default=False)
        """
        self.activation_function = globals()[activation_function]
        self.name = name
        self.debug = False
        self.bias = bias
        self.net = 0
        self.y = 0

    def output(self, x):
        """
        Function for generating neuron output
        :param x: Input vector
        :return: Output value - activation(sum(wx))
        """
        net = 0
        for i in range(len(x)):
            net += x[i]
        # net += (1 * self.weights[i + 1])
        self.net = net
        self.y = self.activation_function(net)
        print_d(f"Output of {self.name} : {self.y}", self.debug)
        return self.y


class NeuralNet:
    def __init__(self, activation_function="sigmoid", learning_rate=0.5, debug=False,
                 input_layer_size=-1, hidden_layer_size=[6, 3]):
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.debug = debug
        self.input_layer = []
        self.ip_W = []
        self.hidden_layer = []
        self.hidden_W = []
        self.output_layer = []
        self.op_W = []
        self.W = {}
        self.training_data = None
        self.test_data = None
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_count = len(self.hidden_layer_size)

    @staticmethod
    def __error(o, t):
        if type(o) is float or type(t) is float or type(o) is numpy.float64:
            return 0.5 * (abs(o - t) ** 2)
        e = 0
        # check
        for i in range(len(o)):
            e += abs(o[i] - t[i]) ** 2
        return 0.5 * e / len(o)

    def __preprocess_data(self):
        self.training_data.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
        self.test_data.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
        self.encoder_geo = LabelEncoderExt()
        self.encoder_gender = LabelEncoderExt()
        self.encoder_geo.fit(self.training_data["Geography"])
        self.encoder_gender.fit(self.training_data["Gender"])
        self.training_data["Gender"] = self.encoder_gender.transform(self.training_data["Gender"])
        self.training_data["Geography"] = self.encoder_geo.transform(self.training_data["Geography"])

        self.test_data["Gender"] = self.encoder_gender.transform(self.test_data["Gender"])
        self.test_data["Geography"] = self.encoder_geo.transform(self.test_data["Geography"])

    def __preprocess_row(self, row):
        row.drop(["RowNumber", "CustomerId", "Surname"], inplace=True, errors="ignore")
        row["Gender"] = self.encoder_gender.transform([row["Gender"]])[0]
        row["Geography"] = self.encoder_geo.transform([row["Geography"]])[0]
        return row

    def __create_input_layer(self):
        if self.input_layer_size == -1:
            self.input_layer_size = len(self.training_data.columns) - 1
        for i in range(self.input_layer_size):
            print_d(f"Added input neuron for feature {self.training_data.columns[i]}", self.debug)
            self.input_layer.append(
                Neuron(activation_function="identity",
                       name=f"input-{i}", debug=True))
            self.W[f"input_{i}_0"] = 1

    def __default_hidden_layer(self):
        prev_layer_size = self.input_layer_size
        for hidden_cnt in range(0, self.hidden_layer_count):
            hidden_layer = []
            for i in range(self.hidden_layer_size[hidden_cnt]):
                neuron = Neuron(activation_function=self.activation_function,
                                name=f"hidden-{i}", bias=1, debug=True)
                hidden_layer.append(neuron)
                self.W[f"hidden_{hidden_cnt}_{i}_b"] = random_list(1)[0]
                for cnt in range(prev_layer_size):
                    self.W[f"hidden_{hidden_cnt}_{i}_{cnt}"] = random_list(1)[0]
            self.hidden_layer.append(hidden_layer)
            prev_layer_size = len(hidden_layer)

    def __create_output_layer(self):
        self.output_layer = [
            Neuron(activation_function=self.activation_function,
                   name=f"output", bias=1, debug=True)
        ]
        self.W[f"output_b"] = random_list(1)[0]
        for i in range(len(self.hidden_layer[-1])):
            self.W[f"output_{i}"] = random_list(1)[0]

    def train(self, training_data, test_data, epochs=100):
        self.training_data = training_data
        self.test_data = test_data
        self.__preprocess_data()
        # Create Neural network structure
        self.__create_input_layer()
        self.__default_hidden_layer()
        self.__create_output_layer()
        # convert ip op to numpy array with proper indexing
        training_data = self.training_data.to_numpy()
        x = np.array([sublist[:-1] for sublist in training_data])
        y = np.array([sublist[-1] for sublist in training_data])
        for epoch in range(epochs):
            op = []
            for i in range(len(x)):
                # Forward Pass
                o = self.predict(x[i])
                op.append(o)
                sys.stdout.flush()
                print_d(f"\rEpoch: {epoch+1} Step: {i} Training Loss: {self.__error(op, y)}", debug=True, end=" ")
                # Backward Pass
                # Calculate d_i for each node
                # update weight for each node
            print_d(f"\rEpoch: {epoch+1} Training Loss: {self.__error(op, y)}", debug=True)

    def test(self):
        test_data = self.test_data.to_numpy()
        x = np.array([sublist[:-1] for sublist in test_data])
        y = np.array([sublist[-1] for sublist in test_data])

        op = []
        for i in range(len(x)):
            op.append(1 if self.predict(x[i]) > 0.5 else 0)
        print_d(f"Test Loss: {self.__error(op, y)}", debug=True)
        print_d(f"Test Accuracy: {len([i for i in range(len(op)) if op[i] == y[i]]) / len(op)}", debug=True)

    def predict(self, x):
        if type(x) is pd.Series:
            x = self.__preprocess_row(x)
            x = x.to_numpy()
            debug = True
        else:
            debug = False
        ip_o = []
        for i in range(len(self.input_layer)):
            ip_o.append(self.input_layer[i].output([x[i]]))
        hidden_o = ip_o
        for hidden_cnt in range(self.hidden_layer_count):
            hidden_o = []
            for i in range(self.hidden_layer_size[hidden_cnt]):
                hidden_ip = [ip_o[j] * self.W[f"hidden_{hidden_cnt}_{i}_{j}"]
                             for j in range(len(hidden_o))]
                hidden_ip.append(self.W[f"hidden_{hidden_cnt}_{i}_b"] * self.hidden_layer[hidden_cnt][i].bias)
                hidden_o.append(self.hidden_layer[hidden_cnt][i].output(hidden_ip))
        o = self.output_layer[0].output(hidden_o)
        print_d(f"Output of neural network : {o}", debug)
        return o


if __name__ == "__main__":
    n1 = Neuron(activation_function="tanh", bias=2,
                name="sample neuron", debug=True)
    n1.output([2, 3])
