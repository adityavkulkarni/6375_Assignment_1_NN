import numpy
import pandas as pd
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
            if str(unique_item) not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x == unique_item else x for x in new_data_list]
        return self.label_encoder.transform(new_data_list)


class Neuron:
    def __init__(self, activation_function, name=None, bias=0, debug=False):
        """
        Class for a single neuron
        :param activation_function: Activation function: (sigmoid|tanh|relu)
        :param name: Name of neuron (optional)
        :param debug: True for print_ding debug messages (optional, default=False)
        """
        self.activation_function = globals()[activation_function]
        self.name = name
        self.debug = debug
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
    def __init__(self, activation_function="sigmoid", debug=False,
                 input_layer_size=-1, hidden_layer_size=None):
        """

        :param activation_function:
        :param debug:
        :param input_layer_size:
        :param hidden_layer_size:
        """
        if hidden_layer_size is None:
            hidden_layer_size = [100]
        self.activation_function = activation_function
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layer_count = len(self.hidden_layer_size)
        self.debug = debug
        self.neuron_count = 1
        self.input_layer = []
        self.hidden_layer = []
        self.output_layer = []
        self.W = {}
        self.training_data = None
        self.test_data = None
        self.min_max_scaler = None

    @staticmethod
    def __error(o, t):
        """
        Function for MSE calculation
        :param o: predicted output
        :param t: actual output
        :return:
        """
        if type(o) is float or type(t) is float or type(o) is numpy.float64:
            return 0.5 * (abs(t - o) ** 2)
        e = 0
        # check
        for i in range(len(o)):
            e += abs(t[i] - o[i]) ** 2
        return 0.5 * e / len(o)

    def __print_performance(self, y_true, y_pred, debug=None):
        if debug is None:
            debug = self.debug
        print_d(f"Test Accuracy: {accuracy_score(y_true, y_pred)}",
                debug=debug)

    def __preprocess_data(self):
        """
        Function for preprocessing training and test data
        :return:
        """
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
        """
        Preprocess a single row of data
        :param row:
        :return:
        """
        row.drop(["RowNumber", "CustomerId", "Surname"], inplace=True, errors="ignore")
        row["Gender"] = self.encoder_gender.transform([row["Gender"]])[0]
        row["Geography"] = self.encoder_geo.transform([row["Geography"]])[0]
        return row

    def __create_input_layer(self):
        """
        Create an input layer with number of features or given input layer size
        :return:
        """
        if self.input_layer_size == -1:
            self.input_layer_size = len(self.training_data.columns) - 1
        for i in range(self.input_layer_size):
            print_d(f"Added input neuron for feature {self.training_data.columns[i]}", self.debug)
            self.input_layer.append(
                Neuron(activation_function="identity",
                       name=self.neuron_count))
            self.W[f"w_{self.neuron_count}"] = 1
            self.neuron_count += 1

    def __default_hidden_layer(self):
        """
        Create a hidden layer with given parameters
        :return:
        """
        prev_layer = self.input_layer
        for hidden_cnt in range(self.hidden_layer_count):
            hidden_layer = []
            for i in range(self.hidden_layer_size[hidden_cnt]):
                neuron = Neuron(activation_function=self.activation_function,
                                name=self.neuron_count, bias=random_list(1)[0])
                hidden_layer.append(neuron)
                for cnt in range(len(prev_layer)):
                    p = prev_layer[cnt].name
                    # self.W[f"hidden_{hidden_cnt}_{i}_{cnt}"] = random_list(1)[0]
                    self.W[f"w_{self.neuron_count}_{p}"] = random_list(1)[0]
                self.W[f"w_{self.neuron_count}_b"] = random_list(1)[0]
                self.neuron_count += 1
            self.hidden_layer.append(hidden_layer)

    def __create_output_layer(self):
        """
        create an output layer with single neuron
        :return:
        """
        self.output_layer = [
            Neuron(activation_function=self.activation_function,
                   name=self.neuron_count, bias=random_list(1)[0])
        ]
        self.W[f"w_{self.neuron_count}_b"] = random_list(1)[0]
        for i in range(len(self.hidden_layer[-1])):
            p = self.hidden_layer[-1][i].name
            self.W[f"w_{self.neuron_count}_{p}"] = random_list(1)[0]
        self.neuron_count += 1

    def train(self, training_data, test_data, learning_rate=0.5, epochs=100):
        """
        Train the neural network
        :param training_data:
        :param test_data:
        :param learning_rate:
        :param epochs:
        :return:
        """
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
        self.min_max_scaler = StandardScaler()
        self.min_max_scaler.fit(x)
        x = self.min_max_scaler.transform(x)
        for epoch in range(epochs):
            op = []
            for i in range(len(x)):
                # Forward Pass
                o = self.__predict(x[i])
                op.append(o)
                # sys.stdout.flush()
                q = int(40 * i / len(x))
                s = '{:6.5f}'.format(self.__error(o, y[i]))
                print(f"\rEpoch: {epoch + 1}{' '*(3 - len(str(epoch + 1)))} "
                      f"Training Loss: {s}   "
                      f"[{u'=' * q}{('.' * (40 - q))}] {i + 1}/{len(x)}",
                      end='', file=sys.stdout, flush=True)
                # Backward Pass
                # Calculate d_i for each node
                d = {}
                dw = {}
                output_neuron = self.output_layer[0]
                d[f"d_{output_neuron.name}"] = output_neuron.y * (1 - output_neuron.y) * (y[i] - output_neuron.y)
                for hidden_neuron in self.hidden_layer[::-1][0]:
                    d[f"d_{hidden_neuron.name}"] = (hidden_neuron.y * (1 - hidden_neuron.y) *
                                                    (self.W[f"w_{output_neuron.name}_{hidden_neuron.name}"] *
                                                     d[f"d_{output_neuron.name}"]))
                    dw[f"w_{output_neuron.name}_{hidden_neuron.name}"] = (learning_rate *
                                                                          d[f"d_{output_neuron.name}"] *
                                                                          hidden_neuron.y)
                    dw[f"w_{hidden_neuron.name}_b"] = (learning_rate * d[f"d_{output_neuron.name}"] *
                                                       output_neuron.bias)
                for input_neuron in self.input_layer:
                    for hidden_neuron in self.hidden_layer[0]:
                        dw[f"w_{hidden_neuron.name}_{input_neuron.name}"] = (learning_rate *
                                                                             d[f"d_{hidden_neuron.name}"] *
                                                                             input_neuron.y)
                        dw[f"w_{hidden_neuron.name}_b"] = (learning_rate *
                                                           d[f"d_{hidden_neuron.name}"] * hidden_neuron.bias)
                # print(dw)
                for key in dw:
                    self.W[key] += dw[key]
                # Update weight for each neuron
            s = '{:6.5f}'.format(self.__error(op, y))
            t = '{:6.5f}'.format(self.test(debug=False))
            print(f"\rEpoch: {epoch + 1}{' '*(3 - len(str(epoch+1)))}"
                  f" Training Loss: {s}   "
                  f"[{u'=' * 40}] {len(x)}/{len(x)}  Test Loss: {t}\n",
                  end='', file=sys.stdout, flush=True)

    def test(self, debug=None):
        """
        Test the neural network on the test/ validation set
        :return:
        """
        if debug is None:
            debug = self.debug
        test_data = self.test_data.to_numpy()
        x = np.array([sublist[:-1] for sublist in test_data])
        y = np.array([sublist[-1] for sublist in test_data])
        x = self.min_max_scaler.fit_transform(x)
        op = []
        for i in range(len(x)):
            op.append(1 if self.__predict(x[i]) > 0.5 else 0)
        print_d(f"Test Loss: {self.__error(op, y)}", debug=debug)
        self.__print_performance(y_true=y, y_pred=op, debug=debug)
        return self.__error(op, y)

    def __predict(self, x, debug=False):
        """
        Give prediction for given input
        :param x:
        :param debug:
        :return:
        """
        if type(x) is pd.Series:
            x = self.__preprocess_row(x)
            x = x.to_numpy()
            x = self.min_max_scaler.transform([x]).reshape((-1, 1))
        ip_o = []
        for i in range(len(self.input_layer)):
            ip_o.append(self.input_layer[i].output([x[i]]))
        hidden_o = ip_o
        for hidden_cnt in range(self.hidden_layer_count):
            hidden_o = []
            for i in range(self.hidden_layer_size[hidden_cnt]):
                current_neuron = self.hidden_layer[hidden_cnt][i]
                hidden_ip = [ip_o[j] * self.W[f"w_{current_neuron.name}_{j+1}"]
                             for j in range(len(self.input_layer))]
                hidden_ip.append(current_neuron.bias * self.W[f"w_{current_neuron.name}_b"])
                hidden_o.append(current_neuron.output(hidden_ip) *
                                self.W[f"w_{self.output_layer[0].name}_{current_neuron.name}"])
        hidden_o.append(self.output_layer[0].bias + self.W[f"w_{self.output_layer[0].name}_b"])
        o = self.output_layer[0].output(hidden_o)
        print_d(f"Output of neural network : {o}", debug)
        return o

    def predict(self, x):
        return 1 if self.__predict(x) > 0.5 else 0


if __name__ == "__main__":
    n1 = Neuron(activation_function="tanh", bias=2,
                name="sample neuron", debug=True)
    n1.output([2, 3])
