from neural_net_exceptions import *
from utils import *


class Neuron:
    def __init__(self, activation_function, input_size, bias=0, name="neuron", debug=False):
        """
        Class for a single neuron
        :param activation_function: Activation function: (sigmoid|tanh|relu)
        :param input_size: Number of inputs
        :param bias: Bias value for neuron (default = 0)
        :param name: Name of neuron (optional)
        :param debug: True for print_ding debug messages (optional, default=False)
        """
        self.activation_function = globals()[activation_function]
        self.weights = random_list(input_size)
        self.bias = bias
        self.name = name
        self.debug = False
        self.net = 0
        self.d = 0
        self.dw = []
        self.y = 0
        print_d(f"Initial weights for {self.name} : {' '.join([str(x) for x in self.weights])}", self.debug)
        print_d(f"Bias for {self.name} : {self.bias}", self.debug)

    def output(self, x):
        """
        Function for generating neuron output
        :param x: Input vector
        :return: Output value
        """
        if len(self.weights) != len(x):
            raise NeuronInputException(f"Neuron input length incorrect. "
                                       f"Expected {len(self.weights)} but got {len(x)}")
        net = self.bias
        for i in range(len(x)):
            net += x[i] * self.weights[i]
        self.net = net
        self.y = self.activation_function(net)
        print_d(f"Output of {self.name} : {self.y}", self.debug)
        return self.y

    def update_weights(self, dw=None):
        """
        Function for updating weights for neuron
        """
        if dw:
            self.dw = dw
        if len(self.dw) != len(self.weights):
            raise NeuronInputException(f"Neuron dw length incorrect. "
                                       f"Expected {len(self.weights)} but got {len(dw)}")
        print_d(f"Weight difference for {self.name}: {' '.join([str(x) for x in self.dw])}", self.debug)
        for i in range(len(self.weights)):
            self.weights[i] += self.dw[i]
        print_d(f"Updated weights for {self.name} : {' '.join([str(x) for x in self.weights])}", self.debug)


class NeuralNet:
    def __init__(self, activation_function="sigmoid", learning_rate=0.01, debug=False):
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.debug = debug
        self.input_layer = []
        self.ip_W = []
        self.hidden_W = []
        self.op_W = []
        self.output_layer = []
        self.hidden_layer = []
        self.training_data = None
        self.test_data = None
        self.input_layer_size = None

    @staticmethod
    def __error(o, t):
        if type(o) is float and type(t) is float:
            return 0.5 * (abs(o - t) ** 2)
        e = 0
        for i in range(len(o)):
            e += abs(o[i] - t[i]) ** 2
        return 0.5 * e

    def __preprocess_data(self):
        self.training_data.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
        self.test_data.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
        # For now
        self.training_data.drop(["Geography", "Gender"], axis=1, inplace=True)
        self.test_data.drop(["Geography", "Gender"], axis=1, inplace=True)

    @staticmethod
    def __preprocess_row(row):
        row.drop(["RowNumber", "CustomerId", "Surname"], inplace=True)
        row.drop(["Geography", "Gender"], inplace=True)
        return row

    def __create_input_layer(self, ):
        if self.input_layer_size is None:
            self.input_layer_size = len(self.training_data.axes[1]) - 1
        for i in range(self.input_layer_size):
            print_d(f"Added input neuron for feature {self.training_data.columns[i]}", self.debug)
            self.input_layer.append(
                Neuron(activation_function=self.activation_function,
                       input_size=1, name=f"input-{i}", debug=True))
            self.ip_W.append(self.input_layer[i].weights)

    def __default_hidden_layer(self):
        hidden_layer = []
        hidden_W = []
        for i in range(3):
            neuron = Neuron(activation_function=self.activation_function,
                            input_size=len(self.input_layer), name=f"hidden-{i}",
                            debug=True)
            hidden_layer.append(neuron)
            hidden_W.append(neuron.weights)
        self.hidden_layer.append(hidden_layer)
        self.hidden_W.append(hidden_W)

    def __create_output_layer(self):
        self.output_layer = [
            Neuron(activation_function=self.activation_function,
                   input_size=len(self.hidden_layer[-1]), name=f"output",
                   debug=True)
        ]
        self.op_W.append(self.output_layer[0].weights)

    def __add_hidden_layer(self, neuron_count=6):
        hidden_layer = []
        hidden_W = []
        for i in range(neuron_count):
            neuron = Neuron(activation_function=self.activation_function,
                            input_size=len(self.input_layer), name=f"hidden-{i}",
                            debug=True)
            hidden_layer.append(neuron)
            hidden_W.append(neuron.weights)
        self.hidden_layer.append(hidden_layer)
        self.hidden_W.append(hidden_W)
        print_d(f"Added hidden layer with {neuron_count} neurons", self.debug)

    def train(self, training_data, test_data, input_layer_size=None):
        self.training_data = training_data
        self.test_data = test_data
        self.__preprocess_data()
        self.input_layer_size = input_layer_size
        x = self.training_data[self.training_data.columns[:-1]]
        y = self.training_data[self.training_data.columns[-1]]
        self.__create_input_layer()
        self.__default_hidden_layer()
        self.__create_output_layer()

        """for epoch in range(self.epochs):
            o = []
            for i in range(len(x)):
                # Forward Pass
                o.append(self.predict(x[i]))
                e = self.__error(o[i], y[i])
                # Backward Pass
            print(f"Epoch: {epoch} Training Loss: {self.__error(o, y)}")"""

    def predict(self, x):
        x = self.__preprocess_row(x)
        ip_o = []
        for i in range(len(self.input_layer)):
            ip_o.append(self.input_layer[i].output([x[i]]))
        h_o = []
        for hidden_layer in self.hidden_layer:
            for hidden_neuron in hidden_layer:
                h_o.append(hidden_neuron.output(ip_o))
        o = self.output_layer[0].output(h_o)
        print_d(f"Output of neural network : {o}", self.debug)
        return o


if __name__ == "__main__":
    n1 = Neuron(activation_function="tanh", bias=2, input_size=2,
                name="sample neuron", debug=True)
    n1.output([2, 3])
    n1.update_weights([0.1, -1])
    n1.output([2, 3])
