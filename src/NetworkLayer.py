import numpy as np

from src.MatrixOperator import MatrixOperator


class NetworkLayer(MatrixOperator):

    def __init__(self, n_input_neurons, n_output_neurons):
        # weights matrix and bias
        # list of n_input_neurons * n_output_neurons weights and n_output_neurons bias
        # proper initialization of weights in accord to ahfb
        self.n_weights = n_input_neurons * n_output_neurons
        self.n_input_neurons = n_input_neurons
        self.n_output_neurons = n_output_neurons
        self.weights = [np.random.randn() * np.sqrt(2 / n_output_neurons)
                        for _ in range(n_input_neurons * n_output_neurons)]
        self.bias = [0. for _ in range(n_output_neurons)]

    def getweights(self):
        # return weights on a sparse matrix
        return self.as_matrix(self.weights, self.n_input_neurons)

    def setweights(self, values):
        self.weights = values.copy()

    def correctweights(self, values):
        self.weights = list(np.add(np.array(self.weights), -values))

    def getbias(self):
        # return bias on a vertical vector
        return self.as_vector(self.bias)

    def setbias(self, values):
        self.bias = values.copy()

    def correctbias(self, values):
        self.bias = list(np.add(np.array(self.bias), -values))

    def getweightslen(self):
        return self.n_weights

    def getbiaslen(self):
        return self.n_output_neurons

    def getinputlen(self):
        return self.n_input_neurons

    def getoutputlen(self):
        return self.n_output_neurons

    def export_data(self):
        return [np.array(self.weights).tolist(), np.array(self.bias).tolist()]

    def load_data(self, weights, bias):
        self.setweights(weights)
        self.setbias(bias)
