from src.MatrixOperator import MatrixOperator
import numpy as np


class NeuronLayer(MatrixOperator):

    def __init__(self, n_neurons):
        # list of neuron_values with n_neurons neurons_values
        self.n_neurons = n_neurons
        self.neuron_values = [0. for _ in range(n_neurons)]

    def length(self):
        # return number of neurons
        return self.n_neurons

    def getvalues(self):
        # return neuron_values in a vertical vector
        return self.as_vector(self.neuron_values)

    def calculatevalues(self, input_neuron_layer, network_layer):
        # calculate neuron_values = [max(0,a) for a in
        # (np.dot(network_layer.getweights(), input_neuron_layer.getvalues()) + network_layer.getbias())]
        # min(1., a) changed for a leaky rectification linear unit.
        self.setvalues([a if a > 0 else a / 20. for a in
                        np.add(self.dot_operator(network_layer.getweights(), input_neuron_layer.getvalues()),
                               network_layer.getbias())])

    def setvalues(self, values_list):
        # set neuron_values = values_list.copy()
        self.neuron_values = values_list.copy()

    def maxvalueindex(self):
        # return the index of max value
        return self.neuron_values.index(max(self.neuron_values))
