from src.NetworkLayer import NetworkLayer
from src.NeuronLayer import NeuronLayer


class NeuralNetwork:

    def __init__(self, layers):
        # init
        self.neuron_layers = [NeuronLayer(i) for i in layers]
        self.network_layers = [NetworkLayer(self.neuron_layers[i].length(), self.neuron_layers[i + 1].length())
                               for i in range(len(self.neuron_layers) - 1)]

    def forward_feed(self):
        for i in range(len(self.neuron_layers) - 1):
            self.neuron_layers[i + 1].calculatevalues(self.neuron_layers[i], self.network_layers[i])
