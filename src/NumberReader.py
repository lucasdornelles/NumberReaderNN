import numpy as np
from functools import reduce

from src.DataPersistence import DataPersistence
from src.NeuralNetwork import NeuralNetwork


class NumberReader(NeuralNetwork, DataPersistence):

    def __init__(self, input_pixels, hidden_layers, neurons_by_hidden_layers):
        self.data_info = [input_pixels, hidden_layers, neurons_by_hidden_layers]
        super().__init__([input_pixels] + [neurons_by_hidden_layers for _ in range(hidden_layers)] + [10])
        self.evolutive_rate = 0.1
        self.momentum = 0.75

    def save_number_reader(self, file):
        data_list = self.pack_data()
        self.save_data(file, data_list)

    def load_number_reader(self, file):
        data_list = self.load_data(file)
        if data_list and len(data_list) == len(self.network_layers) + 1 and \
                reduce(lambda x, y: x and y, map(lambda p, q: p == q, data_list[0], self.data_info), True):
            for i in range(len(self.network_layers)):
                self.network_layers[i].load_data(data_list[i + 1][0], data_list[i + 1][1])
            return True
        else:
            return False

    def pack_data(self):
        # pack layers information in a data_list
        data_list = list()
        data_list.append(self.data_info)
        for i in range(len(self.network_layers)):
            data_list.append(self.network_layers[i].export_data())
        return data_list

    def solve_number_image(self, values_list):
        self.neuron_layers[0].setvalues(values_list)
        self.forward_feed()
        return self.neuron_layers[len(self.neuron_layers) - 1].maxvalueindex()

    def batch_training(self, data_set, batch_size):
        shuffled_data = list(data_set)
        np.random.shuffle(shuffled_data)
        iterations = 0.
        times_correct = 0.
        dcost_dweight = [[0. for _ in range(n)] for n in [layer.getweightslen() for layer in self.network_layers]]
        adcost_dweight = dcost_dweight.copy()
        dcost_dbias = [[0. for _ in range(n)] for n in [layer.getbiaslen() for layer in self.network_layers]]
        adcost_dbias = dcost_dbias.copy()
        reference_dcost_dweight = adcost_dweight.copy()
        reference_dcost_dbias = adcost_dbias.copy()
        for i in range(0, len(shuffled_data), batch_size):
            last_dcost_dweight = reference_dcost_dweight.copy()
            dcost_dweight = [[0. for _ in range(n)] for n in [layer.getweightslen() for layer in self.network_layers]]
            adcost_dweight = dcost_dweight.copy()
            last_dcost_dbias = reference_dcost_dbias.copy()
            dcost_dbias = [[0. for _ in range(n)] for n in [layer.getbiaslen() for layer in self.network_layers]]
            adcost_dbias = dcost_dbias.copy()
            cost_batch = 0
            itimes_correct = 0
            for j in range(i, i + batch_size):
                iterations += 1
                values = shuffled_data[j][0]
                label = shuffled_data[j][1]
                result = self.solve_number_image(values)
                cost = self.calculate_cost_function(label)
                times_correct += 1 if result == label else 0
                itimes_correct += 1 if result == label else 0
                # print((iterations, times_correct, cost, label, result))
                cost_batch = cost_batch + cost
                iteration_dcost_dweight, iteration_dcost_dbias = self.get_dcost_dweights(label)
                aiteration_dcost_dweight = np.array(iteration_dcost_dweight)
                aiteration_dcost_dbias = np.array(iteration_dcost_dbias)
                for k in range(len(adcost_dweight)):
                    adcost_dweight[k] = np.add(adcost_dweight[k], aiteration_dcost_dweight[k])
                    adcost_dbias[k] = np.add(adcost_dbias[k], aiteration_dcost_dbias[k])
            l2_norm = 0
            for j in range(len(adcost_dweight)):
                adcost_dweight[j] = np.add(np.multiply(adcost_dweight[j], (1 / (batch_size * 1)) * self.evolutive_rate),
                                           np.multiply(last_dcost_dweight[j], self.momentum))
                l2_norm = l2_norm + sum([item * item for item in adcost_dweight[j]])
                adcost_dbias[j] = np.add(np.multiply(adcost_dbias[j],(1 / (batch_size * 1)) * self.evolutive_rate),
                                         np.multiply(last_dcost_dbias[j], self.momentum))
                l2_norm = l2_norm + sum([item * item for item in adcost_dbias[j]])

            reference_dcost_dweight = adcost_dweight.copy()
            reference_dcost_dbias = adcost_dbias.copy()

            l2_norm = np.sqrt(l2_norm)
            if l2_norm > 1:
                for j in range(len(adcost_dweight)):
                    adcost_dweight[j] = np.multiply(adcost_dweight[j], 1 / l2_norm)
                    adcost_dbias[j] = np.multiply(adcost_dbias[j], 1 / l2_norm)

            for j in range(len(self.network_layers)):
                self.network_layers[j].correctweights(adcost_dweight[j])
                self.network_layers[j].correctbias(adcost_dbias[j])
            print((iterations, (cost_batch / batch_size), (itimes_correct / batch_size)))

    def calculate_cost_function(self, label):
        return sum(
            [np.square((a - y)) for a, y in zip(self.neuron_layers[-1].getvalues(), self.expected_result(label))])

    def get_dcost_dweights(self, label):
        dcost_dweight = []
        dcost_dbias = []
        for i in range(len(self.network_layers)):
            if i < 1:
                last_layer = self.network_layers[-1]
                al = self.neuron_layers[-1].getvalues()
                c = np.array(self.expected_result(label))
                dcda = np.multiply(np.add(al, -c), 2)
                wa = np.add(np.dot(last_layer.getweights(), self.neuron_layers[-2].getvalues()),
                            last_layer.getbias())
                dadz = np.array([1 if a > 0 else 0.05 for a in wa])
                dadzdcda = np.multiply(dadz, dcda)
                layer_dcost_dbias = dadzdcda.tolist()
                dcdw = np.transpose(np.multiply(np.transpose(np.array([self.neuron_layers[-2].getvalues().tolist()
                                                                       for _ in range(len(dadzdcda))])), dadzdcda))
                layer_dcost_dweight = dcdw.reshape(1, -1)[0].tolist()

            else:
                layer = self.network_layers[-(i + 1)]
                dcda = np.dot(np.transpose(self.network_layers[-i].getweights()), dadzdcda)
                wa = np.add(np.dot(layer.getweights(), self.neuron_layers[-(i + 2)].getvalues()),
                            layer.getbias())
                dadz = np.array([1 if a > 0 else 0.05 for a in wa])
                dadzdcda = np.multiply(dadz, dcda)
                dcdw = np.transpose(np.multiply(np.transpose(np.array([self.neuron_layers[-(i + 2)].getvalues().tolist()
                                                                       for _ in range(len(dadzdcda))])), dadzdcda))
                layer_dcost_dbias = dadzdcda.tolist()
                layer_dcost_dweight = dcdw.reshape(1, -1)[0].tolist()

            dcost_dweight.append(layer_dcost_dweight)
            dcost_dbias.append(layer_dcost_dbias)

        dcost_dweight.reverse()
        dcost_dbias.reverse()
        return dcost_dweight, dcost_dbias

    def expected_result(self, label):
        y = [0. for _ in range(10)]
        y[label] = 1.
        return y
