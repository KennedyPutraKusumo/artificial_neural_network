from an import ArtificialNeuron as ANeuron
from synapse import ArtificialSynapse as ASynapse
import numpy as np


class ArtificialNeuralNetwork:

    def __init__(self, ann_structure):
        # the network's structural attributes
        self.structure = ann_structure
        self.layers = np.arange(len(ann_structure))
        # members making up the network
        self.neurons = None
        self.synapses = None
        # initialize the network
        self.create_neurons()
        self.create_synapses()

    def __str__(self):
        return 'ANN{}'.format(self.structure)

    def __repr__(self):
        return self.__str__()

    def create_neurons(self):
        self.neurons = {}
        for layer in self.layers:
            self.neurons[layer] = np.array([ANeuron(neural_id=(layer, _)) for _ in range(self.structure[layer])])

    def create_synapses(self):
        self.synapses = {}
        for layer_1 in self.layers:
            for layer_2 in self.layers:
                if layer_1 + 1 == layer_2:
                    self.synapses[layer_1, layer_2] = np.array([])
                    for neuron_1 in self.neurons[layer_1]:
                        for neuron_2 in self.neurons[layer_2]:
                            self.synapses[layer_1, layer_2] = np.append(self.synapses[layer_1, layer_2],
                                                                        ASynapse(neuron_1, neuron_2))

    def stimulate_sensory_neurons(self, sensory_input):
        assert sensory_input.shape == self.neurons[0].shape, 'sensory input must be a numpy array with the same ' \
                                                             'shape as the sensory neuron of the network'
        for neuron in self.neurons[0]:
            neuron.stimulation = sensory_input[neuron.id[1]]

    def predict(self, sensory_input):
        # stimulate sensory neurons
        self.stimulate_sensory_neurons(sensory_input)
        # passing signal from sensory neurons down the network layers
        for layer in self.layers:
            # exclude last layer
            if layer == self.layers[-1]:
                pass
            else:
                # call synapses between layers to pass signal from one layer to the next
                for synapse in self.synapses[layer, layer + 1]:
                    synapse.pass_signal()
        # return output from network as the activations of the action neurons
        output = np.array([])
        # fire the action neurons and storing activations as outputs
        for neuron in self.neurons[self.layers[-1]]:
            neuron.fire()
            output = np.append(output, neuron.activation)
        return output

    def update_synapse_weights(self):
        for layer in self.layers:
            self.synapses[layer].weights += self.gradient
