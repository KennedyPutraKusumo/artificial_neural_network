import numpy as np


class ArtificialNeuron:

    def __init__(self, neural_id, model='sigmoid'):
        # identification
        self.id = neural_id
        self.model = model
        # attributes
        self.input_structure = None  # a scalar specifying number of inputs
        self.stimulation = None  # a scalar representing the neuron's "stimulation" state
        self.activation = None  # a scalar representing the activation state of the neuron: between 0 and 1
        self.bias = 0

    def __str__(self):
        return 'n{}'.format(self.id)

    def __repr__(self):
        return self.__str__()

    def activation_function(self, x):
        if self.model == 'sigmoid':
            a = 1 / (1 + np.exp(-x))
        else:
            print('unrecognized or unimplemented model')
        return a

    def fire(self):
        self.activation = self.activation_function(self.stimulation + self.bias)
