import numpy as np


class ArtificialSynapse:

    def __init__(self, presynaptic_neuron, postsynaptic_neuron):
        self.presynaptic_neuron = presynaptic_neuron
        self.postsynaptic_neuron = postsynaptic_neuron
        self.weight = None

        self.init_weight()

    def __str__(self):
        return 'L{0}{1}'.format(self.presynaptic_neuron.id, self.postsynaptic_neuron.id)

    def __repr__(self):
        return self.__str__()

    def pass_signal(self):
        if self.postsynaptic_neuron.stimulation is None:
            self.postsynaptic_neuron.stimulation = 0
        self.presynaptic_neuron.fire()
        self.postsynaptic_neuron.stimulation += self.presynaptic_neuron.activation * self.weight

    def init_weight(self):
        self.weight = np.random.normal(0, 1)
