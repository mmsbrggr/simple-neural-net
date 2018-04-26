import numpy as np


# This class acts as a base class for the more specific layer types (input, output, hidden)
# and contains the common functionality
class Layer:

    def __init__(self, name, number_neurons):
        self.name = name
        self.input_connection = None
        self.output_connection = None
        self.number_neurons = number_neurons
        self.last_outputs = None
        self.last_inputs = None
        self.biases = self.get_initial_biases()

    def get_initial_biases(self):
        return np.full(self.number_neurons, 0)

    def get_number_neurons(self):
        return self.number_neurons

    def get_last_outputs(self):
        return self.last_outputs

    def get_last_inputs(self):
        return self.last_inputs

    def set_input_connection(self, connection):
        self.input_connection = connection
    
    def set_output_connection(self, connection):
        self.output_connection = connection

    def add_bias_deltas(self, bias_deltas):
        self.biases = self.biases + bias_deltas

    # The activation function can vary in the layer types
    def activation(self, x):
        pass

    def dactivation(self, x):
        pass

    def feed_forward(self, inputs):
        pass

    def back_propagate(self, errors):
        if self.input_connection is not None:
            self.input_connection.back_propagate(errors)
