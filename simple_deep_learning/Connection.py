"""
    This file is part of a simple toy neural network library.

    Author: Marcel Moosbrugger
"""

import numpy as np
from simple_deep_learning.Config import Config


# Represents a connection between to
# Contains the weights and the heart of the backpropagation algorithm
class Connection:

    def __init__(self, from_layer, to_layer):
        self.from_layer = from_layer
        self.to_layer = to_layer
        # Matrix in the i-th row there all all weights corresponding the the i-th neuron in the to-layer
        self.weights = self.get_initial_weights()

    # Initializes the weights according to the 'Xavier' method
    def get_initial_weights(self):
        return np.random.normal(
            0,
            np.sqrt(1/self.from_layer.get_number_neurons()),
            (self.to_layer.get_number_neurons(), self.from_layer.get_number_neurons())
        )

    # Constructs the weighted sum for all neurons in the to_layer and passes it forward
    def feed_forward(self, inputs):
        weighted_inputs = np.dot(self.weights, inputs)
        weighted_inputs = np.squeeze(np.asarray(weighted_inputs))
        self.to_layer.feed_forward(weighted_inputs)

    # Calculates the error made in the from_layer with respect to the error in the to_layer
    # and calculates all the deltas to adjust the weights and biases
    def back_propagate(self, errors_to_layer):
        errors_from_layer = np.dot(np.transpose(self.weights), errors_to_layer)
        errors_from_layer = np.squeeze(np.asarray(errors_from_layer))
        self.from_layer.back_propagate(errors_from_layer)

        deltas = np.multiply(errors_to_layer, self.to_layer.dactivation(self.to_layer.get_last_inputs()))
        deltas = Config.learning_rate * deltas
        self.to_layer.add_bias_deltas(deltas)
        weight_deltas = np.dot(np.transpose(np.matrix(deltas)), np.matrix(self.from_layer.get_last_outputs()))

        self.weights = np.add(self.weights, weight_deltas)
