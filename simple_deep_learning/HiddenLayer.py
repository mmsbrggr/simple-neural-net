"""
    This file is part of a simple toy neural network library.

    Author: Marcel Moosbrugger
"""

from simple_deep_learning.Layer import Layer


class HiddenLayer(Layer):

    def feed_forward(self, inputs):
        self.last_inputs = inputs + self.biases
        self.last_outputs = self.activation(self.last_inputs)
        self.output_connection.feed_forward(self.last_outputs)

    # ReLU activation function for hidden layers
    def activation(self, x):
        return x * (x > 0)

    def dactivation(self, x):
        return 1 * (x >= 0)
