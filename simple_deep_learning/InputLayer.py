"""
    This file is part of a simple toy neural network library.

    Author: Marcel Moosbrugger
"""

from simple_deep_learning.Layer import Layer


class InputLayer(Layer):

    def feed_forward(self, inputs):
        self.last_inputs = inputs
        self.last_outputs = inputs
        self.output_connection.feed_forward(inputs)
