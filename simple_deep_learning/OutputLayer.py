"""
    This file is part of a simple toy neural network library.

    Author: Marcel Moosbrugger
"""

import numpy as np
from simple_deep_learning.Layer import Layer


class OutputLayer(Layer):

    def feed_forward(self, inputs):
        self.last_inputs = inputs + self.biases
        self.last_outputs = self.activation(self.last_inputs)

    def get_outputs(self):
        return self.last_outputs

    # logistic activation function for output layer
    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def dactivation(self, x):
        return self.activation(x) * (1 - self.activation(x))
