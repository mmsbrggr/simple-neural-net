"""
    This file is part of a simple toy neural network library.

    Author: Marcel Moosbrugger

    In this example the neural network is used to train the XOR function,
    as it is a very simple function which is not linearly separable.
    Which means that at least one hidden layer is needed in order to
    learn the function.
"""

import random
from simple_deep_learning.NeuralNet import NeuralNet

# (Input(2) -> Hidden(4) -> Hidden(4) -> Output(2))
nn = NeuralNet(2, [4, 4], 2, ["1", "0"])

data = [
    {"input": [0, 0], "label": "0"},
    {"input": [1, 0], "label": "1"},
    {"input": [0, 1], "label": "1"},
    {"input": [1, 1], "label": "0"},
]

# Repeatedly train the network on a random example
for x in range(1, 10000):
    d = random.choice(list(data))
    nn.train(d["input"], d["label"])

# Now the the neural net should be able to predict the XOR function correctly
print("Input [0,0]", nn.feed_forward([0, 0]), nn.predict([0, 0]))
print("Input [0,1]", nn.feed_forward([0, 1]), nn.predict([0, 1]))
print("Input [1,0]", nn.feed_forward([1, 0]), nn.predict([1, 0]))
print("Input [1,1]", nn.feed_forward([1, 1]), nn.predict([1, 1]))
