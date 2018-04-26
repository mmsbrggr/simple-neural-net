import random
from NeuralNet import NeuralNet

nn = NeuralNet(2, [4, 4], 2, ["1", "0"])

data = [
    {"input": [0, 0], "label": "0"},
    {"input": [1, 0], "label": "1"},
    {"input": [0, 1], "label": "1"},
    {"input": [1, 1], "label": "0"},
]

for x in range(1, 10000):
    d = random.choice(list(data))
    nn.train(d["input"], d["label"])

print("Input [0,0]", nn.feed_forward([0, 0]), nn.predict([0, 0]))
print("Input [0,1]", nn.feed_forward([0, 1]), nn.predict([0, 1]))
print("Input [1,0]", nn.feed_forward([1, 0]), nn.predict([1, 0]))
print("Input [1,1]", nn.feed_forward([1, 1]), nn.predict([1, 1]))
