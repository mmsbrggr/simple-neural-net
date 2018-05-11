"""
    This file is part of a simple toy neural network library.

    Author: Marcel Moosbrugger

    In this example we train a deep neural network with two hidden layers
    on the famous MNIST data set of handwritten digits.
"""

from mnist import MNIST
from simple_deep_learning.NeuralNet import NeuralNet
import numpy as np

# 784 input neurons. One neuron for every pixel
# Two hidden layers. The first with 128 neurons. The second with 64 neurons
# One output layer with 10 neurons (one for every digit)
nn = NeuralNet(784, [128, 64], 10, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

mndata = MNIST('./mnist-data')
mndata.gz = True

# First we train the neural net on all training examples individually
images, labels = mndata.load_training()
print("Learning phase (%d examples):" % len(images))
for i in range(0, len(images) - 1):
    # Normalize the image two be in the range 0-1 instead 0-255
    image = np.divide(images[i], 255)
    label = labels[i]
    nn.train(image, label)
    if (i + 1) % 1000 is 0:
        print(i + 1, "examples trained")
print("\n\n")


# Now that the neural net is trained. Let's examine how it performs on the test examples
images, labels = mndata.load_testing()
correct = 0
print("Testing phase (%d examples):" % len(images))
for i in range(0, len(images) - 1):
    image = np.divide(images[i], 255)
    label = labels[i]
    prediction = nn.predict(image)
    if prediction == label:
        correct += 1
    if (i + 1) % 1000 is 0:
        print(i + 1, "examples tested")

print(correct, "test examples predicted correctly")
print("Accuracy on test set %1.2f %%" % (100 * correct/len(images)))
