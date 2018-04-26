from mnist import MNIST
from NeuralNet import NeuralNet
import numpy as np

nn = NeuralNet(784, [128, 64], 10, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

mndata = MNIST('./mnist-data')
mndata.gz = True

print("Learning phase:")
images, labels = mndata.load_training()
for i in range(0, len(images) - 1):
    image = np.divide(images[i], 255)
    label = labels[i]
    nn.train(image, label)
    if i % 1000 is 0:
        print(i, "examples trained")
print("\n\n")


print("Testing phase:")
images, labels = mndata.load_testing()
correct = 0
for i in range(0, len(images) - 1):
    image = np.divide(images[i], 255)
    label = labels[i]
    prediction = nn.predict(image)
    if prediction == label:
        correct += 1
    if i % 1000 is 0:
        print(i, "examples tested")

print(correct, "tests predicted correctly")
