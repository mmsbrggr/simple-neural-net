# Simple Neural Net

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/mmsbrggr/hsudoku/master/LICENSE)

-------
<p align="center">
    <a href="#motivation">Motivation</a> &bull;
    <a href="#installation">Installation</a> &bull;
    <a href="#installation">Usage</a> &bull;
</p>

-------

## Motivation
This repository contains a straightforward implementation of a neural net with the
cabability of feeding data forward through the net and learning from data with the backpropagation algorithm.
The implementation is of course not able to compete with cutting edge neural net frameworks.
The project was a result of the author trying to better understand neural networks and implementing the basic
algorithms for educational purposes.

## Installation

The main requirement is python 3. For using the neural net class the following packages are needed:
* numpy

For running the examples the following packages are needed:
* PIL (or Pillow)
* opencv
* python-mnist
* scipy

To run the handwritten digit examples (`mnist_example.py` and `drawing_fun_example.py`) please download the MNIST data set
from [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/) and place the files into a folder named `mnist-data` on the top level of the folder.

## Usage

The examples can be simply executed by running the corresponding python file:

```shell
python3 xor_example.py
python3 mnist_example.py
python3 drawing_fun_example.py
```

Creating, using and training a neural net with arbitrary hidden layers and neurons is really simple:
```python
from simple_deep_learning.NeuralNet import NeuralNet

# (Input(3 neurons) -> Hidden(4 neurons) -> Hidden(4 neurons) -> Output(3 neurons))
nn = NeuralNet(100, [50, 25], 3, ["Sell stock", "Hold stock", "Buy stock"])

# This should be done not only once but with a lot of data
nn.train(stock_data, correct_action)

# After training we can give it unkown data
nn.predict(new_stock_data)
```

## Licence
This project is licensed under the terms of the MIT license. See the LICENSE file.
