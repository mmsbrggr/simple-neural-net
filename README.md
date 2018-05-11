# Simple Neural Net

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/mmsbrggr/hsudoku/master/LICENSE)

-------
<p align="center">
    <a href="#motivation">Motivation</a> &bull;
    <a href="#requirements">Requirements</a> &bull;
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

To run the handwritten digit examples (mnist_example.py and drawing_fun_example.py) please download the MNIST data set
from [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/) and place the files into a folder named `mnist-data` on the top level of the folder.

## Usage

## Licence
This project is licensed under the terms of the MIT license. See the LICENSE file.
