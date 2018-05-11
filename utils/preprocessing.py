"""
    This file is part of a simple toy neural network library.

    Author: Marcel Moosbrugger

    This module contains a function which essentially does the same preprocessing
    of gray-scale images as it's done on the images of the MNIST data set.
    This helper functions are used to draw our own handwritten digits, preprocess them
    and classify them with our neural network.
"""


import numpy as np
import scipy.ndimage as ndimage
import cv2
import math


# Preprocess the handwritten image-array as described as in
# https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
def preprocess_image(image_array):
    while np.sum(image_array[0]) == 0:
        image_array = image_array[1:]

    while np.sum(image_array[:, 0]) == 0:
        image_array = np.delete(image_array, 0, 1)

    while np.sum(image_array[-1]) == 0:
        image_array = image_array[:-1]

    while np.sum(image_array[:, -1]) == 0:
        image_array = np.delete(image_array, -1, 1)

    rows, cols = image_array.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        image_array = cv2.resize(image_array, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        image_array = cv2.resize(image_array, (cols, rows))

    cols_padding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rows_padding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    image_array = np.lib.pad(image_array, (rows_padding, cols_padding), 'constant')

    shift_x, shift_y = get_best_shift(image_array)
    return shift(image_array, shift_x, shift_y)


def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shift_x = np.round(cols/2.0-cx).astype(int)
    shift_y = np.round(rows/2.0-cy).astype(int)
    return shift_x, shift_y


def shift(img, sx, sy):
    rows, cols = img.shape
    matrix = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, matrix, (cols, rows))
    return shifted
