"""
    This file is part of a simple toy neural network library.

    Author: Marcel Moosbrugger

    In this example we take the neural net from the MNIST example, which
    is trained on the handwritten digits from the MNIST data set and use
    it to predict our own handwritten digits.

    It doesn't work with the same accuracy as for it does for the test set,
    due to the lack of proper preprocessing.
"""

from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
from utils.preprocessing import preprocess_image

# Import the Neural net from the MNIST example trained on handwritten digits
from mnist_example import nn


# Create a 280x280 canvas. The images are later shrinked to 28x28
canvas_width = 28*10
canvas_height = 28*10


# On drag the function paints a 10px circle under the cursor
# Simultaneously it draws the the same circle onto an image in the memory
def paint(event):
    width = 20
    x1, y1 = (event.x - width/2), (event.y - width/2)
    x2, y2 = (event.x + width/2), (event.y + width/2)
    w.create_oval(x1, y1, x2, y2, fill="#000000")
    draw.ellipse([x1, y1, x2, y2], fill="#ffffff")
    predict_digit()


# Clears the whole canvas as well as the image in the memory
# in order to be able to draw a new digit
def clear_canvas():
    w.delete("all")
    draw.rectangle([1, 1, canvas_width, canvas_height], fill="#000000")


# Predicts the drawn digit on the basis of the image in memory
def predict_digit():
    # First resize the image from 280x280 to 28x28
    small_image = image.resize((28, 28))
    image_array = np.asarray(small_image)
    image_array = preprocess_image(image_array)
    image_array = image_array.flatten()
    image_array = image_array / 255

    prediction = nn.predict(image_array)
    predicted_digit.config(text=("Predicted digit: %d" % prediction))


master = Tk()
master.title("Painting using Ovals")
w = Canvas(master,
           width=canvas_width,
           height=canvas_height)
w.pack(expand=YES, fill=BOTH)
w.bind("<B1-Motion>", paint)

message = Label(master, text="Press and Drag the mouse to draw")
message.pack(side=BOTTOM)

predicted_digit = Label(master, text="Predicted digit: _")
predicted_digit.pack(side=BOTTOM)

button_clear = Button(master, text="Clear canvas", command=clear_canvas)
button_clear.pack(side=BOTTOM)

image = Image.new("L", (canvas_width, canvas_height))
draw = ImageDraw.Draw(image)

mainloop()
