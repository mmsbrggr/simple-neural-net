from mnist import MNIST
from NeuralNet import NeuralNet
import numpy as np
import random

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







from tkinter import *
from PIL import Image, ImageDraw

canvas_width = 28*10
canvas_height = 28*10


def paint(event):
    width = 10
    x1, y1 = (event.x - width/2), (event.y - width/2)
    x2, y2 = (event.x + width/2), (event.y + width/2)
    w.create_oval(x1, y1, x2, y2, fill="#000000")
    draw.ellipse([x1, y1, x2, y2], fill="#ffffff")


def clear_canvas():
    w.delete("all")
    draw.rectangle([1, 1, canvas_width, canvas_height], fill="#000000")


def predict_digit():
    print("Predict digit")
    small_image = image.resize((28, 28))
    image_array = np.asarray(np.asmatrix(small_image).flatten())[0]
    print(nn.predict(image_array / 255))


master = Tk()
master.title("Painting using Ovals")
w = Canvas(master,
           width=canvas_width,
           height=canvas_height)
w.pack(expand=YES, fill=BOTH)
w.bind("<B1-Motion>", paint)

message = Label(master, text="Press and Drag the mouse to draw")
message.pack(side=BOTTOM)

button_clear = Button(master, text="Clear", command=clear_canvas)
button_clear.pack(side=BOTTOM)

button_predict = Button(master, text="Predict", command=predict_digit)
button_predict.pack(side=BOTTOM)

image = Image.new("L", (canvas_width, canvas_height))
draw = ImageDraw.Draw(image)

mainloop()

