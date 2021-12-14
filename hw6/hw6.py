import numpy as np
import pandas


def safelog(x):
    return (np.log(x + 1e-100))

images = pandas.read_csv('hw06_images.csv', header=None)
labels = pandas.read_csv('hw06_labels.csv', header=None)

# Divide the data, 30000 to training, 5000 to test
x_train = images[:1000]
y_train = labels[:1000][0]
x_test = images[-4000:]
y_test = labels[-4000:][0]

K = 5
