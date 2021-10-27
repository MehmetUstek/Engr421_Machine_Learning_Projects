import numpy as np
import matplotlib.pyplot as plt
import pandas
import math

from scipy import linalg


def safelog(x):
    return (np.log(x + 1e-100))


images = pandas.read_csv('hw02_images.csv', header=None)
labels = pandas.read_csv('hw02_labels.csv', header=None)

# Divide the data, 30000 to training, 5000 to test
x_train = images[:30000]
y_train = labels[:30000][0]
x_test = images[-5000:]
y_test = labels[-5000:][0]

K = 5

# Estimate the mean parameters
sample_means = [np.mean(x_train[y_train == (c + 1)], axis=0) for c in range(K)]
print("sample_means:")
print(sample_means)

sample_deviations = [np.sqrt(np.mean((x_train[y_train == (c + 1)] - sample_means[c]) ** 2)) for c in range(K)]
print("sample_deviations:")
print(sample_deviations)

class_priors = [np.mean(y_train == (c + 1)) for c in range(K)]

print("class_priors:")
print(class_priors)


# define the sigmoid function
def sigmoid(X, w, w0):
    # print(X)
    return(1 / (1 + np.exp(-(np.matmul(X, w) + w0))))
# define the gradient functions
def gradient_w(X, y_truth, y_predicted):
    return(-np.sum(np.transpose(np.repeat([y_truth - y_predicted], X.shape[1], axis = 0)) * X, axis = 0))

def gradient_w0(y_truth, y_predicted):
    return(-np.sum(y_truth - y_predicted))

# set learning parameters
eta = 0.01
epsilon = 1e-3
# randomly initalize w and w0
np.random.seed(421)
w = np.random.uniform(low = -0.01, high = 0.01, size = x_train.shape[1])
w0 = np.random.uniform(low = -0.01, high = 0.01, size = 1)
# learn w and w0 using gradient descent
iteration = 1
objective_values = []
while 1:
    y_predicted = sigmoid(x_train, w, w0)

    objective_values = np.append(objective_values, -np.sum(y_train * safelog(y_predicted) + (1 - y_train) * safelog(1 - y_predicted)))

    w_old = w
    w0_old = w0

    w = w - eta * gradient_w(x_train, y_train, y_predicted)
    w0 = w0 - eta * gradient_w0(y_train, y_predicted)

    if np.sqrt((w0 - w0_old)**2 + np.sum((w - w_old)**2)) < epsilon:
        break

    iteration = iteration + 1
print(w, w0)


# Confusion Matrix


# confusion_matrix = pandas.crosstab(y_predicted, y_train, rownames=['y_pred'], colnames=['y_truth'])

print("confusion_matrix")
# print(confusion_matrix)
