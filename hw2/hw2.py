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
x_train = images[:1000]
y_train = labels[:1000][0]
x_test = images[-5000:]
y_test = labels[-5000:][0]

K = 5
# Estimate the mean parameters
sample_means = np.array(([np.mean(x_train[y_train == (c + 1)], axis=0) for c in range(K)]))
print("sample_means:")
print(sample_means)

class_priors = [np.mean(y_train == (c + 1)) for c in range(K)]

sample_deviations = np.array(([np.sqrt(np.mean((x_train[y_train == (c + 1)] - sample_means[c]) ** 2)) for c in range(K)]))
print("sample_deviations:")
print(sample_deviations)

print("class_priors:")
print(class_priors)

# data_interval = np.linspace(-7, +7, 1401)

score_values = np.stack([-0.5 * np.log(2 * math.pi * sample_deviations[c]**2)
                         - 0.5 * (x_train - sample_means[c])**2 / sample_deviations[c]**2
                         + np.log(class_priors[c])
                         for c in range(K)])
# score_values = np.array((np.sum(score_values,)))
print("score_values")
print(score_values)


# Confusion Matrix
# define the sigmoid function
def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(X, w) + w0))))

# define the gradient functions
def gradient_w(X, y_truth, y_predicted):
    return np.array((-np.sum(np.transpose(np.repeat([y_truth - y_predicted], X.shape[1], axis = 0)) * X, axis = 0)))

def gradient_w0(y_truth, y_predicted):
    return(-np.sum(y_truth - y_predicted, axis=0))

# set learning parameters
eta = 0.01
epsilon = 1e-3

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

    if np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((w - w_old)**2)) < epsilon:
        break

    iteration = iteration + 1
print("w,w0")
print(w, w0)
# calculate confusion matrix
y_predicted = 1 * (y_predicted > 0.5)
confusion_matrix = pandas.crosstab(y_predicted, y_train, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)
