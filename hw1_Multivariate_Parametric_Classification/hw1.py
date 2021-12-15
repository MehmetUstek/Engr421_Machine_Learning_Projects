import math
import numpy as np
import matplotlib.pyplot as plt
import pandas
import scipy.linalg as linalg

# Parameters
np.random.seed(421)
# generating data
class_means = np.array([[0.0, 2.5], [-2.5, -2.0], [2.5, -2.0]])

class_covariances = np.array([[
    [3.2, 0.0],
    [0.0, 1.2]],
    [
        [1.2, 0.8],
        [0.8, 1.2]
    ],
    [
        [1.2, -0.8],
        [-0.8, 1.2]
    ]
])

# number of data points for each class.
class_sizes = np.array([120, 80, 100])

points1 = np.random.multivariate_normal(class_means[0, :], class_covariances[0, :, :], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1, :], class_covariances[1, :, :], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2, :], class_covariances[2, :, :], class_sizes[2])
points = np.vstack((points1, points2, points3))

# generating the labels.
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2])))

x1 = points[:, 0]
x2 = points[:, 1]
# Plotting Data
plt.figure(figsize=(10, 10))
plt.plot(points1[:, 0], points1[:, 1], "r.", markersize=12)
plt.plot(points2[:, 0], points2[:, 1], "g.", markersize=12)
plt.plot(points3[:, 0], points3[:, 1], "b.", markersize=12)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()

# Part 3 - Estimation

# number of classes
K = np.max(y)
# number of data points
N = points.shape[0]

# Parameter Estimation
# Mean
# axis = 0
sample_means = [np.mean(points[y == (c + 1)], axis=0) for c in range(K)]
print("sample_means")
print(sample_means)

# Covariances
sample_covariances = [
    np.dot(np.transpose(points[y == (c + 1)] - sample_means[c]), points[y == (c + 1)] - sample_means[c]) / class_sizes[
        c] for c in range(K)]
print("sample_covariances")
print(sample_covariances)

# Class Priors
class_priors = [np.mean(y == (c + 1)) for c in range(K)]
print("Class Priors")
print(class_priors)

# Multivariate Parametric Classification
# From class notes on Multivariate Parametric Classification
D = 3
y_predicted = []
for i in range(points.shape[0]):
    wc0 = [
        -1 / 2 * np.dot(np.dot(np.transpose(sample_means[c]), np.linalg.inv(sample_covariances[c])), sample_means[c]) -
        D / 2 * np.log(2 * math.pi) - 1 / 2 * np.log(np.linalg.det(sample_covariances[c])) + np.log(class_priors[c]) for
        c in range(K)]
    wc = [np.dot(np.linalg.inv(sample_covariances[c]), sample_means[c]) for c in range(K)]
    Wc = [-1 / 2 * np.linalg.inv(sample_covariances[c]) for c in range(K)]
    g = [np.dot(np.dot(np.transpose(points[i]), Wc[c]), points[i]) + np.dot(np.transpose(wc[c]), points[i]) + wc0[c] for
         c in range(K)]
    # take the maximum of gc(x)
    y_predicted.append(np.argmax(g) + 1)
y_predicted = np.array(y_predicted)

# Confusion Matrix
confusion_matrix = pandas.crosstab(y_predicted, y, rownames=['y_pred'], colnames=['y_truth'])
print(confusion_matrix)


# Plotting the decision boundaries
# From lab 03
x1_interval = np.linspace(-6, +6, 1201)
x2_interval = np.linspace(-6, +6, 1201)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
# discriminant_values = [Wc[c][0][0] * x1_grid + Wc[c][0][1] * x2_grid + wc[c][0] for c in range(K)]
plt.figure(figsize=(10, 10))
plt.plot(points[y == 1, 0], points[y == 1, 1], "r.", markersize=10)
plt.plot(points[y == 2, 0], points[y == 2, 1], "g.", markersize=10)
plt.plot(points[y == 3, 0], points[y == 3, 1], "b.", markersize=10)
plt.plot(points[y_predicted != y, 0], points[y_predicted != y, 1], "ko", markersize = 12, fillstyle = "none")
# plt.contour(x1_grid, x2_grid, discriminant_values, levels=0, colors="k")
plt.show()
