import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa
from scipy.stats import multivariate_normal

points = np.genfromtxt("hw08_data_set.csv", delimiter=",")
K = 5
x1 = points[:, 0]
x2 = points[:, 1]
# Plotting Data
plt.figure(figsize=(8, 8))
plt.plot(x1, x2, "k.", markersize=12)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()
N = points.shape[0]

class_means_given = np.array([[2.5, 2.5], [-2.5, 2.5], [-2.5, -2.5], [2.5, -2.5], [0.0, 0.0]])

class_covariances_given = np.array([[
    [0.8, -0.6],
    [-0.6, 0.8]],
    [
        [0.8, 0.6],
        [0.6, 0.8]
    ],
    [
        [0.8, -0.6],
        [-0.6, 0.8]
    ],
    [
        [0.8, 0.6],
        [0.6, 0.8]
    ],
    [
        [1.6, 0.0],
        [0.0, 1.6]
    ]
])