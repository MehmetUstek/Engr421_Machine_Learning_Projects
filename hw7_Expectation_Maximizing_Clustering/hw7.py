import numpy as np
import matplotlib.pyplot as plt

np.random.seed(421)
#TODO: K = ?
K = 5
class_means = np.array([[2.5, 2.5], [-2.5, 2.5], [-2.5, -2.5], [2.5, -2.5], [0, 0]])

class_covariances = np.array([[
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

# number of data points for each class.
class_sizes = np.array([50, 50, 50, 50, 100])

points1 = np.random.multivariate_normal(class_means[0, :], class_covariances[0, :, :], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1, :], class_covariances[1, :, :], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2, :], class_covariances[2, :, :], class_sizes[2])
points4 = np.random.multivariate_normal(class_means[3, :], class_covariances[3, :, :], class_sizes[3])
points5 = np.random.multivariate_normal(class_means[4, :], class_covariances[4, :, :], class_sizes[4])

points = np.vstack((points1, points2, points3, points4, points5))

# generating the labels.
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2]), np.repeat(4,class_sizes[3]), np.repeat(5, class_sizes[4])))

x1 = points[:, 0]
x2 = points[:, 1]
# Plotting Data
plt.figure(figsize=(8, 8))
plt.plot(points[:, 0], points[:, 1], "k.", markersize=12)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()

def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(x1, x2, ".", markersize = 10, color = "black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize = 12,
                 markerfacecolor = cluster_colors[c], markeredgecolor = "black")
    plt.xlabel("x1")
    plt.ylabel("x2")