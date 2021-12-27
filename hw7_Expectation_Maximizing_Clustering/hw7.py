import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa
from scipy.stats import multivariate_normal

initial_centroids = np.genfromtxt("hw07_initial_centroids.csv", delimiter=",")
points = np.genfromtxt("hw07_data_set.csv", delimiter=",")
# TODO: K = ?
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


def update_memberships(centroids, X):
    D = spa.distance_matrix(centroids, X)
    memberships = np.argmin(D, axis=0)
    return (memberships)


def e_step(Fi):
    centroids = Fi[0]
    class_covariances = Fi[1]
    probabilities = Fi[2]

    denominator = 0.0
    numerator = np.vstack(
        [multivariate_normal.pdf(points, mean=centroids[k, :], cov=class_covariances[k, :, :]) * probabilities[k] for k
         in range(K)])
    for c in range(K):
        current_gaussian = multivariate_normal.pdf(points, mean=centroids[c, :], cov=class_covariances[c, :, :])
        denominator += current_gaussian * probabilities[c]
    return (numerator / denominator).T


def m_step(X, memberships_probabilities):
    denom = 0.0
    centroids = np.zeros(shape=(5, 2))
    for i in range(N):
        current_point = memberships_probabilities[i]
        for k in range(K):
            centroids[k] += current_point[k] * X[i]
        denom += np.array([np.sum(current_point[k]) for k in range(K)])
    centroids = (centroids.T / denom).T
    denom = 0.0
    class_covariances = np.zeros(shape=(5, 2, 2))
    for i in range(N):
        current_point = memberships_probabilities[i]
        for k in range(K):
            class_covariances[k] += current_point[k] * (X[i] - centroids[k])[None, :] * (X[i] - centroids[k])[:,
                                                                                        None]
        denom += np.array([np.sum(current_point[k]) for k in range(K)])
    class_covariances = (class_covariances.T / denom).T

    probabilities = [0, 0, 0, 0, 0]
    for i in range(N):
        current_point = memberships_probabilities[i]
        probabilities += np.array([np.sum(current_point[k]) for k in range(K)])
    probabilities = probabilities / N

    Fi = centroids, class_covariances, probabilities
    return (Fi)


def plot_current_state(centroids, memberships, X, class_covariances, class_means_given, class_covariances_given):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(x1, x2, ".", markersize=10, color="black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10,
                     color=cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize=12,
                 markerfacecolor=cluster_colors[c], markeredgecolor="black")
    plt.xlabel("x1")
    plt.ylabel("x2")

    x1_interval = np.linspace(-6, +6, 1201)
    x2_interval = np.linspace(-6, +6, 1201)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
    # dstack concatenates these arrays into a third dimension. I found this implementation after
    # long search on documentations of vstack in numpy library.
    positions = np.dstack((x1_grid, x2_grid))
    for k in range(K):
        EM_points = multivariate_normal(centroids[k], class_covariances[k]).pdf(positions)
        plt.contour(x1_grid, x2_grid, EM_points, colors=cluster_colors[k], levels=[0.05])
        given_points = multivariate_normal(class_means_given[k], class_covariances_given[k]).pdf(positions)
        plt.contour(x1_grid, x2_grid, given_points, linestyles='dashed', levels=[0.05], colors='k')


# centroids = None
# memberships = None
# class_covariances = None
# probabilities = None
# memberships_probabilities = None

centroids = initial_centroids
memberships = update_memberships(centroids, points)
class_sizes = np.array(np.bincount(memberships))
class_covariances = np.array([
    np.mat(points[memberships == k, :] - centroids[k, :]).T * np.mat(points[memberships == k, :] - centroids[k, :]) /
    class_sizes[k]
    for k in range(K)])
probabilities = np.array(class_sizes / N)

iteration = 1
Fi = []
for i in range(100):
    print("Iteration#{}:".format(iteration))

    Fi = (centroids, class_covariances, probabilities)
    membership_probabilities = e_step(Fi)

    Fi = m_step(points, membership_probabilities)
    centroids = Fi[0]
    class_covariances = Fi[1]
    probabilities = Fi[2]

    iteration = iteration + 1
print(centroids)
memberships = np.argmax(membership_probabilities, axis=1)
plt.figure(figsize=(8, 8))
plot_current_state(centroids, memberships, points, class_covariances, class_means_given, class_covariances_given)
plt.show()
