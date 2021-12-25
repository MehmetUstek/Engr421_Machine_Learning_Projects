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

class_means = np.array([[2.5, 2.5], [-2.5, 2.5], [-2.5, -2.5],[2.5, -2.5],[0.0, 0.0]])

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

priors = np.array([50, 50, 50, 50, 100])

points1 = np.random.multivariate_normal(class_means[0, :], class_covariances[0, :, :], priors[0])
points2 = np.random.multivariate_normal(class_means[1, :], class_covariances[1, :, :], priors[1])
points3 = np.random.multivariate_normal(class_means[2, :], class_covariances[2, :, :], priors[2])
points4 = np.random.multivariate_normal(class_means[3, :], class_covariances[3, :, :], priors[3])
points5 = np.random.multivariate_normal(class_means[4, :], class_covariances[4, :, :], priors[4])

probabilities = [1/6, 1/6, 1/6, 1/6, 1/3]

# Step 2: M Step
def update_centroids(memberships, X):
    if memberships is None:
        # This is the first iteration.
        # initialize centroids
        centroids = initial_centroids
    else:
        # update centroids
        # Calculate sample mean for K = 0, K= 1, K = 2, K = 3 and done.
        # return centroids array.

        centroids = np.vstack([np.mean(X[memberships == k,:], axis = 0) for k in range(K)])
    return(centroids)
# Step 1: E Step
def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    # Distance between all centroids and data points.
    # D matrix is KxN.
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    # Nearest centroid.
    memberships = np.argmin(D, axis = 0)
    return(memberships)
def e_step(t, k):
    denominator = 0.0
    iteration_gaussian = multivariate_normal.pdf(points, mean= centroids[t, :], cov= class_covariances[t, :, :])
    numerator = iteration_gaussian * probabilities[k]
    for c in range(K):
        # current_point = np.random.multivariate_normal(centroids[t, :], class_covariances[t, :, :], priors[c])
        current_gaussian = multivariate_normal.pdf(points, mean= centroids[t, :], cov= class_covariances[t, :, :])
        denominator += current_gaussian * probabilities[c]
    return numerator / denominator


def m_step(memberships, X, class_covariances, probabilities):
    if memberships is None:
        centroids = initial_centroids
    else:
        centroids = np.vstack([np.mean(X[memberships == k, :], axis=0) for k in range(K)])
        class_covariances = np.vstack([np.mean((X[memberships == k, :] - centroids[k]) * np.transpose(X[memberships == k, :] - centroids[k]), axis=0) for k in range(K)])
        probabilities = np.vstack([np.sum(X[memberships == k, :], axis=0)/ K for k in range(K)])
    return (centroids, class_covariances, probabilities)


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

centroids = None
memberships = None
iteration = 1
while True:
    print("Iteration#{}:".format(iteration))

    old_centroids = centroids
    # centroids = update_centroids(memberships, points)
    centroids, class_covariances, probabilities = m_step(memberships, points, class_covariances, probabilities)
    if np.alltrue(centroids == old_centroids):
        break
    else:
        plt.figure(figsize = (12, 6))
        plt.subplot(1, 2, 1)
        plot_current_state(centroids, memberships, points)

    old_memberships = memberships
    # memberships = update_memberships(centroids, points)
    membership_probs = []
    for k in range(K):
        membership_probs.append(e_step(iteration, k))
    memberships = np.argmax(membership_probs, axis= 0)
    if np.alltrue(memberships == old_memberships):
        plt.show()
        break
    else:
        plt.subplot(1, 2, 2)
        plot_current_state(centroids, memberships, points)
        plt.show()

    iteration = iteration + 1