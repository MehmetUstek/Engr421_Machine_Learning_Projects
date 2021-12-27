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
N = points.shape[0]
# plt.show()

# Step 1: E Step
def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    # Distance between all centroids and data points.
    # D matrix is KxN.
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    # Nearest centroid.
    memberships = np.argmin(D, axis = 0)
    # class_covariances =
    return(memberships)
def e_step(t, Fi):
    centroids = Fi[0]
    class_covariances = Fi[1]
    probabilities = Fi[2]

    denominator = 0.0
    numerator = np.vstack([multivariate_normal.pdf(points, mean= centroids[k, :], cov= class_covariances[k, :, :]) * probabilities[k] for k in range(K)])
    for c in range(K):
        current_gaussian = multivariate_normal.pdf(points, mean= centroids[c, :], cov= class_covariances[c, :, :])
        denominator += current_gaussian * probabilities[c]
    probs = (numerator / denominator).T
    return probs


def m_step(X, memberships_probs):
    denom = 0.0
    centroids = np.zeros(shape=(5,2))
    for i in range(N):
        current_point = memberships_probs[i]
        for k in range(K):
            centroids[k] += current_point[k] * X[i]
        denom += np.array([np.sum(current_point[k]) for k in range(K)])
    centroids = (centroids.T / denom).T
    denom = 0.0
    class_covariances = np.zeros(shape=(5,2,2))
    for i in range(N):
        current_point = memberships_probs[i]
        for k in range(K):
            class_covariances[k] += current_point[k] * (X[i] - centroids[k])[None, :] * (X[i] - centroids[k])[:,
                                                                                        None]
        denom += np.array([np.sum(current_point[k]) for k in range(K)])
    class_covariances = (class_covariances.T / denom).T

    probabilities = [0,0,0,0,0]
    for i in range(N):
        current_point = memberships_probs[i]
        probabilities += np.array([np.sum(current_point[k]) for k in range(K)])
    probabilities = probabilities / N

    Fi = centroids, class_covariances, probabilities
    return (Fi)



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

# centroids = None
# memberships = None
# class_covariances = None
# probabilities = None
# memberships_probs = None

centroids = initial_centroids
memberships = update_memberships(centroids, points)
class_sizes = np.array(np.bincount(memberships))
class_covariances = np.array([
    np.mat(points[memberships == k, :] - centroids[k, :]).T * np.mat(points[memberships == k, :] - centroids[k, :]) / class_sizes[k]
    for k in range(K)])
probabilities = np.array(np.bincount(memberships) / N)

iteration = 1
Fi = []
for i in range(100):
    print("Iteration#{}:".format(iteration))

    Fi = (centroids, class_covariances, probabilities)
    memberships_probs = e_step(iteration, Fi)

    Fi= m_step(points, memberships_probs)
    centroids = Fi[0]
    class_covariances = Fi[1]
    probabilities = Fi[2]

    iteration = iteration + 1
print(centroids)
plt.subplot(1, 2, 2)
# plot_current_state(centroids, class_covariances, memberships, points)
plt.show()