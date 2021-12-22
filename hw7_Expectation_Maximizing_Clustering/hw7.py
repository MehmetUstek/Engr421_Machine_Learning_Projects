import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa

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


# Step 2
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
# Step 1
def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    # Distance between all centroids and data points.
    # D matrix is KxN.
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    # Nearest centroid.
    memberships = np.argmin(D, axis = 0)
    return(memberships)

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
    centroids = update_centroids(memberships, points)
    if np.alltrue(centroids == old_centroids):
        break
    else:
        plt.figure(figsize = (12, 6))
        plt.subplot(1, 2, 1)
        plot_current_state(centroids, memberships, points)

    old_memberships = memberships
    memberships = update_memberships(centroids, points)
    if np.alltrue(memberships == old_memberships):
        plt.show()
        break
    else:
        plt.subplot(1, 2, 2)
        plot_current_state(centroids, memberships, points)
        plt.show()

    iteration = iteration + 1