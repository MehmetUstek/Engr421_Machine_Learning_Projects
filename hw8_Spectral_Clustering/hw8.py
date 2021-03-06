import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa
import scipy.spatial.distance as dt

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


# Step 2
def update_centroids(memberships, X):
    if memberships is None:
        centroids = X[np.array([28, 142, 203, 270, 276]),]
    else:
        centroids = np.vstack([np.mean(X[memberships == k, :], axis=0) for k in range(K)])
    return (centroids)


# Step 1
def update_memberships(centroids, X):
    D = spa.distance_matrix(centroids, X)
    memberships = np.argmin(D, axis=0)
    return (memberships)


def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(X[:, 0], X[:, 1], ".", markersize=10, color="black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10,
                     color=cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize=12,
                 markerfacecolor=cluster_colors[c], markeredgecolor="black")
    plt.xlabel("x1")
    plt.ylabel("x2")


def k_means_clustering(X):
    centroids = None
    memberships = None
    iteration = 1
    while True:
        print("Iteration#{}:".format(iteration))

        old_centroids = centroids
        centroids = update_centroids(memberships, X)
        if np.alltrue(centroids == old_centroids):
            break
        old_memberships = memberships
        memberships = update_memberships(centroids, X)
        if np.alltrue(memberships == old_memberships):
            break

        iteration = iteration + 1
    return memberships


def L_symmetric(D, B):
    identity_matrix = np.identity(D.shape[0])
    D_temp = np.linalg.inv(np.sqrt(D))
    result = np.dot(D_temp, B).dot(D_temp)
    return np.asarray(identity_matrix - result)


def L_not_normalized(D, B):
    return D - B


def L_random_walk(D, B):
    identity_matrix = np.identity(D.shape[0])
    inverse = np.linalg.inv(D)
    result = np.dot(inverse, B)
    return np.asarray(identity_matrix - result)


def bij(X1, X2, threshold):
    D = dt.cdist(X1, X2, "euclidean")
    B = (D <= threshold).astype(int)
    for i in range(B.shape[0]):
        B[i][i] = 0
    return (B)


def get_D_matrix(B):
    D = np.diag(B.sum(axis=1))
    return D


def draw(B, X):
    plt.figure(figsize=(8, 8))
    row_iterator = 0
    for row in B:
        col_iterator = 0
        for j in row:
            if j:
                x1_temp = X[row_iterator]
                x2_temp = X[col_iterator]
                x_values = [x1_temp[0], x2_temp[0]]
                y_values = [x1_temp[1], x2_temp[1]]
                plt.plot(x_values, y_values, "-", color="gray")
            col_iterator += 1
        row_iterator += 1
    x1 = points[:, 0]
    x2 = points[:, 1]
    plt.plot(x1, x2, "k.", markersize=12)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()


def spectral_clustering(X, R=5):
    B = bij(X, X, 1.25)
    draw(B, X)
    D = get_D_matrix(B)
    L = L_symmetric(D, B)
    # L = D - B
    # L = L_random_walk(N, D, B)
    values, vectors = np.linalg.eig(L)
    values = np.real(values)
    vectors = np.real(vectors)

    vectors = vectors[:, np.argsort(values)]
    Z = vectors[:, 1:R + 1]

    memberships = k_means_clustering(Z)
    centroids = update_centroids(memberships, X)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_current_state(centroids, memberships, X)
    plt.show()

spectral_clustering(points)
