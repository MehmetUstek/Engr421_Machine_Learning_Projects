import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spa
from scipy.stats import multivariate_normal
import scipy.linalg as linalg
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
# plt.show()
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
        centroids = X[np.random.choice(range(N), K, False),]
    else:
        centroids = np.vstack([np.mean(X[memberships == k,:], axis = 0) for k in range(K)])
    return(centroids)

# Step 1
def update_memberships(centroids, X):
    D = spa.distance_matrix(centroids, X)
    memberships = np.argmin(D, axis = 0)
    return(memberships)

def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(X[:,0], X[:,1], ".", markersize = 10, color = "black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize = 12,
                 markerfacecolor = cluster_colors[c], markeredgecolor = "black")
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
        else:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plot_current_state(centroids, memberships, X)

        old_memberships = memberships
        memberships = update_memberships(centroids, X)
        if np.alltrue(memberships == old_memberships):
            plt.show()
            break
        else:
            plt.subplot(1, 2, 2)
            plot_current_state(centroids, memberships, X)
            plt.show()

        iteration = iteration + 1

def L_symmetric(N, D, B):
    identity_matrix = np.identity(N)
    result = np.dot(np.linalg.inv(D) ** (1 / 2), B)
    result = np.dot(result, np.linalg.inv(D) ** (1 / 2))
    return identity_matrix - result

# Returns eigenvalues, eigenvectors.
def PCA(X):
    # calculate the covariance matrix
    Sigma_X = np.cov(np.transpose(X))

    # calculate the eigenvalues and eigenvectors
    values, vectors = linalg.eig(Sigma_X)
    values = np.real(values)
    vectors = np.real(vectors)
    return values, vectors

# define Gaussian kernel function
def bij(X1, X2, threshold):
    D = dt.cdist(X1, X2)
    B = (D < threshold).astype(int)
    for i in range(B.shape[0]):
        B[i][i] = 0
    return (B)

def get_D_matrix(B):
    D = np.zeros(shape=(B.shape))
    row_number = 0
    for row in B:
        sum = np.sum(row)
        D[row_number][row_number] = sum
        row_number += 1
    return D



def spectral_clustering(X):
    B = bij(X, X, 1.25)
    #TODO: D matrix now
    D = get_D_matrix(B)
    #TODO: L matrix
    L = L_symmetric(N, D, B)

    return L

spectral_clustering(points)

