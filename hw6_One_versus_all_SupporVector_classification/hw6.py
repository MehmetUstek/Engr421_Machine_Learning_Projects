import numpy as np
import pandas as pd
import scipy.spatial.distance as dt
import matplotlib.pyplot as plt
import cvxopt as cvx

def safelog(x):
    return (np.log(x + 1e-100))

images = pd.read_csv('hw06_images.csv', header=None)
labels = pd.read_csv('hw06_labels.csv', header=None)

# Divide the data, 30000 to training, 5000 to test
X_train = images[:1000]
y_train = labels[:1000][0]
x_test = images[-4000:]
y_test = labels[-4000:][0]


K = 5
# get number of samples and number of features
N_train = len(y_train)
D_train = X_train.shape[1]


# define Gaussian kernel function
def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D**2 / (2 * s**2))
    return(K)


# calculate Gaussian kernel
s = 10
K_train = gaussian_kernel(X_train, X_train, s)
yyK = np.matmul(y_train[:, None], y_train[None, :]) * K_train

# set learning parameters
C = 10
epsilon = 1e-3

P = cvx.matrix(yyK)
q = cvx.matrix(-np.ones((N_train, 1)))
G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
A = cvx.matrix(1.0 * y_train[None, :])
b = cvx.matrix(0.0)

# use cvxopt library to solve QP problems
result = cvx.solvers.qp(P, q, G, h, A, b)
alpha = np.reshape(result["x"], N_train)
# Most of the alpha values should be 0.
alpha[alpha < C * epsilon] = 0
alpha[alpha > C * (1 - epsilon)] = C
# What is the purpose of cutting small values to 0?
# What is the purpose of cutting larger values that are close to C directly to C value 10.
# To find support indices!
# Non zero alpha coefficient vectors are called support vectors!


# find bias parameter
support_indices, = np.where(alpha != 0)
print(support_indices)
active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
# Alpha points between 0 and C. As seen in the ipynb subject to.
w0 = np.mean(
    y_train[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))


# calculate predictions on training samples
f_predicted = np.matmul(K_train, y_train[:,None] * alpha[:,None]) + w0

# calculate confusion matrix

y_predicted = 2 * (f_predicted > 0.0) - 1
confusion_matrix = pd.crosstab(np.reshape(y_predicted, N_train), y_train, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)