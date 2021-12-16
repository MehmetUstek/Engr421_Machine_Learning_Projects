import numpy as np
import pandas as pd
import scipy.spatial.distance as dt
import cvxopt as cvx

def safelog(x):
    return (np.log(x + 1e-100))

images = np.genfromtxt("hw06_images.csv", delimiter = ",")
labels = np.genfromtxt("hw06_labels.csv", delimiter = ",")

# Divide the data, 30000 to training, 5000 to test
X_train = images[:1000]
y_train = labels[:1000].astype(int)
# y_train = 2 * data_set[:,2].astype(int) - 1
x_test = images[-4000:]
y_test = labels[-4000:][0]
K = 5
# get number of samples and number of features
N_train = len(y_train)
D_train = X_train.shape[1]
# print(D_train)


# define Gaussian kernel function
def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D**2 / (2 * s**2))
    return(K)

# set learning parameters
C = 10
epsilon = 1e-3
s = 10
# calculate Gaussian kernel

K_train = gaussian_kernel(X_train, X_train, s)
# print("K_train", K_train)
f_predicted = []
y_predicted = []
# def func(y_truth):
for k in range(1,K+1):
    y_train_indexes = (y_train == k).astype(int)
    for index in range(len(y_train_indexes)):
        if not y_train_indexes[index]:
            y_train_indexes[index] = -1

    yyK = np.matmul(y_train_indexes[:, None], y_train_indexes[None, :]) * K_train
    # print(yyK)
    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N_train, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
    h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
    A = cvx.matrix(1.0 * y_train_indexes[None, :])
    b = cvx.matrix(0.0)

    # use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N_train)
    # Most of the alpha values should be 0.
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C
    # print(alpha)
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
        y_train_indexes[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))

    ## Training performance
    # calculate predictions on training samples
    f_predicted.append(np.matmul(K_train, y_train_indexes[:, None] * alpha[:, None]) + w0)
print(f_predicted)

# calculate confusion matrix

    # y_predicted = 2 * (f_predicted > 0.0) - 1
y_predicted = np.argmax(f_predicted, axis=0)
y_predicted = y_predicted.reshape(-1)
print(y_predicted)
print("shape", y_predicted.shape)
print("shape", y_train.shape)
confusion_matrix = pd.crosstab(y_predicted, y_train, rownames = ['y_predicted'], colnames = ['y_train'])
print(confusion_matrix)



confusion_matrix = pd.crosstab(y_predicted, y_test, rownames=['y_pred'], colnames=['y_truth'])
print("Confusion_matrix for test:")
print(confusion_matrix)
