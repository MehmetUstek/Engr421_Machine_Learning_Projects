import numpy as np
import pandas as pd
import scipy.spatial.distance as dt
import cvxopt as cvx
import matplotlib.pyplot as plt

images = np.genfromtxt("hw06_images.csv", delimiter=",")
labels = np.genfromtxt("hw06_labels.csv", delimiter=",")

# Divide the data, 1000 to training, 4000 to test
X_train = images[:1000]
y_train = labels[:1000].astype(int)
X_test = images[-4000:]
y_test = labels[-4000:].astype(int)
K = 5
# get number of samples and number of features
N_train = len(y_train)
N_test = len(y_test)


# define Gaussian kernel function
def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D ** 2 / (2 * s ** 2))
    return (K)


def func(y_truth, X_given, N_given, C, s, epsilon):
    f_predicted = []
    K_trick = gaussian_kernel(X_given, X_given, s)
    for k in range(1, K + 1):
        y_indexes = (y_truth == k) * 2 - 1
        yyK = np.matmul(y_indexes[:, None], y_indexes[None, :]) * K_trick
        # print(yyK)
        P = cvx.matrix(yyK)
        q = cvx.matrix(-np.ones((N_given, 1)))
        G = cvx.matrix(np.vstack((-np.eye(N_given), np.eye(N_given))))
        h = cvx.matrix(np.vstack((np.zeros((N_given, 1)), C * np.ones((N_given, 1)))))
        A = cvx.matrix(1.0 * y_indexes[None, :])
        b = cvx.matrix(0.0)

        # use cvxopt library to solve QP problems
        result = cvx.solvers.qp(P, q, G, h, A, b)
        alpha = np.reshape(result["x"], N_given)
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
        # print(support_indices)
        active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
        # Alpha points between 0 and C. As seen in the ipynb subject to.
        w0 = np.mean(
            y_indexes[active_indices] * (
                        1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))

        ## Training performance
        # calculate predictions on training samples
        f_predicted.append(np.matmul(K_trick, y_indexes[:, None] * alpha[:, None]) + w0)

    y_predicted = np.argmax(f_predicted, axis=0) + 1
    y_predicted = y_predicted.reshape(-1)
    return y_predicted

def predict(y_predicted_train):
    kernel = gaussian_kernel(X_test, X_train, s)
    # for alpha in kernel:
    #     alpha[alpha < C * epsilon] = 0
    #     alpha[alpha > C * (1 - epsilon)] = C
    #     support_indices, = np.where(alpha != 0)
    #     # print(support_indices)
    #     active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    #     np.argmax(kernel, axis=1)
    y_pred = np.argmax(kernel, axis=1)
    ret_val = y_predicted_train[y_pred]
    return ret_val


C = 10
epsilon = 1e-3
s = 10

y_predicted_train1 = func(y_train, X_train, N_train, C, s, epsilon)
confusion_matrix = pd.crosstab(y_predicted_train1, y_train, rownames=['y_predicted'], colnames=['y_train'])
print("Confusion_matrix for train:")
print(confusion_matrix)

y_predicted_test1 = predict(y_predicted_train1)
confusion_matrix = pd.crosstab(y_predicted_test1, y_test, rownames=['y_predicted'], colnames=['y_test'])
print("Confusion_matrix for test:")
print(confusion_matrix)


# Accuracy score
def accuracy_score(y_pred, y_truth):
    score = 0
    for index in range(len(y_truth)):
        if y_pred[index] == y_truth[index]:
            score += 1
    accuracy_score = float(score / len(y_truth))
    return accuracy_score


# Visualization
C_values = [0.1, 1,10, 100,1000]
s = 1
train_list = []
test_list = []
for c in C_values:
    # Train
    y_predicted_train = func(y_train, X_train, N_train, c, s, epsilon)
    accuracy_score_train = accuracy_score(y_predicted_train, y_train)
    train_list.append(accuracy_score_train)

    # Test
    y_predicted_test = func(y_test, X_test, N_test, c, s, epsilon)
    accuracy_score_test = accuracy_score(y_predicted_test, y_test)
    test_list.append(accuracy_score_test)
    confusion_matrix = pd.crosstab(y_predicted_test, y_test, rownames=['y_predicted'], colnames=['y_test'])
    print("Confusion_matrix for test:")
    print(confusion_matrix)

plt.figure(figsize=(6, 6))
plt.plot(C_values, train_list, "-ob", markersize=4, label='training')
plt.plot(C_values, test_list, "-or", markersize=4, label='test')
plt.xlabel("Regularization Parameter (C)")
plt.ylabel("Accuracy")
plt.legend(loc='upper left')
plt.show()
