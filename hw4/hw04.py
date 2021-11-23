import math
import matplotlib.pyplot as plt
import numpy as np


def safelog(x):
    return (np.log(x + 1e-100))


data_set = np.genfromtxt("hw04_data_set.csv", delimiter=",")
# Divide the data, 150 to training, rest to test
train_set = data_set[1:151]
test_set = data_set[-122:]
# print(train_set)
x_train = train_set[:, 0]
y_train = train_set[:, 1].astype(int)
x_test = test_set[:, 0]
y_test = test_set[:, 1].astype(int)

# Probably will change.
K = np.max(y_train)
N = train_set.shape[0]
# print(K, N)

bin_width = 0.37
origin = 1.5

max_val = max(x_train)

left_borders = np.arange(origin, max_val, bin_width)
right_borders = np.arange(origin + bin_width, max_val + bin_width, bin_width)

##############################################################
## Regressogram
def xi_in_same_bin(x, xi):
    if (left_borders[x] <= xi) and (xi <= right_borders[x]):
        return True
    else:
        return False


def g(x):
    sum = 0
    denominator_sum = 0
    for i in range(len(x_train)):
        xi = x_train[i]
        if xi_in_same_bin(x, xi):
            sum += y_train[i]
            denominator_sum += 1
    return sum / denominator_sum


y_predicted = [g(x) for x in range(len(left_borders))]

plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, "b.", markersize=10, label='training')
plt.plot(x_test, y_test, "r.", markersize=10, label='test')
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [y_predicted[b], y_predicted[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [y_predicted[b], y_predicted[b + 1]], "k-")
plt.legend(loc='upper left')
plt.show()

# Calculating RMSE.
error = 0
for i in range(len(x_test)):
    xi = x_test[i]
    for j in range(len(left_borders)):
        if xi_in_same_bin(j, xi):
            err = (y_test[i] - y_predicted[int((x_test[i]-origin)/bin_width)])**2
            error += err
result = np.sqrt(error / len(x_test))
print("Regressogram => RMSE is", result, " when h is", bin_width)


##############################################################

# Mean Smoother
bin_width = 0.37
def w(u):
    if abs(u) <= 1/2:
        return True
    else:
        return False
step = 1 / 1000
step_inverse = 1000
data_interval = np.arange(origin, max_val, step)

def mean_smoother_g(x):
    sum = 0
    denominator_sum = 0
    for i in range(len(x_train)):
        xi = x_train[i]
        if w((x - xi) / bin_width):
            sum += y_train[i]
            denominator_sum += 1
    return sum / denominator_sum

mean_smoother_y_predicted = [mean_smoother_g(x) for x in data_interval]

plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, "b.", markersize=10, label='training')
plt.plot(x_test, y_test, "r.", markersize=10, label='test')
plt.plot(data_interval, mean_smoother_y_predicted, "k")
plt.legend(loc='upper left')
plt.show()

# Calculating Mean Smoother RMSE:
def calculate_RMSE(smoother):
    error_list = []
    for i in range(len(x_test)):
        error = (y_test[i] - smoother[int((x_test[i] - origin) * step_inverse)]) ** 2
        error_list.append(error)
    rmse = np.sqrt(np.sum(error_list) / len(x_test))
    print("Mean Smoother => RMSE is", rmse, " when h is", bin_width)

calculate_RMSE(mean_smoother_y_predicted)
##############################################################

# Kernel Smoother
bin_width = 0.37
def K(u):
    return (1.0/math.sqrt(2 * math.pi)) * np.exp(- (u**2)/2)

def kernel_smoother_g(x):
    sum = 0
    denominator_sum = 0
    for i in range(len(x_train)):
        xi = x_train[i]
        u = (x - xi) / bin_width
        k = K(u)
        sum += k * y_train[i]
        denominator_sum += k
    return sum / denominator_sum


kernel_smoother_y_predicted = [kernel_smoother_g(x) for x in data_interval]

plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, "b.", markersize=10, label='training')
plt.plot(x_test, y_test, "r.", markersize=10, label='test')
plt.plot(data_interval, kernel_smoother_y_predicted, "k")
plt.legend(loc='upper left')
plt.show()

calculate_RMSE(kernel_smoother_y_predicted)