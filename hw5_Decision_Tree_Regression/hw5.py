import numpy as np
import matplotlib.pyplot as plt

data_set = np.genfromtxt("hw05_data_set.csv", skip_header=1, delimiter=",")
# Divide the data, 150 to training, rest to test
train_set = data_set[:150]
test_set = data_set[-122:]
x_train = train_set[:, 0]
y_train = train_set[:, 1]
x_test = test_set[:, 0]
y_test = test_set[:, 1]
K = np.max(y_train)
N = train_set.shape[0]
# print(K, N)
# get numbers of train and test samples
N_train = len(y_train)
N_test = len(y_test)


# print(len(x_train))
# print(len(x_test))
# print(y_train)


def learn(P):
    # create necessary data structures
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_splits = {}  # w_m0 value in x_j > w_m0 in the lecture notes.
    means = {}

    # put all training instances into the root node
    node_indices[1] = np.array(range(N_train))
    is_terminal[1] = False
    need_split[1] = True
    # learning algorithm
    while True:
        # find nodes that need splitting
        split_nodes = [key for key, value in need_split.items() if value == True]
        # check whether we reach all terminal nodes
        if len(split_nodes) == 0:
            break
        # find best split positions for all nodes
        for split_node in split_nodes:
            data_indices = node_indices[split_node]  # 0,1,...74
            need_split[split_node] = False
            node_mean = np.mean(y_train[data_indices])
            if len(data_indices) <= P:
                is_terminal[split_node] = True
                means[split_node] = node_mean
            else:
                is_terminal[split_node] = False
                unique_values = np.sort(np.unique(x_train[data_indices]))
                # Since we cannot differentiate same value, we want unique values.
                # An overlapping point for example 4.6 cannot give different class values y.
                split_positions = (unique_values[1:len(unique_values)] + unique_values[
                                                                         0:(len(unique_values) - 1)]) / 2
                split_scores = np.repeat(0.0, len(split_positions))
                for s in range(len(split_positions)):  # For all possible splits
                    left_indices = data_indices[x_train[data_indices] >= split_positions[s]]
                    right_indices = data_indices[x_train[data_indices] < split_positions[s]]
                    data_points_size = 1 / len(data_indices)
                    score = 0.0
                    average_left_indices = np.mean(y_train[left_indices])
                    average_right_indices = np.mean(y_train[right_indices])

                    for c in left_indices:
                        score += (y_train[c] - average_left_indices) ** 2
                    for c in right_indices:
                        score += (y_train[c] - average_right_indices) ** 2
                    score = score * data_points_size
                    split_scores[s] = score

                best_splits = split_positions[np.argmin(split_scores)]
                node_splits[split_node] = best_splits  # w_m0 in the lecture notes.

                # create left node using the selected split
                left_indices = data_indices[x_train[data_indices] >= best_splits]  # True branch
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True

                # create right node using the selected split
                right_indices = data_indices[x_train[data_indices] < best_splits]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True
    return is_terminal, node_splits, means


def predict(point, is_terminal, node_splits, means):
    index = 1
    while (True):
        if is_terminal[index] == True:
            return means[index]
        else:
            if point > node_splits[index]:
                index = 2 * index
            else:
                index = 2 * index + 1


# set = data_set[:,0]
max_val = max(x_train)
origin = min(x_train)
data_points = np.linspace(origin - 0.1, max_val + 0.1, 1000)

is_terminal, node_splits, means = learn(25)

y_predicted = [predict(data_points[i], is_terminal, node_splits, means) for i in range(len(data_points))]

plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, "b.", markersize=10, label='training')
plt.plot(x_test, y_test, "r.", markersize=10, label='test')
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(loc='upper left')
plt.plot(data_points, y_predicted, "k")
plt.show()


def calculate_RMSE(P, y_predicted, y_truth, x_length, type, is_print: bool):
    error = 0.0
    for i in range(x_length):
        error += (y_truth[i] - y_predicted[i]) ** 2
    rmse = np.sqrt(error / x_length)
    if is_print:
        print("RMSE on ", type, " set is", rmse, " when P is", P)
    return rmse


y_predicted_train = [predict(i, is_terminal, node_splits, means) for i in x_train]
y_predicted_test = [predict(i, is_terminal, node_splits, means) for i in x_test]
calculate_RMSE(25, y_predicted_train, y_train, N_train, "training", is_print=True)
calculate_RMSE(25, y_predicted_test, y_test, N_test, "test", is_print=True)

# Different P values.
# P = 5
P_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# P = 5
rmse_train_list = []
rmse_test_list = []
for P in P_values:
    is_terminal, node_splits, means = learn(P)
    y_predicted_train = [predict(i, is_terminal, node_splits, means) for i in x_train]
    y_predicted_test = [predict(i, is_terminal, node_splits, means) for i in x_test]
    rmse_train = calculate_RMSE(P, y_predicted_train, y_train, N_train, "training", is_print=False)
    rmse_test = calculate_RMSE(P, y_predicted_test, y_test, N_test, "test", is_print=False)
    rmse_train_list.append(rmse_train)
    rmse_test_list.append(rmse_test)

plt.figure(figsize=(6, 6))
plt.plot(P_values, rmse_train_list, "-ob", markersize=4, label='training')
plt.plot(P_values, rmse_test_list, "-or", markersize=4, label='test')
plt.xlabel("Pre-pruning size (P)")
plt.ylabel("RMSE")
plt.legend(loc='upper right')
plt.show()
