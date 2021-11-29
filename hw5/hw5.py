import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def safelog2(x):
    if x == 0:
        return(0)
    else:
        return(np.log2(x))

data_set = np.genfromtxt("hw05_data_set.csv", delimiter=",")
# Divide the data, 150 to training, rest to test
train_set = data_set[1:151]
test_set = data_set[-122:]
x_train = train_set[:, 0]
y_train = train_set[:, 1]
x_test = test_set[:, 0]
y_test = test_set[:, 1]

K = np.max(y_train)
N = train_set.shape[0]
D = train_set.shape[1]
print(K, D)
# get numbers of train and test samples
N_train = len(y_train)
N_test = len(y_test)

# create necessary data structures
node_indices = {}
is_terminal = {}
need_split = {}

node_features = {} # j value in x_j > w_m0 in the lecture notes.
node_splits = {} # w_m0 value in x_j > w_m0 in the lecture notes.
node_means = {}

# put all training instances into the root node
node_indices[1] = np.array(range(N_train))
is_terminal[1] = False
need_split[1] = True

# print(node_indices)
# All of the indices are received by root node.
P = 25
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
        if len(data_indices) <= P:
            # TODO: Change terminal
            is_terminal[split_node] = True
            node_mean = np.mean(y_train[data_indices])
            node_means[split_node] = node_mean
            # Check whether this leaf is pure.
        else:
            is_terminal[split_node] = False

            best_scores = np.repeat(0.0, D)
            best_splits = np.repeat(0.0, D)
            # Initialize 2 arrays with size of features.
            # For all features, for all possible splits
            # Try splits for 1st feature, 2nd, 3rd, get the best one.
            for d in range(D):  # For all features
                unique_values = np.sort(np.unique(train_set[data_indices, d]))
                # Since we cannot differentiate same value, we want unique values.
                # An overlapping point for example 4.6 cannot give different class values y.
                split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values) - 1)]) / 2
                # split_positions is the mean point to split the data.
                # For example, array = [[2.4, 3.5, 4.5, 5.1]]
                # split positions will be:
                # [2.95, 4. , 4.8]
                # First one is 2.4 + 3.5 /2 and so on.
                split_scores = np.repeat(0.0, len(split_positions))
                for s in range(len(split_positions)):  # For all possible splits
                    left_indices = data_indices[train_set[data_indices, d] > split_positions[s]]
                    right_indices = data_indices[train_set[data_indices, d] <= split_positions[s]]
                    # TODO: Change score.
                    data_points_size = 1 / len(data_indices)
                    score = 0.0
                    average_left_indices = np.mean(y_train[left_indices])
                    average_right_indices = np.mean(y_train[right_indices])
                    for c in left_indices:
                        score += (c - average_left_indices)**2
                    for c in right_indices:
                        score += (c - average_right_indices) ** 2
                    score = score * data_points_size
                    split_scores[s] = score



                best_scores[d] = np.min(split_scores)
                best_splits[d] = split_positions[np.argmin(split_scores)]
            # decide where to split on which feature
            split_d = np.argmin(best_scores)

            node_features[split_node] = split_d  # j index in the lecture notes.
            node_splits[split_node] = best_splits[split_d]  # w_m0 in the lecture notes.

            # create left node using the selected split
            left_indices = data_indices[train_set[data_indices, split_d] > best_splits[split_d]]  # True branch
            node_indices[2 * split_node] = left_indices
            is_terminal[2 * split_node] = False
            need_split[2 * split_node] = True

            # create right node using the selected split
            right_indices = data_indices[train_set[data_indices, split_d] <= best_splits[split_d]]
            node_indices[2 * split_node + 1] = right_indices
            is_terminal[2 * split_node + 1] = False
            need_split[2 * split_node + 1] = True

max_val = max(x_train)
origin = min(x_train)
step = 1 / 2000
step_inverse = 2000
data_interval = np.arange(origin, max_val, step)




plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, "b.", markersize=10, label='training')
plt.plot(x_test, y_test, "r.", markersize=10, label='test')
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(loc='upper left')



# extract rules
terminal_nodes = [key for key, value in is_terminal.items() if value == True]
for terminal_node in terminal_nodes:
    index = terminal_node
    rules = np.array([])
    while index > 1:
        parent = np.floor(index / 2)
        if index % 2 == 0:
            # if node is left child of its parent
            rules = np.append(rules, "x{:d} > {:.2f}".format(node_features[parent] + 1, node_splits[parent]))
        else:
            # if node is right child of its parent
            rules = np.append(rules, "x{:d} <= {:.2f}".format(node_features[parent] + 1, node_splits[parent]))
        index = parent
    rules = np.flip(rules)
    # print(rules)
    print("{} => {}".format(rules, node_means[terminal_node]))

# traverse tree for training data points
y_predicted = np.repeat(0, N_train)
for i in range(N_train):
    index = 1
    while True:
        if is_terminal[index] == True:
            y_predicted[i] = node_means[index]
            break
        else:
            if train_set[i, node_features[index]] > node_splits[index]:
                index = index * 2
            else:
                index = index * 2 + 1

def calculate_RMSE():
    error_list = []
    for i in range(len(x_test)):
        error = (y_test[i] - y_predicted[int((x_test[i]))]) ** 2
        error_list.append(error)
    rmse = np.sqrt(np.sum(error_list) / len(x_test))
    print("RMSE on training set is", rmse, " when P is", P)

calculate_RMSE()
