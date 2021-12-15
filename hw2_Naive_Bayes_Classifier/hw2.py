import numpy as np
import pandas
import math


def safelog(x):
    return (np.log(x + 1e-100))


images = pandas.read_csv('hw02_images.csv', header=None)
labels = pandas.read_csv('hw02_labels.csv', header=None)

# Divide the data, 30000 to training, 5000 to test
x_train = images[:30000]
y_train = labels[:30000][0]
x_test = images[-5000:]
y_test = labels[-5000:][0]

K = 5
# Estimate the parameters
sample_means = np.array(([np.mean(x_train[y_train == (c + 1)], axis=0) for c in range(K)]))
print("sample_means:")
print(sample_means)

class_priors = [np.mean(y_train == (c + 1)) for c in range(K)]

sample_deviations = np.array(
    ([np.sqrt(np.mean((x_train[y_train == (c + 1)] - sample_means[c]) ** 2)) for c in range(K)]))
print("sample_deviations:")
print(sample_deviations)

print("class_priors:")
print(class_priors)


def get_score_values(x):
    return np.sum(np.stack([-0.5 * safelog(2 * math.pi * sample_deviations[c] ** 2)
                            - 0.5 * (x - sample_means[c]) ** 2 / sample_deviations[c] ** 2
                            + safelog(class_priors[c])
                            for c in range(K)]), axis=2)


# ---- Train
score_values1 = get_score_values(x_train)
y_predicted = np.argmax(score_values1, axis=0) + 1

# Confusion Matrix

confusion_matrix = pandas.crosstab(y_predicted, y_train, rownames=['y_pred'], colnames=['y_truth'])
print("Confusion_matrix for training:")
print(confusion_matrix)

# ---- Test
score_values2 = get_score_values(x_test)
score_values2 = np.transpose(score_values2)
y_predicted = np.argmax(score_values2, axis=1) + 1

# Confusion Matrix

confusion_matrix = pandas.crosstab(y_predicted, y_test, rownames=['y_pred'], colnames=['y_truth'])
print("Confusion_matrix for test:")
print(confusion_matrix)
