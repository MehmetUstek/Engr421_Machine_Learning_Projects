import numpy as np
import matplotlib.pyplot as plt
import pandas

def safelog(x):
    return(np.log(x + 1e-100))


images = pandas.read_csv('hw02_images.csv', header=None)
labels = pandas.read_csv('hw02_labels.csv', header=None)

# Divide the data, 30000 to training, 5000 to test
x_train = images[:30000]
y_train = labels[:30000][0]
x_test = images[-5000:]
y_test = labels[-5000:][0]

K = 5

# Estimate the mean parameters
sample_means = [np.mean(x_train[y_train == (c+1)], axis=0) for c in range(K)]
print("sample_means:")
print(sample_means)

sample_deviations = [np.sqrt(np.mean((x_train[y_train == (c + 1)] - sample_means[c])**2)) for c in range(K)]
print("sample_deviations:")
print(sample_deviations)

class_priors = [np.mean(y_train == (c + 1)) for c in range(K)]

print("class_priors:")
print(class_priors)

