import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters
np.random.seed(421)
# generating data
# class_means = np.array([-3.0,-1.0,+3.0])
#
# class_deviations = np.array([1.2, 1.0, 1.3])
# #number of data points for each class.
# class_sizes = np.array([40, 30, 50])
#
# points1 = np.random.normal(class_means[0],class_deviations[0],class_sizes[0])
# points2 = np.random.normal(class_means[1],class_deviations[1],class_sizes[1])
# points3 = np.random.normal(class_means[2],class_deviations[2],class_sizes[2])
# points = np.concatenate((points1,points2,points3))
# # generating the labels.
# y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2])))
#
# # print(points)
# # print(y)
#
# # Exporting data
# # First column is xi, second column is yi column.
# np.savetxt("lab01_data_set.csv", np.stack((points, y), axis=1), delimiter=",")
#
# # Plotting Data
# # Used for drawing purposes.
# data_interval = np.linspace(-7, +7, 1401)
# print(data_interval)
# # Probability density function for the normal distribution
# density1 = stats.norm.pdf(data_interval, loc= class_means[0], scale= class_deviations[0])
# density2 = stats.norm.pdf(data_interval, loc= class_means[1], scale= class_deviations[1])
# density3 = stats.norm.pdf(data_interval, loc= class_means[2], scale= class_deviations[2])
#
# plt.figure(figsize= (10,6))
# plt.plot(data_interval, density1, "r")
# plt.plot(data_interval, density2, "g")
# plt.plot(data_interval, density3, "b")
# plt.plot(points1, np.repeat(-0.01, class_sizes[0]),"r.", markersize= 12)
# plt.plot(points2, np.repeat(-0.02, class_sizes[1]),"g.", markersize= 12)
# plt.plot(points3, np.repeat(-0.03, class_sizes[2]),"b.", markersize= 12)
# plt.xlabel("x")
# plt.ylabel("density")
# plt.show()

#Starting the machine learning
# Importing data
data_set = np.genfromtxt("lab01_data_set.csv", delimiter= ",")
x = data_set[:,0]
y = data_set[:,1].astype(int)

# number of classes
K = np.max(y)
# number of data points
N = data_set.shape[0]

# Parameter Estimation
sample_means = [np.mean(x[y == (c + 1)]) for c in range(K)]
# y == c + 1 to get the x1s for example.
# x[y == ...] thing is where you slice x array into three pieces.
# with mean you get the mean of the data in x1 for example,
# or first 40 items in this case.

print(sample_means)
# parameters that we gave was, -3, -1, 3 so it is close but not perfect.

sample_deviations = [np.sqrt(np.mean((x[y == (c + 1)] - sample_means[c])**2)) for c in range(K)]
print(sample_deviations)
# Correct values were, 1.2, 1.0, 1.3
# Still close but not much.

class_priors = [np.mean(y == (c + 1)) for c in range(K)]
print(class_priors)
# 40 / 120, 30 /120 , 50 /120

# Parametric Classification
# We are giving 1401 data points to test the classification, between the intervals -7 to 7.

data_interval = np.linspace(-7, +7, 1401)

#gc(x) = log(x | y = c) + log P(y=c) = - 1/2 * log(2pi sigmac^2 )
# - (x - mu)^2 / 2 sigmac^2 + log P(y=c)
score_values = np.stack([-0.5 * np.log(2 * math.pi * sample_deviations[c]**2)
                         - 0.5 * (data_interval - sample_means[c])**2 / sample_deviations[c]**2
                         + np.log(class_priors[c])
                         for c in range(K)])
print("score_values:")
print(score_values)

# Score Functions
plt.figure(figsize=(10,6))
plt.plot(data_interval, score_values[0,:], "r")
plt.plot(data_interval, score_values[1,:], "g")
plt.plot(data_interval, score_values[2,:], "b")
plt.xlabel("x")
plt.ylabel("score")
plt.show()
# Based on the graph, you should pick what is on the top.
# So, from -7 to -2 you should pick red since it is on top.
# From -2 to 0.5, green, rest is blue.

# Posterior Probabilities
# calculate log posteriors:
log_posteriors = score_values - [np.max(score_values[:, r]) +
                                 np.log(np.sum(np.exp(score_values[:, r] - np.max(score_values[:, r]))))
                                 for r in range(score_values.shape[1])
                                 ]
print("log_posteriors")
print(log_posteriors)

# Posteriors


plt.figure(figsize = (10, 6))
# plot posterior probability of the first class
plt.plot(data_interval, np.exp(log_posteriors[0,:]), "r")
# plot posterior probability of the second class
plt.plot(data_interval, np.exp(log_posteriors[1,:]), "g")
# plot posterior probability of the third class
plt.plot(data_interval, np.exp(log_posteriors[2,:]), "b")

class_assignments = np.argmax(score_values, axis = 0)

#plot region where the first class has the highest probability
plt.plot(data_interval[class_assignments == 0],
         np.repeat(-0.05, np.sum(class_assignments == 0)), "r.", markersize = 10)
#plot region where the second class has the highest probability
plt.plot(data_interval[class_assignments == 1],
         np.repeat(-0.10, np.sum(class_assignments == 1)), "g.", markersize = 10)
#plot region where the third class has the highest probability
plt.plot(data_interval[class_assignments == 2],
         np.repeat(-0.15, np.sum(class_assignments == 2)), "b.", markersize = 10)

plt.xlabel("x")
plt.ylabel("probability")

plt.show()