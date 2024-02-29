"""
Created on Feb 29, 2024

@author: Ming Xu
"""

import numpy as np
import matplotlib.pyplot as plt


def load_data_set():
    """load data set."""
    data_matrix = []
    label_matrix = []
    fr = open(
        "/home/allride/machine_learning_in_action/logistic_regression/test_set.txt",
        encoding="utf-8",
    )
    for line in fr.readlines():
        line_array = line.strip().split()
        data_matrix.append([1.0, float(line_array[0]), float(line_array[1])])
        label_matrix.append(int(line_array[2]))
    return data_matrix, label_matrix


def sigmoid(in_x):
    """sigmoid function"""
    return 1.0 / (1 + np.exp(-in_x))


def gradient_ascent(data_matrix_in, class_labels):
    """gradient ascent method"""
    data_matrix = np.mat(data_matrix_in)
    label_matrix = np.mat(class_labels).transpose()
    m, n = np.shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = label_matrix - h
        weights = weights + alpha * data_matrix.transpose() * error
    return weights


def plot_best_fit(weights):
    """plot best fit line"""
    data_matrix, label_matrix = load_data_set()
    data_array = np.array(data_matrix)
    n = np.shape(data_array)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_matrix[i]) == 1:
            xcord1.append(data_array[i, 1])
            ycord1.append(data_array[i, 2])
        else:
            xcord2.append(data_array[i, 1])
            ycord2.append(data_array[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c="red", marker="s")
    ax.scatter(xcord2, ycord2, s=30, c="green")
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


def stochastic_gradient_ascent0(data_matrix, class_labels):
    """stochastic gradient ascent"""
    m, n = np.shape(data_matrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_matrix[i]
    return weights


def stochastic_gradient_ascent1(data_matrix, class_labels, num_iter=150):
    """stochastic gradient ascent"""
    m, n = np.shape(data_matrix)
    weights = np.ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            alpha = (
                4 / (1.0 + j + i) + 0.0001
            )  # alpha decreases with iteration, does not go to 0 because of the constant
            rand_index = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]
            del data_index[rand_index]
    return weights


data_matrix_global, label_matrix_global = load_data_set()
# weights_global = gradient_ascent(data_matrix_global, label_matrix_global)
# plot_best_fit(weights_global.getA())
weights_global = stochastic_gradient_ascent1(
    np.array(data_matrix_global), label_matrix_global
)
plot_best_fit(weights_global)
