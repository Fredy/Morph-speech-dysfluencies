import numpy as np
from math import sqrt


def cross_correlation(x_array, y_array):
    """Cross correlation: [-1,1]: [negative relation, positive relation]"""
    x_mean = x_array.mean()
    y_mean = y_array.mean()

    tmp = 0
    x_tmp = 0
    y_tmp = 0
    for x, y in zip(x_array, y_array):
        tmp += (x - x_mean) * (y - y_mean)
        x_tmp += (x - x_mean) ** 2
        y_tmp += (y - y_mean) ** 2

    return tmp / sqrt(x_tmp * y_tmp)


def distance(x_array, y_array):
    """Euclidean distance: [0, X]: [similar, dissimilar]"""
    tmp = 0
    for x, y in zip(x_array, y_array):
        tmp += (x - y) ** 2

    return sqrt(tmp)


def self_similarity(arr, distance_func, normalize=False):
    size = len(arr)
    matrix = np.empty((size, size))

    for i in range(size):
        for j in range(size):
            matrix[i, j] = matrix[j, i] = distance_func(arr[i], arr[j])

    if normalize:
        max_ = matrix[0, 0]  # Diagonal has the highest value
        min_ = matrix.min()

        if max_ == 0:
            # Case for euclidean distance
            max_ = matrix.max()
            matrix = 1 - (matrix - min_) / (max_ - min_)
        else:
            matrix = (matrix - min_) / (max_ - min_)

    return matrix
