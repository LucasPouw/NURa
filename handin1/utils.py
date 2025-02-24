import numpy as np


def cumsum(array):
    return np.array([np.sum(array[:i]) for i in range(1, len(array) + 1)])


def polynomial(x, coeff):
    return np.dot(x[:,np.newaxis]**np.arange(len(coeff)), coeff)


if __name__ == '__main__':
    pass
