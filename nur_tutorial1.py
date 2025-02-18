import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
from matplotlib.image import imread

# 1. Numerical errors

def sinc_numpy(x):
    return np.sin(x) / x


def cumsum(array):
    return np.array([np.sum(array[:i]) for i in range(1, len(array) + 1)])


def log_factorial(array):
    assert np.sum(array < 0) == 0, "Input should be greater than or equal than 0."

    array = np.array(array)  # Force list to array
    max_idx = np.max(array) + 1

    all_factorials = np.zeros(max_idx)
    all_factorials[1:] = cumsum( np.log(np.arange(1, max_idx)) )  # nth element contains log(n!)
    return all_factorials[array]


def factorial(array):
    return np.exp(log_factorial(array))


def sinc(x, n_max):

    # if x == 0:
    #     return 1

    value = 0
    for n in range(n_max):  # Leads to truncation error
        n_times_2 = 2 * n
        print(factorial(n_times_2 + 1))  # Overflow error not fixed yet
        value += (-1)**n * x**n_times_2 * float(factorial(n_times_2 + 1))**(-1)  # Can still be optimized for large n
    return value


# 3. Interpolation

# image = imread('M42_128.jpg')


if __name__ == '__main__':

    arr = np.array([200])
    # print(cumsum(arr))
    print(factorial(arr))

    # x = 7
    # highest_n_max = 100
    # n_max_to_test = np.arange(highest_n_max)
    # sinc2compare = sinc_numpy(x)

    # errors = []
    # for n_max in n_max_to_test:
    #     errors.append(sinc(x, n_max) / sinc2compare)

    # print(errors)

    # fig, ax = plt.figure(figsize=(8,6))
    # ax.scatter(n_max_to_test, errors)
    # plt.show()

