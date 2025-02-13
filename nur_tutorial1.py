import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
from matplotlib.image import imread

# 1. Numerical errors

def sinc_numpy(x):
    return np.sin(x) / x


def prod(array):
    assert isinstance(array, (list, np.ndarray)), "Input should be np.ndarray or list"

    if len(array) == 0:
        raise ValueError('Trying to take the product of an empty array.')
    
    value = 1
    for i in array:
        value *= i
    return value


def cumprod(array):
    return np.array([prod(array[:i]) for i in range(1, len(array) + 1)])


def factorial(array):
    # assert np.all(isinstance(array, int)), "Input should be an integer."
    # assert np.all(array > -1), "Input should be greater than or equal than 0."

    array = np.array(array)  # Force list to array

    all_factorials = cumprod(np.arange(1, np.max(array) + 1))  # nth element contains n!
    all_factorials = np.insert(all_factorials, 0, 1).astype(int)
    requested_factorials = all_factorials[array]  # Will error if array contains floats or give wrong value if array contains negative numbers
    return requested_factorials


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

image = imread('M42_128.jpg')


if __name__ == '__main__':

    print(factorial(np.array([0, 5, 3])))

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

