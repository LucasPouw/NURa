import numpy as np

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
