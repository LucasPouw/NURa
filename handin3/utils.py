import numpy as np


def readfile(filename):
        f = open(filename, 'r')
        data = f.readlines()[3:]  # Skip first 3 lines 
        nhalo = int(data[0])  # Number of halos
        radius = []
        
        for line in data[1:]:
            if line[:-1]!='#':
                radius.append(float(line.split()[0]))
        
        radius = np.array(radius, dtype=float)    
        f.close()
        return radius, nhalo  # Return the virial radius for all the satellites in the file, and the number of halos


def prod(array):
    assert isinstance(array, (list, np.ndarray)), "Input should be np.ndarray or list"

    if len(array) == 0:
        raise ValueError('Trying to take the product of an empty array.')
    
    value = 1
    for i in array:
        value *= i
    return value


def cumsum(array):
    return np.array([np.sum(array[:i]) for i in range(1, len(array) + 1)])


def log_factorial(array):
    '''
    Calculates the logarithm of the factorial of the input array element-wise.

    Input: np.ndarray or list of integers
    Output: np.ndarray of factorials of 
    '''
    assert np.sum(array < 0) == 0, "All elements should be greater than or equal to 0."

    array = np.array(array).astype(np.int32)  # Force list to array of integers
    max_idx = np.max(array) + 1

    all_factorials = np.zeros(max_idx, dtype=np.float32)
    all_factorials[1:] = cumsum( np.log(np.arange(1, max_idx)) )  # nth element contains log(n!)
    return all_factorials[array]  # Requested factorials


if __name__ == '__main__':
     pass