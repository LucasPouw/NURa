import numpy as np
from tqdm import tqdm


def selection_sort(array: np.ndarray) -> None:
    ''' In-place sorting with selection sort algortihm. '''

    array = np.asarray(array)
    if len(array) <= 1:
        return None

    N = len(array)
    for i in tqdm(range(N-1)):
        imin = i
        for j in range(i+1, N):
            if array[j] < array[imin]:
                imin = j
        if imin != i:
            array[i], array[imin] = array[imin], array[i]


def quicksort(array: np.ndarray) -> None:
    ''' In-place sorting with quicksort algortihm '''

    array = np.asarray(array)
    N = len(array)

    if N == 0:
        return

    mid = N//2

    # Sorting first, middle and last element
    if array[0] > array[mid]:
        array[0], array[mid] = array[mid], array[0]
    if array[mid] > array[-1]:
        array[mid], array[-1] = array[-1], array[mid]
    if array[0] > array[mid]:
        array[0], array[mid] = array[mid], array[0]

    pivot = array[mid]

    if len(array) <= 3:  # Sub-array of size 3 is now sorted
        return

    j = N - 1
    i = 0
    i_flag, j_flag = False, False
    while j > i:
        if not i_flag:
            if array[i] >= pivot:
                i_flag = True
            else:
                i += 1

        if not j_flag:
            if array[j] <= pivot:
                j_flag = True
            else:
                j -= 1
        
        if i_flag and j_flag:
            if array[i] != array[j]:  # Only swap if they are not equal to get stable algorithm
                array[i], array[j] = array[j], array[i]
            else:  # Avoid infinite loops that happen when array[i] = array[j]
                i += 1

            i_flag, j_flag = False, False

    # Sorting sub-arrays
    quicksort(array[:i])
    quicksort(array[j+1:])  # Don't include pivot

    
if __name__ == '__main__':

    N = int(1e6)
    array = np.arange(N)
    # array[:100] = N//2
    np.random.shuffle(array)
    print(array)

    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    print('Initial array sorted?', is_sorted(array))

    # selection_sort(array)

    quicksort(array)
    print(array)
    print('Final array sorted?', is_sorted(array))
