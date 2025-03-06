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
    ''' In-place sorting with quicksort algortihm. '''

    array = np.asarray(array)
    print(array)

    N = len(array)
    pivot = np.median([array[0], array[N//2], array[-1]]).astype(array.dtype)

    pivot_idx = np.where(array == pivot)[0][0]
    array[N//2], array[pivot_idx] = array[pivot_idx], array[N//2]
    print('Pivot in middle', array)

    i_flag, j_flag = False, False
    for i, j in zip(range(N), range(N)[::-1]):
        print(i, j)
        if j <= i:
            print('j <= i', array)
            break

        if array[i] >= pivot:
            switch_i = i
            i_flag = True

        if array[j] <= pivot:
            switch_j = j
            j_flag = True
        
        if i_flag * j_flag:
            array[switch_i], array[switch_j] = array[switch_j], array[switch_i]
            print('Switch made:', array)
    
    if len(array[:N//2]) > 1:
        print('Doing left subarray')
        quicksort(array[:N//2])

    if len(array[N//2:]) > 1:
        print('Doing right subarray')
        quicksort(array[N//2:])

    
if __name__ == '__main__':

    N = int(7)
    array = np.arange(N)
    np.random.shuffle(array)

    # array = np.array([60, 24, 7, 1890, 55, 105982340, 0])
    
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    print(is_sorted(array), array)

    # selection_sort(array)
    quicksort(array)
    print(is_sorted(array), array)
