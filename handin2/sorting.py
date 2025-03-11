#%%
%%time

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
    ''' In-place sorting with quicksort algortihm that is not so quick... '''

    array = np.asarray(array)
    N = len(array)

    # Sorting first, middle and last element
    pivot = np.median([array[0], array[N//2], array[-1]]).astype(array.dtype)
    pivot_idx = np.where(array == pivot)[0][0]
    array[N//2], array[pivot_idx] = array[pivot_idx], array[N//2]
    if array[0] > array[-1]:
        array[-1], array[0] = array[0], array[-1]

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
        
        if i_flag & j_flag:
            array[i], array[j] = array[j], array[i]
            i_flag, j_flag = False, False

    # Sorting sub-arrays
    pivot_idx = np.where(array == pivot)[0][0]  # Pivot could have change position
    quicksort(array[:pivot_idx])
    quicksort(array[pivot_idx:])


def quicksort_gpt(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pivot_index = partition(arr, low, high)
        quicksort_gpt(arr, low, pivot_index - 1)
        quicksort_gpt(arr, pivot_index + 1, high)
    
    return arr

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

    
if __name__ == '__main__':

    N = int(1e3)
    array = np.arange(N)
    np.random.shuffle(array)

    unsorted_mine = array.copy()
    unsorted_gpt = array.copy()

    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    print('Initial array sorted?', is_sorted(array))

    # selection_sort(array)

#%%
    %%time
    quicksort(unsorted_mine)
    print('Final array sorted?', is_sorted(unsorted_mine))

#%%
    %%time
    quicksort_gpt(unsorted_gpt)
    print('Final array sorted?', is_sorted(unsorted_gpt))

#%%