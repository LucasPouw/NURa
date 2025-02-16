import numpy as np

def cumsum(array):
    return np.array([np.sum(array[:i]) for i in range(1, len(array) + 1)])

# def bubble_sort(array):
#     n = len(array)
#     for i in range(n):
#         for j in range(0, n-i-1):
#             if array[j] > array[j+1]:
#                 array[j], array[j+1] = array[j+1], array[j]
#     return array

if __name__ == '__main__':
    arr = np.array([5, 3, 2, 7, 8])
    # print(bubble_sort(arr))