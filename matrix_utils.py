import numpy as np


def crout(matrix):

    ''' Returns the LU decomposition of an NxN matrix (2D array)'''

    n_cols = matrix.shape[1]
    assert matrix.shape[0] == n_cols, 'Matrix must be shape NxN'

    decomposition = matrix.astype(float).copy()
    for j in range(n_cols):  # Loop columns

        for i in range(j+1):  # Calculate upper
            
            for k in range(i):
                decomposition[i, j] -= decomposition[i, k] * decomposition[k, j]

        for i in range(j+1, n_cols):  # Calculate lower
                
            for k in range(j):
                decomposition[i, j] -= decomposition[i, k] * decomposition[k, j]

            decomposition[i, j] /= decomposition[j, j]
 
    return decomposition


def check_lu(matrix):
    upper = np.triu(matrix)
    lower = np.tril(matrix)

    for i in range(matrix.shape[0]):
        lower[i,i] = 1

    return lower @ upper

if __name__ == '__main__':

    # matrix = np.array([[1, 2, 3], 
    #                    [4, 5, 6],
    #                    [7, 8, 9]])

    matrix = np.random.randint(low=-10, high=10, size=(3,3))
    
    # matrix = np.array([[7, 8, 9],
    #                    [1, 2, 3], 
    #                    [4, 5, 6]])
    print(matrix, 'ORIGINAL!!!!')

    decomp = crout(matrix)
    print(decomp, 'DECOMPOSITION!!!!')

    check = check_lu(decomp)
    print(check, 'ORIGINAL????????')

    print('work pls', np.allclose(check, matrix))
