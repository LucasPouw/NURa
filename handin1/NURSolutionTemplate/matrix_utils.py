import numpy as np
import sys


def vandermonde(array):
    ''' Returns the NxN Vandermonde matrix '''

    array = np.array(array).astype(np.float64)

    if array.shape == ():  # Single number passed, so x^0 = 1 is returned
        return 1
        
    N = len(array)
    powers = np.arange(N)
    return array[:, np.newaxis]**powers


def crout(matrix):

    ''' Returns the LU decomposition of an NxN matrix (2D array)'''

    n_cols = matrix.shape[1]
    assert matrix.shape[0] == n_cols, 'Matrix must be shape NxN'

    decomposition = matrix.astype(np.float64).copy()
    for j in range(n_cols):  # Loop columns

        for i in range(j+1):  # Calculate upper
            
            for k in range(i):
                decomposition[i, j] -= decomposition[i, k] * decomposition[k, j]

        for i in range(j+1, n_cols):  # Calculate lower
                
            for k in range(j):
                decomposition[i, j] -= decomposition[i, k] * decomposition[k, j]

            if decomposition[j, j] == 0:
                sys.exit('Pivot element is zero. This version of the Crout algorithm fails. Do a permutation of matrix rows and try again.')

            decomposition[i, j] /= decomposition[j, j]
 
    return decomposition


def check_lu(matrix):
    upper = np.triu(matrix)
    lower = np.tril(matrix)

    n = matrix.shape[0]
    lower[range(n), range(n)] = 1

    return lower @ upper


def solve_matrix_equation(A, b, method='LU'):
    ''' Returns value of x in the matrix equation Ax = b '''

    A = np.array(A).astype(np.float64)
    b = np.array(b).astype(np.float64)

    if method == 'LU':
        A = crout(A)
        sol_len = len(b)

        # Forward substitution
        solution = b.copy()  # Assuming alpha[i,i] = 1 in our Crout implementation
        for i in range(sol_len):
            for j in range(i):
                solution[i] -= A[i, j] * solution[j]
        
        # Back-substitution
        for i in range(sol_len)[::-1]:
            for j in range(i+1, sol_len):
                solution[i] -= A[i, j] * solution[j]
            
            if A[i, i] == 0:
                sys.exit('Provided matrix is singular, so no unique solution exists. Exiting...')
                
            solution[i] /= A[i, i]
        
    elif method == 'GJ':
        sys.exit('Gauss-Jordan algorithm not yet implemented.')

    return solution


if __name__ == '__main__':

    N = 3

    A = np.random.randint(1, 9, size=(N, N))
    # A = np.array([[2, 3, 2], [1, 0, 1], [1, 6, 1]])
    b = np.random.randint(1, 9, size=N)
    # A = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 4]])
    # b = np.array([5, 10, 4])

    solution = solve_matrix_equation(A, b)
    check = np.linalg.solve(A, b)
    print(solution, check)
    print('\nSAME?', np.allclose(check, solution))
