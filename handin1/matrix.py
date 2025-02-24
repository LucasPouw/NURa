import numpy as np
import sys


class Matrix:

    '''
    Class for matrix operations and solving sets of linear equations.
    '''

    def __init__(self,
                 matrix: np.ndarray,
                 indexing_array: np.ndarray = None,
                 is_LU: bool = False):

        self.matrix = matrix.astype(np.float64)
        self.indexing_array = indexing_array
        self.is_LU = is_LU
        
    
    @classmethod
    def as_vandermonde(cls, array):
        ''' Initialize Matrix object using a 1D array from which the Vandermonde matrix is calculated. '''
        matrix = np.array(array).astype(np.float64)
        return cls(cls._vandermonde(matrix))


    @classmethod
    def as_LU(cls, array, improved=False):
        ''' 
        Initialize Matrix object and immediately calculate its LU-decomposition.

        Set improved=True to use the improved Crout's algorithm
        '''
        matrix = array.astype(np.float64)  # Assuming you do not want to overwrite the input array when initializing the class
        if improved:
            indexing_array = cls._to_LU_improved_crout(matrix)
        else:
            indexing_array = cls._to_LU_crout(matrix)
        return cls(matrix, indexing_array, is_LU=True)
    

    def to_LU(self, improved=False):
        ''' Set improved=True to use the improved Crout's algorithm '''
        if improved:
            self.indexing_array = self._to_LU_improved_crout(self.matrix)
        else:
            self.indexing_array = self._to_LU_crout(self.matrix)
        self.is_LU = True


    def undo_LU(self):
        if self.is_LU:  # Can only undo LU if it is LU
            P, L, U = self.get_LU()
            
            self.matrix = P.T @ L @ U
            self.indexing_array = np.arange(len(self.indexing_array))
            self.is_LU = False
    

    def get_LU(self):
        self._check_LU()  # Can only return LU if matrix is LU decomposition

        L, U = np.tril(self.matrix), np.triu(self.matrix)
        n = self.matrix.shape[0]
        L[range(n), range(n)] = 1  # Set diagonal of lower matrix to 1

        P = self._get_permutation_matrix()
        return P, L, U
    

    def solve_matrix_equation(self, b, method='LU', n_iterations=0):

        assert len(b) == self.matrix.shape[0], f'Vector b shape {b.shape} must match matrix A shape {self.matrix.shape} in Ax=b'

        match method:

            case 'LU':

                self._check_LU()

                solution = self._solve_equation_LU(b)

                if n_iterations != 0:  # Iteratively improve solution
                    P, L, U = self.get_LU()
                    A = P.T @ L @ U  # Need original matrix
                    for _ in range(n_iterations):
                        db = A @ solution - b
                        solution -= self._solve_equation_LU(db)
            
            case _:
                sys.exit('Method not implemented.')

        return solution
    

    def _solve_equation_LU(self, b):

        sol_len = len(b)

        # Forward substitution
        solution = b.copy()  # Assuming alpha[i,i] = 1 in our Crout implementation
        self._undo_pivoting(solution)  # Correct for pivoting - output solution will be unpivoted

        for i in range(sol_len):
            for j in range(i):
                solution[i] -= self.matrix[i, j] * solution[j]
        
        # Back-substitution
        for i in range(sol_len)[::-1]:
            for j in range(i+1, sol_len):
                solution[i] -= self.matrix[i, j] * solution[j]
            
            if self.matrix[i, i] == 0:
                sys.exit('Provided matrix is singular, so no unique solution exists. Exiting...')
                
            solution[i] /= self.matrix[i, i]

        return solution
    

    def _check_LU(self):
        if not self.is_LU:
            print('WARNING: Matrix has been changed to LU decomposition.')
            self.to_LU()

    
    def _undo_pivoting(self, array):
        for k, imax in enumerate(self.indexing_array):
            if k != imax:
                array[k], array[imax] = array[imax], array[k]


    def _get_permutation_matrix(self):
        ''' Transform indexing array to permutation matrix '''
        n_cols = len(self.indexing_array)
        P = np.eye(n_cols)
        for k, imax in enumerate(self.indexing_array):
            if k != imax:
                for j in range(n_cols):
                        P[imax, j], P[k, j] = P[k, j], P[imax, j]
        return P

    
    @staticmethod
    def _vandermonde(array):
        ''' Returns the NxN Vandermonde matrix '''

        if array.shape == ():  # Single number passed, so x^0 = 1 is returned
            return 1
            
        N = len(array)
        powers = np.arange(N)
        return array[:, np.newaxis]**powers


    @staticmethod
    def _to_LU_crout(matrix):
        ''' Overwrites the input NxN matrix (2D array) with its LU decomposition '''

        n_cols = matrix.shape[1]
        assert matrix.shape[0] == n_cols, 'Matrix must be shape NxN'

        for j in range(n_cols):

            for i in range(j+1):  # Calculate upper
                for k in range(i):
                    matrix[i, j] -= matrix[i, k] * matrix[k, j]

            for i in range(j+1, n_cols):  # Calculate lower
                for k in range(j):
                    matrix[i, j] -= matrix[i, k] * matrix[k, j]

                if matrix[j, j] == 0:
                    sys.exit('Pivot element is zero. Current implementation of the Crout algorithm fails. Try again with improved Crout.')

                matrix[i, j] /= matrix[j, j]

        indexing_array = np.arange(n_cols)
        return indexing_array
    

    @staticmethod
    def _to_LU_improved_crout(matrix):
        ''' Overwrites the input NxN matrix (2D array) with its LU decomposition '''

        n_cols = matrix.shape[1]
        assert matrix.shape[0] == n_cols, 'Matrix must be shape NxN'

        indexing_array = np.zeros(n_cols, dtype=int)
        for k in range(n_cols):
            largest_candidate = -np.inf
            for i in range(k, n_cols):
                current_candidate = matrix[i, k]
                if current_candidate > largest_candidate:
                    largest_candidate = current_candidate
                    indexing_array[k] = i
            
            imax = indexing_array[k]
            if imax != k:  # Not a diagonal element, so pivot
                for j in range(n_cols):
                    matrix[imax, j], matrix[k, j] = matrix[k, j], matrix[imax, j]

            for i in range(k+1, n_cols):
                matrix[i, k] /= matrix[k, k]
                for j in range(k+1, n_cols):
                    matrix[i, j] -= matrix[i, k] * matrix[k, j]

        return indexing_array


if __name__ == '__main__':
    pass