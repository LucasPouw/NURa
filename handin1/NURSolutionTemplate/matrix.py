import numpy as np
import sys


class Matrix:

    '''
    Class for matrix operations and solving sets of linear equations

    Not implemented: pivoting and Gauss-Jordan elimination
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
        array = np.array(array).astype(np.float64)
        return cls(cls._vandermonde(array))


    @classmethod
    def as_LU(cls, array, improved=False):
        matrix = array.astype(np.float64).copy()  # Assuming you do not want to overwrite the input array when initializing the class
        if improved:
            indexing_array = cls._to_LU_improved_crout(matrix)
        else:
            indexing_array = cls._to_LU_crout(matrix)
        return cls(matrix, indexing_array, is_LU=True)


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
    

    def _permute_array(self, array):
        # Correct for pivoting, I know this is inefficient, but I'm tired. TODO
        for k, imax in enumerate(self.indexing_array):
            if k != imax:
                array[k], array[imax] = array[imax], array[k]
    

    def _solve_equation_LU(self, b):

        sol_len = len(b)

        # Forward substitution
        solution = b.copy()  # Assuming alpha[i,i] = 1 in our Crout implementation
        self._permute_array(solution)  # Correct for pivoting

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
       

    def solve_matrix_equation(self, b, method='LU', n_iterations=0):

        assert len(b) == self.matrix.shape[0], f'Vector b shape {b.shape} must match matrix A shape {self.matrix.shape} in Ax=b'

        match method:

            case 'LU':

                self._check_LU()

                solution = self._solve_equation_LU(b)
                for _ in range(n_iterations):  # Improve solution
                    db = self.matrix @ solution - b
                    solution -= self._solve_equation_LU(db)
            
            case _:
                sys.exit('Method not implemented.')

        return solution


    def to_LU(self, improved=False):
        if improved:
            self.indexing_array = self._to_LU_improved_crout(self.matrix)
        else:
            self.indexing_array = self._to_LU_crout(self.matrix)
        self.is_LU = True


    def _get_permutation_matrix(self):
        ''' Transform indexing array to permutation matrix '''
        n_cols = len(self.indexing_array)
        P = np.eye(n_cols)
        for k, imax in enumerate(self.indexing_array):
            if k != imax:
                for j in range(n_cols):
                        P[imax, j], P[k, j] = P[k, j], P[imax, j]
        return P


    def undo_LU(self):
        L, U = self.get_LU()
        P = self._get_permutation_matrix()
        
        self.matrix = P.T @ L @ U
        self.indexing_array = np.arange(len(self.indexing_array))
        self.is_LU = False
    

    def get_LU(self):
        self._check_LU()  # Can only return LU if matrix is LU decomposition

        upper, lower = np.triu(self.matrix), np.tril(self.matrix)
        n = self.matrix.shape[0]
        lower[range(n), range(n)] = 1
        return lower, upper
    
    
    def _to_row_echelon(self):
        NotImplemented
    
    def _to_reduced_row_echelon(self):
        NotImplemented

    def _solve_equation_GJ(self, b):
        ''' Gauss-Jordan algorithm '''
        NotImplemented
    
    def det(self):
        NotImplemented

    def inv(self):  # Needs Gauss-Jordan
        NotImplemented

    def svd(self):
        NotImplemented

    def lsq(self):
        NotImplemented


if __name__ == '__main__':

    import os

    arr = np.array([10,2,3]).astype(np.float64)
    arr2 = np.array([[2, 2,3], [4, 5, 6], [7,29,9]]).astype(np.float64)
    # mat = Matrix(arr2.copy())

    try:
        data = np.genfromtxt(os.path.join(sys.path[0], "Vandermonde.txt"), comments='#', dtype=np.float64)
    except FileNotFoundError:
        data = np.genfromtxt("/net/vdesk/data2/pouw/NUR/NURa/handin1/NURSolutionTemplate/Vandermonde.txt", comments='#', dtype=np.float64)

    x = data[:, 0]
    y = data[:, 1]
    xx = np.linspace(x[0], x[-1], 1001)  # x values to interpolate at

    # Question 2a
    mat = Matrix.as_vandermonde(x)
    mat.to_LU()
    print(mat.solve_matrix_equation(y))

    mat = Matrix.as_vandermonde(x)
    mat.to_LU(improved=True)
    print(mat.solve_matrix_equation(y))

    # mat.undo_LU()
    # print(mat.matrix)

    # matrix2 = Matrix.as_LU(arr2)
    # print(matrix2.is_LU)

    # print(matrix2.matrix)

    # N = 3

    # A = np.random.randint(1, 9, size=(N, N))
    # # A = np.array([[2, 3, 2], [1, 0, 1], [1, 6, 1]])
    # b = np.random.randint(1, 9, size=N)
    # # A = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 4]])
    # # b = np.array([5, 10, 4])

    # solution = solve_matrix_equation(A, b)
    # check = np.linalg.solve(A, b)
    # print(solution, check)
    # print('\nSAME?', np.allclose(check, solution))
