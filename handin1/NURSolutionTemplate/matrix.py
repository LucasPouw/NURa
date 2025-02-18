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
    def as_LU(cls, array):
        matrix = array.astype(np.float64).copy()  # Assuming you do not want to overwrite the input array when initializing the class
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
                    sys.exit('Pivot element is zero. Current implementation of the Crout algorithm fails. Swap matrix rows and try again.')

                matrix[i, j] /= matrix[j, j]

        indexing_array = np.arange(n_cols)  # TODO: Pivoting not implemented yet
        return indexing_array
    

    def _solve_equation_LU(self, b):

        sol_len = len(b)

        # Forward substitution
        solution = b.copy()  # Assuming alpha[i,i] = 1 in our Crout implementation
        solution = solution[self.indexing_array]  # Correct for pivoting - TODO: NOT TESTED FOR ACTUAL PIVOTING
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
    

    def _improve_LU_solution(self, b, solution, n_iterations):
        ''' Assumes A in LU decomposition '''
        for _ in range(n_iterations):
            db = self.matrix @ solution - b
            solution -= self._solve_equation_LU(db)
        return solution
    

    def solve_matrix_equation(self, b, method='LU', n_iterations=0):

        match method:
            case 'LU':
                if not self.is_LU:
                    print('WARNING: Matrix has been changed to LU decomposition.')
                    self.to_LU()

                solution = self._solve_equation_LU(b)

                if n_iterations != 0:
                    self._improve_LU_solution(b, solution, n_iterations)
            
            case _:
                sys.exit('Method not implemented.')

        return solution


    def to_LU(self):
        self.indexing_array = self._to_LU_crout(self.matrix)
        self.is_LU = True


    def undo_LU(self):
        ''' TODO: implement undoing pivoting with self.indexing_array '''
        upper, lower = np.triu(self.matrix), np.tril(self.matrix)
        n = self.matrix.shape[0]
        lower[range(n), range(n)] = 1
        self.is_LU = False
        return lower @ upper
    
    
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

    arr = np.array([1,2,3])
    arr2 = np.array([[1,2,3], [4,5,6], [7,8,9]])

    Mat = Matrix.as_vandermonde(arr)
    print(Mat.is_LU)

    matrix2 = Matrix.as_LU(arr2)
    print(matrix2.is_LU)

    print(matrix2.matrix)

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
