import timeit

setup = """
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matrix import Matrix
from interpolator import Interpolator
from utils import polynomial

data = np.genfromtxt(os.path.join(sys.path[0], "Vandermonde.txt"), comments='#', dtype=np.float64)

x = data[:, 0]
y = data[:, 1]
xx = np.linspace(x[0], x[-1], 1001)  # x values to interpolate at

# Question 2a
V = Matrix.as_vandermonde(x)
V.to_LU(improved=False)
"""

q2a_code2time = """
solution = V.solve_matrix_equation(y, method='LU')
yya = polynomial(xx, solution)
ya = polynomial(x, solution)
"""

q2b_code2time = """
Interp = Interpolator(data)
yyb, yyb_error = Interp.polynomial(xx, order=19)
yb, yb_error = Interp.polynomial(x, order=19)
"""

q2c_code2time = """
improved_solution_10 = V.solve_matrix_equation(y, method='LU', n_iterations=10)
yyc10 = polynomial(xx, improved_solution_10)
yc10 = polynomial(x, improved_solution_10)
"""

def main(n_its=[10000, 100, 1000]):
    for i, (q, code) in enumerate(zip(['Q2a', 'Q2b', 'Q2c'], [q2a_code2time, q2b_code2time, q2c_code2time])):
        time = timeit.timeit(setup=setup, stmt=code, number=n_its[i])
        print(f"{q} runtime for {n_its[i]} iterations = {time:.3f} s, averaging {time/n_its[i]:.3e} s/it \n")


if __name__ == "__main__":
    main()
