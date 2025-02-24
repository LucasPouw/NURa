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
solution = V.solve_matrix_equation(y, method='LU')
print(f'The solution vector is c = {solution}')
yya = polynomial(xx, solution)
ya = polynomial(x, solution)

# Question 2b
Interp = Interpolator(data)
yyb, yyb_error = Interp.polynomial(xx, order=19)
yb, yb_error = Interp.polynomial(x, order=19)

# Question 2c
improved_solution_1 = V.solve_matrix_equation(y, method='LU', n_iterations=1)
improved_solution_10 = V.solve_matrix_equation(y, method='LU', n_iterations=10)

yyc1 = polynomial(xx, improved_solution_1)
yc1 = polynomial(x, improved_solution_1)
yyc10 = polynomial(xx, improved_solution_10)
yc10 = polynomial(x, improved_solution_10)

# Plot of points with absolute difference shown on a log scale (Q2a)
fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.0,1.0])
axs = gs.subplots(sharex=True, sharey=False)
axs[0].plot(x, y, marker='o', linewidth=0)
plt.xlim(-1, 101)
axs[0].set_ylim(-400, 400)
axs[0].set_ylabel('$y$')
axs[1].set_ylabel('$|y-y_i|$')
axs[1].set_xlabel('$x$')
axs[1].set_yscale('log')
line, = axs[0].plot(xx, yya, color='orange')
line.set_label('Via LU decomposition')
axs[0].legend(frameon=False, loc="lower left")
axs[1].plot(x, abs(y-ya), color='orange')
plt.savefig('plots/my_vandermonde_sol_2a.png', dpi=600)

# Q2b
line, = axs[0].plot(xx, yyb, linestyle='dashed', color='green')
line.set_label('Via Neville\'s algorithm')
axs[0].legend(frameon=False, loc="lower left")
axs[1].plot(x, abs(y-yb), linestyle='dashed', color='green')
plt.savefig('plots/my_vandermonde_sol_2b.png', dpi=600)

# Q2c
line, = axs[0].plot(xx, yyc1, linestyle='dotted', color='red')
line.set_label('LU with 1 iteration')
axs[1].plot(x, abs(y-yc1), linestyle='dotted', color='red')
line, = axs[0].plot(xx, yyc10, linestyle='dashdot', color='purple')
line.set_label('LU with 10 iterations')
axs[1].plot(x, abs(y-yc10), linestyle='dashdot', color='purple')
axs[0].legend(frameon=False, loc="lower left")
plt.savefig('plots/my_vandermonde_sol_2c.png', dpi=600)
