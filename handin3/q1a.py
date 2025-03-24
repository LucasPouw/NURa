import numpy as np
from minimize import golden_section
import matplotlib.pyplot as plt
import os, sys


NSAT = 100
NORM = 0.2 * 256 / np.pi**(3/2)
A = 2.4
B = 0.25
C = 1.6
XMIN = 1e-4
XMAX = 5


def prefactor(norm=NORM, nsat=NSAT):
    return 4 * np.pi * norm * nsat


def N_of_x_without_prefactor(x, a=C, b=C, c=C):
    return x**2 * ((x / b) ** (a - 3)) * np.exp(-((x / b) ** c))


func2minimize = lambda x: -N_of_x_without_prefactor(x)
minimum = golden_section(func2minimize, a=XMIN, b=XMAX)
print(minimum, prefactor() * N_of_x_without_prefactor(minimum))


plt.figure()
xx = np.linspace(XMIN, XMAX, 1000)
plt.plot(xx, prefactor() * N_of_x_without_prefactor(xx))
plt.scatter(minimum, prefactor() * N_of_x_without_prefactor(minimum), color='red', marker='o')
plt.savefig(os.path.join(sys.path[0], 'plots/numgals.png'), bbox_inches='tight', dpi=600)
plt.show()