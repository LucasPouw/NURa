import numpy as np
from minimize import golden_section

NSAT = 100
NORM = 0.2 * 256 / np.pi**(3/2)
A = 2.4
B = 0.25
C = 1.6
XMIN = 0
XMAX = 5
prefactor = 4 * np.pi * NORM * NSAT


def N_of_x_without_prefactor(x, a=A, b=B, c=C):
    '''Unnormalized x^2 n(x)'''
    return  x ** (a - 1) * b ** (3 - a) * np.exp(-((x / b) ** c))


func2minimize = lambda x: -N_of_x_without_prefactor(x)
extremum_x = golden_section(func2minimize, a=XMIN, b=XMAX)
func_at_maximum = prefactor * N_of_x_without_prefactor(extremum_x)
print(f'$x_{{\\rm max}}$ = {extremum_x}, $N(x_{{\\rm max}})$ = {func_at_maximum}')
