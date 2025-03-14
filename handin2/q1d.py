#!/usr/bin/env python
import numpy as np
from integrate import romberg
from differentiate import ridder


NSAT = 100
A = 2.4
B = 0.25
C = 1.6
XMIN, XMAX = 10**-4, 5


def numdens(x, norm, Nsat=NSAT, a=A, b=B, c=C):
    return norm * Nsat * ((x / b) ** (a - 3)) * np.exp(-((x / b) ** c))

func2integrate = lambda x: numdens(x, Nsat=1, norm=1) * x**2 * 4 * np.pi  # Nsat = 1, because it divides out
integral = romberg(func2integrate, XMIN, XMAX, order=14)  # Wolfram: 0.108756
normalization = 1/integral[0]

# 1d starts here
def numdens_derivative(x, norm=normalization, Nsat=NSAT, a=A, b=B, c=C):
    return (
        (
            norm 
            * Nsat 
            * b ** 3 
            * np.exp(-(x / b) ** c) 
            * (x / b) ** a
            * (-3 + a - c * (x / b) ** c)
        )
        / x ** 4
    )

f = lambda x: numdens(x, norm=normalization)
numerical_result = ridder(f, x=1, m=7, target_error=1e-12)
analytical_result = numdens_derivative(x=1)
print(f'\nRidder differentiation of n(x) at x=1 gives {numerical_result[0][0]} with an error of {numerical_result[1]}')
print(f'\nThe analytical result is {analytical_result}')
