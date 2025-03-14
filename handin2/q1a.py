#!/usr/bin/env python
import numpy as np
from integrate import romberg


NSAT = 100
A = 2.4
B = 0.25
C = 1.6
XMIN, XMAX = 10**-4, 5


def numdens(x, norm, Nsat=NSAT, a=A, b=B, c=C):
    return norm * Nsat * ((x / b) ** (a - 3)) * np.exp(-((x / b) ** c))


func2integrate = lambda x: numdens(x, Nsat=1, norm=1) * x**2 * 4 * np.pi  # Nsat = 1 such that norm = 1/integral
integral = romberg(func2integrate, XMIN, XMAX, order=14)  # Wolfram: 0.108756
normalization = 1/integral[0]
print(f'Romberg integration of p(x) for $A$=1 gives {integral[0]} with an error of {integral[1]}\n')
print(r'Therefore, the required normalization for x between $10^{-4}$ and 5 is $A=$', normalization)
