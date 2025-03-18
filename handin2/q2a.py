#!/usr/bin/env python
import numpy as np
from rootfinding import false_position
import time

k = 1.38e-16  # erg/K
PSI = 0.929
TC = 1e4  # K
METALLICITY = 0.015


def equilibrium1(T, Z=METALLICITY, Tc=TC, psi=PSI):
    return psi * Tc * k - (0.684 - 0.0416 * np.log(T / (1e4 * Z * Z))) * T * k


if __name__ == '__main__':
    start = time.time()
    root, (abs_err, rel_err) = false_position(equilibrium1, bracket=[1, 1e7])
    t = time.time() - start
    print(f'\nEquilibrium temperature is $T$ = {root}\n')
    print(f'\nAbsolute error: {abs_err:.6e}\n')
    print(f'\nRelative error: {rel_err:.6e}\n')
    print(f'\nThat took {t} s\n')
