#!/usr/bin/env python
import numpy as np
from rootfinding import newton_raphson

k = 1.38e-16  # erg/K
aB = 2e-13  # cm^3 / s
PSI = 0.929
TC = 1e4  # K
METALLICITY = 0.015
CONST = 5e-10  # erg
XI = 1e-15  # s^-1


def equilibrium2(T, nH, Z=METALLICITY, Tc=TC, psi=PSI, A=CONST, xi=XI):
    return (
        (
            psi * Tc
            - (0.684 - 0.0416 * np.log(T / (1e4 * Z * Z))) * T
            - 0.54 * (T / 1e4) ** 0.37 * T
        )
        * k
        * nH
        * aB
        + A * xi
        + 8.9e-26 * (T / 1e4)
    )


def derivative_equilibrium2(T, nH, Z=METALLICITY):
    return (
        (
            -0.684
            + 0.0416 * (np.log(T / (1e4 * Z * Z)) + 1)
            - 0.54 * 1.37 * (T / 1e4) ** 0.37
        )
        * k
        * nH
        * aB
        + 8.9e-30
    )


if __name__ == '__main__':

    for dens in [1e-4, 1, 1e4]:
        print(r'$n_{\rm H} = $', dens)
        func = lambda x: equilibrium2(x, nH=dens)
        fprime = lambda x: derivative_equilibrium2(x, nH=dens)

        root, (abs_err, rel_err) = newton_raphson(func, fprime, initial_guess=5e14, target_abs=np.inf, target_rel=1e-10, max_it=int(1e5))
        print(f'Equilibrium temperature is $T$ = {root}')
        print(f'Absolute error: {abs_err:.6e}')
        print(f'Relative error: {rel_err:.6e}\n')
