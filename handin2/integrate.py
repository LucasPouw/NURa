import numpy as np


def trapezoid(func, start, stop, step):
    ''' Integrate func from start to stop with step-size step '''
    endpoint_values = (func(start) + func(stop)) * 0.5
    x = np.arange(start, stop, step)
    center_values = np.sum( func(x[1:]) )
    return step * (endpoint_values + center_values)


def simpson(func, start, stop, N):
    h0 = (stop - start) / N
    S0 = trapezoid(func, start, stop, h0)
    h1 = h0 * 0.5
    S1 = trapezoid(func, start, stop, h1)
    S = (4 * S1 - S0) / 3
    return S


def romberg(func, start, stop, order):
    h = stop - start
    r = np.zeros(order)
    r[0] = 0.5 * h * (func(start) + func(stop))

    Np = 1
    for i in range(1, order):
        Delta = h
        h *= 0.5

        x = np.arange(start + h, Np * Delta, Delta)
        r[i] += np.sum(func(x))

        r[i] = 0.5 * (r[i-1] + Delta * r[i])
        Np *= 2

    Np = 1
    for i in range(1, order):
        Np *= 4
        for j in range(order - i):
            r[j] = (Np * r[j+1] - r[j]) / (Np - 1)

    return r[0], abs(r[0] - r[1])  # Solution, error


if __name__ == '__main__':

    func = lambda x: 3 * np.exp(-2 * x) + 0.0001 * x**4
    true_value = 3.4999999969082696

    for N in [10, 100, 1000, 10000]:
        trapz = trapezoid(func, 0, 10, 1/N)
        simp = simpson(func, 0, 10, N)
        print(f'Trapezoid for N = {N} gives {trapz}. So the error is {abs(true_value - trapz)}.')
        print(f'Simpson for N = {N} gives {simp}. So the error is {abs(true_value - simp)}.')

    for m in [2, 4, 6]:
        romb = romberg(func, 0, 10, m)[0]
        print(f'Romberg for m = {m} gives {romb}. So the error is {abs(true_value - romb)}.')
