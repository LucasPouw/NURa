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

    for i in range(1, order):
        Delta = h
        h *= 0.5
        x = np.arange(start + h, stop, Delta)
        r[i] = 0.5 * (r[i-1] + Delta * np.sum(func(x)))

    Np = 1
    for i in range(1, order):
        Np *= 4
        for j in range(order - i):
            r[j] = (Np * r[j+1] - r[j]) / (Np - 1)

    return r[0], abs(r[0] - r[1])  # Solution, error


if __name__ == '__main__':
    pass
