import numpy as np
import sys


def secant(func, bracket, target_abs=1e-10, target_rel=1e-5, max_it=100):
    a, b = bracket
    for i in range(max_it):
        func_b = func(b)

        if (abs(a - b) <= target_abs) and (abs((a - b) / b) <= target_rel):
            print(f'Target accuracy reached in {i + 1} iterations.')
            return b
        
        b, a = b - (b - a) / (func_b - func(a)) * func_b, b

    print(f'Maximum number of iterations ({max_it}) reached.')
    return b


def false_position(func, bracket, target_abs=1e-10, target_rel=1e-5, max_it=100):
    a, b = bracket
    best_guess = np.inf
    for i in range(max_it):
        func_b = func(b)
        
        new_guess = b - (b - a) / (func_b - func(a)) * func_b

        if func(a) * func(new_guess) < 0:
            b = new_guess
        elif func(new_guess) * func(b) < 0:
            a = new_guess
        else:
            sys.exit('No bracket, what happened? Are you ok?')

        if (abs(best_guess - new_guess) <= target_abs) and (abs((best_guess - new_guess) / best_guess) <= target_rel):
            print(f'Target accuracy reached in {i + 1} iterations.')
            return new_guess
        
        best_guess = new_guess

    print(f'Maximum number of iterations ({max_it}) reached.')
    return best_guess


def newton_raphson(f, fprime, initial_guess, target_abs=1e-10, target_rel=1e-5, max_it=100):
    best_guess = initial_guess

    for i in range(max_it):

        new_guess = best_guess - f(best_guess) / fprime(best_guess)

        if (abs(best_guess - new_guess) <= target_abs) and (abs((best_guess - new_guess) / best_guess) <= target_rel):
            print(f'Target accuracy reached in {i + 1} iterations.')
            return new_guess
        
        best_guess = new_guess

    print(f'Maximum number of iterations ({max_it}) reached.')
    return best_guess


if __name__ == '__main__':
    pass