import numpy as np
from utils import cumsum


def log_factorial(array):
    assert np.sum(array < 0) == 0, "Input should be greater than or equal than 0."

    array = np.array(array).astype(np.int32)  # Force list to array of integers
    max_idx = np.max(array) + 1

    all_factorials = np.zeros(max_idx, dtype=np.float32)
    all_factorials[1:] = cumsum( np.log(np.arange(1, max_idx)) )  # nth element contains log(n!)
    return all_factorials[array]  # Requested factorials


def poisson(lam, k):
    lam = np.array(lam).astype(np.float32)
    k = np.array(k).astype(np.int32)
    log_poisson = k * np.log(lam).astype(np.float32) - lam - log_factorial(k)
    return np.exp(log_poisson.astype(np.float32))


if __name__ == '__main__':
    
    lambdas = np.array([1, 5, 3, 2.6, 100, 101])
    ks = np.array([0, 10, 21, 40, 5, 200])
    results = poisson(lambdas, ks)
    for i, result in enumerate(results):
        print(f'$(\\lambda, k)$ = ({lambdas[i]}, {ks[i]}) -> $P_{{\\lambda}}(k)$ = {results[i]:.6e}')
