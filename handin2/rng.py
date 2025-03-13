import numpy as np
import time


TWOTOPOWER32 = np.uint64(1) << np.uint64(32)
TWOTOPOWER64 = 2**64


def current_milli_time():
    return np.uint64(time.time() * 1000)


def xorshift64(seed, a1, a2, a3):
    '''64-bit XOR-shift method'''
    x = np.uint64(seed)
    assert x != np.uint64(0), 'Seed must be non-zero.'

    a1 = np.uint64(a1)
    a2 = np.uint64(a2)
    a3 = np.uint64(a3)
    
    x = x ^ (x >> a1)
    x = x ^ (x << a2)
    x = x ^ (x >> a3)
    return np.uint64(x)


def mwc32(seed, a):
    '''32-bit Multiply with Carry method'''
    x = np.uint64( np.uint64(seed) & np.uint64(TWOTOPOWER32 - np.uint64(1)) )  # Ensure initial state < 2**32
    a = np.uint64(a)
    assert x != np.uint64(0), 'Seed must be non-zero.'
    x = a * (x & (TWOTOPOWER32 - np.uint64(1))) + (x >> np.uint64(32))
    return np.uint32( x & (TWOTOPOWER32 - np.uint64(1)) )  # Only provide lowest 32 bits


def rng32(size, 
        seed=current_milli_time(), 
        mwcparam=3238579223, 
        xorshiftparam=[17, 37, 3]):  # Primes
    '''Generates uniformly distributed pseudo-random unsigned 32-bit integers in the interval [0, 2**32).'''
    
    result = np.zeros(size, dtype=np.uint32)
    mwc_state = seed.copy()
    xor_state = seed.copy()
    for i in range(size):
        xor_state = xorshift64(xor_state, *xorshiftparam)
        mwc_state = mwc32(mwc_state, mwcparam)

        value = np.uint32(xor_state ^ mwc_state)
        
        result[i] = value
    return result


def uniform(low=0., high=1., size=1, **kwargs):
    '''Generates uniformly distributed pseudo-random 32-bit floats in the interval (low, high)'''
    result = rng32(size, **kwargs) / TWOTOPOWER32 * (high - low) + low
    return result.astype(np.float32)


def randint(low, high, size=1, **kwargs):
    '''Generates uniformly distributed pseudo-random signed 32-bit ints in the interval [low, high)'''
    requested_range_values = uniform(low, high, size, **kwargs)
    return np.floor(requested_range_values).astype(np.int32)


if __name__ == '__main__':

    seed = current_milli_time()
    for i in range(10):
        print(uniform(seed=seed+i))

    # import matplotlib.pyplot as plt

    # N = 1000
    # size = 100000

    # # arr = choice(np.arange(N), size=size)
    # # print(arr, 'result')

    # minval = 0
    # maxval = 100
    
    # rng_arr = randint(minval, maxval, size)
    # edges = np.linspace(minval, maxval, int(maxval - minval + 1))
    # # print(np.min(rng_arr), np.max(rng_arr))
    # plt.figure()
    # plt.hist(rng_arr, bins=edges, density=True)
    # # plt.vlines(maxnum, 0, 1/maxnum, color='red')
    # plt.show()

