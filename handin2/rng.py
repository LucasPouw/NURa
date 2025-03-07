import numpy as np
import time


TWOTOPOWER32 = np.uint64(1) << np.uint64(32)
RNGMAX = 1 << 64


def current_milli_time():
    return np.uint64(time.time() * 1000)


def xorshift(seed, a1=np.uint64(21), a2=np.uint64(35), a3=np.uint64(4)):
    '''64-bit XOR-shift method'''
    x = np.uint64(seed)
    assert x != np.uint64(0), 'Seed must be non-zero.'

    a1 = np.uint64(a1)
    a2 = np.uint64(a2)
    a3 = np.uint64(a3)
    
    x = x ^ (x >> a1)
    x = x ^ (x << a2)
    x = x ^ (x >> a3)
    return x


def mwc(seed, a=np.uint64(4294957665)):
    '''64-bit Multiply with Carry method'''
    x = np.uint64(np.uint64(seed) & np.uint64(TWOTOPOWER32 - np.uint64(1)))  # Ensure seed < 2**32
    a = np.uint64(a)
    assert x != np.uint64(0), 'Seed must be non-zero.'
    
    x = a * (x & (TWOTOPOWER32 - np.uint64(1))) + (x >> np.uint64(32))
    return x & (TWOTOPOWER32 - np.uint64(1))


def rng(size, 
        seed=current_milli_time(), 
        mwcparam=np.uint64(4294957665), 
        xorshiftparam=[np.uint64(21), np.uint64(35), np.uint64(4)]):
    '''Generates uniformly distributed pseudo-random unsigned 64-bit integers in the interval [0, 2**64).'''

    a1, a2, a3 = xorshiftparam
    
    result = np.zeros(size, dtype=np.uint64)
    for i in range(size):
        x1 = mwc(seed, mwcparam)
        x2 = xorshift(x1, a1, a2, a3)
        
        result[i] = x2
        seed = x2.copy()  # Use current number to generate next number

    return result


def uniform(low=0., high=1., size=1, **kwargs):
    '''Generates uniformly distributed pseudo-random 64-bit floats in the interval (low, high)'''
    assert low < high, 'Lower bound must be smaller than upper bound.'  # Not strictly necessary, but just makes sense
    result = rng(size, **kwargs) / RNGMAX * (high - low) + low
    return result.astype(np.float64)


def randint(low, high, size, **kwargs):
    '''Generates uniformly distributed pseudo-random signed 32-bit ints in the interval [low, high]'''
    low, high = np.int32(low), np.int32(high)
    assert low < high, 'Lower bound must be smaller than upper bound.'
    full_range_values = rng(size, **kwargs)

    # requested_range_values=np.zeros(size)
    # for i in range(size):
    #     requested_range_values[i] = (full_range_values[i] >> np.uint64(64)) * (high - low + 1) + low
    requested_range_values = full_range_values / RNGMAX * (high - low + 1) + low
    return requested_range_values.astype(np.int32)


# def choice(array: np.ndarray, size=1, replace=False, **kwargs):
#     '''Sample elements from array with equal probability'''

#     array = np.asarray(array)
#     N = len(array)

#     if not replace:
#         assert size <= N, 'Cannot draw more samples than elements in array without replacement.'

#         random_indeces = np.zeros(size, dtype=np.int32)

#         # Is this still uniform or am I doing the Monty Hall problem?
#         allowed_indeces = np.arange(N)
#         for i in range(size):

#             if i == N-1:  # Only one element left to choose from
#                 random_indeces[i] = allowed_indeces[0]
#                 continue

#             idx = np.int32( np.rint(uniform(low=0, high=N-1-i, size=1, **kwargs)) )
#             random_indeces[i] = allowed_indeces[idx]
#             allowed_indeces = np.delete(allowed_indeces, idx)
#     else:
#         random_floats = uniform(low=0, high=N-1, size=size, **kwargs)
#         random_indeces = np.int32( np.rint(random_floats) )

#     return array[random_indeces]


# def uniform_on_sphere():
#     return


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    N = 1000
    size = 1000000

    # arr = choice(np.arange(N), size=size)
    # print(arr, 'result')
    
    rng_arr = randint(-10, 50, size)
    
    # rng_arr = uniform(size=size, low=-10, high=50)

    # maxnum = 1
    plt.figure()
    plt.hist(rng_arr, bins=50, density=True)
    # plt.vlines(maxnum, 0, 1/maxnum, color='red')
    plt.show()

