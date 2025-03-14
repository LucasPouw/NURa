import numpy as np
from rng import *
import sys


def choice(array: np.ndarray, size=1, replace=False, **kwargs):
    '''
    Sample elements from array with equal probability using a Knuth shuffle method
    '''

    array = np.asarray(array)
    N = len(array)

    if not replace:
        assert size <= N, 'Cannot draw more samples than elements in array without replacement.'

        # Do the Knuth shuffle (sounds like a funky dance move)
        result = array.copy()
        for i in range(size):
            random_idx = randint(low=i, high=N)[0]
            result[i], result[random_idx] = result[random_idx], result[i]
        return result[:size]

    else:
        random_indeces = randint(low=0, high=N, size=size, **kwargs)
        return array[random_indeces]
    

def rejection_sampling(target_func, xmin, xmax, fmax, size=1, samps_per_it=int(1e3), max_it=int(1e5)):
    '''
    Rejection sampling from target_func.
    We create two random deviates. 
    - The first is uniform between xmin and xmax.    
    - The second is uniform between 0 and fmax.
    '''

    accepted_samps = np.zeros(size)
    n_accepted = 0
    it_counter = 0
    seed = current_milli_time()
    while n_accepted < size:
        it_counter += 1
        seed += 1  # Because the time doesn't update fast enough
        if it_counter > max_it:
            sys.exit('Maximum number of iterations reached. Exiting...')
            break

        # Rejection-sample the target distribution
        unif1 = uniform(low=xmin, high=xmax, size=samps_per_it, seed=seed)
        unif2 = uniform(low=0, high=fmax, size=samps_per_it, seed=seed+1)
        accept_these = np.where(unif2 < target_func(unif1))[0]

        # Handle overshooting the requested number of accepted samples, which arises due to the chunking
        trial_n_total = n_accepted + len(accept_these)
        if trial_n_total > size:  # We have more samples than needed in the final iteration
            print(f'Accepted {trial_n_total - size} too many valid samples, removing them.')
            accept_these = accept_these[:size - n_accepted]  # Don't include the samples that were not requested

        accepted_samps[n_accepted:trial_n_total] = unif1[accept_these]
        n_accepted += len(accept_these)
        
        if not it_counter % 10:
            print(f'\nAccepted:{n_accepted}/{size}')

    print(f'\nAccepted:{n_accepted}/{size} in {it_counter} iterations.\n')

    return accepted_samps
    

if __name__ == '__main__':
    pass