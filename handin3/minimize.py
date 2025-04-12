import numpy as np
from sorting import quicksort
from utils import log_factorial
from matrix import Matrix


GOLDEN_RATIO = 0.5 * (1 + np.sqrt(5))


def abscissa_of_minimum(fa, fb, fc, a, b, c):
    b_minus_a = b - a
    fb_minus_fc = fb - fc
    b_minus_c = b - c
    fb_minus_fa = fb - fa
    return b - 0.5 * (b_minus_a**2 * fb_minus_fc - b_minus_c**2 * fb_minus_fa) / (b_minus_a * fb_minus_fc - b_minus_c * fb_minus_fa)


def bracket_minimum(func, a, b, w=GOLDEN_RATIO, max_it=int(1e4)):
    '''Finding the third point of a bracket.'''

    fa = func(a)
    fb = func(b)

    if fa < fb:  # Need f(b) < f(a)
        fa, fb = fb, fa
        a, b = b, a

    c = b + (b - a) * w
    fc = func(c)

    for _ in range(max_it):

        if fc > fb:
            return [a, b, c]
        
        d = abscissa_of_minimum(fa, fb, fc, a, b, c)
        fd = func(d)

        if b < d < c:

            if fd < fc:
                return [b, d, c]
            elif fd > fb:
                return [a, b, d]
            else:
                d = c + (c - b) * w  # Fit was bad, take another step

        elif d > c:

            thresh = 100 * abs(c - b)
            if abs(d - b) > thresh:
                d = c + (c - b) * w  # Don't trust result, take another step

        else:
            print('Improve the initial bracket')
            return None

        a, b, c = b, c, d
        fa = func(a)
        fb = func(b)
        fc = func(c)

    print(f'Maximum number of iterations reached: {max_it}')


def golden_section(func, a, b, target_accuracy=1e-10, w_bracket=GOLDEN_RATIO, max_it=int(1e5)):
    
    w = 2 - GOLDEN_RATIO

    a, b, c = bracket_minimum(func, a, b, w_bracket, max_it)
    if a > c:  # May happen due to bracketing function implementation
        a, c = c, a

    for _ in range(max_it):
        right_size = abs(c - b)
        left_size = abs(b - a)
        if right_size > left_size:
            d = b + (c - b) * w
        else:
            d = b + (a - b) * w

        fb = func(b)
        fd = func(d)

        if abs(c - a) < target_accuracy:
            if fd < fb:
                return d
            else:
                return b
            
        if fd < fb:
            if b < d < c:
                a, b = b, d
            elif a < d < b:
                c, b = b, d
            else:
                print('Golden section minimization failed.')
                return None
        else:
            if b < d < c:
                c = d
            elif a < d < b:
                a = d
            else:
                print('Golden section minimization failed.')
                return None
    
    print(f'Maximum number of iterations reached: {max_it}')
    if fd < fb:
        return d
    else:
        return b
    

def simplex_volume(simplex):
    """Calculating the volume of a simplex of shape (ndim + 1, ndim) using https://en.wikipedia.org/wiki/Simplex#Volume"""
    ndim = simplex.shape[1]
    A = simplex[1:] - simplex[0]  # Matrix with as each column a vector that points from vertex v0 to another vertex vk.
    G = Matrix.as_LU( np.dot(A, A.T) )  # Gram matrix
    volume = np.sqrt( np.abs(G.determinant()) ) * np.exp(-log_factorial(ndim))
    return volume


def downhill_simplex(func, simplex, target_fractional_accuracy=1e-10, max_it=int(1e2), init_volume_thresh=1e-10):
    '''
    Simplex should be 2d array of shape (ndim+1, ndim), 
    func takes n_args arguments and we have n_points datapoints.
    '''

    volume = simplex_volume(simplex)
    assert volume > init_volume_thresh, f"Volume of initial simplex is too small. Got {volume}, required {init_volume_thresh}"
    print(f'\n----- Starting Downhill simplex minimization with initial simplex volume {volume:.3e} -----\n')

    ndim = simplex.shape[1]
    it = 0
    while it < max_it:
        # Sort x-values based on y-values
        y_vals = func(simplex)
        mapping = {y_val: index for index, y_val in enumerate(y_vals)}  # Link y-vals to their idx
        quicksort(y_vals)  # We sort the rows of x such that f(x0) <= f(x1) <= ... <= f(xN)
        indexing_array = [mapping[key] for key in y_vals]  # Get new idx order
        simplex = simplex[indexing_array,:]

        # Terminate if below target accuracy
        fractional_accuracy = 2 * abs(y_vals[-1] - y_vals[0]) / abs(y_vals[-1] + y_vals[0])

        if not it % 20:
            print(f'Downhill simplex iteration {it}\n')
            print(f'Current optimal parameters: {simplex[0,:]}\n')
            print(f'Current minimum: {y_vals[0]}\n')
            print(f'Fractional accuracy: {fractional_accuracy}\n')

        if fractional_accuracy < target_fractional_accuracy:
            print(f'Fractional accuracy reached: {fractional_accuracy}')
            return simplex[0,:], func(np.array([simplex[0,:]]))

        # Calculate centroid of first N points, so excluding worst one
        centroid = np.sum(simplex[:-1, :], axis=0) / ndim
        
        # Propose new point by reflecting x_N
        x_try = np.array([2 * centroid - simplex[-1, :]])
        y_try = func(x_try)

        if y_vals[0] <= y_try < y_vals[-1]:  # New point is better, but not the best
            simplex[-1,:] = x_try  # Accept it

        elif y_try < y_vals[0]:  # New point is the best
            x_exp = 2 * x_try - centroid  # Propose expanded point
            y_exp = func(x_exp)
            if y_exp < y_try:  # Even better point found
                simplex[-1,:] = x_exp
            else:  # Expanded point is not better
                simplex[-1,:] = x_try

        else:  # Now it must be that f(x_try) >= f(x_N), so propose new point
            x_try = np.array([0.5 * (centroid + simplex[-1,:])])
            y_try = func(x_try)

            if y_try < y_vals[-1]:
                simplex[-1,:] = x_try  # Accept it

            else:  # All points were bad, so contract
                for i in range(1, ndim):
                    simplex[i,:] = 0.5 * (simplex[0,:] + simplex[i,:])

        it += 1

    print(f'Maximum number of iterations reached with a fractional accuracy of {fractional_accuracy}.')
    return simplex[0,:], func(np.array([simplex[0,:]]))


if __name__ == '__main__':
    pass