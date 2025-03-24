import numpy as np
from sorting import quicksort


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

    # TODO: optimize by removing left-right check at start of loop
    
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
                print('You fucked up')
                return None
        else:
            if b < d < c:
                c = d
            elif a < d < b:
                a = d
            else:
                print('You fucked up')
                return None
    
    print(f'Maximum number of iterations reached: {max_it}')
    if fd < fb:
        return d
    else:
        return b
    

def brent(func, bracket, target_accuracy=1e-10, max_it=int(1e5)):
    return


def downhill_simplex(func, simplex, it=0, target_fractional_accuracy=1e-10, max_it=int(1e3)):
    '''
    simplex should be 2d array of shape (ndim+1, ndim), 
    where func takes n_args arguments and we have n_points datapoints.

    We sort the rows of x such that f(x0) <= f(x1) <= ... <= f(xN)    
    '''

    # TODO: assert non-degenerate simplex

    ndim = simplex.shape[1]

    while it < max_it:
        # Sort x-values based on y-values
        y_vals = func(simplex)
        mapping = {y_val: index for index, y_val in enumerate(y_vals)}  # Link y-vals to their idx
        quicksort(y_vals)  # In-place sorted
        indexing_array = [mapping[key] for key in y_vals]  # Get new idx order
        simplex = simplex[indexing_array,:]

        # Terminate if below target accuracy
        fractional_accuracy = 2 * abs(y_vals[-1] - y_vals[0]) / abs(y_vals[-1] + y_vals[0])
        # print(fractional_accuracy)
        if fractional_accuracy < target_fractional_accuracy:
            print('Fractional accuracy reached.')
            return simplex[0,:]

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

    print('Maximum number of iterations reached.')
    return simplex[0,:]


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from scipy.optimize import minimize

    func = lambda x: -(x**4 + 10 * x**3 + 10 * (x - 2)**2)
    # print(minimize(func, x0=-1.84), 'TARGET')

    # a, b, c = -4, 0, 6.47
    # fa, fb, fc = func(a), func(b), func(c)
    # d = abscissa_of_minimum(fa, fb, fc, a, b, c)
    # fd = func(d)

    # xx = np.linspace(a, c, 100)
    # plt.figure()
    # plt.plot(xx, func(xx))
    # plt.scatter(np.array([a, b, c]), np.array([fa, fb, fc]), marker='o', color='black')
    # plt.scatter(d, fd, marker='X', color='red')
    # plt.savefig('test.pdf')
    # plt.show()
    
    # print(bracket_minimum(func, a=-2, b=0))

    soln = golden_section(func, a=-3, b=0)
    print(soln)