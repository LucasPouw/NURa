import numpy as np
from sorting import quicksort
from integrate import romberg


# TWOSQRT3 = 2 * np.sqrt(3)

# def func(x_vec):
#     '''f(x, y) -> make $f(\vec{x})$, with \vec{x} = (x, y)'''
#     x = x_vec[:,0]
#     y = x_vec[:,1]
#     return -np.exp(-(x**2 + y**2))

# def func2(x_vec):
#     x = x_vec[:,0]
#     y = x_vec[:,1]
#     return np.sqrt((5 + TWOSQRT3) * (x-1)**2 - 4 * (x-1) * y + (5 - TWOSQRT3) * y**2)


XMIN = 1e-4
XMAX = 5


def func2norm(x, a, b, c):
    return 4 * np.pi * x**2 * ((x / b) ** (a - 3)) * np.exp(-((x / b) ** c))


def normalization(a, b, c):
    func = lambda x: func2norm(x, a, b, c)
    return 1 / romberg(func, XMIN, XMAX, order=10)[0]


def numdens(x, a, b, c):
    '''TODO: think about NSAT'''
    a, b, c = np.atleast_1d(a), np.atleast_1d(b), np.atleast_1d(c)
    N = len(a)
    A = np.zeros(N)
    for i in range(N):
        A[i] = normalization(a[i], b[i], c[i])
    return A * func2norm(x[:, np.newaxis], a, b, c)


def poisson_log_llh(xdat, param_vec, model=numdens):
    model_params = [param_vec[:, i] for i in range(param_vec.shape[1])]
    llh = np.log(model(xdat, *model_params))
    return -np.sum(llh, axis=0)


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


def readfile(filename):
        f = open(filename, 'r')
        data = f.readlines()[3:]  # Skip first 3 lines 
        nhalo = int(data[0])  # Number of halos
        radius = []
        
        for line in data[1:]:
            if line[:-1]!='#':
                radius.append(float(line.split()[0]))
        
        radius = np.array(radius, dtype=float)    
        f.close()
        return radius, nhalo  # Return the virial radius for all the satellites in the file, and the number of halos


if __name__ == '__main__':

    #Call this function as: 
    #radius, nhalo = readfile('satgals_m15.txt')
    
    A = 2.4
    B = 0.25
    C = 1.6

    test_xdat = np.load(r'C:\Users\lucas\OneDrive\Documenten\Uni\stk Master\Jaar 2 master\NUR\NURa\handin3\testdata.npy')

    init_simplex = np.array([[1.5, 2.5, 0.2],
                             [1, 0.1, 1.8],
                             [1.8, 1.6, 1],
                             [0.1, 2.3, 5.5]])
    
    llh = lambda p: poisson_log_llh(param_vec=p, xdat=test_xdat, model=numdens)
    minimum = downhill_simplex(llh, init_simplex)
    print(minimum)

    import matplotlib.pyplot as plt

    # 21 edges of 20 bins in log-space
    edges = 10 ** np.linspace(np.log10(XMIN), np.log10(XMAX), 51)
    hist = np.histogram(test_xdat, bins=edges)[0]
    hist_scaled = hist / np.diff(edges) / len(test_xdat)
    xx = np.linspace(XMIN, XMAX, 10000)  # Range for plotting

    fig1b, ax = plt.subplots()
    ax.stairs(hist_scaled, edges=edges, fill=True, label="Satellite galaxies")
    plt.plot(xx, numdens(xx, *minimum), "r-", label="Maximum likelihood model")
    plt.plot(xx, numdens(xx, A, B, C), "b-", label="True model")
    ax.set(
        xlim=(XMIN, XMAX),
        ylim=(10 ** (-3), 10),
        yscale="log",
        xscale="log",
        xlabel="Relative radius",
        ylabel="Number of galaxies",
    )
    ax.legend()
    plt.show()
    # plt.savefig(os.path.join(sys.path[0], "plots/my_solution_1b.png"), dpi=600)


