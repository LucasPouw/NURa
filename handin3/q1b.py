import numpy as np
from utils import readfile
from integrate import romberg
from minimize import downhill_simplex
import matplotlib.pyplot as plt
import os, sys
import time


XMIN_PLOT = 1e-4
XMIN_INTEGRAL = 0
XMAX = 5


def func2norm(x, a, b, c):
    '''Unnormalized n(x)'''
    return x**(a - 1) * b ** (3 - a) * np.exp(-((x / b) ** c))


def normalization(a, b, c):
    func = lambda x: func2norm(x, a, b, c)
    return 1 / romberg(func, XMIN_INTEGRAL, XMAX, order=10)[0]


def num_gal_pdf(x, a, b, c, nsat):
    '''Returns normalized pdf p(x|a,b,c) = 4pi x^2 n(x|a,b,c)'''
    return  nsat * normalization(a, b, c) * func2norm(x, a, b, c)


def chi_squared(edges, bin_heights, param_vec, model, **model_kwargs):
    '''Chi^2 calculation to use in combination with downhill-simplex'''
    
    chi2_arr = np.zeros(param_vec.shape[0])
    for k in range(param_vec.shape[0]):  # Loop over all the points of the simplex
        model_params = param_vec[k,:]
        # model_params = np.array([1.31420016, 1.11502032, 3.13383544])
        print(model_params)

        # Apply priors to a and b
        if model_params[0] < 1:
            return np.tile(np.inf, param_vec.shape[0])
        if model_params[1] < 0:
            return np.tile(np.inf, param_vec.shape[0])
        
        func = lambda x: model(x, *model_params, **model_kwargs)  # p(x|a,b,c) to integrate over bins

        binned_model_values = np.zeros(len(edges)-1)
        for i in range(len(edges)-1):  # Loop over all bins
            binned_model_values[i] = romberg(func, edges[i], edges[i+1], order=10)[0]
        
        variance = bin_heights.copy()
        variance[variance == 0] = 1
        chi2_arr[k] = np.sum( (bin_heights - binned_model_values)**2 / variance )
    return chi2_arr

filenames = ['satgals_m11.txt', 'satgals_m12.txt', 'satgals_m13.txt', 'satgals_m14.txt', 'satgals_m15.txt']
fig1b, ax = plt.subplots(3, 2, figsize=(6.4, 8.0))
for i, filename in enumerate(filenames):
    radius, nhalo = readfile(filename)
    
    nbins = 100  # More bins is better bins
    Nsat = len(radius) / nhalo  # Mean number of satellites per halo
    print('Mean number of satellites per halo:', Nsat)
    logbins = np.linspace(np.log10(XMIN_PLOT), np.log10(XMAX), nbins + 1)
    edges = 10**logbins
    centers = 10**(logbins[:-1] + np.diff(logbins) * 0.5)
    binned_data = np.histogram(radius, bins=edges)[0] / nhalo / np.diff(edges)

    init_simplex = np.array([[1.31420016, 1.11502032, 3.13383544], 
                            [1.31420016, 1.11502032, 3.13383544], 
                            [1.31420016, 1.11502032, 3.13383544], 
                            [1.31420016, 1.11502032, 3.13383544]]) * np.random.uniform(0.95, 1.05, size=(4,3))
    
    # chi2 = lambda p: chi_squared(edges, binned_data, param_vec=p, model=num_gal_pdf, nsat=Nsat)
    
    # start = time.time()
    # minimum = downhill_simplex(chi2, init_simplex, target_fractional_accuracy=1e-8)
    # print(minimum, f'that took {time.time() - start} s')

    minimum = np.array([1.31420016, 1.11502032, 3.13383544])

    row = i // 2
    col = i % 2
    ax[row,col].step(edges[:-1], binned_data, where='post', label='binned data')
    ax[row,col].step(edges[:-1], num_gal_pdf(centers, *minimum, nsat=Nsat), where='post', label='Best-fit profile', color='red', linewidth=1)
    ax[row,col].set(yscale='log', xscale='log', xlabel='x', ylabel='N', title=f"$M_h \\approx 10^{{{11+i}}} M_{{\\odot}}/h$")
    ax[row, col].set_ylim(1e-3, 10 * Nsat)
ax[2,1].set_visible(False)
plt.tight_layout()
handles,labels=ax[2,0].get_legend_handles_labels()
plt.figlegend(handles, labels, loc=(0.65, 0.15))
plt.savefig(os.path.join(sys.path[0], f'plots/q1b.png'), bbox_inches='tight', dpi=600)
