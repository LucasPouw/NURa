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
    '''Unnormalized x^2 n(x)'''
    return  x ** (a - 1) * b ** (3 - a) * np.exp(-((x / b) ** c))


def normalization(a, b, c):
    func = lambda x: func2norm(x, a, b, c)
    return 1 / romberg(func, XMIN_INTEGRAL, XMAX, order=6)[0]


def num_gal_pdf(x, a, b, c, nsat):
    return nsat * normalization(a, b, c) * func2norm(x, a, b, c)


def ntilde(func, edges, order):
    binned_model_values = np.zeros(len(edges)-1)
    for i in range(len(edges)-1):  # Loop over all bins
        binned_model_values[i] = romberg(func, edges[i], edges[i+1], order=order)[0]
    return  binned_model_values


def chi_squared(edges, bin_heights, param_vec, model, **model_kwargs):
    '''Chi^2 calculation to use in combination with downhill-simplex'''
    chi2_arr = np.zeros(param_vec.shape[0])
    for k in range(param_vec.shape[0]):  # Loop over all the points of the simplex
        model_params = param_vec[k,:]

        # Apply priors to a and b
        if model_params[0] < 1:
            return np.tile(np.inf, param_vec.shape[0])
        if model_params[1] < 0:
            return np.tile(np.inf, param_vec.shape[0])
        
        func = lambda x: model(x, *model_params, **model_kwargs)  # p(x|a,b,c) to integrate over bins
        binned_model_values = ntilde(func, edges, order=6)
        chi2_arr[k] = np.sum( (bin_heights - binned_model_values)**2 / binned_model_values )
    return chi2_arr


filenames = ['satgals_m11.txt', 'satgals_m12.txt', 'satgals_m13.txt', 'satgals_m14.txt', 'satgals_m15.txt']
ymin = [1e-8, 1e-7, 1e-5, 1e-3, 1e-1]  # For plotting
ymax = [1e-2, 1e-1, 1e0, 1e1, 1e2]

# minima = np.array([[1.30767685, 1.11882273, 3.15335137],
#                    [1.5918548, 0.91906036, 3.47614821],
#                    [1.48406971, 0.80283839, 2.87626552],
#                    [1.94381939, 0.60776245, 2.50984679],
#                    [1.99410978, 0.70836665, 2.04264211]])

fig1b, ax = plt.subplots(3, 2, figsize=(6.4, 8.0))
for i, filename in enumerate(filenames):
    radius, nhalo = readfile(filename)
    
    nbins = 100  # More bins is better bins
    dof = nbins - 4
    Nsat = len(radius) / nhalo  # Mean number of satellites per halo
    logbins = np.linspace(np.log10(XMIN_PLOT), np.log10(XMAX), nbins + 1)
    edges = 10**logbins
    centers = 10**(logbins[:-1] + np.diff(logbins) * 0.5)
    binned_data = np.histogram(radius, bins=edges)[0] / nhalo  # Integrates to Nsat

    init_simplex = np.array([[1.8, 1.9, 2.6], 
                            [2., 0.7, 3.2], 
                            [2.5, 0.8, 2.4], 
                            [1.9, 1.6, 3.8]])
    
    chi2 = lambda p: chi_squared(edges, binned_data, param_vec=p, model=num_gal_pdf, nsat=Nsat)
    
    start = time.time()
    minimum, best_chi2 = downhill_simplex(chi2, init_simplex, target_fractional_accuracy=1e-8, init_volume_thresh=0.1)
    stop = time.time() - start
    print(f'----- CHI^2 RESULTS FOR FILE {filename} -----')
    print(f'Mean number of satellites per halo: {Nsat}')
    print('\nBest-fit parameters are:\n')
    print(f'a = {minimum[0]}, \nb = {minimum[1]}, \nc = {minimum[2]}')
    print('\nMinimum chi-squared is:\n')
    print(f'chi^2 = {best_chi2[0] * nhalo}, \nchi^2 / k = {best_chi2[0] * nhalo / dof}')
    print(f'\nThis took {stop:.2f} s\n')

    func = lambda x: num_gal_pdf(x, *minimum, nsat=Nsat)
    binned_model_values = ntilde(func, edges, order=6)
    # func_llh = lambda x:num_gal_pdf(x, *minima[i], nsat=Nsat)
    # llh_model_values = ntilde(func_llh, edges, order=6)

    row = i // 2
    col = i % 2
    ax[row,col].step(edges[:-1], binned_data, where='post', color='black', label='binned data')
    ax[row,col].step(edges[:-1], binned_model_values, where='post', label=r'Best-fit $\chi^{2}$', color='red', linewidth=1)
    # ax[row,col].step(edges[:-1], llh_model_values, where='post', label=r'Best-fit $\ln \mathcal{L}$', color='blue', linewidth=1)
    ax[row,col].set(yscale='log', xscale='log', xlabel='x', ylabel='N', title=f"$M_h \\approx 10^{{{11+i}}} M_{{\\odot}}/h$")
    ax[row,col].set_ylim(ymin[i], ymax[i])
ax[2,1].set_visible(False)
plt.tight_layout()
handles,labels=ax[2,0].get_legend_handles_labels()
plt.figlegend(handles, labels, loc=(0.65, 0.15))
plt.savefig(os.path.join(sys.path[0], f'plots/q1b.png'), bbox_inches='tight', dpi=600)
