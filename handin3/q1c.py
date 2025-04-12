import numpy as np
import matplotlib.pyplot as plt
from integrate import midpoint_romberg as romberg
from minimize import downhill_simplex
from utils import readfile
import os, sys
import time
from utils import log_factorial
from scipy.special import gammainc


XMIN_PLOT = 1e-4
XMIN_INTEGRAL = 0
XMAX = 5


def func2norm(x, a, b, c):
    return x**(a - 1) * b ** (3 - a) * np.exp(-((x / b) ** c))


def normalization(a, b, c, xmin=XMIN_INTEGRAL, xmax=XMAX):
    func = lambda x: func2norm(x, a, b, c)
    return 1 / romberg(func, xmin, xmax, order=10)[0]


def num_gal_pdf(x, a, b, c):
    x, a, b, c = np.atleast_1d(x), np.atleast_1d(a), np.atleast_1d(b), np.atleast_1d(c)
    N = len(a)
    A = np.zeros(N)
    for i in range(N):
        A[i] = normalization(a[i], b[i], c[i])
    return A * func2norm(x[:, np.newaxis], a, b, c)


def ntilde(func, edges, order):
    binned_model_values = np.zeros(len(edges)-1)
    for i in range(len(edges)-1):  # Loop over all bins
        binned_model_values[i] = romberg(func, edges[i], edges[i+1], order=order)[0]
    return  binned_model_values


def poisson_log_llh(data, param_vec, model=num_gal_pdf):
    model_params = [param_vec[:, i] for i in range(param_vec.shape[1])]
    model_value = model(data, *model_params)
    llh = np.log(model_value)
    return -np.sum(llh, axis=0)  # Maximize L = minimize -L


def unnormalized_poisson_log_llh(bin_heights, binned_model):
    return np.sum(-bin_heights * np.log(binned_model) + binned_model + log_factorial(bin_heights))


def chi2_cdf(x, k):
    "Returns P(chi^2 <= x) for k DoF"
    return gammainc(0.5 * k, 0.5 * x)


def Gtest(observed, expected, dof):
    remove_zeros = (observed != 0)  # Empty bins have G = 0
    G = 2 * np.sum( observed[remove_zeros] * (np.log(observed[remove_zeros]) - np.log(expected[remove_zeros])) )
    p_value = 1 - chi2_cdf(G, dof)
    return G, p_value


filenames = ['satgals_m11.txt', 'satgals_m12.txt', 'satgals_m13.txt', 'satgals_m14.txt', 'satgals_m15.txt']
ymin = [1e-8, 1e-7, 1e-5, 1e-3, 1e-1]  # For plotting
ymax = [1e-2, 1e-1, 1e0, 1e1, 1e2]

fig1c, ax = plt.subplots(3,2,figsize=(6.4,8.0))
for i, filename in enumerate(filenames):
    radius, nhalo = readfile(filename)

    init_simplex = np.array([[1.8, 1.9, 2.6], 
                            [2., 0.7, 3.2], 
                            [2.5, 0.8, 2.4], 
                            [1.9, 1.6, 3.8]])
    
    llh = lambda p: poisson_log_llh(data=radius, param_vec=p, model=num_gal_pdf)
    
    start = time.time()
    minimum, best_log_llh = downhill_simplex(llh, init_simplex, target_fractional_accuracy=1e-0, init_volume_thresh=0.1)
    stop = time.time() - start

    nbins = 100
    logbins = np.linspace(np.log10(XMIN_PLOT), np.log10(XMAX), nbins + 1)
    edges = 10**(logbins)
    centers = 10**(logbins[:-1] + np.diff(logbins) * 0.5)
    binned_data = np.histogram(radius, bins=edges)[0] / nhalo
    func = lambda x: num_gal_pdf(x, *minimum) * len(radius) / nhalo
    binned_model_values = ntilde(func, edges, order=6)

    print(f'\n----- $-\\ln \\mathcal{{L}}$ RESULTS FOR FILE $\\rm {filename}$ -----\n')
    print('Best-fit parameters are:\n')
    print(f'a = {minimum[0]}, \nb = {minimum[1]}, \nc = {minimum[2]}')
    print('\nMinimum $-\\ln \\mathcal{{L}}$ on normalized model is:\n')
    print(f'$-\\ln \\mathcal{{L}}$ = {best_log_llh[0] + 1}')
    print('\nMinimum $-\\ln \\mathcal{{L}}$ value on binned, unscaled model is: \n')
    print(f'$-\\ln \\mathcal{{L}}$ = {unnormalized_poisson_log_llh(binned_data * nhalo, binned_model_values * nhalo)}')
    print('\nG-test results using the untouched model: \n')
    g, p = Gtest(binned_data * nhalo, binned_model_values * nhalo, dof=nbins-4)
    print(f'$G$ = {g}, p-value = {p}')
    print('\nG-test results using the renormalized model: \n')
    g2, p2 = Gtest(binned_data * nhalo, binned_model_values / np.sum(binned_model_values) * np.sum(binned_data) * nhalo, dof=nbins-4)
    print(f'$G$ = {g2}, p-value = {p2}')
    print(f'\nOptimization took {stop:.2f} s\n')

    row = i // 2
    col = i % 2
    ax[row,col].step(edges[:-1], binned_data, where='post', label='binned data', color='black', linewidth=2)
    ax[row,col].step(edges[:-1], binned_model_values, where='post', label='Best-fit profile', color='red', linewidth=1)
    ax[row,col].set(yscale='log', xscale='log', xlabel='x', ylabel='N', title=f"$M_h \\approx 10^{{{11+i}}} M_{{\\odot}}/h$")
    ax[row,col].set_ylim(ymin[i], ymax[i])
ax[2,1].set_visible(False)
plt.tight_layout()
handles, labels = ax[2,0].get_legend_handles_labels()
plt.figlegend(handles, labels, loc=(0.65,0.15))
plt.savefig(os.path.join(sys.path[0], f"plots/q1c.png"), dpi=600)
