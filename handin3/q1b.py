import numpy as np
from utils import readfile
from integrate import romberg
from minimize import downhill_simplex
from sorting import quicksort
import matplotlib.pyplot as plt
import os, sys
import time


XMIN = 1e-4
XMAX = 5


def func2norm(x, a, b, c):
    '''Requires multiplication by normalization() function'''
    return x**2 * ((x / b) ** (a - 3)) * np.exp(-((x / b) ** c))  # 4pi factor not necessary


def normalization(a, b, c):
    func = lambda x: func2norm(x, a, b, c)
    return 1 / romberg(func, XMIN, XMAX, order=10)[0]


def num_gal_pdf(x, a, b, c, nsat):
    '''Returns normalized pdf p(x|a,b,c) = 4pi n(x|a,b,c)x^2'''
    return nsat * normalization(a, b, c) * func2norm(x, a, b, c)


def chi_squared(edges, bin_heights, param_vec, model, **model_kwargs):
    '''Chi^2 calculation to use in combination with downhill-simplex'''
    
    chi2_arr = np.zeros(param_vec.shape[0])
    for k in range(param_vec.shape[0]):  # Loop over all the points of the simplex
        model_params = param_vec[k,:]
        func = lambda x: model(x, *model_params, **model_kwargs)  # p(x|a,b,c) to integrate over bins

        binned_model_values = np.zeros(len(edges)-1)
        for i in range(len(edges)-1):  # Loop over all bins
            binned_model_values[i] = romberg(func, edges[i], edges[i+1], order=10)[0]   
        chi2_arr[k] = np.sum( (bin_heights - binned_model_values)**2 / bin_heights )
    print(param_vec)
    return chi2_arr


filenames = ['satgals_m11.txt', 'satgals_m12.txt', 'satgals_m13.txt', 'satgals_m14.txt', 'satgals_m15.txt']
fig1b, ax = plt.subplots(3, 2, figsize=(6.4, 8.0))
for i, filename in enumerate(filenames):
    radius, nhalo = readfile(filename)
    
    nbins = 100  # More bins is better bins
    Nsat = len(radius) / nhalo  # Mean number of satellites per halo
    print('Mean number of satellites per halo:', Nsat)
    logbins = np.linspace(np.log10(XMIN), np.log10(XMAX), nbins + 1)
    edges = 10**logbins
    centers = 10**(logbins[:-1] + np.diff(logbins) * 0.5)
    binned_data = np.histogram(radius, bins=edges)[0] / nhalo / np.diff(edges)
    # print(np.sum(binned_data * np.diff(edges)) / Nsat)

    init_simplex = np.array([[1.5, 2.5, 0.2],
                            [1, 0.5, 1.8],
                            [1.8, 1.6, 1],
                            [0.8, 2.3, 5.5]]) * 10
    
    chi2 = lambda p: chi_squared(edges, binned_data, param_vec=p, model=num_gal_pdf, nsat=Nsat)
    
    start = time.time()
    minimum = downhill_simplex(chi2, init_simplex)
    print(minimum, f'that took {time.time() - start} s')

    # Ntilda = num_gal_pdf(edges[:-1], *minimum)


#     row = i // 2
#     col = i % 2
#     ax[row,col].step(edges[:-1], binned_data, where='post', label='binned data')
#     ax[row,col].scatter(centers, binned_data, color='black', zorder=3, marker='.', label='fitted points')
#     # ax[row,col].step(edges[:-1], Ntilda, where='post', label='best-fit profile')
#     ax[row,col].set(yscale='log', xscale='log', xlabel='x', ylabel='N', title=f"$M_h \\approx 10^{{{11+i}}} M_{{\\odot}}/h$")

#     xx = np.linspace(XMIN, XMAX, 10000)  # Range for plotting

# ax[2,1].set_visible(False)
# plt.tight_layout()
# handles,labels=ax[2,0].get_legend_handles_labels()
# plt.figlegend(handles, labels, loc=(0.65, 0.15))
# plt.savefig(os.path.join(sys.path[0], f'plots/q1b_{filename[:-4]}.png'), bbox_inches='tight', dpi=600)
