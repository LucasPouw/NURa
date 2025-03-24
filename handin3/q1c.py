import numpy as np
import matplotlib.pyplot as plt
from integrate import romberg
from minimize import downhill_simplex
from utils import readfile
import os, sys
import time


XMIN = 1e-4
XMAX = 5


def func2norm(x, a, b, c):
    return 4 * np.pi * x**2 * ((x / b) ** (a - 3)) * np.exp(-((x / b) ** c))


def normalization(a, b, c):
    func = lambda x: func2norm(x, a, b, c)
    return 1 / romberg(func, XMIN, XMAX, order=10)[0]


def numdens(x, a, b, c):
    a, b, c = np.atleast_1d(a), np.atleast_1d(b), np.atleast_1d(c)
    N = len(a)
    A = np.zeros(N)
    for i in range(N):
        A[i] = normalization(a[i], b[i], c[i])
    return A * func2norm(x[:, np.newaxis], a, b, c)


def poisson_log_llh(xdat, param_vec, model=numdens):
    model_params = [param_vec[:, i] for i in range(param_vec.shape[1])]
    model_value = model(xdat, *model_params)
    llh = np.log(model_value)
    return -np.sum(llh, axis=0)


filenames = ['satgals_m11.txt', 'satgals_m12.txt', 'satgals_m13.txt', 'satgals_m14.txt', 'satgals_m15.txt']
for i, filename in enumerate(filenames):
    radius, _ = readfile(filename)

    init_simplex = np.array([[1.5, 2.5, 0.2],
                            [1, 0.5, 1.8],
                            [1.8, 1.6, 1],
                            [0.8, 2.3, 5.5]])
    
    llh = lambda p: poisson_log_llh(param_vec=p, xdat=radius, model=numdens)
    
    start = time.time()
    minimum = downhill_simplex(llh, init_simplex)
    print(minimum, f'that took {time.time() - start} s')

    edges = 10 ** np.linspace(np.log10(XMIN), np.log10(XMAX), 51)
    hist = np.histogram(radius, bins=edges)[0]
    hist_scaled = hist / np.diff(edges) / len(radius)
    xx = np.linspace(XMIN, XMAX, 10000)  # Range for plotting

    fig1b, ax = plt.subplots()
    ax.stairs(hist_scaled, edges=edges, fill=True, label="Satellite galaxies")
    plt.plot(xx, numdens(xx, *minimum), "r-", label="Maximum likelihood model")
    ax.set(
        xlim=(XMIN, XMAX),
        ylim=(10 ** (-3), 10),
        yscale="log",
        xscale="log",
        xlabel="Relative radius",
        ylabel="Number of galaxies",
    )
    ax.legend()
    plt.savefig(os.path.join(sys.path[0], f"plots/test_1c_{filename[:-4]}.png"), dpi=600)
    




# A = 2.4
# B = 0.25
# C = 1.6

# test_xdat = np.load('/net/vdesk/data2/pouw/NUR/NURa/handin3/testdata.npy')

# init_simplex = np.array([[1.5, 2.5, 0.2],
#                             [1, 0.1, 1.8],
#                             [1.8, 1.6, 1],
#                             [0.1, 2.3, 5.5]])

# llh = lambda p: poisson_log_llh(param_vec=p, xdat=test_xdat, model=numdens)
# minimum = downhill_simplex(llh, init_simplex)
# print(minimum)

# # 21 edges of 20 bins in log-space
# edges = 10 ** np.linspace(np.log10(XMIN), np.log10(XMAX), 51)
# hist = np.histogram(test_xdat, bins=edges)[0]
# hist_scaled = hist / np.diff(edges) / len(test_xdat)
# xx = np.linspace(XMIN, XMAX, 10000)  # Range for plotting

# fig1b, ax = plt.subplots()
# ax.stairs(hist_scaled, edges=edges, fill=True, label="Satellite galaxies")
# plt.plot(xx, numdens(xx, *minimum), "r-", label="Maximum likelihood model")
# plt.plot(xx, numdens(xx, A, B, C), "b-", label="True model")
# ax.set(
#     xlim=(XMIN, XMAX),
#     ylim=(10 ** (-3), 10),
#     yscale="log",
#     xscale="log",
#     xlabel="Relative radius",
#     ylabel="Number of galaxies",
# )
# ax.legend()
# plt.savefig(os.path.join(sys.path[0], "plots/test_1c.png"), dpi=600)





# # Plot of binned data with the best fit (question 1b.4 and 1c)
# # As always, feel free to replace by your own plotting routines if you want
# xmin, xmax = 1e-4, 5. # replace by your choices
# n_bins = 100 # replace by your binning
# edges = np.exp(np.linspace(np.log(xmin), np.log(xmax), n_bins+1))

# fig1b, ax = plt.subplots(3,2,figsize=(6.4,8.0))
# for i in range(5):
#     Nsat = 100 # replace by actual appropriate number for mass bin i
#     x_radii = np.random.rand(10000) * (xmax-xmin) # replace by actual data for mass bin i
#     Ntilda = np.sort(np.random.rand(n_bins)) * (xmax-xmin) # replace by fitted model for mass bin i integrated per radial bin
#     binned_data=np.histogram(x_radii,bins=edges)[0]/Nsat
#     row=i//2
#     col=i%2
#     ax[row,col].step(edges[:-1], binned_data, where='post', label='binned data')
#     ax[row,col].step(edges[:-1], Ntilda, where='post', label='best-fit profile')
#     ax[row,col].set(yscale='log', xscale='log', xlabel='x', ylabel='N', title=f"$M_h \\approx 10^{{{11+i}}} M_{{\\odot}}/h$")
# ax[2,1].set_visible(False)
# plt.tight_layout()
# handles,labels=ax[2,0].get_legend_handles_labels()
# plt.figlegend(handles, labels, loc=(0.65,0.15))
# plt.savefig('plots/my_solution_1b.png', dpi=600)

# # Plot 1c (same code as above)
# fig1c, ax = plt.subplots(3,2,figsize=(6.4,8.0))
# for i in range(5):
#     Nsat = 100 # replace by actual appropriate number for mass bin i
#     x_radii = np.random.rand(10000) * (xmax-xmin) # replace by actual data for mass bin i
#     Ntilda = np.sort(np.random.rand(n_bins)) * (xmax-xmin) # replace by fitted model for mass bin i integrated per radial bin
#     binned_data=np.histogram(x_radii,bins=edges)[0]/Nsat
#     row=i//2
#     col=i%2
#     ax[row,col].step(edges[:-1], binned_data, where='post', label='binned data')
#     ax[row,col].step(edges[:-1], Ntilda, where='post', label='best-fit profile')
#     ax[row,col].set(yscale='log', xscale='log', xlabel='x', ylabel='N', title=f"$M_h \\approx 10^{{{11+i}}} M_{{\\odot}}/h$")
# ax[2,1].set_visible(False)
# plt.tight_layout()
# handles,labels=ax[2,0].get_legend_handles_labels()
# plt.figlegend(handles, labels, loc=(0.65,0.15))
# plt.savefig('plots/my_solution_1c.png', dpi=600)