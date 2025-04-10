import numpy as np
import matplotlib.pyplot as plt
from integrate import romberg
from minimize import downhill_simplex
from utils import readfile
from sorting import quicksort
import os, sys
import time


XMIN_PLOT = 1e-4
XMIN_INTEGRAL = 0
XMAX = 5


def func2norm(x, a, b, c):
    return x**(a - 1) * b ** (3 - a) * np.exp(-((x / b) ** c))


def normalization(a, b, c, xmin=XMIN_INTEGRAL, xmax=XMAX):
    func = lambda x: func2norm(x, a, b, c)
    return 1 / romberg(func, xmin, xmax, order=10)[0]


def num_gal_pdf(x, a, b, c):
    a, b, c = np.atleast_1d(a), np.atleast_1d(b), np.atleast_1d(c)
    N = len(a)
    A = np.zeros(N)
    for i in range(N):
        A[i] = normalization(a[i], b[i], c[i])
    return A * func2norm(x[:, np.newaxis], a, b, c)


def poisson_log_llh(data, param_vec, model=num_gal_pdf):
    model_params = [param_vec[:, i] for i in range(param_vec.shape[1])]
    model_value = model(data, *model_params)
    llh = np.log(model_value)
    return -np.sum(llh, axis=0)  # Maximize L = minimize -L


fig1c, ax = plt.subplots(3,2,figsize=(6.4,8.0))
filenames = ['satgals_m11.txt', 'satgals_m12.txt', 'satgals_m13.txt', 'satgals_m14.txt', 'satgals_m15.txt']
for i, filename in enumerate(filenames):
    radius, _ = readfile(filename)

    init_simplex = np.array([[1.5, 2.5, 0.2],
                            [1, 0.5, 1.8],
                            [1.8, 1.6, 1],
                            [1.1, 2.3, 5.5]])
    
    llh = lambda p: poisson_log_llh(data=radius, param_vec=p, model=num_gal_pdf)
    
    start = time.time()
    minimum = downhill_simplex(llh, init_simplex)
    print(minimum, f'that took {time.time() - start} s')

    nbins = 100
    logbins = np.linspace(np.log10(XMIN_PLOT), np.log10(XMAX), nbins + 1)
    edges = 10**(logbins)
    centers = 10**(logbins[:-1] + np.diff(logbins) * 0.5)
    hist = np.histogram(radius, bins=edges)[0]
    hist_scaled = hist / np.diff(edges) / len(radius)

    row = i // 2
    col = i % 2
    ax[row,col].step(edges[:-1], hist_scaled, where='post', label='binned data', linewidth=2)
    ax[row,col].step(edges[:-1], num_gal_pdf(centers, *minimum), where='post', label='Best-fit profile', color='red', linewidth=1)
    ax[row,col].set(yscale='log', xscale='log', xlabel='x', ylabel='N', title=f"$M_h \\approx 10^{{{11+i}}} M_{{\\odot}}/h$")
    ax[row,col].set_ylim(1e-4, 5)
ax[2,1].set_visible(False)
plt.tight_layout()
handles, labels = ax[2,0].get_legend_handles_labels()
plt.figlegend(handles, labels, loc=(0.65,0.15))
plt.savefig(os.path.join(sys.path[0], f"plots/q1c.png"), dpi=600)
