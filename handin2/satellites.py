#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from integrate import romberg
# from rng import uniform
import sys, os

uniform = np.random.uniform   # TODO: CHANGE TO MY OWN

NSAT = 100
A = 2.4
B = 0.25
C = 1.6

XMIN, XMAX = 10**-4, 5
xx = np.linspace(XMIN, XMAX, 10000)  # Range for plotting and integrating

N_GENERATE = 10000


def numdens(x, norm, Nsat=NSAT, a=A, b=B, c=C):
    return norm * Nsat * ((x / b) ** (a - 3)) * np.exp(-((x / b) ** c))


# Q1a
func2integrate = lambda x: numdens(x, Nsat=1, norm=1) * x**2 * 4 * np.pi  # Nsat = 1 such that norm = 1/integral
integral = romberg(func2integrate, XMIN, XMAX, order=14)  # Wolfram: 0.108756
normalization = 1/integral[0]
print(integral, 'integral of p(x) for norm=1')
print(normalization, 'required normalization on x=[0, 5]')

# Q1b
def beta(a=A, b=B, c=C):
    '''ugly math is ugly'''
    a_minus_one = a - 1
    a_minus_one_divby_c = a_minus_one / c
    result = a_minus_one * ( np.exp(-a_minus_one_divby_c) * a_minus_one_divby_c ** a_minus_one_divby_c * a / (b ** (-a_minus_one)) ) ** (-a / a_minus_one)
    return result


def func2samp(x, norm=normalization, a=A, b=B, c=C):
    return 4 * np.pi * norm * b**(3 - a) * x**(a - 1) / (1 + beta(a, b, c) * x**a)


def p_of_x(x, norm=normalization, a=A, b=B, c=C):
    return 4 * np.pi * norm * b**(3 - a) * x**(a - 1) * np.exp(- (x / b)**c)


def cdf(x, norm=normalization, a=A, b=B, c=C):
    betaval = beta(a, b, c)
    return 4 * np.pi * norm * b ** (3 - a) / (betaval * a) * np.log(1 + betaval * x ** a)


def inverse_cdf(x, norm=normalization, a=A, b=B, c=C):
    betaval = beta(a, b, c)
    result = ((np.exp(betaval * a * x / (4 * np.pi * norm * b ** (3 - a))) - 1) / betaval) ** (1 / a)
    return result


def pmax(norm=normalization, a=A, b=B, c=C):
    return 4 * np.pi * norm * b**2 * ((a-1)/c)**((a-1)/c) * np.exp((1-a)/c)


#################### Rejection sampling ####################
accepted_samps = np.zeros(N_GENERATE)
samps_per_it = int(1e3)  # Chunking speeds things up
n_accepted = 0
it_counter = 0
max_it = int(1e5)
max_value_cdf = cdf(XMAX)
while n_accepted < N_GENERATE:
    it_counter += 1
    if it_counter > max_it:
        sys.exit('Maximum number of iterations reached. Exiting...')
        break

    # First sample from f(x) using inverse transform sampling
    unif1 = inverse_cdf( uniform(low=0, high=max_value_cdf, size=samps_per_it) )
    # for x in unif1:
    #     print('high', func2samp(x))
    #     print(x, p_of_x(x))
    # print(np.max(func2samp(fx_samp)), unif1[np.argmax(func2samp(fx_samp))], 'AA')

    # Then rejection-sample p(x)
    # unif1 = uniform(low=XMIN, high=XMAX, size=samps_per_it)
    # unif2 = uniform(low=0, high=pmax(), size=samps_per_it)
    unif2 = uniform(low=0, high=func2samp(unif1), size=samps_per_it)
    accept_these = np.where(unif2 < p_of_x(unif1))[0]

    trial_n_total = n_accepted + len(accept_these)
    if trial_n_total > N_GENERATE:  # We have more samples than needed in the final iteration
        print(f'Found {trial_n_total - N_GENERATE} too many valid samples, removing them.')
        accept_these = accept_these[:N_GENERATE - n_accepted]  # Don't include the samples that were not requested

    accepted_samps[n_accepted:trial_n_total] = unif1[accept_these]
    n_accepted += len(accept_these)
    
    if not it_counter % 10:
        print(f'Accepted:{n_accepted}/{N_GENERATE}')

print(f'Accepted:{n_accepted}/{N_GENERATE} in {it_counter} iterations.')
############################################################

x_samps = inverse_cdf( uniform(low=0, high=max_value_cdf, size=N_GENERATE ) )

plt.figure()
plt.plot(xx, cdf(xx), color='black', label=r'CDF')
plt.plot(xx, inverse_cdf(xx), color='grey', label=r'ICDF')
plt.plot(xx, func2samp(xx), color='red', zorder=6, label=r'$f(x)$')
plt.plot(xx, p_of_x(xx), color='blue', zorder=5, label=r'$p(x)$')

edges = 10 ** np.linspace(np.log10(XMIN), np.log10(XMAX), 51)
hist, _ = np.histogram(x_samps, bins=edges, density=True)
plt.stairs(hist * max_value_cdf, edges=edges, linewidth=2, color='coral', label='Inverse transform sampled')
plt.hist(accepted_samps, density=True, histtype='step', color='skyblue', linewidth=2, bins=edges, label='Rejection sampled')
plt.ylim(10 ** (-3), 5)
plt.xlim(XMIN, XMAX)
plt.loglog()
# plt.semilogy()
plt.legend()
plt.savefig(os.path.join(sys.path[0], "plots/testing1blogax_manysamp.png"), dpi=600)

# 21 edges of 20 bins in log-space
edges = 10 ** np.linspace(np.log10(XMIN), np.log10(XMAX), 21)
hist = np.histogram(accepted_samps, bins=edges)[0]
hist_scaled = hist / np.diff(edges) / (N_GENERATE / NSAT ) / NSAT   # TODO: figure out why the two factors of 100 are necessary to normalize the histogram -> I probably forgot an <Nsat> somewhere

relative_radius = xx
analytical_function = p_of_x(xx)

fig1b, ax = plt.subplots()
ax.stairs(
    hist_scaled, edges=edges, fill=True, label="Satellite galaxies"
)  # just an example line, correct this!
plt.plot(
    relative_radius, analytical_function, "r-", label="Analytical solution"
)  # correct this according to the exercise!
ax.set(
    xlim=(XMIN, XMAX),
    ylim=(10 ** (-3), 10),
    yscale="log",
    xscale="log",
    xlabel="Relative radius",
    ylabel="Number of galaxies",
)
ax.legend()
plt.savefig(os.path.join(sys.path[0], "plots/my_solution_1b.png"), dpi=600)

# # Cumulative plot of the chosen galaxies (1c)
# chosen = XMIN + np.sort(np.random.rand(NSAT)) * (XMAX - XMIN)  # replace!
# fig1c, ax = plt.subplots()
# ax.plot(chosen, np.arange(100))
# ax.set(
#     xscale="log",
#     xlabel="Relative radius",
#     ylabel="Cumulative number of galaxies",
#     xlim=(XMIN, XMAX),
#     ylim=(0, 100),
# )
# plt.savefig("figures/my_solution_1c.png", dpi=600)
