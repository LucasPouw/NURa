#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

A = 1.0  # to be computed
Nsat = 100
a = 2.4
b = 0.25
c = 1.6


def n(x, A, Nsat, a, b, c):
    return A * Nsat * ((x / b) ** (a - 3)) * np.exp(-((x / b) ** c))


# Plot of histogram in log-log space with line (question 1b)
xmin, xmax = 10**-4, 5
N_generate = 10000

# 21 edges of 20 bins in log-space
edges = 10 ** np.linspace(np.log10(xmin), np.log10(xmax), 21)
hist = np.histogram(
    xmin + np.sort(np.random.rand(N_generate)) * (xmax - xmin), bins=edges
)[
    0
]  # replace!
hist_scaled = (
    1e-3 * hist
)  # replace; this is NOT what you should be plotting, this is just a random example to get a plot with reasonable y values (think about how you *should* scale hist)

relative_radius = edges.copy()  # replace!
analytical_function = edges.copy()  # replace

fig1b, ax = plt.subplots()
ax.stairs(
    hist_scaled, edges=edges, fill=True, label="Satellite galaxies"
)  # just an example line, correct this!
plt.plot(
    relative_radius, analytical_function, "r-", label="Analytical solution"
)  # correct this according to the exercise!
ax.set(
    xlim=(xmin, xmax),
    ylim=(10 ** (-3), 10),
    yscale="log",
    xscale="log",
    xlabel="Relative radius",
    ylabel="Number of galaxies",
)
ax.legend()
plt.savefig("figures/my_solution_1b.png", dpi=600)

# Cumulative plot of the chosen galaxies (1c)
chosen = xmin + np.sort(np.random.rand(Nsat)) * (xmax - xmin)  # replace!
fig1c, ax = plt.subplots()
ax.plot(chosen, np.arange(100))
ax.set(
    xscale="log",
    xlabel="Relative radius",
    ylabel="Cumulative number of galaxies",
    xlim=(xmin, xmax),
    ylim=(0, 100),
)
plt.savefig("figures/my_solution_1c.png", dpi=600)
