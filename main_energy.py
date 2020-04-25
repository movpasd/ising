"""Calculating heat capacity and comparison to F.D. theorem"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ising import datagen, thermo


datapath = Path(__file__).parents[0] / "data/energy"
mmagdatapath = Path(__file__).parents[0] / "data/meanmag"


# print("Loading data")


# dataset = datagen.DataSet(mmagdatapath, load=True)

# # Just copying bs over to datapath
# bs = np.load(mmagdatapath / "bs.npy")
# np.save(datapath / "bs.npy", bs)
# energies = []  # these will be avgd over time, but not over systems
# flucts = []  # same with this


# print("Calculating")


# for k, ens in enumerate(dataset.ensembles):

#     print(end=".")
#     ener_arr = thermo.energy(ens.asarray())  # indexed by time, system
#     energies.append(np.mean(ener_arr, axis=0))
#     flucts.append(np.std(ener_arr, axis=0))

# print()

# # These guys are saved to file indexed as [ensemble, system]
# energies = np.stack(energies, axis=0)
# flucts = np.stack(flucts, axis=0)
# np.save(datapath / "energies.npy", energies)
# np.save(datapath / "flucts.npy", flucts)


print("Plotting")


bs = np.load(datapath / "bs.npy")
energies = np.load(datapath / "energies.npy")
flucts = np.load(datapath / "flucts.npy")

sysnum = energies.shape[1]

# Best estimates obtained by averaging over each ensemble
est_energies = np.mean(energies, axis=1)
err_energies = np.std(energies, axis=1) / np.sqrt(sysnum)

# Take midpoints of each b value and associated temperatures
# These are the arguments of dE_db calculated below
midbs = (bs[1:] + bs[:-1]) / 2
Ts = 1 / midbs

# Then calculate dE/db and use chain rule to get
# heat capacity = caps = dE/dT = db/dT dE/db = -b**2 dE/dB
dE_db = np.diff(est_energies) / np.diff(bs)
caps = -midbs**2 * dE_db

# Now for everyone's favourite: error propagation!
# Since the steps in b are pretty big, it's clear that the dominant
# source of error is the random error across the ensemble, not
# numerical errors. So we can just use regular error propagation
# as applied to the finite-difference derivative formula,
#
#             f(b) - f(a)
#    df/dx ~= -----------
#                b - a
#
# b and a are known exactly, so it's just the numerator error we need

rel_caperrors = (np.sqrt(err_energies[1:]**2 + err_energies[:-1]**2)
                 / np.abs(np.diff(est_energies)))

caperrors = caps * rel_caperrors

# Second method for calculating the heat capacity, using the
# fluctuation dissipation theorem

# A bit confusing - we have fluctuations which are estimated using
# a standard deviation, and then we have errors on those fluctuations
# which are also obtained using a standard deviation. But the former
# is obtained with std over time and the latter with std over ensemble.
est_flucts = np.mean(flucts, axis=1)
err_flucts = np.std(flucts, axis=1) / np.sqrt(sysnum)

# Note "kb = 1" because of the units we chose to measure temp. with
cap2s = est_flucts**2 * bs**2
cap2errors = 2 * est_flucts * bs**2 * err_flucts  # basic err. prop.

plt.figure(figsize=(12, 8))

plt.errorbar(Ts, caps, caperrors, fmt="kx", ms=8, ecolor="r", elinewidth=1.5)
plt.errorbar(1 / bs, cap2s, cap2errors,
             fmt="bx", ms=8, ecolor=(.7, .8, 0), elinewidth=1.5)

plt.legend(["numerical", "F.-D. theorem"])

plt.title("Heat capacity versus temperature, N=30, "
          "calculated via 2 methods")
plt.xlabel("$T$")
plt.ylabel("C")

plt.show()
