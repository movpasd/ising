"""Investigating finite-sized scaling"""


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ising import simulator, plotter, thermo, datagen


datapath = Path(__file__).parents[0] / "data/scaling"
resultspath = Path(__file__).parents[0] / "results/scaling"


def find_energy(N, T, tol, sysnum=20):
    """Find mean energy and fluctuations to specified tolerance"""

    # Basic working:
    # Create ensemble with given parameters.
    # Keep simulating, periodically calculating the energy.
    # Error on energy is estimated by avging over ensemble.
    # As soon as the energy is found to a suitable tolerance,
    # return that value.

    print(f"Finding energy, N={N}, T={T:.2f}, {sysnum} systems\n")

    relaxtime = 150
    maxtime = 10000
    checktime = 100
    sysnum = 20

    b = 1 / T
    ensemble = datagen.Ensemble(N, sysnum, p=1, b=b, h=0, randflip=True)

    # initial simulation to reach equilibrium
    ensemble.simulate(relaxtime + 10)
    ensemble.trim_init(relaxtime)

    total_iterations = 10

    done = False

    while total_iterations < maxtime and not done:

        print(end=f"Simulating {total_iterations} -> "
              f"{total_iterations + checktime}: ")

        ensemble.simulate(checktime, reset=False)

        # Now check if we have energy to given tolerance

        arr = ensemble.asarray()
        energies = np.mean(thermo.energy(arr), axis=0)  # avg over time

        est_energy = np.mean(energies)
        err_energy = np.std(energies, ddof=1) / np.sqrt(sysnum)

        rel_err = np.abs(err_energy / est_energy)

        print(
            f"|sigma / E| = |{err_energy:.3f} / {est_energy:.3f}| = {rel_err:.3f}")

        if rel_err < tol:
            done = True

        total_iterations += checktime

    if not done:
        print(f"Exceeded max time of {maxtime}")

    return est_energy, err_energy


def calculate(Ns, Ts, tol):

    for N in Ns:

        if N == 3:

            ests = []
            errs = []

            for T in Ts:

                est, err = find_energy(N, T, tol)
                ests.append(est)
                errs.append(err)

                print()

            np.save(datapath / f"ests-N{N}.npy", np.array(ests))
            np.save(datapath / f"errs-N{N}.npy", np.array(errs))


def results(Ns, Ts):

    plt.figure(figsize=(12, 8))

    # colours
    cs = [(.9, 0, 0), (.8, .7, 0), (0, .8, 0), (0, .7, .7),
          (0, 0, .9), (.7, 0, .7), (.5, .5, .5)]

    for k, N in enumerate(Ns):

        ests = np.load(datapath / f"ests-N{N}.npy")
        errs = np.load(datapath / f"errs-N{N}.npy")

        # plt.errorbar(Ts, est / N**2, err / N**2,
        #              fmt="x", color=cs[k], markersize=8,
        #              ecolor=cs[k] + (0.5,), elinewidth=1)

        midTs = (Ts[1:] + Ts[:-1]) / 2
        caps = np.diff(ests) / np.diff(Ts) / N**2
        caperrs = np.sqrt(errs[1:]**2 + errs[:-1]**2) / \
            np.abs(np.diff(Ts)) / N**2

        # Average over rolling window of width w
        w = 20

        # Average over appropriate range
        ks = np.argwhere(np.logical_and(2 <= midTs, midTs <= 4))[:, 0]
        ki, kf = ks[0], ks[-1] + 1
        
        sqcaperrs = caperrs**2

        w_midTs = thermo.rolling_average(midTs[ki:kf], w)
        midTs = np.concatenate([midTs[:ki], w_midTs, midTs[kf:]])
        w_caps = thermo.rolling_average(caps[ki:kf], w)
        caps = np.concatenate([caps[:ki], w_caps, caps[kf:]])
        w_sqcaperrs = thermo.rolling_average(sqcaperrs[ki:kf], w) / w
        sqcaperrs = np.concatenate([sqcaperrs[:ki], w_sqcaperrs, sqcaperrs[kf:]])
        caperrs = np.sqrt(sqcaperrs)

        plt.errorbar(midTs, caps, caperrs,
                     fmt="-", color=cs[k], ecolor=cs[k] + (0.2,), elinewidth=3)

    T_ons = 2 / np.log(1 + np.sqrt(2))
    plt.plot([T_ons, T_ons], [0, 1.25], "k--")

    plt.legend(["$T_{ons}$"] + [f"N = {N}" for N in Ns])

    plt.title("Specific heat capacity versus temperature for various N")
    plt.xlabel("Temperature")
    plt.ylabel("Specific heat capacity $C/N^2$")

    plt.savefig(resultspath / "caps.pdf")
    plt.show()
    plt.close()


ranges = [
    (1, 2, 0.2),
    (2, 4, 0.025),
    (4, 5, 0.2)
]

Ns = [12, 8, 6, 5, 4, 3, 2]
Ts = np.concatenate([np.arange(*r) for r in ranges])

# calculate(Ns, Ts, tol=0.005)
results(Ns, Ts)
