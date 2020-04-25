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
    cs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.7, 0.7, 0)]

    for k, N in enumerate(Ns):

        est = np.load(datapath / f"ests-N{N}.npy")
        err = np.load(datapath / f"errs-N{N}.npy")

        plt.errorbar(Ts, est / N**2, err / N**2, 
            fmt="x", color=cs[k], markersize=8,
            ecolor=cs[k] + (0.5,), elinewidth=1)

    plt.legend([f"N = {N}" for N in Ns])

    plt.title("Average specific energy versus temperature for various N")
    plt.xlabel("Temperature")
    plt.ylabel("Specific energy $E/N^2$")

    plt.savefig("energy.pdf")
    plt.show()
    plt.close()

    

ranges = [
    (1, 2, 0.2),
    (2, 4, 0.025),
    (4, 5, 0.2)
]

Ns = [12, 8, 6, 5, 4, 3, 2]
Ts = np.concatenate([np.arange(*r) for r in ranges])

calculate(Ns, Ts, tol=0.005)
results(Ns, Ts)