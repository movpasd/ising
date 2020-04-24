"""Calculate and plot auto-correlations"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from warnings import warn

from ising import datagen, loadingbar, plotter, simulator, thermo


datapath = Path(__file__).parents[1] / "data/autoc"
resultspath = Path(__file__).parents[1] / "results/autoc"


# GENERATION PARAMETERS
# ----------------------------------------------------------------------------

# The systems are indexed by id_N and id_b for the indices of N and b
# and this allows conversion between (id_N, id_b) <--> k, the index
# which labels ensembles in the datagen.DataSet.

Ns = [5, 10, 30, 50, 100]
bs = [0, 0.2, 0.4, 0.42, 0.43, 0.44, 0.45, 0.46, 0.5, 0.8, 1.0]

# These arrays help quickly switch from (id_N, id_b) indexing to k-indexing
Nb_to_ks = [[i * len(bs) + j for j in range(len(bs))] for i in range(len(Ns))]
k_to_Nbs = [(Ns[k // len(bs)], bs[k % len(bs)])
            for k in range(len(Ns) * len(bs))]

sysnum = 1
relaxtime = 200
iternum = 500
b_crit = 0.44


# ANALYSIS PARAMETERS
# ----------------------------------------------------------------------------

# Largest autocorrelation lag to calculate up to
# The bigger this is, the less time over which M'(t)M'(t+tau) is averaged
maxtau = 300


def generate(wipe):
    """Generate and save all required data"""

    dataset = datagen.DataSet(datapath)

    assert type(wipe) is bool

    if wipe == True:

        print("Wiping dataset")
        dataset.wipe()

        print("Creating new dataset")
        for k, Nb in enumerate(k_to_Nbs):

            N, b = Nb

            # Aligned initial conditions for cold, randomised for hot
            # This provides the fastest convergence to equilibrium
            p = 1 if b >= b_crit else 0.5

            print(f"k: {k} >> N={N}, b={b:.2f}")
            ens = datagen.Ensemble(N, sysnum=sysnum, p=p, b=b, h=0)
            ens.simulate(iternum + relaxtime, verbose=True)

            # Remove the relaxation time
            ens.trim_init(relaxtime)

            dataset.add_ensemble(ens, save=True)

    else:

        print("Loading dataset")
        dataset.load()

        print("Updating dataset")
        for k, ens in enumerate(dataset.ensembles):

            print(f"k: {k} >> N={ens.grid_shape[0]}, b={ens.b:.2f}")
            ens.simulate(iternum, reset=False, verbose=True)

            dataset.save(ens_index=k)


def display_mosaic(k):
    """For testing"""

    dataset = datagen.DataSet(datapath)
    dataset.load()

    ens = dataset.ensembles[k]

    print(f"Nb: {k_to_Nbs[k]}")

    ak = {"interval": 1}
    fig, _, _ = plotter.animate_mosaic(ens, timestamp=True, show=True,
                                       anim_kwargs=ak)
    plt.close(fig)


def analyse():
    """Analyse the data"""

    print("Loading data")
    dataset = datagen.DataSet(datapath)
    dataset.load()

    # e-folding times
    tau_es = []

    print("Calculating")

    bar = loadingbar.LoadingBar(len(dataset.ensembles))

    # For each run (varying N and b)...
    for k, ens in enumerate(dataset.ensembles):

        bar.print_next()

        ens_arr = ens.asarray()

        # 1. Calculate the magnetisation as a function of time
        #    and save to file

        # the magic 0 picks out the sole system in the ensemble
        # (that axis corresponds to sysnum)
        mags = thermo.magnetisation(ens_arr)[:, 0]
        np.save(datapath / f"mags-{k}.npy", mags)

        # 2. Calculate the autocorrelation as a function of tau (time lag)
        #    and save to file

        autocs = thermo.autocorrelation(mags, maxtau)
        np.save(datapath / f"autocs-{k}.npy", autocs)

        # 3. Calculate the e-folding time
        ids = np.argwhere(autocs < 1 / np.e)
        if len(ids) == 0:
            warn("No e-folding length"
                 f"k={k} N={ens.grid_shape[0]} b={ens.b:.2f}")
            tau_es.append(None)
        else:
            tau_es.append(ids[0])

    # Save the e-folding lengths
    np.save(datapath / "tau_es.npy", np.array(tau_es))


    print("Done!")


