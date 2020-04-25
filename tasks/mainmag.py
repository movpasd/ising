"""Calculation of mean magnetisation variation with temperature"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from warnings import warn

from ising import datagen, loadingbar, plotter, simulator, thermo


# In this task I re-use the data from tasks.autoc, except I only use
# N = 30 and include values for b far from critical. 


autocdatapath = Path(__file__).parents[1] / "data/autoc-night"
datapath = Path(__file__).parents[1] / "data/mainmag"


def analyse():

    dataset = datagen.DataSet(autocdatapath)
    dataset.load()

    # Only consider the 30x30 grids
    ensembles = filter(lambda ens: ens.N == 30,dataset.ensembles)
    assert len(ensembles) > 0

    # It doesn't matter if you average over time or the ensemble first,
    # hence the lack of axis specified in np.mean
    avg_mags = [np.mean(ens.asarray()) for ens in ensembles]
    avg_sqmags = [np.mean(ens.asarray()**2) for ens in ensembles]

    np.save(datapath / "avg_mags.npy", np.array(avg_mags))
    np.save(datapath / "avg_sqmags.npy", np.array(avg_sqmags))

    avg_mag_flucts = [np.std(ens.asarray()) for ens in ensembles]
    avg_sqmag_flucts = [np.std(ens.asarray()**2) for ens in ensembles]

    np.save(datapath / "avg_mag_flucts.npy", np.array(avg_mag_flucts))
    np.save(datapath / "avg_sqmag_flucts.npy", np.array(avg_sqmag_flucts))

    bs = [ens.b for ens in ensembles]
    np.save(datapath / "bs.npy", np.array(bs))


def results():

    # Load up all the data

    bs = np.load(datapath / "bs.npy")

    avg_mags = np.load(datapath / "avg_mags.npy")
    avg_sqmags = np.load(datapath / "avg_sqmags.npy")

    avg_mag_flucts = np.load(datapath / "avg_mag_flucts.npy")
    avg_sqmag_flucts = np.load(datapath / "avg_sqmag_flucts.npy")


    # Plot it

    plt.plot(bs, avg_mags, "x")
    plt.show()


