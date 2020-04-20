"""Calculate and plot auto-correlations"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ising import datagen, loadingbar, plotter, simulator, thermo


def ensemble_autoc(ensemble, maxtau=400, func=thermo.magnetisation):
    """
    Calculates autocorrelations for every system of an ensemble

    RETURNS: (taunum, sysnum)-array
        where taunum goes from 0 to maxtau
    """

    ens_arr = ensemble.asarray()
    ens_mags = func(ens_arr)

    assert len(ens_mags.shape) == 2

    ens_autocs = thermo.autocorrelation(ens_mags, maxtau=maxtau, axis=0)

    # I made the unfortunate choice when writing datagen.py of indexing the
    # out put of Ensemble.asarray() as (iternum, sysnum, Nx, Ny), when
    # putting the sysnum first makes more sense...
    
    # Oh well, just be careful when indexing the output array!

    return ens_autocs

