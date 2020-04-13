"""Determining thermodynamic properties from Ising model simulation"""

import numpy as np


def magnetisation(a):
    """
    Calculate mean magnetisation

    a: int (Nx, Ny)- OR (iternum, Nx, Ny)-array
    RETURNS: float or float (iternum,)-array
    """

    return np.mean(a, axis=(-1, -2))


def autocorrelation(samples, maxtau=None):
    """
    Calculate the auto-correlation of a sampled function of time

    samples: float (iternum,)-array
    maxtau: int <= iternum
    """

    if maxtau is None:
        maxtau = samples // 2
    else:
        assert type(maxtau) is int and maxtau <= iternum

    taus = np.arange(maxtau)
