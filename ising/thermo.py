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


def isflat(testfunc, ensemble, timescale, tolerance, absolute=True):
    """
    Determines when and if a test function is ~constant over time
    over an ensemble

    testfunc: callable -- must take a microstate
    ensemble: datagen.Ensemble
    tolerance: float
    absolute: bool -- whether to interpret tolerance as absolute or relative
        defaults to True, i.e.: absolute tolerance

    RETURNS: (ensemble.iternum, - timescale)-array
    """

    # First, we calculate the ensemble average of the quantity
    # as it varies over time
    ens_avgs = np.array(ensemble.ensemble_avg(testfunc))

    # Next, we look at variations in that ensemble average over time
    diffs = np.diff(ens_avgs)

    # Take a rolling average of the changes over the timescale
    cs = np.cumsum(diffs)
    smoothed_diffs = (cs[timescale:] - cs[:timescale]) / timescale

    # If the smoothed difference is less than the tolerance,
    # the test function can be said to be ~constant.
    #
    # i.e.: if
    if absolute:
        isflats = smoothed_diffs < tolerance
    else:
        cs = np.cumsum(ens_avgs)
        smoothed_avgs = (cs[timescale:] - cs[:timescale]) / timescale
        isflats = smoothed_diffs / smoothed_avgs < tolerance

    return isflats
