"""Simulator for the 2D Ising model using a Metropolis method"""

import numpy as np
import numpy.random as npr
from pathlib import Path

from . import loadingbar


nwxs = np.newaxis


def new_grid(grid_shape, p=0.5):
    """
    Create new random initial state

    grid_shape: (int, int)
    p: float -- percentage of spin up
    """

    assert 0 <= p <= 1

    if type(grid_shape) is int:
        grid_shape = (grid_shape, grid_shape)

    return 2 * (npr.rand(*grid_shape) > p) - 1


def new_ensemble(grid_shape, sysnum, p=0.5, identical=False):
    """
    Create a new statistical ensemble of initial states

    grid_shape: (int, int)
    sysnum: int -- number of independent systems in the ensemble
    p: float -- percentage of spin up

    RETURNS: (sysnum, Nx, Ny)-array
    """

    assert 0 <= p <= 1

    if type(grid_shape) is int:
        grid_shape = (grid_shape, grid_shape)

    if identical:
        a = 2 * (npr.rand(*grid_shape) > p) - 1
        return np.repeat(a[np.newaxis, ...], sysnum, axis=0)
    else:
        return 2 * (npr.rand(sysnum, *grid_shape) > p) - 1


def _rand_flip_spin(spins, bs, hs, Nx, Ny):
    """
    Helper function for iterate()

    [!] modifies spins array in-place
    """

    # Pick out one random spin
    i = int(npr.rand() * Nx)
    j = int(npr.rand() * Ny)
    s = spins[i, j]
    # Find the spin's neighbours
    neighbours = (
        spins[(i + 1) % Nx, j],
        spins[i - 1, j],
        spins[i, (j + 1) % Ny],
        spins[i, j - 1]
    )

    # Calculate Delta E
    dE = (np.sum(neighbours) + hs[i, j]) * s

    # Choose whether to flip it or not!
    if np.exp(-2 * bs[i, j] * dE) > npr.rand():
        spins[i, j] *= -1


def iterate(spins, b=1, h=0):
    """
    Step through one iteration on the spins.

    spins: int (Nx, Ny)-array
    b: float -- kinetic/temperature parameter, b = J/kT
    h: float -- field parameter, h = muH/J

    RETURNS: int (Nx, Ny)-array
    """

    # I'm going to be modifying spins in-place, so first I make a copy
    # of the spins grid.
    spins = np.copy(spins)

    Nx, Ny = spins.shape

    # I wanted this to work for spatially varying h, so the general
    # case is to just cast it to an array of the same size as spins.

    bs = np.broadcast_to(b, spins.shape)
    hs = np.broadcast_to(h, spins.shape)

    for k in range(spins.size):
        _rand_flip_spin(spins, bs, hs, Nx, Ny)

    return spins


def iterate_ensemble(ensemble, b=1, h=0):
    """
    Step through one iteration on an ensemble.
    """

    ensemble = np.copy(ensemble)

    sysnum = ensemble.shape[0]

    for k in range(sysnum):
        ensemble[k] = iterate(ensemble[k], b, h)

    return ensemble


def _cast(a, output_shape):
    """Helper function for run()"""

    iternum, Nx, Ny = output_shape
    tp = type(a)

    if tp is np.ndarray:

        # Numpy broadcasting works from right to left in indices.

        if a.shape == (iternum,):
            return np.broadcast_to(a[:, nwxs, nwxs], output_shape)
        elif a.shape == (Nx, Ny) or a.shape == (1,):
            return np.broadcast_to(a, output_shape)

    elif tp == int or tp == float:

        return np.broadcast_to(a, output_shape)

    else:

        raise ValueError(f"Can't cast this type {tp} to {output_shape}:\n{a}")


def run(init_spins, iternum, b=1, h=0, verbose=False, filename=None):
    """
    Run a simulation and return it as an array indexed over time and space

    init_spins: int (Nx, Ny)-array
    iternum: int -- number of iterations
    filename: str OR Path OR None -- if None, do not save to file
    b, h: float (Nx, Ny)- OR (iternum,)- OR (iternum, Nx, Ny)- array OR float
        -- accepts space/time varying arrays

    RETURNS simulation: int (iternum, Nx, Ny)-array
    """

    if verbose:
        bar = loadingbar.LoadingBar(iternum)

    # Useful throughout the function
    Nx, Ny = init_spins.shape
    output_shape = (iternum, Nx, Ny)

    # Convert b and h to appropriately-sized arrays:
    bs = _cast(b, output_shape)
    hs = _cast(h, output_shape)

    # Actually calculate simulation

    if verbose:
        print(f"Running simulation, {iternum} iterations")
        bar.print_init()

    # Initial state
    simulation = np.empty(output_shape)
    spins = init_spins
    simulation[0] = spins
    if verbose:
        bar.print_next()

    # Step through simulation
    for k in range(1, iternum):

        if verbose:
            bar.print_next()

        spins = iterate(spins, bs[k], hs[k])
        simulation[k] = spins

    # If a filename is given, save it to that file
    if filename is not None:

        if filename.endswith(".npy"):
            filename = filename[:-4]

        np.save(filename, simulation)

    return simulation
