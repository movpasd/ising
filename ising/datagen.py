"""Module for generation and analysis of large data sets"""

import numpy as np
import json
from pathlib import Path
from warnings import warn

from . import simulator, thermo, loadingbar


class DataSet:
    """Collection of simulated ensembles, allows saving/loading"""

    def __init__(self, path, ensembles=None, load=False):
        """path: str"""

        if ensembles is None:
            self.ensembles = []
        else:
            self.ensembles = list(ensembles)

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        if load:
            self.load()

    def save(self, ens_index=None):

        with open(self.path / "metadata.json", "w") as outfile:
            json.dump(self.get_metadata(), outfile, indent=4)

        if ens_index is None:

            for k in range(len(self.ensembles)):
                np.save(self.path / ("ens-" + str(k) + ".npy"),
                        self.ensembles[k].asarray())

        else:

            np.save(self.path / ("ens-" + str(ens_index) + ".npy"),
                    self.ensembles[ens_index].asarray())

    def get_metadata(self):

        return [
            {
                "grid_shape": ens.grid_shape,
                "sysnum": ens.sysnum,
                "identical": ens.identical,
                "p": ens.p,
                "b": ens.b,
                "h": ens.h,
                "iternum": ens.iternum
            } for ens in self.ensembles
        ]

    def load(self):

        self.ensembles = []

        with open(self.path / "metadata.json", "r") as mdfile:
            metadata = json.load(mdfile)

        ensemble_count = len(metadata)

        for k in range(ensemble_count):

            md = metadata[k]
            iternum = md.pop("iternum")
            ens = Ensemble(**md)

            ens_data = np.load(self.path / f"ens-{k}.npy")

            # Check data integrity
            if not ens_data.shape == (iternum, ens.sysnum, *ens.grid_shape):
                error_message = (
                    f"Integrity check failure\n"
                    f"data path: {self.path}\n"
                    f"ens. index: {k}\n\n"
                    f"Ensemble data had shape {ens_data.shape} "
                    f"when {(iternum, ens.sysnum, *ens.grid_shape)} was expected!"
                )
                raise RuntimeError(error_message)

            ens.init_state = ens_data[0]
            ens.iterations = list(ens_data)
            ens.iternum = iternum

            self.add_ensemble(ens)

    def wipe(self):
        """Wipes all the data from the folder"""

        with open(self.path / "metadata.json", "r") as mdfile:

            metadata = json.load(mdfile)

            for k in range(len(metadata)):

                try:

                    (self.path / f"ens-{k}.npy").unlink()

                except FileNotFoundError:

                    warn(str(self.path / f"ens-{k}.npy") + " not found :(")

        with open(self.path / "metadata.json", "w") as mdfile:

            mdfile.write("[]")

        self.load()

    def add_ensemble(self, ens, save=False):

        self.ensembles.append(ens)

        if save:

            k = len(self.ensembles) - 1

            with open(self.path / "metadata.json", "w") as outfile:
                json.dump(self.get_metadata(), outfile, indent=4)

            np.save(self.path / ("ens-" + str(k) + ".npy"), ens.asarray())


class Ensemble:
    """Ensemble of states with similar init conditions and parameters"""

    def __init__(self, grid_shape, sysnum, p, b, h, identical=False, initialise=True):
        """
        grid_shape: (int, int)
        sysnum: int -- number of systems in the ensemble
        p, b, h: floats -- proportion of initial spins, 1/temp, applied field
        identical: bool -- whether the systems should be initialised identically
        initialise: bool -- set to False to manually initialise iternum and such
        """

        if type(grid_shape) is int:
            grid_shape = (grid_shape, grid_shape)

        self.grid_shape = grid_shape
        self.sysnum = sysnum
        self.identical = identical
        self.p = p
        self.b = b
        self.h = h

        if initialise:
            self.reset(regen_init=True)

    def simulate(self, iternum, reset=True, regen_init=False, verbose=False):

        if reset:
            self.reset(regen_init=regen_init)
            iternum -= 1  # This makes sure self.iternum matches up with
                          # the parameter passed onto this function

        if verbose:
            bar = loadingbar.LoadingBar(iternum)

        for k in range(iternum):
            self.next()
            if verbose:
                bar.print_next()

    def next(self):

        ens_state = self.iterations[-1]
        self.iterations.append(
            simulator.iterate_ensemble(ens_state, b=self.b, h=self.h))
        self.iternum += 1

    def reset(self, regen_init=False):
        """
        Erase simulation data

        regen_init: bool -- regenerate initial state? 
            defaults to False
        """

        if regen_init:
            self.init_state = simulator.new_ensemble(
                self.grid_shape, self.sysnum, self.p, self.identical)

        self.iterations = [self.init_state]
        self.iternum = 1
        self.final_state = self.init_state

    def ensemble_avg(self, func):
        """
        Calculates avg of property across ensemble over time

        func: callable -- takes a microstate
        """

        return [
            np.mean([func(microstate) for microstate in ens_state])
            for ens_state in self.iterations
        ]

    def ensemble_stdev(self, func, std_kwargs=None):
        """
        Calculates stdev of property across ensemble over time

        func: callable -- must take a microstate
        """

        if std_kwargs is None:
            std_kwargs = {}

        std_kwargs.setdefault("ddof", 1)

        return [
            np.std([func(microstate) for microstate in ens_state],
                   **std_kwargs)
            for ens_state in self.iterations
        ]

    def asarray(self):
        """
        Get ensemble over time as array

        Indexing: (iternum, sysnum, Nx, Ny)
        """

        return np.array(self.iterations)

    def trim_init(self, trimcount):
        """
        Remove the first trimcount elements of data

        Useful to get rid of relaxation time
        """

        self.iternum -= trimcount
        self.iterations = self.iterations[trimcount:]
        self.init_state = self.iterations[0]
