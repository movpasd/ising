"""Module for generation and analysis of large data sets"""

import numpy as np
import json
from pathlib import Path

from . import simulator, thermo, loadingbar


class DataSet:
    """Collection of simulated ensembles, allows saving/loading"""

    def __init__(self, path, ensembles=None):
        """path: str"""

        if ensembles is None:
            self.ensembles = []
        else:
            self.ensembles = list(ensembles)

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def save(self):

        with open(self.path / "metadata.json", "w") as outfile:
            json.dump(self.get_metadata(), outfile, indent=4)

        for k in range(len(self.ensembles)):
            np.save(self.path / ("ens-" + str(k) + ".npy"),
                    self.ensembles[k].asarray())

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

        with open(self.path / "metadata.json", "r") as infile:
            metadata = json.load(infile)

        ensemble_count = len(metadata)

        for k in range(ensemble_count):

            md = metadata[k]
            iternum = md.pop("iternum")
            ens = Ensemble(**md)

            ens_data = np.load(self.path / f"ens-{k}.npy")

            # Check data integrity
            assert ens_data.shape == (iternum, ens.sysnum, *ens.grid_shape)

            ens.init_state = ens_data[0]
            ens.iterations = list(ens_data)
            ens.iternum = iternum

            self.add_ensemble(ens)

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

    def simulate(self, iternum, reset=True, regen_init=False, printbar=False):

        if printbar:
            bar = loadingbar.LoadingBar(iternum)

        if reset:
            self.reset(regen_init=regen_init)

        if printbar:
            bar.print_init()

        for k in range(iternum):
            self.next()
            if printbar:
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

        func: callable -- takes a microstate
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
