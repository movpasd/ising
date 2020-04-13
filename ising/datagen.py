"""Module for generation and analysis of large data sets"""

import numpy as np
import json
from . import simulator, thermo


class DataSet:
    """Interfaces collections of simulated ensembles"""

    def __init__(self):

        pass

    def save(self, path):

        pass

    def load(self, path):

        pass

    def add_ensemble(self):
        # should take either an Ensemble object or a set of ensemble kwargs?
        pass


class Ensemble:
    """Ensemble of states with similar init conditions and parameters"""

    def __init__(self, grid_shape, sysnum, p, b, h, identical=False):

        if type(grid_shape) is int:
            grid_shape = (grid_shape, grid_shape)

        self.grid_shape = grid_shape
        self.sysnum = sysnum
        self.identical = identical

        self.reset(regen_init=True)

    def simulate(self, iternum, reset=True, regen_init=False):

        if reset:
            self.reset(regen_init=regen_init)

        state = self.iterations[-1]

        for k in range(iternum):

            self.iterations.append(
                simulator.iterate_ensemble(state), b=self.b, h=self.h)
            self.iternum += 1

    def reset(self, regen_init=False):
        """
        Erase simulation data

        regen_init: bool -- regenerate initial state? 
            defaults to False
        """

        if regen_init:

            self.init_state = simulator.new_ensemble(
                grid_shape, sysnum, p, identical)

            self.iterations = [self.init_state]

        self.iternum = 1
        self.final_state = self.init_state
