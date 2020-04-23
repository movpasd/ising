"""Calculate and plot auto-correlations"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ising import datagen, loadingbar, plotter, simulator, thermo


datapath = Path(__file__).parents[1] / "data/autoc"
resultspath = Path(__file__).parents[1] / "results/autoc"


def generate():

    pass