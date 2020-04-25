"""Studying heat capacity across different Ns"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ising import datagen, thermo


datapath = Path(__file__).parents[0] / "data/energy"
autocdatapath = Path(__file__).parents[0] / "data/autoc"


# Unpacking the autoc data is a bit more complicated than in main_energy.py
# because we have to sort the data by N _and_ by b.
