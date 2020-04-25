"""Main script for auto-correlation tasks"""

from ising import simulator, plotter, thermo, datagen, loadingbar
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from tasks import autoc


autoc.generate(wipe=False, iternum=500)

# autoc.analyse()
# autoc.results()
# autoc.mosaics()
