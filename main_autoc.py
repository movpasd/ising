"""Main script for auto-correlation tasks"""

from ising import simulator, plotter, thermo, datagen, loadingbar
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from tasks import relaxation