"""Investigating hysteresis effects"""


import numpy as np
import matplotlib.pyplot as plt

from ising import datagen, thermo, plotter


hparams = {
    "maxh": 1,
    "period": 200
}

relaxtime = 200


def get_hysteresis_loop(b, N, iternum, sysnum, hparams):
    """
    Compute and return magnetisation v time values

    RETURNS: mags -- (iternum, sysnum)-array 
             hs -- (iternum,)-array
    """

    a, p = hparams["maxh"], hparams["period"]
    hs = a * np.sin(2 * np.pi * np.arange(0, p) / p)

    # In this case, domain walls probably actually lead to
    # interesting behaviour, so set p = 0.5

    # Initially the applied field is zero, but we'll turn it on
    # once the relaxation time has elapsed
    ensemble = datagen.Ensemble(N, sysnum, p=0.5, b=b, h=np.zeros(1))

    ensemble.simulate(relaxtime + 1)
    ensemble.trim_init(relaxtime)

    ensemble.hs = hs
    ensemble.simulate(iternum - 1, reset=False, verbose=True)
    arr = ensemble.asarray()

    mags = thermo.magnetisation(arr)

    return np.resize(hs, iternum), mags


def generate(bs, N, iternum, sysnum):

    for b in bs:




bs = np.arange(0, 1.1, 0.1)
N = 25
iternum = 2 * hparams["period"]
sysnum = 10

hs, mags = get_hysteresis_loop(b, N, iternum, sysnum, hparams)

avg_mags = np.mean(mags, axis=1)

plt.savefig("test.pdf")
plt.plot(hs, avg_mags)
plt.show()
