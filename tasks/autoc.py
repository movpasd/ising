"""Calculate and plot auto-correlations"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path
from warnings import warn

from ising import datagen, loadingbar, plotter, simulator, thermo


datapath = Path(__file__).parents[1] / "data/autoc"
resultspath = Path(__file__).parents[1] / "results/autoc"


# GENERATION PARAMETERS
# ----------------------------------------------------------------------------

# The systems are indexed by id_N and id_b (also called i and j),
# we can convert between (id_N, id_b) <--> k, the index
# which labels ensembles in the datagen.DataSet.

Ns = []
bs = []
kcount = len(Ns) * len(bs)

# These arrays help quickly switch from (id_N, id_b) indexing to k-indexing
Nb_to_ks = [[i * len(bs) + j for j in range(len(bs))] for i in range(len(Ns))]
k_to_Nbs = [(Ns[k // len(bs)], bs[k % len(bs)])
            for k in range(kcount)]

sysnum = 30
b_crit = 0  # the relaxation time is short enough that it doesn't really matter


# ANALYSIS PARAMETERS
# ----------------------------------------------------------------------------

# Largest autocorrelation lag to calculate up to
# The bigger this is, the less time over which M'(t)M'(t+tau) is averaged
maxtau = 300


def generate(wipe, iternum, relaxtime=None):
    """Generate and save all required data"""

    dataset = datagen.DataSet(datapath)

    assert type(wipe) is bool

    if wipe == True:

        if relaxtime is None:
            raise ValueError("need a relaxtime")

        print("Wiping dataset")
        dataset.wipe()

        print("Creating new dataset")
        for k, Nb in enumerate(k_to_Nbs):

            N, b = Nb

            # Aligned initial conditions for cold, randomised for hot
            # This provides the fastest convergence to equilibrium
            p = 1 if b >= b_crit else 0.5

            print(f"k: {k} >> N={N}, b={b:.2f}")
            ens = datagen.Ensemble(N, sysnum=sysnum, p=p,
                                   b=b, h=0, randflip=True)
            ens.simulate(iternum + relaxtime, verbose=True)

            # Remove the relaxation time
            ens.trim_init(relaxtime)

            dataset.add_ensemble(ens, save=True)

    else:

        print("Loading dataset")
        dataset.load()

        print("Updating dataset")
        for k, ens in enumerate(dataset.ensembles):

            b = ens.b
            N = ens.grid_shape[0]

            print(f"k: {k} >> N={N}, b={b:.2f}")
            ens.simulate(iternum, reset=False, verbose=True)
            dataset.save(ens_index=k)


def display_mosaic(k):
    """For testing"""

    dataset = datagen.DataSet(datapath)
    dataset.load()

    ens = dataset.ensembles[k]

    print(f"Nb: {k_to_Nbs[k]}")

    ak = {"interval": 1}
    fig, _, _ = plotter.animate_mosaic(ens, timestamp=True, show=True,
                                       anim_kwargs=ak)
    plt.close(fig)


def analyse():
    """Analyse the data"""

    print("Loading data")
    dataset = datagen.DataSet(datapath)
    dataset.load()

    # e-folding times
    tau_es = []

    print("Calculating")

    bar = loadingbar.LoadingBar(len(dataset.ensembles))

    # For each run (varying N and b)...
    for k, ens in enumerate(dataset.ensembles):

        bar.print_next()

        ens_arr = ens.asarray()

        # 1. Calculate the magnetisation as a function of time
        #    and save to file

        mags = thermo.magnetisation(ens_arr)
        np.save(datapath / f"mags-{k}.npy", mags)

        # 2. Calculate the autocorrelation as a function of tau (time lag)
        #    and save to file

        # Calculate autocorrelation of each member of the ensembles and
        # then take average over ensemble
        autocs = thermo.autocorrelation(mags, maxtau, axis=0)
        autocs = np.mean(autocs, axis=1)
        np.save(datapath / f"autocs-{k}.npy", autocs)

        # 3. Calculate the e-folding time

        # First, simply try finding the time step at which the
        # autoc drops below 1/e

        ids = np.argwhere(autocs < 1 / np.e)
        if len(ids) > 0:

            tau_es.append(ids[0])

        else:

            tau_es.append(None)

            # If that fails, then take the logarithm of the values and
            # calculate a linear regression

            # logs = np.log(autocs)
            # reg = linregress(range(autocs.shape[-1]), logs)

            # print(f"oops N{ens.grid_shape}, b{ens.b}")
            # print(f"slope {reg[0]:.4f}, r {reg[2]:.4f}")

            # slope = reg[0]

            # tau_es.append(-1 / slope)

    # Save the e-folding lengths
    np.save(datapath / "tau_es.npy", np.array(tau_es))

    print("Done!")


def results():
    """Generate human-readable content from analysed results"""

    # # 1. tau_e graph
    # # ------------------------------------------------------------

    tau_es = np.load(datapath / "tau_es.npy", allow_pickle=True)

    # I want to plot tau_e against b for various Ns. Annoyingly this
    # means I have to do some index juggling.

    # This is all because of the way I set up datagen.DataSet... oh well.

    for i, N in enumerate(Ns):

        # values to plot against b for the specific N
        vals = []

        for j, b in enumerate(bs):

            k = Nb_to_ks[i][j]
            vals.append(tau_es[k])

        plt.plot(bs, vals, "-")

    plt.title("Auto-correlation e-folding timelag for "
              "variable temperatures, grid sizes")

    plt.xlabel("$\\beta$")
    plt.ylabel("$\\tau_e$")

    plt.legend([f"N={N}" for N in Ns])

    plt.savefig(resultspath / "tau_es.pdf")
    # plt.show()
    plt.close()

    # 2. magnetisation graphs
    # ------------------------------------------------------------

    mags_list = [np.load(datapath / f"mags-{k}.npy") for k in range(kcount)]

    for i, N in enumerate(Ns):

        plt.title(f"Square magnetisations N={N}")
        plt.xlabel("t")
        plt.ylabel("M")

        for j, b in enumerate(bs):

            c = np.max([0, np.min([1, 10 * (b - 0.4)])])

            k = Nb_to_ks[i][j]
            vals = np.mean(mags_list[k]**2, axis=1)
            plt.plot(vals, color=(1 - c, 0, c))

        plt.savefig(resultspath / f"mags-{N}.pdf")
        # plt.show()
        plt.close()

    # 3. autoc graphs
    # ------------------------------------------------------------

    autocs_list = [
        np.load(datapath / f"autocs-{k}.npy") for k in range(kcount)]

    for i, N in enumerate(Ns):

        plt.title(f"Auto-correlation N={N}")
        plt.xlabel("$ \\tau $")
        plt.ylabel("$ A(\\tau) $")

        for j, b in enumerate(bs):

            c = np.max([0, np.min([1, 10 * (b - 0.4)])])

            k = Nb_to_ks[i][j]
            vals = autocs_list[k]
            plt.plot(vals, color=(1 - c, 0, c))

        plt.legend(bs)

        plt.savefig(resultspath / f"autocs-{N}.pdf")
        # plt.show()
        plt.close()


def mosaics():

    dataset = datagen.DataSet(datapath)
    dataset.load()

    for k, ens in enumerate(dataset.ensembles):

        N, b = ens.grid_shape[0], ens.b

        if (N < 35) and 0.435 < b < 0.445:

            print(f"k{k} | N{ens.grid_shape[0]} b{ens.b}")
            fig, _, _ = plotter.animate_mosaic(
                ens, timestamp=True,
                saveas=resultspath / f"mosaic-{k}.mp4"
            )


def randflip():

    # I had to write this function because the data I generated didn't
    # properly randomise the initial conditions (half systems fully aligned
    # spin up, half systems fully aligned spin down)

    dataset = datagen.DataSet(datapath)
    dataset.load()

    for k, ens in enumerate(dataset.ensembles):

        print(k)
        ens.do_randflip()
        dataset.save(k)
