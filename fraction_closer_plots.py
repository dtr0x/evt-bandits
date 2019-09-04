import numpy as np
import matplotlib.pyplot as plt
import os, sys
from tce import tce_gpd, tce_lnorm, tce_weibull

def fraction_closer(ev_data, sa_data, true_val):
    ev_diff = np.abs(ev_data - true_val)
    sa_diff = np.abs(sa_data - true_val)
    ev_sa_diff = ev_diff - sa_diff
    f = lambda z: np.where(z < 0)[0].size/np.where(z !=0)[0].size
    return np.apply_along_axis(f, 0, ev_sa_diff)

if __name__ == "__main__":
    dirname = sys.argv[1]
    params = dirname.split('_')
    dist = params[0]
    p1 = float(params[1])
    p2 = float(params[2])
    alph = float(params[3])

    if dist == "gpd":
        true_val = tce_gpd(alph, p1, p2)
    elif dist == "lnorm":
        true_val = tce_lnorm(alph, p1, p2)
    elif dist == "weibull":
        true_val = tce_weibull(alph, p1, p2)

    # SA data
    arrs = []
    data_path = os.path.join("fraction_closer", dirname, "sa")
    for file in os.listdir(data_path):
        arrs.append(np.load(os.path.join(data_path, file), allow_pickle=True))
    sa_data = np.vstack(tuple(arrs))

    # EVT data
    arrs = []
    data_path = os.path.join("fraction_closer", dirname, "ev")
    for file in os.listdir(data_path):
        arrs.append(np.load(os.path.join(data_path, file), allow_pickle=True))
    ev_data = np.vstack(tuple(arrs))

    fc = fraction_closer(ev_data, sa_data, true_val)

    print(fc)

    x_vals = range(249, 2001, 250)

    plt.plot(x_vals, fc)
    plt.xlabel("Sample Size")
    plt.ylabel("Fraction of times EVT closer for TCE("+params[3]+")")
    plt.title(dirname)
    plt.savefig(os.path.join("fraction_closer/plots", dirname+".png"), bbox_inches="tight")
    plt.clf()