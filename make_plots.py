import numpy as np
import matplotlib.pyplot as plt
import os, sys
from tce import tce_gpd, tce_lnorm, tce_weibull

def rmse(data, true_val):
    f = lambda z: np.sqrt(np.mean((z - true_val)**2))
    return np.apply_along_axis(f, 0, data)

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
    data_path = os.path.join("data", dirname, "sa")
    for file in os.listdir(data_path):
        arrs.append(np.load(os.path.join(data_path, file), allow_pickle=True))
    sa_data = np.vstack(tuple(arrs))

    # EVT data
    arrs = []
    data_path = os.path.join("data", dirname, "ev")
    for file in os.listdir(data_path):
        arrs.append(np.load(os.path.join(data_path, file), allow_pickle=True))
    ev_data = np.vstack(tuple(arrs))

    x_vals = range(100, 5001, 50)

    # Plot RMSE
    sa_rmse = rmse(sa_data, true_val)
    ev_rmse = rmse(ev_data, true_val)
    plt.plot(x_vals, sa_rmse, 'b')
    plt.plot(x_vals, ev_rmse, 'r')
    plt.xlabel("Sample Size")
    plt.ylabel("RMSE")
    plt.legend(labels = ["Sample Average", "Extreme Value"])
    plt.savefig(os.path.join("plots", dirname+"_rmse.png"), bbox_inches="tight")
    plt.clf()

    # Plot Fraction Closer
    fc = fraction_closer(ev_data, sa_data, true_val)
    plt.plot(x_vals, fc, 'k-')
    plt.xlabel("Sample Size")
    plt.ylabel("Fraction Closer")
    plt.savefig(os.path.join("plots", dirname+"_fc.png"), bbox_inches="tight")
    plt.clf()

    # Plot Average Threshold Percentile
    arrs = []
    data_path = os.path.join("data", dirname, "thresh")
    for file in os.listdir(data_path):
        arrs.append(np.load(os.path.join(data_path, file), allow_pickle=True))
    tp_data = np.vstack(tuple(arrs))
    plt.plot(x_vals, np.mean(tp_data, axis=0), 'k-')
    plt.xlabel("Sample Size")
    plt.ylabel("Average Threshold Percentile")
    plt.savefig(os.path.join("plots", dirname+"_tp.png"), bbox_inches="tight")
    plt.clf()

    # Plot Average Rejection Rate
    arrs = []
    data_path = os.path.join("data", dirname, "n_rejected")
    for file in os.listdir(data_path):
        arrs.append(np.load(os.path.join(data_path, file), allow_pickle=True))
    n_rejected = np.vstack(tuple(arrs))
    plt.plot(x_vals, np.mean(n_rejected, axis=0)/40, 'k-')
    plt.xlabel("Sample Size")
    plt.ylabel("Average Rejection Rate")
    plt.savefig(os.path.join("plots", dirname+"_nr.png"), bbox_inches="tight")
    plt.clf()
