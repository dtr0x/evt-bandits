import numpy as np
import matplotlib.pyplot as plt
import os, sys
from tce import tce_gpd, tce_lnorm, tce_weibull
import matplotlib.ticker

def rmse(data, true_val):
    f = lambda z: np.sqrt(np.mean((z - true_val)**2))
    return np.apply_along_axis(f, 0, data)

if __name__ == "__main__":
    dirname = "gpd_0.9_1_0.999_search_0.9_100_0.1_0.9_forward_stop"#sys.argv[1]
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

    nan_idx = np.where(np.isnan(ev_data))
    ev_data[nan_idx] = sa_data[nan_idx]

    sa_rmse = rmse(sa_data, true_val)
    ev_rmse = rmse(ev_data, true_val)

    x_vals = range(100, 10001, 100)

    fig, ax = plt.subplots()
    plt.plot(x_vals, sa_rmse, 'k-')
    plt.plot(x_vals, ev_rmse, 'k--')
    plt.xlabel("Sample Size")
    plt.ylabel("RMSE")
    plt.legend(labels = ["Sample Average", "Extreme Value"])
    plt.yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.minorticks_off()
    plt.yticks([2048,4096,8192])
    plt.show()
    #plt.savefig(os.path.join("plots/new", dirname+".png"), bbox_inches="tight")
    #plt.clf()