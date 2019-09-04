import numpy as np
import matplotlib.pyplot as plt
import os, sys
import matplotlib.ticker

dirname = "lnorm_0.5_1_10_0.999"
vals = np.load(os.path.join("data", "bandits", dirname, "tce.npy"))
best_arm = np.argmin(vals)

def percent_best_action(data, best_arm):
    f = lambda z: np.sum((z == best_arm).astype(int))/len(z)
    return np.apply_along_axis(f, 0, data)

if __name__ == "__main__":
    # SA data
    arrs = []
    data_path = os.path.join("data", "bandits", dirname, "sa")
    for file in os.listdir(data_path):
        arrs.append(np.load(os.path.join(data_path, file), allow_pickle=True))
    sa_data = np.vstack(tuple(arrs))

    # EV data
    arrs = []
    data_path = os.path.join("data", "bandits", dirname, "ev")
    for file in os.listdir(data_path):
        arrs.append(np.load(os.path.join(data_path, file), allow_pickle=True))
    ev_data = np.vstack(tuple(arrs))

    x_vals = range(0, 10001, 100)

    sa_pba = percent_best_action(sa_data, best_arm)
    ev_pba = percent_best_action(ev_data, best_arm)

    fig, ax = plt.subplots()
    plt.plot(x_vals, sa_pba, 'k-')
    plt.plot(x_vals, ev_pba, 'k--')
    plt.xlabel("Sample Size")
    plt.ylabel("Percent Best Action")
    #plt.title(dirname)
    plt.legend(labels = ["Sample Average", "Extreme Value"])
    #plt.yscale("log")
    #ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #plt.minorticks_off()
    #plt.yticks([0.25, 0.5, 0.75, 1])
    plt.show()
    #plt.savefig(os.path.join("plots", "bandits", dirname+".png"), bbox_inches="tight")
    #plt.clf()