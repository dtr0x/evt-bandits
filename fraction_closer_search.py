from tce import *
import multiprocessing as mp
import os, sys

def fraction_closer(ev_data, sa_data, true_val):
    ev_diff = np.abs(ev_data - true_val)
    sa_diff = np.abs(sa_data - true_val)
    ev_sa_diff = ev_diff - sa_diff
    f = lambda z: np.where(z < 0)[0].size/np.where(z !=0)[0].size
    return np.apply_along_axis(f, 0, ev_sa_diff)

def compute_sa(x):
    sa_row_dat = []
    for i in range(249, 2001, 250):
        sa_row_dat.append(tce_sa(x[:i], alph))
    return sa_row_dat

def compute_ev(x):
    ev_row_dat = []
    for i in range(249, 2001, 250):
        ev_row_dat.append(tce_ad(x[:i], alph, tp_init, tp_num, signif, cutoff))
    return ev_row_dat

if __name__ == "__main__":
    dirname = sys.argv[1]
    params = dirname.split('_')
    dist = params[0]
    p1 = float(params[1])
    p2 = float(params[2])
    alph = float(params[3])
    tp_init = float(params[4])
    tp_num = int(params[5])
    signif = float(params[6])
    cutoff = float(params[7])

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", default='0')
    ncpus = int(os.environ.get("SLURM_CPUS_PER_TASK", default=1))
    np.random.seed(int(task_id))
    pool = mp.Pool(processes=ncpus)

    data = []

    if dist == "gpd":
        true_val = tce_gpd(alph, p1, p2), #xi, sigma
        data = genpareto.rvs(p1, 0, p2, (ncpus,2000))
    elif dist == "lnorm":
        true_val = tce_lnorm(alph, p1, p2) #sigma, mu
        data = lognorm.rvs(p1, 0, np.exp(p2), (ncpus,2000))
    elif dist == "weibull":
        true_val = tce_weibull(alph, p1, p2) #k, lambda
        data = weibull_min.rvs(p1, 0, p2, (ncpus,2000))

    results = [pool.apply_async(compute_sa, args=(x,)) for x in data]
    vals = np.asarray([p.get() for p in results])
    np.save(os.path.join("fraction_closer", dirname, "sa", task_id), vals)

    results = [pool.apply_async(compute_ev, args=(x,)) for x in data]
    vals = np.asarray([p.get() for p in results])
    np.save(os.path.join("fraction_closer", dirname, "ev", task_id), vals)