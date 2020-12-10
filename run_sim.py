from tce import *
import multiprocessing as mp
import os, sys

def tce_sim(x, alph, tce_func, start=99, step=50, **kwargs):
    tce_dat = []
    tp_dat = []
    n_rejected_dat = []
    for i in range(start, x.shape[0], step):
        if tce_func == tce_ad:
            tce, tp, n_rejected = tce_func(x[:i], alph, **kwargs, return_thresh=True)
            tce_dat.append(tce)
            tp_dat.append(tp)
            n_rejected_dat.append(n_rejected)
        else:
            tce_dat.append(tce_func(x[:i], alph, **kwargs))

    if tce_func == tce_ad:
        return tce_dat, tp_dat, n_rejected_dat
    else:
        return tce_dat

if __name__ == "__main__":
    dirname = sys.argv[1]
    params = dirname.split('_')
    dist, p1, p2, alph, tp_select = params[:5]
    tce_args = params[5:]
    if tp_select == "fixed":
        tce_func = tce_ev
        if len(tce_args) == 1:
            tce_args = {'tp': float(tce_args[0])}
        elif len(tce_args) == 2:
            tce_args = {'tp': float(tce_args[0]), 'cutoff': float(tce_args[1])}
    elif tp_select == "search":
        tce_func = tce_ad
        if len(tce_args) == 3:
            tce_args = {'tp_init': float(tce_args[0]), 'tp_num': int(tce_args[1]), 'signif': float(tce_args[2])}
        elif len(tce_args) == 4:
            tce_args = {'tp_init': float(tce_args[0]), 'tp_num': int(tce_args[1]), 'signif': float(tce_args[2]), 'cutoff': float(tce_args[3])}
        elif len(tce_args) == 6:
            stop_rule = globals()[tce_args[4] + '_' + tce_args[5]]
            tce_args = {'tp_init': float(tce_args[0]), 'tp_num': int(tce_args[1]), 'signif': float(tce_args[2]), 'cutoff': float(tce_args[3]), 'stop_rule': stop_rule}

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", default='0')
    ncpus = int(os.environ.get("SLURM_CPUS_PER_TASK", default=4))

    np.random.seed(int(task_id))

    n_plays = 5000

    if dist == "gpd":
        data = genpareto.rvs(float(p1), 0, float(p2), (ncpus, n_plays))
    elif dist == "lnorm":
        data = lognorm.rvs(float(p1), 0, np.exp(float(p2)), (ncpus, n_plays))
    elif dist == "weibull":
        data = weibull_min.rvs(float(p1), 0, float(p2), (ncpus, n_plays))

    pool = mp.Pool(processes=ncpus)

    # SA data
    results_sa = [pool.apply_async(tce_sim, args=(x, float(alph), tce_sa)) for x in data]
    vals = np.asarray([p.get() for p in results_sa])
    np.save(os.path.join("data", dirname, "sa", task_id), vals)

    # EVT data
    if not tce_args:
        results_ev = [pool.apply_async(tce_sim, args=(x, float(alph), tce_func)) for x in data]
    else:
        results_ev = [pool.apply_async(tce_sim, args=(x, float(alph), tce_func), kwds=tce_args) for x in data]
        tce_est = []
        thresh = []
        n_rejected = []
        for p in results_ev:
            tce_est.append(p.get()[0])
            thresh.append(p.get()[1])
            n_rejected.append(p.get()[2])
        tce_est = np.asarray(tce_est)
        thresh = np.asarray(thresh)
        n_rejected = np.asarray(n_rejected)

    np.save(os.path.join("data", dirname, "ev", task_id), tce_est)
    np.save(os.path.join("data", dirname, "thresh", task_id), thresh)
    np.save(os.path.join("data", dirname, "n_rejected", task_id), n_rejected)
