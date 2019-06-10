from tce import *
import multiprocessing as mp
import os, sys

def tce_sim(x, alph, tce_func, start=99, step=100, **kwargs):
    tce_dat = []
    for i in range(start, x.shape[0], step):
        tce_dat.append(tce_func(x[:i], alph, **kwargs))
    return tce_dat

def main(seed = 7):
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

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", default='0')
    ncpus = int(os.environ.get("SLURM_CPUS_PER_TASK", default=1))

    np.random.seed(seed)
    if dist == "gpd":
        data = genpareto.rvs(float(p1), 0, float(p2), ncpus*10000).reshape(ncpus, 10000)
    elif dist == "lnorm":
        data = lognorm.rvs(float(p2), float(p1), 1, ncpus*10000).reshape(ncpus, 10000)
    elif dist == "weibull":
        data = weibull_min.rvs(float(p1), 0, float(p2), ncpus*10000).reshape(ncpus, 10000)

    pool = mp.Pool(processes=ncpus)

    # SA data
    results = [pool.apply_async(tce_sim, args=(x, float(alph), tce_sa)) for x in data]
    vals = np.asarray([p.get() for p in results])
    np.save(os.path.join("data", dirname, "sa", task_id), vals)

    # EVT data
    if not tce_args:
        results = [pool.apply_async(tce_sim, args=(x, float(alph), tce_func)) for x in data]
    else:
         results = [pool.apply_async(tce_sim, args=(x, float(alph), tce_func), kwds=tce_args) for x in data]
    vals = np.asarray([p.get() for p in results])
    np.save(os.path.join("data", dirname, "ev", task_id), vals)

if __name__ == "__main__":
    main()