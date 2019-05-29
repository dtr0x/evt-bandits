from tce import *
import multiprocessing as mp
import os

def tce_sim(x, alph, tce_func, start=99, step=100):
	tce_dat = []
	for i in range(start, x.shape[0], step):
		tce_dat.append(tce_func(x[:i], alph))
	return tce_dat

def main():
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", default=0)
    ncpus = int(os.environ.get("SLURM_CPUS_PER_TASK", default=1))
    data = lognorm.rvs(2, 0, 1, ncpus*10000).reshape(ncpus, 10000)
    pool = mp.Pool(processes=ncpus)

    # SA data
    results = [pool.apply_async(tce_sim, args=(x, 0.99, tce_sa)) for x in data]
    vals = np.asarray([p.get() for p in results])
    np.save(os.path.join("data", "sa", task_id), vals)

    # EVT data
    results = [pool.apply_async(tce_sim, args=(x, 0.99, tce_ev)) for x in data]
    vals = np.asarray([p.get() for p in results])
    np.save(os.path.join("data", "ev", task_id), vals)

if __name__ == "__main__":
    main() 