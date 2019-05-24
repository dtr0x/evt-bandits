import multiprocessing as mp
import os
import tce

#ncpus = int(os.environ.get("SLURM_CPUS_PER_TASK",default=1))

def tce_sim(x, alph, tce_func, start=99, step=100):
	tce_dat = []
	for i in range(start, x.shape[0], step):
		tce_dat.append(tce_func(x[:i], alph))
	return tce_dat


ncpus = 1000
pool = mp.Pool(processes=ncpus)
data = genpareto.rvs(0.5, 0, 1, 10*10000).reshape(1000, 10000)
results = [pool.apply_async(tce_sim, args=(x, 0.99, tce_sa)) for x in data]
vals = np.asarray([p.get() for p in results])
print(vals)