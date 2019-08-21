from tce import *
import multiprocessing as mp
import os, sys

dirname = sys.argv[1]
params = dirname.split('_')

dist = params[0]
p_min = float(params[1])
p_max = float(params[2])
n_arms = int(params[3])
alph = float(params[4])
arm_vals = np.linspace(p_min, p_max, n_arms)

trueTCEs = []

if dist == "gpd":
        gen_func = lambda shape, size: genpareto.rvs(shape, 0, 1, size)
        trueTCEs = tce_gpd(alph, arm_vals)
elif dist == "lnorm":
        gen_func = lambda shape, size: lognorm.rvs(shape, 0, 1, size)
        trueTCEs = tce_lnorm(alph, arm_vals)
elif dist == "weibull":
        gen_func = lambda shape, size: weibull_min.rvs(shape, 0, 1, size)
        trueTCEs = tce_weibull(alph, arm_vals)

np.save(os.path.join("data", "bandits", dirname, "tce"), trueTCEs)

n_plays = 10000
step = 100
eps = np.concatenate((np.linspace(1, 0.1, 1000, endpoint=False), np.linspace(0.1, 0, 9001)))

def bandit_one_run(x, tce_func):
    armSelected = []
    for i in range(0, n_plays+1, step):
        if np.random.uniform() <= eps[i]: # pick random arm
            arm = np.random.randint(n_arms)
        else:
            armTCEs =  np.apply_along_axis(lambda z: tce_func(z, alph), 1, x[:, :i])
            arm  = np.argmin(armTCEs)
        armSelected.append(arm)
    return armSelected

task_id = os.environ.get("SLURM_ARRAY_TASK_ID", default='0')
n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", default=1))

np.random.seed(int(task_id))
data = np.zeros((n_cpus, n_arms, n_plays))
for i in range(n_arms):
    arm_dat = gen_func(arm_vals[i], (n_cpus, n_plays))
    data[:,i,:] = arm_dat

pool = mp.Pool(processes=n_cpus)

# SA data
results = [pool.apply_async(bandit_one_run, args=(x, tce_sa)) for x in data]
vals = np.asarray([p.get() for p in results])
np.save(os.path.join("data", "bandits", dirname, "sa", task_id), vals)

# EV data
results = [pool.apply_async(bandit_one_run, args=(x, tce_ad)) for x in data]
vals = np.asarray([p.get() for p in results])
np.save(os.path.join("data", "bandits", dirname, "ev", task_id), vals)
