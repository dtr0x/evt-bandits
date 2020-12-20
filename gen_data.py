import numpy as np
from frechet import Frechet
from burr import Burr
from half_t import HalfT
import multiprocessing as mp
from cvar_iter import cvar_iter
from cvar import cvar_sa, cvar_ad

# generate samples from distributions
def gen_samples(dists, s, n):
    data = []
    for d in dists:
        data.append(d.rand((s, n)))
    return np.array(data)

# generate CVaR estimates from sample data
def get_cvars(dist_data, alph, sampsizes):
    n_cpus = mp.cpu_count()
    pool = mp.Pool(n_cpus)
    cvars_evt = [] # extreme value theory estimates
    cvars_sa = [] # sample average estimates
    tp = [] # threshold percentiles chosen
    n_rejected = [] # rejection rate from mle estimation

    for d in dist_data:
        # evt cvar
        result1 = [pool.apply_async(cvar_iter, args=(x, alph, sampsizes, cvar_ad)) for x in d]
        # sa cvar
        result2 = [pool.apply_async(cvar_iter, args=(x, alph, sampsizes, cvar_sa)) for x in d]

        # evt cvar
        c_est1 = np.array([r.get() for r in result1])
        cvars_evt.append(c_est1[:,:,0])
        tp.append(c_est1[:,:,1])
        n_rejected.append(c_est1[:,:,2])

        # sa cvar
        c_est2 = np.array([r.get() for r in result2])
        cvars_sa.append(c_est2)

    return np.array([cvars_evt, cvars_sa, tp, n_rejected])

if __name__ == '__main__':
    # Burr distributions
    c = [0.75, 1, 2, 3, 4]
    d = [2, 1.5, 1, 0.75, 0.5]
    params = [(i,j) for i,j in zip(c,d)]
    burr_dists = [Burr(*p) for p in params]

    # Frechet distributions
    gamma = [1.25, 1.5, 2, 2.5, 3]
    frec_dists = [Frechet(p) for p in gamma]

    # half-t distributions
    df = [1.25, 1.5, 2, 2.5, 3]
    t_dists = [HalfT(p) for p in df]

    # sample sizes to test CVaR estimation
    sampsizes = np.linspace(2000, 20000, 10).astype(int)
    n_max = sampsizes[-1]

    # number of independent runs
    s = 1000

    # generate data
    np.random.seed(7)
    burr_data = gen_samples(burr_dists, s, n_max)
    frec_data = gen_samples(frec_dists, s, n_max)
    t_data = gen_samples(t_dists, s, n_max)

    # CVaR level
    alph = 0.998

    burr_result = get_cvars(burr_data, alph, sampsizes)
    frec_result = get_cvars(frec_data, alph, sampsizes)
    t_result = get_cvars(t_data, alph, sampsizes)

    np.save('data/burr_samples.npy', burr_data)
    np.save('data/frec_samples.npy', frec_data)
    np.save('data/t_samples.npy', t_data)
    np.save('data/burr_cvars.npy', burr_result)
    np.save('data/frec_cvars.npy', frec_result)
    np.save('data/t_cvars.npy', t_result)
