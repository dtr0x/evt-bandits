import numpy as np
from burr import Burr
from frechet import Frechet
from half_t import HalfT
from light_tailed import *

if __name__ == '__main__':
    alph = 0.998

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

    # Lognormal distributions
    lnorm_dists = [Lognormal(5, 0.25), Lognormal(4, 0.5), Lognormal(2.5, 0.75),
        Lognormal(2, 1), Lognormal(1, 1.5)]

    # Weibull distributions
    weib_dists = [Weibull(0.5, 1), Weibull(0.75, 2), Weibull(1, 3),
         Weibull(1.25, 4), Weibull(1.5, 5)]

    dists = burr_dists + frec_dists + t_dists + lnorm_dists + weib_dists

    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=1000)
    np.set_printoptions(threshold=10000)

    burr_cvars = np.load('data/burr_cvars.npy')
    frec_cvars = np.load('data/frec_cvars.npy')
    t_cvars = np.load('data/t_cvars.npy')
    lnorm_cvars = np.load('data/lnorm_cvars.npy')
    weib_cvars = np.load('data/weib_cvars.npy')

    cvars_all = [burr_cvars, frec_cvars, t_cvars, lnorm_cvars, weib_cvars]

    cvars_sa = np.asarray([c[1] for c in cvars_all])
    cvars_evt = np.asarray([c[0] for c in cvars_all])
    chosen_tp = np.asarray([c[2] for c in cvars_all])
    rejection_rate = np.asarray([c[3] for c in cvars_all])/20

    labels = [d.get_label() for d in dists]
    cvars_true = np.around([d.cvar(alph) for d in dists], 2)
    avg_sa = np.around(np.nanmean(cvars_sa, axis=2)[:,:,-1].flatten(), 2)
    avg_evt = np.around(np.nanmean(cvars_evt, axis=2)[:,:,-1].flatten(), 2)
    avg_tp = np.around(np.nanmean(chosen_tp, axis=2)[:,:,-1].flatten(), 2)
    avg_rr = np.around(np.nanmean(rejection_rate, axis=2)[:,:,0].flatten(), 2)

    for l,t,s,e,tp,rr in zip(labels,cvars_true,avg_sa,avg_evt,avg_tp,avg_rr):
        print("{} & {} & {} & {} & {} & {} \\\\".format(l,t,s,e,tp,rr))
