import numpy as np
import pandas as pd
from frechet import Frechet
from burr import Burr
from half_t import HalfT

if __name__ == '__main__':
    # Burr distributions
    c = [0.75, 1, 2, 3, 4]
    d = [2, 1.5, 1, 0.75, 0.5]
    params = [(i,j) for i,j in zip(c,d)]
    burr_dists = [Burr(*p) for p in params]
    burr_labels = ['Burr({}, {})'.format(i,j) for i,j in params]

    # Frechet distributions
    gamma = [1.25, 1.5, 2, 2.5, 3]
    frec_dists = [Frechet(p) for p in gamma]
    frec_labels = ['Frechet({})'.format(p) for p in gamma]

    # half-t distributions
    df = [1.25, 1.5, 2, 2.5, 3]
    t_dists = [HalfT(p) for p in df]
    t_labels = ['half-t({})'.format(p) for p in df]

    # CVaR level
    alph = 0.998

    sampsizes = np.linspace(2000, 20000, 10).astype(int)

    burr_result = np.load('data/burr_cvars.npy')
    frec_result = np.load('data/frec_cvars.npy')
    t_result = np.load('data/t_cvars.npy')

    labels = burr_labels + frec_labels + t_labels

    #np.around(np.nanmean(burr_result[2], axis=1), 3)
