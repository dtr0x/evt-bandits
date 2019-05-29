import numpy as np
import os

arrs = []
for file in os.listdir("data/sa"):
    arrs.append(np.load("data/sa/" + file))
sa_dat = np.vstack(tuple(arrs))

arrs = []
for file in os.listdir("data/ev"):
    arrs.append(np.load("data/ev/" + file))
ev_dat = np.vstack(tuple(arrs))

nan_idx = np.where(np.isnan(ev_dat))
ev_dat[nan_idx] = sa_dat[nan_idx]

sa_means = np.mean(sa_dat, axis=0)
ev_means = np.mean(ev_dat, axis=0)

