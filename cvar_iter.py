import numpy as np

# define function for processing CVaR estimates in parallel

def cvar_iter(x, alph, sampsizes, cvar_fn):
    cvars = []
    for n in sampsizes:
        c = cvar_fn(x[:n], alph)
        cvars.append(c)
    return np.array(cvars)
