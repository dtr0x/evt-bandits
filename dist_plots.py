import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from frechet import Frechet
from burr import Burr
from half_t import HalfT
from light_tailed import *

def rmse(x, true):
    return np.sqrt(np.mean((x-true)**2, axis=0))

def bias(x, true):
    return np.abs(np.mean(x, axis=0) - true)

def cvar_rmse(cvars, cvars_true):
    r = []
    for i in range(len(cvars_true)):
        r1 = rmse(cvars[0,i], cvars_true[i])
        r2 = rmse(cvars[1,i], cvars_true[i])
        r.append([r1,r2])
    return np.array(r)

def cvar_bias(cvars, cvars_true):
    b = []
    for i in range(len(cvars_true)):
        b1 = bias(cvars[0,i], cvars_true[i])
        b2 = bias(cvars[1,i], cvars_true[i])
        b.append([b1,b2])
    return np.array(b)

def get_means(cvars):
    m = np.nanmean(cvars[:,:,:,-1], axis=2)
    f = lambda x: ' & '.join(np.around(x, 2).astype(str))
    m_str = np.apply_along_axis(f, 1, m.transpose())
    for s in m_str:
        print(s)

if __name__ == '__main__':
    burr_cvars = np.load('data/burr_cvars.npy')[:2]
    frec_cvars = np.load('data/frec_cvars.npy')[:2]
    t_cvars = np.load('data/t_cvars.npy')[:2]
    lnorm_cvars = np.load('data/lnorm_cvars.npy')[:2]
    weib_cvars = np.load('data/weib_cvars.npy')[:2]

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

    # sample sizes to test CVaR estimation
    sampsizes = np.linspace(2000, 20000, 10).astype(int)

    # CVaR level
    alph = 0.998
    burr_cvars_true = [d.cvar(alph) for d in burr_dists]
    frec_cvars_true = [d.cvar(alph) for d in frec_dists]
    t_cvars_true = [d.cvar(alph) for d in t_dists]
    lnorm_cvars_true = [d.cvar(alph) for d in lnorm_dists]
    weib_cvars_true = [d.cvar(alph) for d in weib_dists]

    # rmse
    burr_rmse = cvar_rmse(burr_cvars, burr_cvars_true)
    frec_rmse = cvar_rmse(frec_cvars, frec_cvars_true)
    t_rmse = cvar_rmse(t_cvars, t_cvars_true)
    lnorm_rmse = cvar_rmse(lnorm_cvars, lnorm_cvars_true)
    weib_rmse = cvar_rmse(weib_cvars, weib_cvars_true)

    # bias
    burr_bias = cvar_bias(burr_cvars, burr_cvars_true)
    frec_bias = cvar_bias(frec_cvars, frec_cvars_true)
    t_bias = cvar_bias(t_cvars, t_cvars_true)
    lnorm_bias = cvar_bias(lnorm_cvars, lnorm_cvars_true)
    weib_bias = cvar_bias(weib_cvars, weib_cvars_true)

    plt.style.use('seaborn')
    plt.rc('axes', titlesize=8)     # fontsize of the axes title
    plt.rc('axes', labelsize=6)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=4)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=4)    # fontsize of the tick labels
    plt.rc('legend', fontsize=5)    # fontsize of the tick labels
    plt.rc('font', family='serif')

    # uncomment this line for Latex rendering
    #plt.rc('text', usetex=True)

    n_classes = 5
    n_dists = 5
    figsize = (7, 4)
    burr_titles = [d.get_label() for d in burr_dists]
    frec_titles = [d.get_label() for d in frec_dists]
    t_titles = [d.get_label() for d in t_dists]
    lnorm_titles = [d.get_label() for d in lnorm_dists]
    weib_titles = [d.get_label() for d in weib_dists]

    # Plot RMSE

    fig, axs = plt.subplots(n_classes, n_dists, sharex=True, figsize=figsize)

    for i in range(n_dists):
        # Burr plots
        axs[0,i].plot(sampsizes, burr_rmse[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=4, color='r')
        axs[0,i].plot(sampsizes, burr_rmse[i,1], linestyle=':', linewidth=0.5, marker='s', markersize=2, color='b')
        # Frechet plots
        axs[1,i].plot(sampsizes, frec_rmse[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=4, color='r')
        axs[1,i].plot(sampsizes, frec_rmse[i,1], linestyle=':', linewidth=0.5, marker='s', markersize=2, color='b')
        # Half-t plots
        axs[2,i].plot(sampsizes, t_rmse[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=4, color='r')
        axs[2,i].plot(sampsizes, t_rmse[i,1], linestyle=':', linewidth=0.5, marker='s', markersize=2, color='b')
        # Lognormal plots
        axs[3,i].plot(sampsizes, lnorm_rmse[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=4, color='r')
        axs[3,i].plot(sampsizes, lnorm_rmse[i,1], linestyle=':', linewidth=0.5, marker='s', markersize=2, color='b')
        # Weibull plots
        axs[4,i].plot(sampsizes, weib_rmse[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=4, color='r')
        axs[4,i].plot(sampsizes, weib_rmse[i,1], linestyle=':', linewidth=0.5, marker='s', markersize=2, color='b')

        axs[4,i].set_xlabel('sample size')

        axs[0,i].set_title(burr_titles[i])
        axs[1,i].set_title(frec_titles[i])
        axs[2,i].set_title(t_titles[i])
        axs[3,i].set_title(lnorm_titles[i])
        axs[4,i].set_title(weib_titles[i])

    for i in range(n_classes):
        axs[i,0].set_ylabel('RMSE')
        axs[i,0].legend(['EVT', 'SA'])

    plt.tight_layout(pad=0.5)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    fig.savefig('plots/rmse.pdf', format='pdf', bbox_inches='tight')

    plt.clf()

    # Plot Bias

    fig, axs = plt.subplots(n_classes, n_dists, sharex=True, figsize=figsize)

    for i in range(n_dists):
        # Burr plots
        axs[0,i].plot(sampsizes, burr_bias[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=4, color='r')
        axs[0,i].plot(sampsizes, burr_bias[i,1], linestyle=':', linewidth=0.5, marker='s', markersize=2, color='b')
        # Frechet plots
        axs[1,i].plot(sampsizes, frec_bias[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=4, color='r')
        axs[1,i].plot(sampsizes, frec_bias[i,1], linestyle=':', linewidth=0.5, marker='s', markersize=2, color='b')
        # Half-t plots
        axs[2,i].plot(sampsizes, t_bias[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=4, color='r')
        axs[2,i].plot(sampsizes, t_bias[i,1], linestyle=':', linewidth=0.5, marker='s', markersize=2, color='b')
        # Lognormal plots
        axs[3,i].plot(sampsizes, lnorm_bias[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=4, color='r')
        axs[3,i].plot(sampsizes, lnorm_bias[i,1], linestyle=':', linewidth=0.5, marker='s', markersize=2, color='b')
        # Weibull plots
        axs[4,i].plot(sampsizes, weib_bias[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=4, color='r')
        axs[4,i].plot(sampsizes, weib_bias[i,1], linestyle=':', linewidth=0.5, marker='s', markersize=2, color='b')

        axs[4,i].set_xlabel('sample size')

        axs[0,i].set_title(burr_titles[i])
        axs[1,i].set_title(frec_titles[i])
        axs[2,i].set_title(t_titles[i])
        axs[3,i].set_title(lnorm_titles[i])
        axs[4,i].set_title(weib_titles[i])

    for i in range(n_classes):
        axs[i,0].set_ylabel('absolute bias')
        axs[i,0].legend(['EVT', 'SA'])

    plt.tight_layout(pad=0.5)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    fig.savefig('plots/bias.pdf', format='pdf', bbox_inches='tight')

    plt.clf()
