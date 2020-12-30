import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from light_tailed import Lognormal, Weibull

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
    lnorm_cvars = np.load('data/lnorm_cvars.npy')

    weib_cvars = np.load('data/weib_cvars.npy')

    # Lognormal distributions
    lnorm_dists = [Lognormal(0, 0.25), Lognormal(0, 0.5), Lognormal(0, 1), Lognormal(0, 1.5)]

    # Weibull distributions
    weib_dists = [Weibull(0.5, 1), Weibull(1, 1), Weibull(2, 1), Weibull(2, 5)]

    # sample sizes to test CVaR estimation
    sampsizes = np.linspace(2000, 20000, 10).astype(int)

    # CVaR level
    alph = 0.998
    lnorm_cvars_true = [d.cvar(alph) for d in lnorm_dists]
    weib_cvars_true = [d.cvar(alph) for d in weib_dists]

    # rmse
    lnorm_rmse = cvar_rmse(lnorm_cvars, lnorm_cvars_true)
    weib_rmse = cvar_rmse(weib_cvars, weib_cvars_true)
    # bias
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

    n_dists = len(lnorm_dists)
    fig, axs = plt.subplots(4, n_dists, sharex=True, figsize=(7, 6))

    lnorm_params = [d.sigma for d in lnorm_dists]
    weib_params = [(d.shape, d.scale) for d in weib_dists]
    lnorm_titles = ['Lognormal(0, {})'.format(i) for i in lnorm_params]
    weib_titles = ['Weibull({}, {})'.format(i,j) for i,j in weib_params]

    for i in range(len(lnorm_dists)):
        # lnorm plots
        # RMSE
        axs[0,i].plot(sampsizes, lnorm_rmse[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
        axs[0,i].plot(sampsizes, lnorm_rmse[i,1], linestyle=':', linewidth=0.5, marker='.', markersize=5, color='b')
        # Bias
        axs[1,i].plot(sampsizes, lnorm_bias[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='red')
        axs[1,i].plot(sampsizes, lnorm_bias[i,1], linestyle=':', linewidth=0.5, marker='.', markersize=5, color='b')

        # weibull plots
        # RMSE
        axs[2,i].plot(sampsizes, weib_rmse[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
        axs[2,i].plot(sampsizes, weib_rmse[i,1], linestyle=':', linewidth=0.5, marker='.', markersize=5, color='b')
        # Bias
        axs[3,i].plot(sampsizes, weib_bias[i,0], linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
        axs[3,i].plot(sampsizes, weib_bias[i,1], linestyle=':', linewidth=0.5, marker='.', markersize=5, color='b')

        axs[3,i].set_xlabel('sample size')

        axs[0,i].set_title(lnorm_titles[i])
        axs[2,i].set_title(weib_titles[i])

    axs[0,0].set_ylabel('RMSE')
    axs[1,0].set_ylabel('absolute bias')
    axs[2,0].set_ylabel('RMSE')
    axs[3,0].set_ylabel('absolute bias')
    axs[0,0].legend(['EVT', 'SA'])
    axs[1,0].legend(['EVT', 'SA'])
    axs[2,0].legend(['EVT', 'SA'])
    axs[3,0].legend(['EVT', 'SA'])

    plt.tight_layout(pad=0.5)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    fig.savefig('plots/light_tail_plots.pdf', format='pdf', bbox_inches='tight')

    plt.show()
    plt.clf()
