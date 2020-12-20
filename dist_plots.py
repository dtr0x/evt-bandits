import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from frechet import Frechet
from burr import Burr
from half_t import HalfT

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

    # CVaR level
    alph = 0.998
    burr_cvars_true = [d.cvar(alph) for d in burr_dists]
    frec_cvars_true = [d.cvar(alph) for d in frec_dists]
    t_cvars_true = [d.cvar(alph) for d in t_dists]

    # rmse
    burr_rmse = cvar_rmse(burr_cvars, burr_cvars_true)
    frec_rmse = cvar_rmse(frec_cvars, frec_cvars_true)
    t_rmse = cvar_rmse(t_cvars, t_cvars_true)
    # bias
    burr_bias = cvar_bias(burr_cvars, burr_cvars_true)
    frec_bias = cvar_bias(frec_cvars, frec_cvars_true)
    t_bias = cvar_bias(t_cvars, t_cvars_true)

    plt.style.use('seaborn')
    plt.rc('axes', titlesize=8)     # fontsize of the axes title
    plt.rc('axes', labelsize=6)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=4)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=4)    # fontsize of the tick labels
    plt.rc('legend', fontsize=5)    # fontsize of the tick labels
    plt.rc('font', family='serif')

    # uncomment this line for Latex rendering
    #plt.rc('text', usetex=True)

    n_dists = len(burr_dists)
    fig, axs = plt.subplots(6, n_dists, sharex=True, figsize=(7, 6))

    burr_titles = ["Burr({}, {})".format(c,d) for c,d in np.around(params, 2)]
    frec_titles = ["Frechet({})".format(g) for g in np.around(gamma, 2)]
    t_titles = ["half-t({})".format(p) for p in np.around(df, 2)]

    for i in range(len(burr_dists)):
        # Burr plots
        # RMSE
        axs[0,i].plot(sampsizes, burr_rmse[i,0], linestyle='solid', linewidth=1, marker='o', markersize=3, color='darkorange')
        axs[0,i].plot(sampsizes, burr_rmse[i,1], linestyle='dashed', linewidth=1, marker='D', markersize=3, color='cornflowerblue')
        # Bias
        axs[1,i].plot(sampsizes, burr_bias[i,0], linestyle='solid', linewidth=1, marker='o', markersize=3, color='darkorange')
        axs[1,i].plot(sampsizes, burr_bias[i,1], linestyle='dashed', linewidth=1, marker='D', markersize=3, color='cornflowerblue')

        # Frechet plots
        # RMSE
        axs[2,i].plot(sampsizes, frec_rmse[i,0], linestyle='solid', linewidth=1, marker='o', markersize=3, color='darkorange')
        axs[2,i].plot(sampsizes, frec_rmse[i,1], linestyle='dashed', linewidth=1, marker='D', markersize=3, color='cornflowerblue')
        # Bias
        axs[3,i].plot(sampsizes, frec_bias[i,0], linestyle='solid', linewidth=1, marker='o', markersize=3, color='darkorange')
        axs[3,i].plot(sampsizes, frec_bias[i,1], linestyle='dashed', linewidth=1, marker='D', markersize=3, color='cornflowerblue')

        # t plots
        # RMSE
        axs[4,i].plot(sampsizes, t_rmse[i,0], linestyle='solid', linewidth=1, marker='o', markersize=3, color='darkorange')
        axs[4,i].plot(sampsizes, t_rmse[i,1], linestyle='dashed', linewidth=1, marker='D', markersize=3, color='cornflowerblue')
        # Bias
        axs[5,i].plot(sampsizes, t_bias[i,0], linestyle='solid', linewidth=1, marker='o', markersize=3, color='darkorange')
        axs[5,i].plot(sampsizes, t_bias[i,1], linestyle='dashed', linewidth=1, marker='D', markersize=3, color='cornflowerblue')

        axs[5,i].set_xlabel('sample size')

        axs[0,i].set_title(burr_titles[i])
        axs[2,i].set_title(frec_titles[i])
        axs[4,i].set_title(t_titles[i])

    axs[0,0].set_ylabel('RMSE')
    axs[1,0].set_ylabel('absolute bias')
    axs[2,0].set_ylabel('RMSE')
    axs[3,0].set_ylabel('absolute bias')
    axs[4,0].set_ylabel('RMSE')
    axs[5,0].set_ylabel('absolute bias')
    axs[0,0].legend(['EVT', 'SA'])
    axs[1,0].legend(['EVT', 'SA'])
    axs[2,0].legend(['EVT', 'SA'])
    axs[3,0].legend(['EVT', 'SA'])
    axs[4,0].legend(['EVT', 'SA'])
    axs[5,0].legend(['EVT', 'SA'])

    plt.tight_layout(pad=0.5)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    fig.savefig('plots/dist_plots.pdf', format='pdf', bbox_inches='tight')

    plt.show()
    plt.clf()
