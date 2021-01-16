import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from frechet import Frechet
from burr import Burr
from half_t import HalfT

def error_prob(result, best_arm):
    n_trials = result.shape[1]
    return 1 - np.asarray(result == best_arm).sum(axis=1)/n_trials

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
    budgets = np.linspace(5000, 25000, 5).astype(int)

    # CVaR level
    alph = 0.998

    # get bandit experiment results
    arms_selected = np.load('data/arms_selected.npy')
    n_trials = 1000

    # get best arms
    burr_optim = np.argmin([d.cvar(alph) for d in burr_dists])
    frec_optim = np.argmin([d.cvar(alph) for d in frec_dists])
    t_optim = np.argmin([d.cvar(alph) for d in t_dists])

    # error probability metrics
    burr_evt = error_prob(arms_selected[0], burr_optim)
    burr_sa = error_prob(arms_selected[1], burr_optim)
    frec_evt = error_prob(arms_selected[2], frec_optim)
    frec_sa = error_prob(arms_selected[3], frec_optim)
    t_evt = error_prob(arms_selected[4], t_optim)
    t_sa = error_prob(arms_selected[5], t_optim)

    plt.style.use('seaborn')
    plt.rc('axes', titlesize=8)     # fontsize of the axes title
    plt.rc('axes', labelsize=6)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=4)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=4)    # fontsize of the tick labels
    plt.rc('legend', fontsize=5)    # fontsize of the tick labels
    plt.rc('font', family='serif')

    # uncomment this line for Latex rendering
    #plt.rc('text', usetex=True)

    fig, axs = plt.subplots(1, 3, figsize=(7, 2.5))

    # Burr plots
    axs[0].plot(budgets, burr_evt, linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
    axs[0].plot(budgets, burr_sa, linestyle=':', linewidth=0.5, marker='.', markersize=5, color='b')
    axs[0].set_title('Burr bandit')

    # Frechet plots
    axs[1].plot(budgets, frec_evt, linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
    axs[1].plot(budgets, frec_sa, linestyle=':', linewidth=0.5, marker='.', markersize=5, color='b')
    axs[1].set_title('Frechet bandit')

    # T plots
    axs[2].plot(budgets, t_evt, linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
    axs[2].plot(budgets, t_sa, linestyle=':', linewidth=0.5, marker='.', markersize=5, color='b')
    axs[2].set_title('half-t bandit')

    for i in range(3):
        axs[i].set_xlabel('budget')
        axs[i].legend(['EVT', 'SA'])
        axs[i].set_xticks(budgets)
        axs[i].ticklabel_format(axis='x', style='sci', scilimits=(0,0))


    plt.tight_layout(pad=0.5)
    axs[0].set_ylabel('probability of error')
    fig.savefig('plots/bandit_plots.pdf', format='pdf', bbox_inches='tight')

    plt.show()
    plt.clf()
