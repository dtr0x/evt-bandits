import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from light_tailed import *

def error_prob(result, best_arm):
    n_trials = result.shape[1]
    return 1 - np.asarray(result == best_arm).sum(axis=1)/n_trials

if __name__ == '__main__':
    # Lognormal distributions
    lnorm_dists = [Lognormal(5, 0.25), Lognormal(4, 0.5), Lognormal(2.5, 0.75),
        Lognormal(2, 1), Lognormal(1, 1.5)]

    # Weibull distributions
    weib_dists = [Weibull(0.5, 1), Weibull(0.75, 2), Weibull(1, 3),
         Weibull(1.25, 4), Weibull(1.5, 5)]

    # sample sizes to test CVaR estimation
    budgets = np.linspace(5000, 25000, 5).astype(int)

    # CVaR level
    alph = 0.998

    # get bandit experiment results
    arms_selected = np.load('data/arms_selected_light.npy')
    n_trials = 1000

    # get best arms
    lnorm_optim = np.argmin([d.cvar(alph) for d in lnorm_dists])
    weib_optim = np.argmin([d.cvar(alph) for d in weib_dists])

    # error probability metrics
    lnorm_evt = error_prob(arms_selected[0], lnorm_optim)
    lnorm_sa = error_prob(arms_selected[1], lnorm_optim)
    weib_evt = error_prob(arms_selected[2], weib_optim)
    weib_sa = error_prob(arms_selected[3], weib_optim)

    plt.style.use('seaborn')
    plt.rc('axes', titlesize=8)     # fontsize of the axes title
    plt.rc('axes', labelsize=6)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=4)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=4)    # fontsize of the tick labels
    plt.rc('legend', fontsize=5)    # fontsize of the tick labels
    plt.rc('font', family='serif')

    # uncomment this line for Latex rendering
    #plt.rc('text', usetex=True)

    fig, axs = plt.subplots(1, 2, figsize=(7, 2.5))

    # Lognormal plots
    axs[0].plot(budgets, lnorm_evt, linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
    axs[0].plot(budgets, lnorm_sa, linestyle=':', linewidth=0.5, marker='.', markersize=5, color='b')
    axs[0].set_title('Lognormal bandit')

    # Weibull plots
    axs[1].plot(budgets, weib_evt, linestyle='--', linewidth=0.5, marker='.', markersize=5, color='r')
    axs[1].plot(budgets, weib_sa, linestyle=':', linewidth=0.5, marker='.', markersize=5, color='b')
    axs[1].set_title('Weibull bandit')

    for i in range(2):
        axs[i].set_xlabel('budget')
        axs[i].legend(['EVT', 'SA'])
        axs[i].set_xticks(budgets)
        axs[i].ticklabel_format(axis='x', style='sci', scilimits=(0,0))


    plt.tight_layout(pad=0.5)
    axs[0].set_ylabel('probability of error')
    fig.savefig('plots/bandit_plots_light.pdf', format='pdf', bbox_inches='tight')

    plt.show()
    plt.clf()
