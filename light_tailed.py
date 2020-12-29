import numpy as np
from scipy.stats import lognorm, norm, gamma as pgamma, weibull_min
from scipy.special import gamma
from gen_data import gen_samples, get_cvars
from distribution import Distribution

class Lognormal(Distribution):
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def var(self, alph):
        mu = self.mu
        sigma = self.sigma
        return lognorm.ppf(alph, sigma, 0, np.exp(mu))

    def cvar(self, alph):
        mu = self.mu
        sigma = self.sigma
        q = lognorm.ppf(alph, sigma, 0, np.exp(mu))
        a = norm.cdf((mu+sigma**2-np.log(q))/sigma)
        b = 1 - norm.cdf((np.log(q)-mu)/sigma)
        return np.exp(mu + sigma**2/2) * a/b


class Weibull(Distribution):
    def __init__(self, shape, scale=1):
        self.shape = shape
        self.scale = scale

    def _gamma_inc(self, a, x):
        return pgamma.sf(x, a) * gamma(a)

    def var(self, alph):
        return weibull_min.ppf(alph, self.shape, 0, self.scale)

    def cvar(self, alph):
        shape = self.shape
        scale = self.scale
        return scale/(1-alph) * self._gamma_inc(1+1/shape, -np.log(1-alph))

if __name__ == '__main__':
    # Lognormal distributions
    lnorm_dists = [Lognormal(0, 0.25), Lognormal(0, 0.5), Lognormal(0, 1), Lognormal(1.5)]

    # Weibull distributions
    weib_dists = [Weibull(0.5, 1), Weibull(1, 1), Weibull(2, 1), Weibull(2, 5)]

    # sample sizes to test CVaR estimation
    sampsizes = np.linspace(2000, 20000, 10).astype(int)
    n_max = sampsizes[-1]

    # number of independent runs
    s = 1000

    # generate data
    np.random.seed(0)
    lnorm_data = gen_samples(lnorm_dists, s, n_max)
    weib_data = gen_samples(weib_dists, s, n_max)

    # CVaR level
    alph = 0.998

    lnorm_result = get_cvars(lnorm_data, alph, sampsizes)
    weib_result = get_cvars(weib_data, alph, sampsizes)

    np.save('data/lnorm_samples.npy', lnorm_data)
    np.save('data/weib_samples.npy', weib_data)
    np.save('data/lnorm_cvars.npy', lnorm_result)
    np.save('data/weib_cvars.npy', weib_result)
