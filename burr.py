import numpy as np
from scipy.special import hyp2f1, beta
from distribution import Distribution

class Burr(Distribution):
    def __init__(self, c, d):
        self.c = c
        self.d = d
        self.xi = 1/c/d

    def U(self, t):
        c = self.c
        d = self.d
        return (t**(1/d) - 1)**(1/c)

    def cdf(self, x):
        c = self.c
        d = self.d
        return 1 - (1 + x**c)**(-d)

    def pdf(self, x):
        c = self.c
        d = self.d
        return c*d*x**(c-1)/(1 + x**c)**(d+1)

    def var(self, alph):
        c = self.c
        d = self.d
        return ((1-alph)**(-1/d) - 1)**(1/c)

    def cvar(self, alph):
        c = self.c
        d = self.d
        q = self.var(alph)
        return 1/(1-alph) * (d * ((1/q)**c)**(d-1/c))/(d-1/c) * \
                hyp2f1(d-1/c, 1+d, d-1/c+1, -1/q**c)
