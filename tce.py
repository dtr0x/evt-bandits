import numpy as np
import torch
from scipy.stats import genpareto, lognorm, norm
from scipy.interpolate import interp1d

ad_quantiles = np.loadtxt("ADQuantiles.csv", delimiter = ",", dtype = float, skiprows = 1, usecols = range(1,1000))
ad_pvals = np.round(np.linspace(0.999, 0.001, 1000), 3) #col names
ad_shape = np.round(np.linspace(-0.5, 1, 151), 2) #row names

def tce_sa(x, alph):
	q = np.quantile(x, alph)
	y = x[x >= q]

	return np.mean(y)

def tce_ev_params(alph, u, scale, shape, tp):
	q = u + scale/shape*((1-(alph-tp)/(1-tp))**(-shape) - 1)
	return q + (scale + shape*(q - u))/(1 - shape)

def tce_ev(x, alph, tp = 0.95):
    u = np.quantile(x, tp)
    y = x[x > u] - u
    shape, loc, scale = genpareto.fit(y, floc=0)

    if shape > 1:
        return np.nan
    else:
        return tce_ev_params(alph, u, scale, shape, tp)    

def tce_gpd(alph, shape, scale = 1):
    q = genpareto.ppf(alph, shape, loc=0, scale=scale)
    return (q+scale)*(1+shape*q/scale)**(-1/shape)/((1-alph)*(1-shape))

def tce_lnorm(alph, mu = 0, sigm = 1):
	q = lognorm.ppf(alph, sigm, mu, 1)
	a = norm.cdf((mu+sigm**2-np.log(q))/sigm)
	b = 1 - norm.cdf((np.log(q)-mu)/sigm)
	return np.exp(mu + sigm**2/2) * a/b


def tce_weibull(alph, shape, scale = 1):
	pass

def gpd_ad(x, tp = 0.95):
    u = np.quantile(x, tp)
    y = x[x > u] - u
    shape, loc, scale = genpareto.fit(y, floc=0)
    z = genpareto.cdf(y, shape, loc=0, scale=scale)
    z = np.sort(z)
    n = len(z)
    print(z)
    i = np.linspace(1, n, n)
    stat = -n - (1/n) * np.sum((2 * i - 1) * (np.log(z) + np.log1p(-z[::-1])))

    return stat, shape, scale

def ad_pvalue(stat, shape):
    row = np.where(ad_shape == max(round(shape, 2), -0.5))[0][0]

    if stat > ad_quantiles[row, -1]:
    	xdat = ad_quantiles[row, 950:999]
    	ydat = -np.log(ad_pvals[950:999])
    	lfit = np.polyfit(xdat, ydat, 1)
    	m = lfit[0]
    	b = lfit[1]
    	p = np.exp(-(m*stat+b))
    else:
    	bound_idx = min(np.where(stat < ad_quantiles[row, ])[0])
    	bound = ad_pvals[bound_idx]
    	if bound == 0.999:
    		p = bound
    	else:
    		x1 = ad_quantiles[row, bound_idx-1]
    		x2 = ad_quantiles[row, bound_idx]
    		y1 = -np.log(ad_pvals[bound_idx-1])
    		y2 = -np.log(ad_pvals[bound_idx])
    		lfit = interp1d([x1, x2], [y1, y2])
    		p = np.exp(-lfit(stat))

    return p

def tce_ad(x, alph, tp_init = 0.9, tp_num = 100, signif = 0.2):
    tps = np.linspace(tp_init, alph, tp_num)
    tps_valid = []
    ad_tests = []
    pvals = []
    for tp in tps:
    	stat, shape, scale = gpd_ad(x, tp)
    	if shape < 1:
    	    ad_tests.append([tp, shape, scale])
    	    pvals.append(ad_pvalue(stat, shape))
    
    if(len(ad_tests) == 0):
        return np.nan

    ad_tests = np.asarray(ad_tests)
    pvals = np.asarray(pvals)

    kf = []
    for i in range(1, len(pvals)):
    	kf.append(-np.mean(np.log1p(-pvals[:i])))
    kf = np.asarray(kf)
    if np.where(kf <= signif)[0].size == 0:
        tp = ad_tests[0, 0]
        u = np.quantile(x, tp)
        shape = ad_tests[0, 1]
        scale = ad_tests[0, 2]
        return tce_ev_params(alph, u, scale, shape, tp)
    else:
    	stop = max(np.where(kf <= signif)[0])

    tp = ad_tests[stop, 0]
    u = np.quantile(x, tp)
    shape = ad_tests[stop, 1]
    scale = ad_tests[stop, 2]
    return tce_ev_params(alph, u, scale, shape, tp)