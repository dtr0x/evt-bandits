import numpy as np
from scipy.stats import genpareto, lognorm, norm, gamma as pgamma, weibull_min
from scipy.special import gamma
from scipy.interpolate import interp1d

ad_quantiles = np.loadtxt("ADQuantiles.csv", delimiter = ",", dtype = float, skiprows = 1, usecols = range(1,1000))
ad_pvals = np.round(np.linspace(0.999, 0.001, 1000), 3) #col names
ad_shape = np.round(np.linspace(-0.5, 1, 151), 2) #row names

def tce_sa(x, alph):
	q = np.quantile(x, alph)
	y = x[x >= q]

	return np.mean(y)

def tce_ev_params(alph, u, shape, scale, tp):
	q = u + scale/shape*((1-(alph-tp)/(1-tp))**(-shape) - 1)
	return q + (scale + shape*(q - u))/(1 - shape)

def tce_ev(x, alph, tp=0.95, cutoff=0.99):
    u = np.quantile(x, tp)
    y = x[x > u] - u
    shape, loc, scale = genpareto.fit(y, floc=0)

    if shape > cutoff:
        return np.nan
    else:
        return tce_ev_params(alph, u, shape, scale, tp)    

def tce_gpd(alph, shape, scale=1):
    q = genpareto.ppf(alph, shape, loc=0, scale=scale)
    return (q+scale)*(1+shape*q/scale)**(-1/shape)/((1-alph)*(1-shape))

def tce_lnorm(alph, sigm, mu=0):
	q = lognorm.ppf(alph, sigm, 0, np.exp(mu))
	a = norm.cdf((mu+sigm**2-np.log(q))/sigm)
	b = 1 - norm.cdf((np.log(q)-mu)/sigm)
	return np.exp(mu + sigm**2/2) * a/b

def gamma_inc(a, x):
    return pgamma.sf(x, a) * gamma(a)

def tce_weibull(alph, shape, scale=1):
	return scale/(1-alph) * gamma_inc(1+1/shape, -np.log(1-alph))

def gpd_ad(x, tp=0.95):
    u = np.quantile(x, tp)
    y = x[x > u] - u
    shape, loc, scale = genpareto.fit(y)
    z = genpareto.cdf(y, shape, loc, scale)
    z = np.sort(z)
    n = len(z)
    i = np.linspace(1, n, n)
    stat = -n - (1/n) * np.sum((2 * i - 1) * (np.log(z) + np.log1p(-z[::-1])))

    return u, stat, shape, scale

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

def forward_stop(pvals, signif):
    pvals_transformed = []
    for i in range(1, len(pvals)+1):
        pvals_transformed.append(-np.mean(np.log1p(-pvals[:i])))
    pvals_transformed = np.asarray(pvals_transformed)

    kf = np.where(pvals_transformed <= signif)[0]
    if kf.size == 0:
        stop = 0
    else:
        stop = max(kf) + 1
    if stop == pvals.size:
        stop -= 1
    return stop

def raw_up(pvals, signif):
    pvals_idx = np.where(pvals > signif)[0]
    if pvals_idx.size == 0:
        stop = -1
    else:
        stop = pvals_idx[0]
    return stop

def raw_down(pvals, signif):
    pvals_idx = np.where(pvals <= signif)[0]
    if pvals_idx.size == 0:
        stop = 0
    else:
        stop = max(pvals_idx) + 1
    if stop == pvals.size:
        stop -= 1
    return stop

def tce_ad(x, alph, tp_init=0.9, tp_num=40, signif=0.1, cutoff=0.9, stop_rule=forward_stop):
    tps = np.linspace(tp_init, min(0.99, alph), tp_num)
    ad_tests = []
    pvals = []
    for tp in tps:
        try:
            u, stat, shape, scale = gpd_ad(x, tp)
            if shape <= cutoff:
                ad_tests.append([u, shape, scale, tp])
                pvals.append(ad_pvalue(stat, shape))
        except ValueError:
            pass
    
    if(len(ad_tests) == 0):
        #return np.nan
        return tce_sa(x, alph)

    ad_tests = np.asarray(ad_tests)
    pvals = np.asarray(pvals)

    stop = stop_rule(pvals, signif)
 
    u, shape, scale, tp = ad_tests[stop, ]
    return tce_ev_params(alph, u, shape, scale, tp)