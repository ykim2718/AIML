"""
y, scipy.stats - lognorm.py, 2017.11.21, 11.26 - 27

scipy, lognormal distribution - parameters
------------------------------------------
https://stackoverflow.com/questions/8747761/scipy-lognormal-distribution-parameters

"""

import numpy as np
from scipy import stats

linsample = stats.norm.rvs(loc=2, scale=1, size=100)  # linsample ~ N(mu=2, sigma=1)
loc, scale = stats.norm.fit(linsample)
print("{:-^32}".format('normal'))
print("loc={l}, scale={s}".format(l=loc, s=scale))

logsample = np.exp(linsample)  # sample ~ lognormal(mu=2, sigma=1)
shape, loc, scale = stats.lognorm.fit(logsample, floc=0)  # hold location to 0 while fitting
print("{:-^32}".format('lognormal'))
print("shape={sh}, loc={l}, scale={sc}".format(l=loc, sc=scale, sh=shape))
mu, sigma = np.log(scale), shape
print("norm: mu={mu}, sigma={sigma}".format(mu=mu, sigma=sigma))

ln_mode = np.exp(mu - sigma ** 2)
ln_mean = np.exp(mu + (sigma ** 2) / 2)
ln_sigma = np.sqrt(np.abs(np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2))
print("lognorm: mode={mode:.3g}, mean={mean:.3g}, sigma={sigma:.3g}".format(
    mode=ln_mode, mean=ln_mean, sigma=ln_sigma))
print("lognorm: mean={:.3g}".format(stats.lognorm.mean(sigma, 0, np.exp(mu))))

print("cdf at x=exp(mu)={:.3g}".format(np.exp(mu)))
print(stats.lognorm.cdf(np.exp(mu), shape, loc, scale), 'Vs expectation 0.5')  # (ln(x)-mu)=0 at x=exp(mu) in CDF eq.
print(stats.lognorm.cdf(np.exp(mu), sigma, 0, np.exp(mu)), 'Vs expectation 0.5')

print('ppf (inverse cdf) at x=0.5')
print(np.log(stats.lognorm.ppf(0.5, shape, loc, scale)), 'Vs 2 expected')
print(np.log(stats.lognorm.ppf(0.5, sigma, 0, np.exp(mu))), 'Vs 2 expected')

# print('pdf')
# print(stats.lognorm.pdf(mu, shape, loc, scale))
# print(stats.lognorm.pdf(mu, sigma, 0, np.exp(mu)))

import matplotlib.pyplot as plt
import pandas as pd

x = np.logspace(np.log10(1e-1), np.log10(1e2), 100)
y1 = stats.norm.pdf(x, mu, sigma)
y2 = stats.lognorm.pdf(x, sigma, 0, np.exp(mu))
frame = pd.DataFrame(dict(x=x, y1=y1, y2=y2))
ax1 = frame.plot(x='x', y='y1', c='r', logx=True, label='norm', ylim=(0, np.max([y1, y2]) * 1.1))
ax1 = frame.plot(x='x', y='y2', c='g', logx=True, label='lognorm', ax=ax1)
ax1.axvline(x=mu, color='r')
ax1.axvline(x=ln_mode, color='g')
ax1.axvline(x=ln_mean, color='b', ymax=0.7)
ax1.set_title("lognorm({n_mu:.3g},${n_sigma:.3g}^2$)".format(n_mu=mu, n_sigma=sigma), loc='center')
summary = pd.DataFrame(columns=['norm', 'lognorm'])
summary.loc['mean'] = [mu, ln_mean]
summary.loc['mode'] = [mu, ln_mode]
summary.loc['sigma'] = [sigma, ln_sigma]
# matplotlib.pyplot.table (https://matplotlib.org/api/_as_gen/matplotlib.pyplot.table.html)
pd.plotting.table(ax1, np.round(summary, 3), loc='upper right', colWidths=[0.1, 0.15])
ax1.legend(loc='upper left')
plt.show()
