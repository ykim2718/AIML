"""
y, 2016.7.6 - 7.23, minitab normal distribution.py
[scipy.stats.norm]
http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
"""

import numpy as np
from scipy.stats import norm, skew
from scipy.special import gamma
import matplotlib.pyplot as plt


def minitab_percentile(a_list):
    """
    y (copyRight) 2016.7.7
    :returns: percentile
    """
    size = len(a_list)
    if size == 0:
        raise AssertionError('argument list is empty')
    ranks = np.array(a_list).argsort().argsort()
    percentiles = [(index_b0 + 0.7) / (size + 0.4) for index_b0 in ranks]
    return percentiles


def minitab_norm_confidence_intervals(
        loc, scale, skewness, sample_size,
        confidence=0.95, start_percentile=0.01, stop_percentile=0.99, size=100
):
    """
    y (copyRight) 2016.7.8 - 7.23
    :returns: ci lower boundary, ci upper boundary, z-score, and percentile

    - Minitab Help: Confidence intervals - probability plot
            --> Minitab calculates pointwise confidence intervals
    - Minitab Methods and Formulas: Confidence limits for percentiles
    - 2003, Jul 24, Sangtae Ahn and Jefreey A. Fessler, Univ of Michigan, Standard Errors of
            Mean, Variance, and Standard Deviation Estimators.pdf
    - 2007,61(2),159-60, The American Statistician, Lingyun Zhang, Sample mean and
            Sample variance Their Covariance and ++.pdf
    - 2008,7(2),408-15, J Modern Appl Stat Methods, Ramalingam Shanmugam, Correlation between
            the Sample Mean and Sample Variance.pdf
    """
    kn = np.sqrt((sample_size - 1) / 2) * gamma((sample_size - 1) / 2) / gamma(sample_size / 2)
    y_hacking = 2  # 2016.7.23, hacked it but don't reason it; why it needs 2
    var_loc = (y_hacking * scale / np.sqrt(sample_size)) ** 2  # UMVU approx, Ahn2003
    var_scale = (scale * np.sqrt(2 / (sample_size - 1))) ** 2  # UMVU approx, Ahn2003
    cov_loc_scale = skewness / sample_size  # normal approx, Zhang2007, Shanmugam2008
    cov_loc_scale = 0  # y, 2016.7.23
    # 2016.7.15, let me study Fisher information matrix for cov_loc_scale
    za = (1 + confidence) / 2
    if True:  # True for debugging
        print('var_loc {:.3e}, var_scale {:.3e}, cov_loc_scale {:.3e}, za {:.3e}'.format(
            var_loc, var_scale, cov_loc_scale, za
        ))
        zp = norm.ppf([0.1, 0.5, 0.99])
        percentile = norm.cdf(zp)
        var_xp = var_loc + zp ** 2 * var_scale + 2 * zp * cov_loc_scale
        print('percentile {}, data {}, ci_lb {}, ci_ub {}'.format(
            percentile, zp,
            (loc + zp * scale) - za * np.sqrt(var_xp),
            (loc + zp * scale) + za * np.sqrt(var_xp)
        ))
    zp = np.linspace(norm.ppf(start_percentile), norm.ppf(stop_percentile), size)
    var_xp = var_loc + zp ** 2 * var_scale + 2 * zp * cov_loc_scale
    xp = loc + zp * scale
    xpl = xp - za * np.sqrt(var_xp)
    xpu = xp + za * np.sqrt(var_xp)
    percentile = norm.cdf(zp)
    return xpl, xpu, zp, percentile


def get_variate_and_z_arrays(
        loc, scale, start_percentile=0.01, stop_percentile=0.99, size=100
):
    variate = np.linspace(norm.ppf(start_percentile, loc, scale),
                          norm.ppf(stop_percentile, loc, scale), size)
    z_score = [(x - loc) / scale for x in variate]
    return variate, z_score


x = [-2, 0, 2]
x_percentile = minitab_percentile(x)
x_icdf = norm.ppf(x_percentile, 0, 1)
mu, sigma = norm.fit(x)  # stdev.p
sigma *= np.sqrt(len(x) / (len(x) - 1))  # stdev.s
print('mu, sigma = {:.3f}, {:.3f}'.format(mu, sigma))

skewness = skew(x)
print('skew {}'.format(skewness))

fig, (ax1, ax2) = plt.subplots(1, 2)

tick_val = [0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.9]
tick_label = [str(x) for x in tick_val]
tick_pos = norm.ppf(np.array(tick_val) / 100)
tick_linspace = np.linspace(norm.ppf(0.001), norm.ppf(0.999), 100)

ax1.plot(tick_linspace, norm.pdf(tick_linspace, 0, 1), 'r-', lw=5, alpha=0.6, label='norm.pdf')
ax1.hist(x, weights=np.ones_like(x) / len(x), alpha=0.2)
ax1.set_title('norm PDF', loc='center')

ci = 0.95
ci_lb, ci_ub, zp, percentile = minitab_norm_confidence_intervals(
    mu, sigma, skewness, len(x), ci, 0.01, 0.99)
fit_variate, fit_zscore = get_variate_and_z_arrays(mu, sigma, 0.01, 0.99, 10)

ax2.plot(x, x_icdf, 'ko', lw=0, label='x_icdf')
ax2.plot(fit_variate, fit_zscore, 'g-', lw=5, alpha=0.6, label='fit (%.3f, %.3f)' % (mu, sigma))
ax2.plot(ci_lb, zp, 'b-', lw=5, alpha=0.6, label='%.0f%% CI(LB)' % (ci * 100))
ax2.plot(ci_ub, zp, 'm-', lw=5, alpha=0.6, label='%.0f%% CI(UB)' % (ci * 100))
ax2.set_yticks(tick_pos)
ax2.set_yticklabels(tick_label)
ax2.legend(loc='best', frameon=False)
ax2.set_title('norm CDF', loc='center')
ax2.set_xlim([-6, 6])
ax2.set_ylim(norm.ppf([0.01, 0.99], 0, 1))

plt.show()
