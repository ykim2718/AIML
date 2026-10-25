"""
y, 2016.7.1 - 7.23, 8.1, minitab gev distribution.py

https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
★https://en.wikipedia.org/wiki/Gumbel_distribution
http:/docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.genextreme.html#scipy.stats.genextreme
http://wwww.weibull.com/hotwire/issue128/relbasics128.htm
http://stats.stackexchange.com/questions/71197/usable-estimators-for-parameters-in-gumbel-distribution
https://en.wikipedia.org/wiki/Distribution_fitting
http://reliawiki.org/index.php/The_Gumbel/SEV_Distribution
http://www.math.utah.edu/~lzhang/teaching/3070spring2009/Daily%20Updates/mar9/mar9.pdf ★★★
★https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot
★http://math.stackexchange.com/questions/72975/variance_of_sample_variance
★2007,61(2),159-60, The American Statistician, Lingyun Zhang, Sample mean and Sample variance Their Covariance and ++.pdf
★2008,7(2),408-15, J Modern Appl Stat Methods, Ramalingam Shanmugam, Correlation between the Sample Mean and Sample Variance.pdf
"""
import numpy as np
from scipy.stats import gumbel_l, gumbel_r, norm, anderson
import matplotlib.pyplot as plt


def MinitabSkewness(sample):
    """ y (copyRight) 2016.7.22 """
    mean, stdev_p, stdev_s = NormalDistributionParameters(sample)
    n = len(sample)
    skewness = 0
    for i in range(n):
        skewness += ((sample[i] - mean) / stdev_s) ** 3
    skewness += n / (n - 1) / (n - 2)
    return skewness


def NormalDistributionParameters(sample):
    """ y (copyRight) 2016.7.22 """
    mean, stdev_p = norm.fit(sample)
    stdev_s = stdev_p * np.sqrt(len(sample)) / (len(sample) - 1)
    return mean, stdev_p, stdev_s


sev_tick_percentile = [0.1, 1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 99.9]
sev_tick_label = [str(x) for x in sev_tick_percentile]
lev_tick_percentile = [0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 98, 99, 99.9]
lev_tick_label = [str(x) for x in lev_tick_percentile]


def get_evd_tick_position(loc=0, scale=1, evd='lev'):
    if evd == 'lev':
        tick_pos = [gumbel_r.ppf(p / 100, loc, scale) for p in lev_tick_percentile]
        return tick_pos
    elif evd == 'sev':
        tick_pos = [gumbel_l.ppf(p / 100, loc, scale) for p in sev_tick_percentile]
        return tick_pos
    else:
        raise AssertionError("evd should be ['sev', 'lev], not %s" % evd)


def get_evd_tick_label(evd='lev'):
    return {
        'sev': sev_tick_label, 'lev': lev_tick_label
    }[evd]


def get_evd_tick_pos_and_label(loc=0, scale=1, evd='lev'):
    return get_evd_tick_position(loc, scale, evd), get_evd_tick_label(evd)


def MinitabPercentile(a_list):
    size = len(a_list)
    if size == 0:
        raise AssertionError('argument list is empty')
    ranks = np.array(a_list).argsort().argsort()
    percentiles = [(index_b0 + 0.7) / (size + 0.4) for index_b0 in ranks]
    return percentiles


def MinitabCIsFromStartEndPercentiles(
        loc, scale, skewness, size,
        distribution='norm', confidence=0.95, start_percentile=0.01, stop_percentile=0.99
):
    """
    y (copyRight) 2016.7.8 - 7.23
    "returns: ci lower boudnary, ci upper boundary, z-score, and percentile
    """
    _norm = 'norm'
    _sev = 'sev'
    _lev = 'lev'
    if distribution not in [_norm, _sev, _lev]: return ()
    ppf = {_norm: norm.ppf, _sev: gumbel_l.ppf, _lev: gumbel_r.ppf}[distribution]
    cdf = {_norm: norm.cdf, _sev: gumbel_l.cdf, _lev: gumbel_r.cdf}[distribution]

    z_scores = np.linspace(ppf(start_percentile), ppf(stop_percentile), 100)
    percentiles = cdf(z_scores)
    return yMinitabConfidenceIntervalsFromPercentiles(
        loc, scale, skewness, size, distribution, confidence, percentiles
    )


def yMinitabConfidenceIntervalsFromPercentiles(loc, scale, skewness, size,
                                               distribution='norm', confidence=0.95, percentiles=[]):
    """
    y (copyRight)
        2016.7.8 - 7.23

    "returns: ci lower boudnary, ci upper boundary, z-socre, and percentile

    - Minitab Help: Confidence intervals - probaility plot
            -> Minitab calculates pointwise confidence intervals
    - Minitab Methods and Formulas: Confidence limits for percentiles
    - 2003, Jul 24, Sangtae Ahn and Jefreey A. Fessler, Univ of Michigan, Standard Errors of
            Mean, Variance, and Standard Deviation Estimators.pdf
    - 2007,61(2),159-60, The American Statistician, Lingyun Zhang, Sample mean and
            Sample variance Their Covariance and ++.pdf
    - 2008,7(2),408-15, J Modern Appl Stat Methods, Ramalingam Shanmugam, Correlation between
            the Sample Mean and Sample Variance.pdf
    """

    _norm = 'norm'
    _sev = 'sev'
    _lev = 'lev'
    if distribution not in [_norm, _sev, _lev]: return ()
    ppf = {_norm: norm.ppf, _sev: gumbel_l.ppf, _lev: gumbel_r.ppf}[distribution]
    cdf = {_norm: norm.cdf, _sev: gumbel_l.cdf, _lev: gumbel_r.cdf}[distribution]

    """
    if distribution in [_sev, _lev]:
        loc = loc + scale * 0.5772      # wikipedia, assuming input loc is mode
        scale = scale * np.pi / np.sqrt(6)    # wikipedia
    """

    if len(percentiles) == 0:
        percentiles = [x / 100 for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60,
                                         70, 80, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]

    y_hacking = 2  # 2016.7.23, hacked it but don't reason it
    var_loc = (y_hacking * scale / np.sqrt(size)) ** 2  # UMVU approx, Ahn2003
    var_scale = (scale * np.sqrt(2 / (size - 1))) ** 2  # UMVU approx, Ahn2003
    cov_loc_scale = skewness / size  # Normal approx, Zhang2007, Shanmugam2008
    # 2016.7.15, let me study Fisher information matrix for cov_loc_scale
    cov_loc_scale = 1.14 / size  # Zhang2007 + wiki/Gumbel_distribution
    cov_loc_scale = 0  # y, 2016.7.23
    za = (1 + confidence) / 2  # Minitab help - Methods and Formulas
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
    zp = ppf(percentiles, 0, 1)  # Minitab help - Methods and Formulas
    xp = loc + zp * scale  # Minitab help - Methods and Formulas
    var_xp = var_loc + 2 * zp * cov_loc_scale + zp ** 2 * var_scale  # Minitab help - Methods and Formulas
    xpl = xp - za * np.sqrt(var_xp)  # Minitab help - Methods and Formulas
    xpu = xp + za * np.sqrt(var_xp)  # Minitab help - Methods and Formulas
    # print('var_ratio (%s/%s)' % (var_loc, var_scale), var_loc/var_scale)
    variate_lbs, variate_ubs, z_scores = xpl, xpu, zp
    return variate_lbs, variate_ubs, z_scores, percentiles


def get_variate_and_z_arrays(
        loc, scale, start_percentile=0.01, stop_percentile=0.99, size=100, distribution='norm'
):
    """ y (copyRight) 2016.7.1 - 7.14 """
    ppf = {'norm': norm.ppf, 'sev': gumbel_l.ppf, 'lev': gumbel_r.ppf}[distribution]
    variate = np.linspace(ppf(start_percentile, loc, scale),
                          ppf(stop_percentile, loc, scale), size)
    z_score = [(x - loc) / scale for x in variate]
    return variate, z_score


def MinitabCIInsideBooleans(sample, distribution, loc=0, scale=0, confidence=0.95):
    """
    y (copyRight)
        2016.7.1 - 7.14, 7.28
    """

    _norm = 'norm'
    _sev = 'sev'
    _lev = 'lev'
    if distribution not in [_norm, _sev, _lev]:
        raise AssertionError('distribution %s is not supported' % distribution)
    fit = {_norm: norm.fit, _sev: gumbel_l.fit, _lev: gumbel_r.fit}[distribution]
    sample_p = MinitabPercentile(sample)
    if loc == 0 and scale == 0:  # 2016.7.28
        loc, scale = fit(sample)
    skewness = MinitabSkewness(sample)
    ci_lb, ci_ub, _, _ = yMinitabConfidenceIntervalsFromPercentiles(
        loc, scale, skewness, len(sample), distribution, confidence, sample_p)
    ci_inside_bools = []
    for i, x in enumerate(sample):
        ci_inside_bools.append(ci_lb[i] <= x <= ci_ub[i])
    return ci_inside_bools


def MinitabCIOuts(sample, distribution, loc=0, scale=0, confidence=0.95):
    """
    y (copyRight)
        2016.7.1 - 7.14, 7.28, 8.1
    """

    try:
        ppf = {'norm': norm.ppf, 'sev': gumbel_l.ppf, 'lev': gumbel_r.ppf}[distribution]
    except:
        return [], []
    sample_p = MinitabPercentile(sample)
    ci_out_x = []
    ci_out_y = []
    ci_booleans = MinitabCIInsideBooleans(sample, distribution, loc, scale, confidence)
    ci_out_positions = [i for i, x in enumerate(ci_booleans) if x == False]
    for pos in ci_out_positions:
        ci_out_x.append(sample[pos])
        ci_out_y.append(sample_p[pos])
    ci_out_y = ppf(ci_out_y, 0, 1)
    return ci_out_x, ci_out_y


def MinitabEVDChartAxes(ax, title, sample, distribution, loc=0, scale=0, confidence=0.95):
    """
    y (copyRight)
        2016.7.1 - 7.14, 8.1
        2017.10.25
    """

    assert distribution in ['norm', 'sev', 'lev'], 'invalid distribution name {d}'.format(distribution)

    ppf = {'norm': norm.ppf, 'sev': gumbel_l.ppf, 'lev': gumbel_r.ppf}[distribution]

    sev_tick_percentile = [0.1, 1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 99.9]
    sev_tick_position = [ppf(p / 100, 0, 1) for p in sev_tick_percentile]
    sev_tick_label = [str(x) for x in sev_tick_percentile]
    lev_tick_percentile = [0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 98, 99, 99.9]
    lev_tick_position = [ppf(p / 100, 0, 1) for p in lev_tick_percentile]
    lev_tick_label = [str(x) for x in lev_tick_percentile]

    def _variate_and_z_arrays(loc, scale, start_percentile=0.01, stop_percentile=0.99, size=100):

        variate = np.linspace(ppf(start_percentile, loc, scale),
                              ppf(stop_percentile, loc, scale), size)
        z_score = [(x - loc) / scale for x in variate]
        return variate, z_score

    sample_p = MinitabPercentile(sample)
    sample_icdf = ppf(sample_p, 0, 1)
    sample_skewness = 0
    fit_variate, fit_zscore = _variate_and_z_arrays(loc, scale, 0.01, 0.99, 100)
    ci_lb, ci_ub, ci_z, ci_p = MinitabCIsFromStartEndPercentiles(
        loc, scale, sample_skewness, len(sample), distribution, confidence, 0.01, 0.99)
    ax.plot(sample, sample_icdf, 'ko', lw=0, label='sample_p')
    ax.plot(fit_variate, fit_zscore, 'g-', lw=5, alpha=0.6, label='fit (%.3f, %.3f)' % (loc, scale))
    ax.plot(ci_lb, ci_z, 'b-', lw=5, alpha=0.6, label='%.0f%% CI(LB)' % (confidence * 100))
    ax.plot(ci_ub, ci_z, 'm-', lw=5, alpha=0.6, label='%.0f%% CI(UB)' % (confidence * 100))
    tick_position = {'sev': sev_tick_position, 'lev': lev_tick_position}[distribution]
    tick_label = {'sev': sev_tick_label, 'lev': lev_tick_label}[distribution]
    ax.set_yticks(tick_position)
    ax.set_yticklabels(tick_label)
    ax.legend(loc='upper left', frameon=False)
    ax.set_title(title, loc='center')
    ax.set_ylim(ppf([0.01, 0.99], 0, 1))
    ax.grid(True)


def demo_minitab_gev_distribution():

    g_debugging = False

    # Minitab, LEV (Loc 3, Scale 2)
    # SEV Fit -> Loc 3.884, Scale 2.121, N 10, AD 0.647, P-Value 0.079
    # LEV Fit -> Loc 2.014, Scale 1.374, N 10, AD 0.312, P-Value >0.250
    sample_lev = [0.76324, 4.60384, 3.19784, 0.95884, 1.51730, 1.85600, 1.43297,
                  3.05610, 4.19548, 7.08785]
    sample_p = MinitabPercentile(sample_lev)
    print('sample_lev', [float('%.4f' % x) for x in sample_lev])
    print('  sample_p', [float('%.4f' % x) for x in sample_p])
    skewness = MinitabSkewness(sample_lev)
    print('  skewness', skewness)

    results = anderson(sample_lev, dist='gumbel')
    print('anderson-daring test, p=', results[0], results[1], results[2])
    print(results)

    confidence_level = 0.95
    print('sev pci inside (ci=%s)' % confidence_level,
          MinitabCIInsideBooleans(sample_lev, 'sev', confidence_level))
    print('lev pci inside (ci=%s)' % confidence_level,
          MinitabCIInsideBooleans(sample_lev, 'lev', confidence_level))

    fig, (ax1, ax2) = plt.subplots(1, 2)

    loc, scale = gumbel_l.fit(sample_lev)
    print('gumbel_l.fit %.3f, %.3f Vs. minitab 3.884, 2.121' % (loc, scale))
    sev_icdf = gumbel_l.ppf(sample_p, 0, 1)
    fit_variate, fit_zscore = get_variate_and_z_arrays(loc, scale, 0.01, 0.99, 10, distribution='sev')
    ci_lb, ci_ub, ci_z, ci_p = MinitabCIsFromStartEndPercentiles(
        loc, scale, skewness, len(sample_lev), 'sev', confidence_level, 0.01, 0.99)
    if g_debugging:
        _lb, _ub, _z, _p = yMinitabConfidenceIntervalsFromPercentiles(
            loc, scale, skewness, len(sample_lev), 'sev', confidence_level)
        for lb, ub, z, p in zip(_lb, _ub, _z, _p):
            print('sev {:.2f} {:.4f} {:.4f} {:.4f}'.format(p, z, lb, ub))
    ax1.plot(sample_lev, sev_icdf, 'ko', lw=0, label='sample_p')
    ax1.plot(fit_variate, fit_zscore, 'g-', lw=5, alpha=0.6, label='fit (%.3f, %.3f)' % (loc, scale))
    ax1.plot(ci_lb, ci_z, 'b-', lw=5, alpha=0.6, label='%.0f%% CI(LB)' % (confidence_level * 100))
    ax1.plot(ci_ub, ci_z, 'm-', lw=5, alpha=0.6, label='%.0f%% CI(UB)' % (confidence_level * 100))
    tick_pos, tick_label = get_evd_tick_pos_and_label(0, 1, 'sev')
    ax1.set_yticks(tick_pos)
    ax1.set_yticklabels(tick_label)
    ax1.legend(loc='best', frameon=False)
    ax1.set_title('SEV (gumbel_l)', loc='center')
    ax1.set_ylim(gumbel_l.ppf([0.01, 0.99], 0, 1))
    ax1.grid(True)

    loc, scale = gumbel_r.fit(sample_lev)
    loc, scale = 2, 1
    MinitabEVDChartAxes(ax2, 'LEV (gumbel_r)', sample_lev, 'lev', loc, scale)

    ci_out_x, ci_out_y = MinitabCIOuts(sample_lev, 'lev', loc, scale)
    print('ci_out_x {}, ci_out_y {}'.format(len(ci_out_x), len(ci_out_y)))
    for xy in zip(ci_out_x, ci_out_y):
        ax2.annotate(
            'ci out',
            xy=(xy[0], xy[1]), xycoords='data',
            xytext=(-20, 20), textcoords='offset points',
            arrowprops=dict(facecolor='red', edgecolor='none', shrink=0.05, width=2, headwidth=6),
            horizontalalignment='right', verticalalignment='bottom',
            color='red', weight='semibold'
        )

    plt.show()


if __name__ == '__main__':

    demo_minitab_gev_distribution()

