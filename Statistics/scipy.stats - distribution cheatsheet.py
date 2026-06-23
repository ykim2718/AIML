"""
scipy.stats - distribution cheatsheet.py
A grid of common probability distributions, after Rasmus Baath (2012).

Continuous distributions are drawn as PDF curves, discrete ones as stems (PMF).
Each panel is labelled only with its parameter symbols, like the original poster.
Parameter symbols are placed at meaningful positions on the panel (e.g. the
location parameter under the peak, scale parameters on the shoulders).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# (title, kind, x-range, callable -> y, [(x, y, label), ...])
#   kind   = 'c' for continuous PDF, 'd' for discrete PMF
#   labels = parameter symbols with (x, y) positions in axes fractions
PANELS = [
    # --- continuous distributions (PDF curves) ---
    ("normal",                'c', (-4, 4),  lambda x: stats.norm.pdf(x, 0, 1),
        [(0.24, 0.66, r"$\sigma_2$"), (0.50, 0.40, r"$\mu$"), (0.76, 0.66, r"$\sigma_1$")]),
    ("t distrib.",            'c', (-4, 4),  lambda x: stats.t.pdf(x, df=3),
        [(0.20, 0.52, r"$df_2$"), (0.50, 0.34, r"$\mu$"), (0.80, 0.52, r"$df_1$")]),
    ("uniform",               'c', (-4, 4),  lambda x: stats.uniform.pdf(x, -2, 4),
        [(0.28, 0.80, r"$L$"), (0.72, 0.80, r"$H$")]),
    ("beta",                  'c', (0, 1),   lambda x: stats.beta.pdf(x, 2, 2),
        [(0.64, 0.78, r"$a,\ b$")]),
    ("gamma",                 'c', (0, 12),  lambda x: stats.gamma.pdf(x, a=2, scale=1.5),
        [(0.62, 0.62, r"$S,\ R$")]),
    ("inv-gamma",             'c', (0, 6),   lambda x: stats.invgamma.pdf(x, a=3, scale=2),
        [(0.60, 0.68, r"$\alpha,\ \beta$")]),
    ("folded t",              'c', (0, 6),   lambda x: stats.foldnorm.pdf(x, 0.1),
        [(0.18, 0.60, r"$\mu$"), (0.48, 0.72, r"$\sigma$"), (0.80, 0.50, r"$df$")]),
    ("chi-square",            'c', (0, 12),  lambda x: stats.chi2.pdf(x, df=3),
        [(0.60, 0.60, r"$k$")]),
    ("noncentral chi-square", 'c', (0, 20),  lambda x: stats.ncx2.pdf(x, df=4, nc=2),
        [(0.70, 0.78, r"$k,\ \lambda$")]),
    ("double exp.",           'c', (-4, 4),  lambda x: stats.laplace.pdf(x),
        [(0.22, 0.58, r"$\tau_2$"), (0.50, 0.40, r"$\mu$"), (0.78, 0.58, r"$\tau_1$")]),
    ("exponential",           'c', (0, 6),   lambda x: stats.expon.pdf(x),
        [(0.45, 0.75, r"$\lambda$")]),
    ("shifted exp.",          'c', (0, 6),   lambda x: stats.expon.pdf(x, loc=1),
        [(0.50, 0.75, r"$\lambda$")]),
    ("F dist.",               'c', (0, 5),   lambda x: stats.f.pdf(x, 5, 10),
        [(0.62, 0.55, r"$df_1,\ df_2$")]),
    ("gen. gamma",            'c', (0, 6),   lambda x: stats.gengamma.pdf(x, a=3, c=2),
        [(0.70, 0.82, r"$r,\ \lambda,\ b$")]),
    ("logistic",              'c', (-5, 5),  lambda x: stats.logistic.pdf(x),
        [(0.22, 0.55, r"$\tau_2$"), (0.50, 0.35, r"$\mu$"), (0.78, 0.55, r"$\tau_1$")]),
    ("log-normal",            'c', (0, 6),   lambda x: stats.lognorm.pdf(x, s=0.5),
        [(0.42, 0.60, r"$\mu$"), (0.66, 0.68, r"$\sigma$")]),
    ("Pareto",                'c', (1, 6),   lambda x: stats.pareto.pdf(x, b=2),
        [(0.50, 0.78, r"$\mu,\ \sigma$")]),
    ("Weibull",               'c', (0, 4),   lambda x: stats.weibull_min.pdf(x, c=2),
        [(0.68, 0.72, r"$v,\ \lambda$")]),
    ("r-cens. normal",        'c', (-4, 4),  lambda x: stats.norm.pdf(x),
        [(0.22, 0.66, r"$\sigma_2$"), (0.46, 0.40, r"$\mu$"), (0.70, 0.66, r"$\sigma_1$"), (0.86, 0.28, r"$c$")]),
    ("l-cens. normal",        'c', (-4, 4),  lambda x: stats.norm.pdf(x),
        [(0.14, 0.28, r"$c$"), (0.30, 0.66, r"$\sigma_2$"), (0.54, 0.40, r"$\mu$"), (0.78, 0.66, r"$\sigma_1$")]),
    ("Cauchy",                'c', (-5, 5),  lambda x: stats.cauchy.pdf(x),
        [(0.22, 0.55, r"$\gamma_2$"), (0.50, 0.32, r"$x_0$"), (0.78, 0.55, r"$\gamma_1$")]),
    ("half-t",                'c', (0, 6),   lambda x: stats.halfnorm.pdf(x),
        [(0.40, 0.68, r"$\sigma$"), (0.68, 0.55, r"$df$")]),
    ("half-Cauchy",           'c', (0, 8),   lambda x: stats.halfcauchy.pdf(x),
        [(0.45, 0.72, r"$\gamma$")]),
    ("half-normal",           'c', (0, 5),   lambda x: stats.halfnorm.pdf(x),
        [(0.45, 0.72, r"$\sigma$")]),
    # GEV (generalized extreme value); scipy uses c = -xi.
    # LED = Lower End-point Distribution (Frechet, xi>0, c<0): support x >= mu + sigma/c
    ("GEV-LED",               'c', (-2, 8),  lambda x: stats.genextreme.pdf(x, c=-0.5),
        [(0.62, 0.80, r"$\mu,\ \sigma,\ \xi$")]),
    # UED = Upper End-point Distribution (rev. Weibull, xi<0, c>0): support x <= mu + sigma/c
    ("GEV-UED",               'c', (-8, 2),  lambda x: stats.genextreme.pdf(x, c=0.5),
        [(0.32, 0.80, r"$\mu,\ \sigma,\ \xi$")]),

    # --- discrete distributions (PMF stems) ---
    ("binomial",              'd', (0, 12),  lambda k: stats.binom.pmf(k, 12, 0.5),
        [(0.72, 0.72, r"$p,\ n$")]),
    ("beta-binomial",         'd', (0, 10),  lambda k: stats.betabinom.pmf(k, 10, 4, 4),
        [(0.50, 0.68, r"$a,\ b$")]),
    ("Bernoulli",             'd', (0, 1),   lambda k: stats.bernoulli.pmf(k, 0.3),
        [(0.50, 0.70, r"$\theta$")]),
    ("neg. binomial",         'd', (0, 14),  lambda k: stats.nbinom.pmf(k, 5, 0.5),
        [(0.62, 0.72, r"$p,\ r$")]),
    ("categorical",           'd', (0, 5),   lambda k: np.array([.15, .1, .25, .1, .3, .1])[np.asarray(k, int)],
        [(0.50, 0.78, r"$v,\ \lambda$")]),
    ("noncentral hypergeom.", 'd', (0, 10),  lambda k: stats.nhypergeom.pmf(k, 30, 10, 12),
        [(0.46, 0.80, r"$n_1, n_2, m_1, \psi$")]),
    ("Poisson",               'd', (0, 12),  lambda k: stats.poisson.pmf(k, 3),
        [(0.55, 0.78, r"$\lambda$")]),
]

N_COLS = 5
N_ROWS = -(-len(PANELS) // N_COLS)  # ceil
COLOR = "#7FB9DE"

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(12, 2.2 * N_ROWS))

for ax, (title, kind, (lo, hi), fn, labels) in zip(axes.flat, PANELS):
    if kind == 'c':
        x = np.linspace(lo, hi, 400)
        ax.plot(x, fn(x), color=COLOR, lw=2)
        ax.fill_between(x, fn(x), color=COLOR, alpha=0.0)
    else:
        k = np.arange(lo, hi + 1)
        markerline, stemlines, baseline = ax.stem(k, fn(k), basefmt=" ")
        plt.setp(stemlines, color=COLOR, lw=4)
        plt.setp(markerline, visible=False)

    ax.set_title(title, fontsize=10, pad=2, y=0.04)  # label just under the axis
    for px, py, text in labels:
        ax.text(px, py, text, transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.margins(y=0.05)

# blank out any unused trailing cells
for ax in axes.flat[len(PANELS):]:
    ax.axis("off")

fig.tight_layout()
plt.show()
