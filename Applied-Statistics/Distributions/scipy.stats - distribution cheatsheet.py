"""
scipy.stats - distribution cheatsheet.py
A grid of common probability distributions, after Rasmus Baath (2012).

Continuous distributions are drawn as PDF curves, discrete ones as stems (PMF).
Each panel is labelled only with its parameter symbols, like the original poster.

Label positions are given in axes fractions. A y-value written as ("c", dy)
means "anchor to the curve": the label is placed dy above the curve at that x,
so scale symbols hug the line instead of floating in empty space.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

C = lambda dy=0.06: ("c", dy)  # curve-anchored y with offset dy

# (title, kind, x-range, callable -> y, [(x, y, label), ...])
#   kind   = 'c' for continuous PDF, 'd' for discrete PMF
#   labels = parameter symbols; y is a fraction, or C(dy) to hug the curve
PANELS = [
    # --- continuous distributions (PDF curves) ---
    ("normal",                'c', (-4, 4),  lambda x: stats.norm.pdf(x, 0, 1),
        [(0.36, C(), r"$\sigma_2$"), (0.50, 0.42, r"$\mu$"), (0.64, C(), r"$\sigma_1$")]),
    ("t distrib.",            'c', (-4, 4),  lambda x: stats.t.pdf(x, df=3),
        [(0.22, C(0.05), r"$\sigma_2$"), (0.37, C(0.05), r"$df_2$"), (0.50, 0.34, r"$\mu$"),
         (0.63, C(0.05), r"$df_1$"), (0.78, C(0.05), r"$\sigma_1$")]),
    ("uniform",               'c', (-4, 4),  lambda x: stats.uniform.pdf(x, -2, 4),
        [(0.27, C(0.03), r"$L$"), (0.73, C(0.03), r"$H$")]),
    ("beta",                  'c', (0, 1),   lambda x: stats.beta.pdf(x, 2, 2),
        [(0.50, C(0.03), r"$a,\ b$")]),
    ("gamma",                 'c', (0, 12),  lambda x: stats.gamma.pdf(x, a=2, scale=1.5),
        [(0.42, C(), r"$S,\ R$")]),
    ("inv-gamma",             'c', (0, 6),   lambda x: stats.invgamma.pdf(x, a=3, scale=2),
        [(0.22, C(), r"$\alpha,\ \beta$")]),
    ("folded t",              'c', (0, 6),   lambda x: stats.foldnorm.pdf(x, 0.1),
        [(0.12, C(0.05), r"$\mu$"), (0.34, C(0.05), r"$\sigma$"), (0.58, C(0.05), r"$df$")]),
    ("chi-square",            'c', (0, 12),  lambda x: stats.chi2.pdf(x, df=3),
        [(0.40, C(), r"$k$")]),
    ("noncentral chi-square", 'c', (0, 20),  lambda x: stats.ncx2.pdf(x, df=4, nc=2),
        [(0.38, C(), r"$k,\ \lambda$")]),
    ("double exp.",           'c', (-4, 4),  lambda x: stats.laplace.pdf(x),
        [(0.36, C(0.05), r"$\tau_2$"), (0.50, 0.42, r"$\mu$"), (0.64, C(0.05), r"$\tau_1$")]),
    ("exponential",           'c', (0, 6),   lambda x: stats.expon.pdf(x),
        [(0.18, C(), r"$\lambda$")]),
    ("shifted exp.",          'c', (0, 6),   lambda x: stats.expon.pdf(x, loc=1),
        [(0.30, C(), r"$\lambda$")]),
    ("F dist.",               'c', (0, 5),   lambda x: stats.f.pdf(x, 5, 10),
        [(0.40, C(), r"$df_1,\ df_2$")]),
    ("gen. gamma",            'c', (0, 6),   lambda x: stats.gengamma.pdf(x, a=3, c=2),
        [(0.42, C(), r"$r,\ \lambda,\ b$")]),
    ("logistic",              'c', (-5, 5),  lambda x: stats.logistic.pdf(x),
        [(0.36, C(0.05), r"$\tau_2$"), (0.50, 0.38, r"$\mu$"), (0.64, C(0.05), r"$\tau_1$")]),
    ("log-normal",            'c', (0, 6),   lambda x: stats.lognorm.pdf(x, s=0.5),
        [(0.20, C(0.05), r"$\mu$"), (0.42, C(0.05), r"$\sigma$")]),
    ("Pareto",                'c', (1, 6),   lambda x: stats.pareto.pdf(x, b=2),
        [(0.08, C(), r"$\mu,\ \sigma$")]),
    ("Weibull",               'c', (0, 4),   lambda x: stats.weibull_min.pdf(x, c=2),
        [(0.45, C(), r"$v,\ \lambda$")]),
    ("r-cens. normal",        'c', (-4, 4),  lambda x: stats.norm.pdf(x),
        [(0.36, C(), r"$\sigma_2$"), (0.48, 0.42, r"$\mu$"), (0.62, C(), r"$\sigma_1$"), (0.82, 0.20, r"$c$")]),
    ("l-cens. normal",        'c', (-4, 4),  lambda x: stats.norm.pdf(x),
        [(0.18, 0.20, r"$c$"), (0.38, C(), r"$\sigma_2$"), (0.52, 0.42, r"$\mu$"), (0.66, C(), r"$\sigma_1$")]),
    ("Cauchy",                'c', (-5, 5),  lambda x: stats.cauchy.pdf(x),
        [(0.36, C(0.05), r"$\gamma_2$"), (0.50, 0.40, r"$x_0$"), (0.64, C(0.05), r"$\gamma_1$")]),
    ("half-t",                'c', (0, 6),   lambda x: stats.halfnorm.pdf(x),
        [(0.20, C(0.05), r"$\sigma$"), (0.45, C(0.05), r"$df$")]),
    ("half-Cauchy",           'c', (0, 8),   lambda x: stats.halfcauchy.pdf(x),
        [(0.18, C(), r"$\gamma$")]),
    ("half-normal",           'c', (0, 5),   lambda x: stats.halfnorm.pdf(x),
        [(0.20, C(), r"$\sigma$")]),
    # GEV (generalized extreme value); scipy uses c = -xi.
    # LED = Lower End-point Distribution (Frechet, xi>0, c<0): support x >= mu + sigma/c
    ("GEV-LED",               'c', (-2, 8),  lambda x: stats.genextreme.pdf(x, c=-0.5),
        [(0.42, C(), r"$\mu,\ \sigma,\ \xi$")]),
    # UED = Upper End-point Distribution (rev. Weibull, xi<0, c>0): support x <= mu + sigma/c
    ("GEV-UED",               'c', (-8, 2),  lambda x: stats.genextreme.pdf(x, c=0.5),
        [(0.80, C(), r"$\mu,\ \sigma,\ \xi$")]),

    # --- discrete distributions (PMF stems) ---
    ("binomial",              'd', (0, 12),  lambda k: stats.binom.pmf(k, 12, 0.5),
        [(0.74, 0.60, r"$p,\ n$")]),
    ("beta-binomial",         'd', (0, 10),  lambda k: stats.betabinom.pmf(k, 10, 4, 4),
        [(0.50, 0.58, r"$a,\ b$")]),
    ("Bernoulli",             'd', (0, 1),   lambda k: stats.bernoulli.pmf(k, 0.3),
        [(0.50, 0.60, r"$\theta$")]),
    ("neg. binomial",         'd', (0, 14),  lambda k: stats.nbinom.pmf(k, 5, 0.5),
        [(0.60, 0.58, r"$p,\ r$")]),
    ("categorical",           'd', (0, 5),   lambda k: np.array([.15, .1, .25, .1, .3, .1])[np.asarray(k, int)],
        [(0.50, 0.70, r"$v,\ \lambda$")]),
    ("noncentral hypergeom.", 'd', (0, 10),  lambda k: stats.nhypergeom.pmf(k, 30, 10, 12),
        [(0.46, 0.72, r"$n_1, n_2, m_1, \psi$")]),
    ("Poisson",               'd', (0, 12),  lambda k: stats.poisson.pmf(k, 3),
        [(0.38, 0.60, r"$\lambda$")]),
]

N_COLS = 5
N_ROWS = -(-len(PANELS) // N_COLS)  # ceil
COLOR = "#7FB9DE"

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(12, 2.2 * N_ROWS))

for ax, (title, kind, (lo, hi), fn, labels) in zip(axes.flat, PANELS):
    if kind == 'c':
        x = np.linspace(lo, hi, 400)
        ax.plot(x, fn(x), color=COLOR, lw=2)
    else:
        k = np.arange(lo, hi + 1)
        markerline, stemlines, baseline = ax.stem(k, fn(k), basefmt=" ")
        plt.setp(stemlines, color=COLOR, lw=4)
        plt.setp(markerline, visible=False)

    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.margins(y=0.05)
    ax.set_title(title, fontsize=10, pad=2, y=0.04)  # label just under the axis

    ymin, ymax = ax.get_ylim()
    for px, py, text in labels:
        if isinstance(py, tuple):          # curve-anchored: ("c", dy)
            val = float(fn(np.array([lo + px * (hi - lo)]))[0])
            py = max((val - ymin) / (ymax - ymin) + py[1], 0.28)  # keep clear of title
        ax.text(px, py, text, transform=ax.transAxes,
                ha="center", va="center", fontsize=11)

# blank out any unused trailing cells
for ax in axes.flat[len(PANELS):]:
    ax.axis("off")

fig.tight_layout()

# save a PNG next to this script
out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "scipy.stats - distribution cheatsheet.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print("saved:", out_png)

plt.show()
