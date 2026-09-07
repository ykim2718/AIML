__author__ = 'yRocket'
__version__ = "0.0.0.2026.9.7"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)
"""
Render the three panels that support the polynomial feature expansion document and print every number
the document quotes.

panel (a)   = degree and extrapolation. One-variable fits on x in [0, 1], drawn out to [-0.4, 1.4].
panel (b)   = condition number of the degree-d design matrix, raw offset units against centred units.
panel (c)   = held-out RMSE against degree for OLS and for ridge, on 60 rows of a five-variable
              response whose only non-linear part is one interaction.
"""
import pathlib
from math import comb

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

matplotlib.use('Agg')

N_TRAIN = 60                                     # samples of the one-variable demonstration
N_ROW = 60                                       # rows of the five-variable demonstration
N_VARIABLE = 5                                   # variables of the five-variable demonstration
X_OFFSET = 10.0                                  # location offset that stands for a physical unit
NOISE_SIGMA = 0.08                               # noise of the one-variable demonstration
DEGREE_DRAWN = [2, 5, 9]                         # degrees drawn in panel (a)
DEGREE_RANGE = list(range(1, 9))                 # degrees swept in panel (b)
DEGREE_SWEEP = list(range(1, 5))                 # degrees swept in panel (c)
RIDGE_ALPHA = 1.0                                # ridge penalty on standardized expanded columns
SEED = 0
VARIABLE_COUNT = [5, 10, 20, 50, 100]            # variable counts of the term-count table
TERM_DEGREE = [2, 3]                             # degrees of the term-count table

FIGSIZE: tuple = (13.5, 4.4)
REFERENCE_WIDTH: float = 13.5                    # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 11.0
FONT_SIZE = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
OUT_PATH = pathlib.Path(__file__).parent / 'polynomial-feature-expansion_fig' / 'fig1.png'

INK_COLOR = '#333333'
MUTED_COLOR = '#767676'
DEGREE_COLOR = [TABLEAU_COLORS['tab:blue'], TABLEAU_COLORS['tab:orange'], TABLEAU_COLORS['tab:red']]
RAW_COLOR = TABLEAU_COLORS['tab:red']
CENTRED_COLOR = TABLEAU_COLORS['tab:blue']
OLS_COLOR = TABLEAU_COLORS['tab:red']
RIDGE_COLOR = TABLEAU_COLORS['tab:blue']


def truth_1d(x: np.ndarray) -> np.ndarray:
    """Smooth one-variable response the polynomial fits approximate."""
    return np.sin(2.0 * np.pi * x) * 0.5 + 0.3 * x


def truth_2d(x: np.ndarray) -> np.ndarray:
    """Five-variable response whose only non-linear part is one interaction term."""
    return 1.0 + 2.0 * x[:, 0] - 3.0 * x[:, 1] + 0.5 * x[:, 2] + 4.0 * x[:, 0] * x[:, 1]


def design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """Column-wise powers 1 ... degree of x, with the intercept column in front."""
    return np.column_stack([np.ones_like(x)] + [x ** power for power in range(1, degree + 1)])


def term_count_full(variable: int, degree: int) -> int:
    """Number of monomials of degree 1 ... degree in the given number of variables."""
    return comb(variable + degree, degree) - 1


def term_count_interaction(variable: int, degree: int) -> int:
    """Number of products of distinct variables, up to the given degree."""
    return sum(comb(variable, order) for order in range(1, min(degree, variable) + 1))


rng = np.random.default_rng(SEED)

x_unit = np.sort(rng.uniform(0.0, 1.0, N_TRAIN))
y_1d = truth_1d(x_unit) + rng.normal(0.0, NOISE_SIGMA, N_TRAIN)
x_grid = np.linspace(-0.4, 1.4, 400)

x_pair = rng.uniform(-1.0, 1.0, (N_ROW * 2, N_VARIABLE))
y_2d = truth_2d(x_pair) + rng.normal(0.0, 0.3, N_ROW * 2)
x_fit, x_test = x_pair[:N_ROW], x_pair[N_ROW:]
y_fit, y_test = y_2d[:N_ROW], y_2d[N_ROW:]

fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)

ax = axes[0]
ax.plot(x_grid, truth_1d(x_grid), color=MUTED_COLOR, linewidth=1.2, linestyle='--', label='truth')
ax.scatter(x_unit, y_1d, s=12, color=INK_COLOR, zorder=3, label='train')
for degree, color in zip(DEGREE_DRAWN, DEGREE_COLOR):
    coefficient = np.linalg.lstsq(design_matrix(x_unit, degree), y_1d, rcond=None)[0]
    ax.plot(x_grid, design_matrix(x_grid, degree) @ coefficient, color=color, linewidth=1.4,
            label=f"degree {degree}")
ax.axvspan(0.0, 1.0, color='#000000', alpha=0.05)
ax.set_ylim(-1.6, 1.6)
ax.set_xlabel('x', fontsize=FONT_SIZE, color=INK_COLOR)
ax.set_ylabel('y', fontsize=FONT_SIZE, color=INK_COLOR)
ax.legend(fontsize=FONT_SIZE * 0.85, frameon=False, loc='lower center', ncol=2)

ax = axes[1]
condition_raw, condition_centred = [], []
for degree in DEGREE_RANGE:
    x_shifted = x_unit + X_OFFSET
    condition_raw.append(np.linalg.cond(design_matrix(x_shifted, degree)))
    x_scaled = (x_shifted - x_shifted.mean()) / x_shifted.std()
    condition_centred.append(np.linalg.cond(design_matrix(x_scaled, degree)))
ax.semilogy(DEGREE_RANGE, condition_raw, marker='o', color=RAW_COLOR, linewidth=1.4, label='raw x + 10')
ax.semilogy(DEGREE_RANGE, condition_centred, marker='s', color=CENTRED_COLOR, linewidth=1.4,
            label='centred and scaled')
ax.set_xlabel('degree', fontsize=FONT_SIZE, color=INK_COLOR)
ax.set_ylabel('condition number', fontsize=FONT_SIZE, color=INK_COLOR)
ax.set_xticks(DEGREE_RANGE)
ax.legend(fontsize=FONT_SIZE * 0.85, frameon=False, loc='upper left')

ax = axes[2]
rmse_ols, rmse_ridge = [], []
for degree in DEGREE_SWEEP:
    for model, holder in ((LinearRegression(), rmse_ols), (Ridge(alpha=RIDGE_ALPHA), rmse_ridge)):
        pipeline = make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree, include_bias=False),
                                 StandardScaler(), model)
        pipeline.fit(x_fit, y_fit)
        holder.append(root_mean_squared_error(y_test, pipeline.predict(x_test)))
ax.plot(DEGREE_SWEEP, rmse_ols, marker='o', color=OLS_COLOR, linewidth=1.4, label='OLS')
ax.plot(DEGREE_SWEEP, rmse_ridge, marker='s', color=RIDGE_COLOR, linewidth=1.4,
        label=f"ridge, alpha = {RIDGE_ALPHA:g}")
ax.set_xlabel('degree', fontsize=FONT_SIZE, color=INK_COLOR)
ax.set_ylabel('held-out RMSE', fontsize=FONT_SIZE, color=INK_COLOR)
ax.set_xticks(DEGREE_SWEEP)
ax.legend(fontsize=FONT_SIZE * 0.85, frameon=False, loc='lower right')

for ax in axes:
    ax.tick_params(labelsize=FONT_SIZE * 0.9, colors=MUTED_COLOR)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#d6d6d6')

fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.22, wspace=0.28)
for ax, label in zip(axes, ('(a)', '(b)', '(c)')):
    box = ax.get_position()
    fig.text(box.x0 + box.width / 2.0, 0.045, label, ha='center', va='center', fontsize=FONT_SIZE,
             color=INK_COLOR)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=300)

x_shifted = x_unit + X_OFFSET
x_centred = x_shifted - x_shifted.mean()
print("correlation between a variable and its square")
print(f"  raw x + {X_OFFSET:g} : {np.corrcoef(x_shifted, x_shifted ** 2)[0, 1]:.4f}")
print(f"  centred      : {np.corrcoef(x_centred, x_centred ** 2)[0, 1]:.4f}")

print("\ncondition number of the design matrix")
print(f"{'degree':>7}{'raw':>14}{'centred':>14}")
for degree, raw, centred in zip(DEGREE_RANGE, condition_raw, condition_centred):
    print(f"{degree:>7}{raw:>14.3e}{centred:>14.3e}")

print("\nheld-out RMSE against degree")
print(f"{'degree':>7}{'OLS':>10}{'ridge':>10}")
for degree, ols, ridge in zip(DEGREE_SWEEP, rmse_ols, rmse_ridge):
    print(f"{degree:>7}{ols:>10.3f}{ridge:>10.3f}")

print("\nterm count without the bias column")
print(f"{'n':>5}" + ''.join(f"{'d=' + str(degree) + ' full':>14}{'d=' + str(degree) + ' inter':>15}"
                            for degree in TERM_DEGREE))
for variable in VARIABLE_COUNT:
    row = ''.join(f"{term_count_full(variable, degree):>14}{term_count_interaction(variable, degree):>15}"
                  for degree in TERM_DEGREE)
    print(f"{variable:>5}" + row)

print(f"\nsaved {OUT_PATH}")
