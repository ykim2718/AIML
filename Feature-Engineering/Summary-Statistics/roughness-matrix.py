__author__ = 'yRocket'
__version__ = "0.0.1.2026.8.14"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)
"""
Render a 5 x 5 matrix chart of sine traces with additive white noise and report slsr on each panel.

slsr = sqrt(MSSD / 2) / s, where MSSD is the mean square successive difference and s the sample
standard deviation. The ratio equals sqrt(1 - rho_1), so it reads 1 for white noise, below 1 for a smooth
trace and above 1 for sample-level oscillation.
"""
import pathlib

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

matplotlib.use('Agg')

N_SAMPLE = 300                                   # samples per trace
AMPLITUDE = 1.0                                  # sine amplitude
CYCLE_LIST = [1, 2, 5, 12, 30]                   # sine cycles over the trace, one per column
NOISE_LIST = [0.0, 0.05, 0.2, 0.6, 2.0]          # noise sigma relative to the amplitude, one per row
SEED = 0
OUT_PATH = pathlib.Path(__file__).with_suffix('.png')

LINE_COLOR = TABLEAU_COLORS['tab:blue']
INK_COLOR = '#333333'
MUTED_COLOR = '#767676'


def slsr(x: np.ndarray) -> float:
    """Short-term sigma over sample standard deviation."""
    dx = np.diff(x)
    mssd = np.mean(dx ** 2)
    return float(np.sqrt(mssd / 2.0) / np.std(x, ddof=1))


def make_trace(cycle: int = 1, noise: float = 0.0, rng: np.random.Generator = None) -> np.ndarray:
    """Sine of the given cycle count over the trace, plus white noise of the given sigma."""
    phase = 2.0 * np.pi * cycle * np.arange(N_SAMPLE) / N_SAMPLE
    return AMPLITUDE * np.sin(phase) + noise * rng.standard_normal(N_SAMPLE)


rng = np.random.default_rng(SEED)
fig, axes = plt.subplots(nrows=len(NOISE_LIST), ncols=len(CYCLE_LIST), figsize=(13.0, 9.5), sharex=True)

for row, noise in enumerate(NOISE_LIST):
    for col, cycle in enumerate(CYCLE_LIST):
        ax = axes[row, col]
        x = make_trace(cycle=cycle, noise=noise, rng=rng)
        ratio = slsr(x)

        ax.plot(x, color=LINE_COLOR, linewidth=0.9)
        ax.text(0.04, 0.95, f"$\\sigma_{{st}}/s$ = {ratio:.2f}", transform=ax.transAxes,
                ha='left', va='top', fontsize=10, color=INK_COLOR)

        span = x.max() - x.min()                          # headroom keeps the label clear of the trace
        ax.set_ylim(x.min() - 0.06 * span, x.max() + 0.42 * span)
        ax.set_yticks([])
        ax.set_xticks([])
        for side in ('top', 'right', 'left', 'bottom'):
            ax.spines[side].set_color('#d6d6d6')

        if row == 0:
            label = "1 cycle" if cycle == 1 else f"{cycle} cycles"
            ax.set_title(label, fontsize=11, color=INK_COLOR, pad=8)
        if col == 0:
            ax.set_ylabel(f"noise {noise:.2f}", fontsize=11, color=MUTED_COLOR, labelpad=10)

fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.03, wspace=0.08, hspace=0.12)
fig.savefig(OUT_PATH, dpi=300)

print(f"{'noise':>7}" + ''.join(f"{cycle:>10}" for cycle in CYCLE_LIST))
rng = np.random.default_rng(SEED)
for noise in NOISE_LIST:
    row_value = [slsr(make_trace(cycle=cycle, noise=noise, rng=rng)) for cycle in CYCLE_LIST]
    print(f"{noise:>7.2f}" + ''.join(f"{value:>10.2f}" for value in row_value))
print(f"saved {OUT_PATH}")
