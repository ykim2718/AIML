__author__ = 'yRocket'
__version__ = "0.1.0.2026.8.15"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)
"""
Render a 5 x 5 matrix chart of sine traces with additive white noise and report the three roughness
features on each panel.

slsr        = sqrt(MSSD / 2) / s, where MSSD is the mean square successive difference and s the sample
              standard deviation. It equals sqrt(1 - rho_1), so it reads 1 for white noise.
zcr         = center-line crossing rate with a hysteresis dead band of 2 * sigma_st.
cycle_count = number of peaks whose prominence reaches 3 * sigma_st.
"""
import pathlib

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from scipy.signal import find_peaks

matplotlib.use('Agg')

N_SAMPLE = 300                                   # samples per trace
AMPLITUDE = 1.0                                  # sine amplitude
CYCLE_LIST = [1, 2, 5, 12, 30]                   # sine cycles over the trace, one per column
NOISE_LIST = [0.0, 0.05, 0.2, 0.6, 2.0]          # noise sigma relative to the amplitude, one per row
DEAD_BAND_FACTOR = 2.0                           # hysteresis width in units of sigma_st
PROMINENCE_FACTOR = 3.0                          # peak threshold in units of sigma_st
SEED = 0
DOC_STEM = 'semiconductor-equipment-trace-non-integral-summary-statistics'
OUT_PATH = pathlib.Path(__file__).parent / f"{DOC_STEM}_fig" / 'fig1.png'

LINE_COLOR = TABLEAU_COLORS['tab:blue']
INK_COLOR = '#333333'
MUTED_COLOR = '#767676'


def short_term_sigma(x: np.ndarray) -> float:
    """SPC short-term sigma, sqrt(MSSD / 2)."""
    return float(np.sqrt(np.mean(np.diff(x) ** 2) / 2.0))


def hysteresis_sign(x: np.ndarray, center: float = 0.0, dead_band: float = 0.0) -> np.ndarray:
    """Sign of x - center, holding the previous state while inside the dead band."""
    d = x - center
    state = np.empty(len(d), dtype=int)
    current = 1 if d[0] >= 0 else -1
    for i, value in enumerate(d):
        if value > dead_band / 2:
            current = 1
        elif value < -dead_band / 2:
            current = -1
        state[i] = current
    return state


def roughness_features(x: np.ndarray) -> dict:
    """Return slsr, zcr and cycle_count with the thresholds derived from the short-term sigma."""
    sigma_st = short_term_sigma(x)
    g = hysteresis_sign(x=x, center=float(x.mean()), dead_band=DEAD_BAND_FACTOR * sigma_st)
    peak_index, _ = find_peaks(x, prominence=PROMINENCE_FACTOR * sigma_st)
    return {
        'slsr': sigma_st / float(np.std(x, ddof=1)),
        'zcr': float(np.mean(g[:-1] != g[1:])),
        'cycle_count': len(peak_index),
    }


def make_trace(cycle: int = 1, noise: float = 0.0, rng: np.random.Generator = None) -> np.ndarray:
    """Sine of the given cycle count over the trace, plus white noise of the given sigma."""
    phase = 2.0 * np.pi * cycle * np.arange(N_SAMPLE) / N_SAMPLE
    return AMPLITUDE * np.sin(phase) + noise * rng.standard_normal(N_SAMPLE)


rng = np.random.default_rng(SEED)
fig, axes = plt.subplots(nrows=len(NOISE_LIST), ncols=len(CYCLE_LIST), figsize=(13.0, 11.0), sharex=True)
panel_feature = {}                                    # (noise, cycle) -> features drawn on that panel

for row, noise in enumerate(NOISE_LIST):
    for col, cycle in enumerate(CYCLE_LIST):
        ax = axes[row, col]
        x = make_trace(cycle=cycle, noise=noise, rng=rng)
        feature = roughness_features(x)
        panel_feature[(noise, cycle)] = feature
        label = (f"slsr {feature['slsr']:.2f}\n"
                 f"zcr {feature['zcr']:.3f}\n"
                 f"cycle_count {feature['cycle_count']}")

        ax.plot(x, color=LINE_COLOR, linewidth=0.9)
        ax.text(0.04, 0.96, label, transform=ax.transAxes, ha='left', va='top',
                fontsize=9, color=INK_COLOR, linespacing=1.4)

        span = x.max() - x.min()                          # headroom keeps the label clear of the trace
        ax.set_ylim(x.min() - 0.06 * span, x.max() + 0.95 * span)
        ax.set_yticks([])
        ax.set_xticks([])
        for side in ('top', 'right', 'left', 'bottom'):
            ax.spines[side].set_color('#d6d6d6')

        if row == 0:
            title = "1 cycle" if cycle == 1 else f"{cycle} cycles"
            ax.set_title(title, fontsize=11, color=INK_COLOR, pad=8)
        if col == 0:
            ax.set_ylabel(f"noise {noise:.2f}", fontsize=11, color=MUTED_COLOR, labelpad=10)

fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.02, wspace=0.08, hspace=0.12)
fig.savefig(OUT_PATH, dpi=300)

for name in ('slsr', 'zcr', 'cycle_count'):
    print(f"\n{name}")
    print(f"{'noise':>7}" + ''.join(f"{cycle:>10}" for cycle in CYCLE_LIST))
    for noise in NOISE_LIST:
        row_value = [panel_feature[(noise, cycle)][name] for cycle in CYCLE_LIST]
        print(f"{noise:>7.2f}" + ''.join(f"{value:>10.3f}" for value in row_value))
print(f"\nsaved {OUT_PATH}")
