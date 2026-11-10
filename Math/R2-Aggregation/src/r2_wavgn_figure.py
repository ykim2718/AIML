#!/usr/bin/env python3
"""Draw a schematic that compares the plain and the SST-weighted average of location R2.

Each measurement location j is a circle whose area is SST_j, with an inner circle whose area is SSE_j, so the
blue ring is the part the prediction explains and R2_j is one minus the inner area over the outer area.
Panel (a) draws every location at the same size, as the plain average R2_avgn counts them.
Panel (b) draws every location at its true size and pools them into one circle, as R2_wavgn counts them.

Changelog:
    0.3.0 drop panel (c) and add radius arrows for SST and SSE to the first circle of panel (a)
    0.2.0 add panel (c) for the SSE-weighted average
    0.1.0 draw a circle schematic of an illustrative two-location case instead of bars of the worked example
"""
__author__ = 'yRocket'
__version__ = "0.3.1+20261110"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.patches import Circle, FancyArrowPatch

__all__ = ['location_r2', 'draw_schematic']

FIGSIZE: tuple = (12.0, 5.6)
REFERENCE_WIDTH: float = 12.0    # the width FONT_SIZES were chosen for
FONT_SIZES: dict = {'text': 12.0, 'axis_label': 14.0, 'text_box': 12.0, 'pane_caption': 14.0}  # points
DPI: int = 300
EXPLAINED_COLOR: str = TABLEAU_COLORS['tab:blue']
ERROR_COLOR: str = TABLEAU_COLORS['tab:orange']
EQUAL_RADIUS: float = 1.0        # radius every location gets in panel (a)
AREA_TO_RADIUS: float = 0.13     # radius = AREA_TO_RADIUS * sqrt(area) in panel (b)
# an illustrative case: location 1 varies widely between wafers, location 2 hardly varies
SST: np.ndarray = np.array([100.0, 1.0])
SSE: np.ndarray = np.array([10.0, 0.5])


def location_r2(sst: np.ndarray = None, sse: np.ndarray = None) -> tuple:
    """Return (R2_j array, R2_avgn, R2_wavgn, SSE-weighted average) for SST_j and SSE_j arrays of length N."""
    if sst is None or sse is None:
        raise ValueError('sst and sse are both required.')
    if sst.shape != sse.shape or sst.ndim != 1 or sst.size < 2:
        raise ValueError(f"need two equal 1-D arrays of length N >= 2: {sst.shape=}, {sse.shape=}")
    if np.any(sst <= 0) or np.any(sse <= 0):
        raise ValueError(f"SST_j and SSE_j must be positive: {sst=}, {sse=}")
    r2_j = 1.0 - sse / sst
    return r2_j, r2_j.mean(), 1.0 - sse.sum() / sst.sum(), (sse / sse.sum() * r2_j).sum()


def _location_circle(ax: plt.Axes = None, center: tuple = None, radius: float = None, error_share: float = None,
                     label: str = None, font_size: float = None) -> None:
    """Draw one location: an outer circle of the given radius and an inner circle holding error_share of its area."""
    if error_share > 1.0:
        raise ValueError(f"the inner circle cannot exceed the outer one: {error_share=}")
    ax.add_patch(Circle(center, radius, facecolor=EXPLAINED_COLOR, edgecolor='white', linewidth=2))
    ax.add_patch(Circle(center, radius * np.sqrt(error_share), facecolor=ERROR_COLOR, edgecolor='white', linewidth=1))
    ax.text(center[0], center[1] - radius - 0.25, label, ha='center', va='top', fontsize=font_size)


def _radius_arrows(ax: plt.Axes = None, center: tuple = None, radius: float = None, error_share: float = None,
                   font_size: float = None) -> None:
    """Draw arrows from the center to the outer edge (SST) and to the inner edge (SSE) of one location circle."""
    inner = radius * np.sqrt(error_share)
    for length, angle, label in ((radius, np.pi / 4, 'SST'), (inner, -np.pi / 2, 'SSE')):
        tip = (center[0] + length * np.cos(angle), center[1] + length * np.sin(angle))
        ax.add_patch(FancyArrowPatch(center, tip, arrowstyle='-|>', mutation_scale=12, color='black', linewidth=1.5,
                                     shrinkA=0, shrinkB=0))
    ax.text(center[0] + 0.62 * radius * np.cos(np.pi / 4) - 0.08, center[1] + 0.62 * radius * np.sin(np.pi / 4) + 0.08,
            'SST', ha='right', va='bottom', color='white', fontsize=font_size)
    ax.text(center[0] - 0.05, center[1] - inner - 0.05, 'SSE', ha='right', va='top', color='white', fontsize=font_size)


def draw_schematic(sst: np.ndarray = None, sse: np.ndarray = None, out_path: pathlib.Path = None) -> pathlib.Path:
    """Draw panel (a) without weighting and panel (b) with SST weighting, and save the figure."""
    if out_path is None:
        raise ValueError('out_path is required.')
    r2_j, r2_avgn, r2_wavgn, _ = location_r2(sst, sse)
    scale = FIGSIZE[0] / REFERENCE_WIDTH
    text_size = FONT_SIZES['text_box'] * scale
    plt.rcParams.update({'font.size': FONT_SIZES['text'] * scale, 'axes.labelsize': FONT_SIZES['axis_label'] * scale})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={'width_ratios': [1.0, 1.35]})
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.14, wspace=0.05)
    for k in range(sst.size):    # (a) every location at the same size: each counts 1/N
        _location_circle(ax_a, (2.6 * k, 0.0), EQUAL_RADIUS, sse[k] / sst[k],
                         f"Location {k + 1}\n$R^2_{k + 1}$ = {r2_j[k]:.2f}", text_size)
    _radius_arrows(ax_a, (0.0, 0.0), EQUAL_RADIUS, sse[0] / sst[0], text_size)
    terms = ' + '.join(f"{r:.2f}" for r in r2_j)
    ax_a.text(1.3, 1.55, f"$R^2_{{avgn}}$ = ({terms}) / {sst.size} = {r2_avgn:.2f}", ha='center', fontsize=text_size)
    ax_a.set_xlim(-1.4, 4.0)
    radii = AREA_TO_RADIUS * np.sqrt(sst)
    pooled_radius = AREA_TO_RADIUS * np.sqrt(sst.sum())
    x = 0.0
    for k in range(sst.size):    # (b) every location at its true size: each counts SST_j / sum SST
        x += radii[k] + 0.35
        _location_circle(ax_b, (x, 0.0), radii[k], sse[k] / sst[k], f"Location {k + 1}", text_size)
        x += radii[k] + 0.35
    pooled_x = x + 1.4 + pooled_radius
    ax_b.add_patch(FancyArrowPatch((x - 0.1, 0.0), (pooled_x - pooled_radius - 0.2, 0.0), arrowstyle='-|>',
                                   mutation_scale=20 * scale, color='0.35'))
    ax_b.text((x + pooled_x - pooled_radius) / 2, 0.2, 'pool', ha='center', va='bottom', fontsize=text_size)
    _location_circle(ax_b, (pooled_x, 0.0), pooled_radius, sse.sum() / sst.sum(), 'Pooled', text_size)
    ax_b.text((pooled_x + 0.0) / 2, 1.55,
              f"$R^2_{{wavgn}}$ = 1 - {sse.sum():g} / {sst.sum():g} = {r2_wavgn:.3f}", ha='center', fontsize=text_size)
    ax_b.set_xlim(-0.2, pooled_x + pooled_radius + 0.3)
    for ax in (ax_a, ax_b):
        ax.set_ylim(-2.1, 1.9)
        ax.set_aspect('equal')
        ax.axis('off')
    handles = [Circle((0, 0), 1, facecolor=EXPLAINED_COLOR), Circle((0, 0), 1, facecolor=ERROR_COLOR)]
    fig.legend(handles, ['Explained part: area SST - SSE', 'Error part: area SSE'], loc='lower center',
               bbox_to_anchor=(0.5, 0.085), ncol=2, frameon=False)
    for ax, caption in ((ax_a, '(a) Without weighting'), (ax_b, '(b) With SST weighting')):
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.02, caption, ha='center', fontsize=FONT_SIZES['pane_caption'] * scale)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


parser = argparse.ArgumentParser(prog=pathlib.Path(__file__).name,
                                 description=f"{pathlib.Path(__file__).name} {__version__}: draw the R2_wavgn schematic.")
parser.add_argument('-v', '--version', action='version', version=f"%(prog)s {__version__}")
parser.add_argument('-o', '--out', type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent.parent / 'r2-aggregation-ko_fig' / 'fig1-weighting.png',
                    help='output PNG path')

if __name__ == '__main__':
    args = parser.parse_args()
    saved = draw_schematic(SST, SSE, args.out)
    for label, sse_case in (('given', SSE), ('SSE_2 x4', SSE * np.array([1.0, 4.0]))):
        r2, avgn, wavgn, sse_avg = location_r2(SST, sse_case)
        print(f"{label}: SSE_j={sse_case}, R2_j={r2}, R2_avgn={avgn:.4f}, R2_wavgn={wavgn:.4f}, "
              f"SSE-weighted={sse_avg:.4f}, SSE weights={sse_case / sse_case.sum()}")
    print(f"saved {saved}")
    sys.exit(0)
