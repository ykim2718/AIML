#!/usr/bin/env python3
"""Draw the sums of squares of the worked example to show what R2_wavgn and R2_avgn average.

Panel (a) stacks, for each measurement location, the explained part SST_j - SSE_j and the error part SSE_j.
Panel (b) pours the N locations into one bar, whose explained share is R2_wavgn.
"""
__author__ = 'yRocket'
__version__ = "0.0.0+20261110"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

__all__ = ['location_sums', 'draw_figure']

FIGSIZE: tuple = (10.0, 5.0)
REFERENCE_WIDTH: float = 10.0    # the width FONT_SIZES were chosen for
FONT_SIZES: dict = {'text': 12.0, 'axis_label': 14.0, 'text_box': 12.0, 'pane_caption': 14.0}  # points
DPI: int = 300
BAR_WIDTH: float = 0.5
EXPLAINED_COLOR: str = TABLEAU_COLORS['tab:blue']
ERROR_COLOR: str = TABLEAU_COLORS['tab:orange']
# Table 2 of the document: rows are wafers i, columns are measurement locations j
MEASURED: np.ndarray = np.array([[10.0, 12.0], [13.0, 15.0], [16.0, 20.0]])
PREDICTED: np.ndarray = np.array([[11.0, 12.0], [13.0, 14.0], [15.0, 19.0]])


def location_sums(measured: np.ndarray = None, predicted: np.ndarray = None) -> tuple:
    """Return (SST_j, SSE_j) as two arrays of length N for an M x N measured and predicted array."""
    if measured is None or predicted is None:
        raise ValueError('measured and predicted are both required.')
    if measured.shape != predicted.shape or measured.ndim != 2 or measured.shape[0] < 2:
        raise ValueError(f"need two equal M x N arrays with M >= 2: {measured.shape=}, {predicted.shape=}")
    sst = ((measured - measured.mean(axis=0)) ** 2).sum(axis=0)
    sse = ((measured - predicted) ** 2).sum(axis=0)
    if np.any(sst <= 0):
        raise ValueError(f"SST_j must be positive at every location: {sst=}")
    return sst, sse


def draw_figure(sst: np.ndarray = None, sse: np.ndarray = None, out_path: pathlib.Path = None) -> pathlib.Path:
    """Draw panel (a) per location and panel (b) pooled over the N locations, and save the figure."""
    if sst is None or sse is None or out_path is None:
        raise ValueError('sst, sse and out_path are all required.')
    scale = FIGSIZE[0] / REFERENCE_WIDTH
    plt.rcParams.update({'font.size': FONT_SIZES['text'] * scale, 'axes.labelsize': FONT_SIZES['axis_label'] * scale})
    r2_j = 1.0 - sse / sst
    r2_avgn = r2_j.mean()
    r2_wavgn = 1.0 - sse.sum() / sst.sum()
    top = sst.sum() * 1.18
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True, gridspec_kw={'width_ratios': [len(sst), 1.6]})
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.24, wspace=0.08)
    x = np.arange(len(sst))
    ax_a.bar(x, sst - sse, BAR_WIDTH, color=EXPLAINED_COLOR, label='Explained  SST - SSE')
    ax_a.bar(x, sse, BAR_WIDTH, bottom=sst - sse, color=ERROR_COLOR, label='Error  SSE')
    for k in range(len(sst)):
        ax_a.text(x[k], sst[k] + top * 0.02, f"$R^2_{k + 1}$ = {r2_j[k]:.4f}", ha='center', va='bottom',
                  fontsize=FONT_SIZES['text_box'] * scale)
    ax_a.text(x.mean(), top * 0.92, f"$R^2_{{avgn}}$ = mean = {r2_avgn:.4f}", ha='center', va='top',
              fontsize=FONT_SIZES['text_box'] * scale)
    ax_a.set_xticks(x, [f"Location {k + 1}" for k in range(len(sst))])
    ax_a.set_ylabel('Sum of squares')
    ax_a.set_ylim(0.0, top)
    ax_a.legend(loc='center left', frameon=False)
    bottom = 0.0
    for k in range(len(sst)):    # stack each location's explained part so its share of the pooled bar is visible
        ax_b.bar(0, sst[k] - sse[k], BAR_WIDTH, bottom=bottom, color=EXPLAINED_COLOR, edgecolor='white', linewidth=2)
        ax_b.text(0, bottom + (sst[k] - sse[k]) / 2, f"Location {k + 1}", ha='center', va='center', color='white',
                  fontsize=FONT_SIZES['text_box'] * scale)
        bottom += sst[k] - sse[k]
    ax_b.bar(0, sse.sum(), BAR_WIDTH, bottom=bottom, color=ERROR_COLOR, edgecolor='white', linewidth=2)
    ax_b.text(0, sst.sum() + top * 0.02, f"$R^2_{{wavgn}}$ = {r2_wavgn:.4f}", ha='center', va='bottom',
              fontsize=FONT_SIZES['text_box'] * scale)
    ax_b.set_xticks([0], ['Pooled'])
    ax_b.set_xlim(-0.8, 0.8)
    for ax in (ax_a, ax_b):
        ax.spines[['top', 'right']].set_visible(False)
    for ax, caption in ((ax_a, '(a) Each measurement location'), (ax_b, '(b) N locations pooled')):
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.04, caption, ha='center', fontsize=FONT_SIZES['pane_caption'] * scale)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


parser = argparse.ArgumentParser(prog=pathlib.Path(__file__).name,
                                 description=f"{pathlib.Path(__file__).name} {__version__}: draw the R2_wavgn figure.")
parser.add_argument('-v', '--version', action='version', version=f"%(prog)s {__version__}")
parser.add_argument('-o', '--out', type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent.parent / 'r2-aggregation-ko_fig' / 'fig1.png',
                    help='output PNG path')

if __name__ == '__main__':
    args = parser.parse_args()
    sst_j, sse_j = location_sums(MEASURED, PREDICTED)
    saved = draw_figure(sst_j, sse_j, args.out)
    print(f"SST_j={sst_j}, SSE_j={sse_j}, R2_j={1 - sse_j / sst_j}, "
          f"R2_avgn={(1 - sse_j / sst_j).mean():.4f}, R2_wavgn={1 - sse_j.sum() / sst_j.sum():.4f}")
    print(f"saved {saved}")
    sys.exit(0)
