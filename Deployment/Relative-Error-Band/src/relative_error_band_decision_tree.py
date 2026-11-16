#!/usr/bin/env python3
"""Draw the decision tree for choosing the p95 method as an image for documents and presentation slides.

The tree asks whether raw pairs are available, whether the relative error is normal and whether its mean is zero, and
ends at the equation and the section to use. The figure is saved as PNG (300 dpi) and SVG; the SVG can be inserted in
a slide and scaled without loss.

Requires matplotlib.
"""
__author__ = 'yRocket'
__version__ = "0.0.1+20261116"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib
import sys
from dataclasses import dataclass

from matplotlib import colors as mcolors
from matplotlib import patches
from matplotlib import pyplot as plt

__all__ = ['Node', 'NODES', 'EDGES', 'NOTE', 'draw']

FIGURE_STEM: str = 'relative-error-band-fig1-decision-tree'
FIGURE_SUFFIXES: tuple = ('.png', '.svg')
FIGSIZE: tuple = (13.33, 7.5)  # 16:9 slide
REFERENCE_WIDTH: float = 13.33
FONT_SIZES: dict = {'text': 12.0, 'axis_label': 14.0, 'text_box': 14.0, 'pane_caption': 14.0}  # points
CANVAS: tuple = (100.0, 56.0)  # data units of the drawing area (width, height)
BOX_PAD: float = 0.8
DPI: int = 300
FILL_ALPHA: float = 0.15
STYLE_COLORS: dict = {'question': mcolors.TABLEAU_COLORS['tab:blue'],
                      'exact': mcolors.TABLEAU_COLORS['tab:green'],
                      'bound': mcolors.TABLEAU_COLORS['tab:orange'],
                      'note': mcolors.TABLEAU_COLORS['tab:gray']}
EDGE_COLOR: str = mcolors.TABLEAU_COLORS['tab:gray']


@dataclass(frozen=True)
class Node:
    """One box of the tree: centre (x, y), size (w, h) in canvas units, text and style key of STYLE_COLORS."""
    x: float
    y: float
    w: float
    h: float
    text: str
    style: str


NODES: dict = {
    'raw': Node(50, 51, 30, 5, r'Raw pairs $(x_i, y_i)$ available?', 'question'),
    'empirical': Node(17, 37, 30, 9, '$p_{95}$ = 0.95 quantile of $|e_i|$\n'
                                     'Eq. (8); rank $k$ for a new pair, Eq. (9)\nSec. 4.2', 'exact'),
    'normal': Node(65, 37, 28, 5, r'Is $e_i$ normally distributed?', 'question'),
    'mean': Node(50, 23, 22, 5, r'Is the mean $\mu_e = 0$?', 'question'),
    'markov': Node(84, 23, 26, 9, r'$p_{95} \leq \mathrm{RMSRE}\,/\,\sqrt{0.05}$' '\n'
                                  'upper bound only, Eq. (12)\nSec. 4.5', 'bound'),
    'zero_mean': Node(37, 7.5, 22, 9, r'$p_{95} = 1.96\ \mathrm{RMSRE}$' '\nEq. (10)\nSec. 4.3', 'exact'),
    'nonzero_mean': Node(63, 7.5, 22, 9, 'solve Eq. (11)\n' r'with $\mu_e$ and $\sigma_e$' '\nSec. 4.4', 'exact'),
}
EDGES: tuple = (('raw', 'empirical', 'yes'), ('raw', 'normal', 'no'),
                ('normal', 'mean', 'yes'), ('normal', 'markov', 'no / unknown'),
                ('mean', 'zero_mean', 'yes'), ('mean', 'nonzero_mean', 'no'))
NOTE: Node = Node(17, 21, 30, 7, 'Only $R^2$ and RMSE known:\n$p_{95}$ cannot be determined.\n'
                                r'With min $y_i$: RMSRE $\leq$ RMSE / min $y_i$,' '\nthen Eq. (12), Sec. 4.6',
                  'note')
LABEL_OFFSET: float = 1.0  # horizontal gap between an edge and its label


def _box(ax: plt.Axes = None, node: Node = None, fontsize: float = None) -> None:
    """Draw one rounded box with its centred text."""
    color = STYLE_COLORS[node.style]
    ax.add_patch(patches.FancyBboxPatch(
        (node.x - node.w / 2, node.y - node.h / 2), node.w, node.h, boxstyle=f"round,pad={BOX_PAD}",
        facecolor=mcolors.to_rgba(color, FILL_ALPHA), edgecolor=color, linewidth=1.8))
    ax.text(node.x, node.y, node.text, ha='center', va='center', fontsize=fontsize, linespacing=1.5)


def draw(paths: list = None) -> None:
    """Save the decision tree to every path in paths; the file type follows each suffix."""
    if not paths:
        raise ValueError('paths is required.')
    scale = FIGSIZE[0] / REFERENCE_WIDTH
    plt.rcParams.update({'font.size': FONT_SIZES['text'] * scale, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(0, CANVAS[0])
    ax.set_ylim(0, CANVAS[1])
    ax.axis('off')
    for parent, child, label in EDGES:
        a, b = NODES[parent], NODES[child]
        start, end = (a.x, a.y - a.h / 2 - BOX_PAD), (b.x, b.y + b.h / 2 + BOX_PAD)
        ax.annotate('', xy=end, xytext=start,
                    arrowprops={'arrowstyle': '-|>', 'color': EDGE_COLOR, 'linewidth': 1.8, 'mutation_scale': 18,
                                'shrinkA': 0, 'shrinkB': 0})
        side = 1.0 if b.x > a.x else -1.0  # put the label on the outer side of the edge
        ax.text((start[0] + end[0]) / 2 + side * LABEL_OFFSET, (start[1] + end[1]) / 2, label,
                ha='left' if side > 0 else 'right', va='center',
                fontsize=FONT_SIZES['text'] * scale, color=EDGE_COLOR, fontweight='bold')
    for node in NODES.values():
        _box(ax=ax, node=node, fontsize=FONT_SIZES['text_box'] * scale)
    _box(ax=ax, node=NOTE, fontsize=FONT_SIZES['text'] * scale)
    fig.tight_layout(pad=0.2)
    for path in paths:
        fig.savefig(path, dpi=DPI)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{pathlib.Path(__file__).name} {__version__}\n"
                    'Decision tree for choosing the p95 method, saved as PNG and SVG.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--output-folder', type=pathlib.Path, help='folder for the PNG and SVG files')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    if args.output_folder is None or not args.output_folder.is_dir():
        parser.error(f"--output-folder must be an existing folder: {args.output_folder}")
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    PATHS = [ARGS.output_folder / f"{FIGURE_STEM}{SUFFIX}" for SUFFIX in FIGURE_SUFFIXES]
    draw(paths=PATHS)
    for PATH in PATHS:
        print(f"wrote {PATH}")
