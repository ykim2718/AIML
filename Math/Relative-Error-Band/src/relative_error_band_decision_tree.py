#!/usr/bin/env python3
"""Draw the decision tree for choosing the p95 method with Graphviz, for documents and presentation slides.

The tree asks whether raw pairs are available, whether the relative error is normal and whether its mean is zero, and
ends at the equation and the section to use. The tree itself is the Graphviz source
relative_error_band_decision_tree.dot next to this script. It is rendered by the cairo backend of `dot`, so the SVG
stores every glyph as a path and does not depend on the fonts of the viewer, and it is saved as SVG, or as PNG
(300 dpi) when the SVG is larger.

Requires Graphviz (the `dot` command), e.g. `apt-get install graphviz`.

Changelog:
    0.1.0  One file, SVG or PNG by size, instead of both; text is drawn as paths.
    0.2.0  Drawn with Graphviz from relative_error_band_decision_tree.dot instead of matplotlib.
"""
__author__ = 'yRocket'
__version__ = "0.2.0+20261116"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib
import shutil
import subprocess
import sys

from relative_error_band_figure import write_smaller

__all__ = ['render', 'draw']

DOT_SOURCE: pathlib.Path = pathlib.Path(__file__).with_name('relative_error_band_decision_tree.dot')
FIGURE_STEM: str = 'relative-error-band-fig1-decision-tree'  # suffix .svg or .png chosen by size
DPI: int = 300
DOT_FORMATS: dict = {'svg': 'svg:cairo', 'png': 'png:cairo'}  # cairo draws glyphs as paths in the SVG


def render(source: pathlib.Path = DOT_SOURCE, fmt: str = None, dpi: int = DPI) -> bytes:
    """Return the Graphviz rendering of source in fmt ('svg' or 'png')."""
    if fmt not in DOT_FORMATS:
        raise ValueError(f"fmt must be one of {list(DOT_FORMATS)}: {fmt!r}")
    dot = shutil.which('dot')
    if dot is None:
        raise RuntimeError('Graphviz is not installed: the `dot` command was not found.')
    if not source.is_file():
        raise FileNotFoundError(f"Graphviz source not found: {source}")
    result = subprocess.run([dot, f"-T{DOT_FORMATS[fmt]}", f"-Gdpi={dpi}" if fmt == 'png' else '-Gdpi=72', str(source)],
                            capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"dot failed ({result.returncode}): {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def draw(stem: pathlib.Path = None) -> pathlib.Path:
    """Save the decision tree to stem + '.svg' or '.png' and return the written path."""
    if stem is None:
        raise ValueError('stem is required.')
    return write_smaller(svg=render(fmt='svg'), png=render(fmt='png'), stem=stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{pathlib.Path(__file__).name} {__version__}\n"
                    'Decision tree for choosing the p95 method, drawn with Graphviz and saved as SVG, '
                    'or as PNG when the SVG is larger.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--output-folder', type=pathlib.Path, help='folder for the figure file')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    if args.output_folder is None or not args.output_folder.is_dir():
        parser.error(f"--output-folder must be an existing folder: {args.output_folder}")
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    draw(stem=ARGS.output_folder / FIGURE_STEM)
