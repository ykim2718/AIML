"""Worked example of the ceiling on the classical z-score, on a 15-observation current sample.

The sample is read from a CSV so that the document and any other script work from the same file.
It sits within a tenth of a percent of the ceiling its size allows, which is what makes it useful
here: the same measurement in a sample of fourteen could not be flagged by a rule at 3.5 at all.
One figure is drawn, the ceiling against the sample size beside the scores the sample produces,
and the points behind both panels are written out as CSV.

Changelog:
    0.0.0 - Initial release.
"""

__author__ = 'yRocket'
__version__ = "0.0.0.2026.8.18"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd

from z_score_ceiling import DEFAULT_DDOF, classical_z_scores, max_attainable_z

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

# The measurement sample under test lives beside the documents that quote it.
DEFAULT_SAMPLE_CSV = pathlib.Path(__file__).resolve().parents[1] / 'data' / '1d_esc_current.csv'

# The cut-off conventionally used with a z-score rule, and the sizes the document tabulates.
CONVENTIONAL_THRESHOLD = 3.5
TABULATED_SIZES = (10, 14, 15, 20, 54)

COLOR_CEILING = TABLEAU_COLORS['tab:blue']
COLOR_THRESHOLD = TABLEAU_COLORS['tab:red']
COLOR_POINT = TABLEAU_COLORS['tab:gray']
COLOR_EXTREME = TABLEAU_COLORS['tab:orange']


def load_sample(input_csv: pathlib.Path = None, column: str = None) -> np.ndarray:
    """Read the column to score from a CSV.

    Args:
        input_csv: the CSV holding the sample.
        column: the column to read.

    Returns:
        The sample, shape (n,).
    """
    frame = pd.read_csv(input_csv)
    if column not in frame.columns:
        raise ValueError(f"column '{column}' is absent from {input_csv}; available: {list(frame.columns)}")
    values = frame[column].to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f"column '{column}' of {input_csv} is empty.")
    return values


def ceiling_frame(sizes: np.ndarray = None, threshold: float = CONVENTIONAL_THRESHOLD) -> pd.DataFrame:
    """Tabulate the ceiling against sample size, and whether the threshold sits under it.

    Args:
        sizes: the sample sizes to evaluate.
        threshold: the cut-off a z-score rule would use.

    Returns:
        A pd.DataFrame indexed by 'size', with columns 'ceiling' and 'threshold_reachable'.
    """
    counts = np.asarray(sizes, dtype=int)
    ceilings = np.array([max_attainable_z(size=int(n)) for n in counts])
    return pd.DataFrame({'ceiling': ceilings, 'threshold_reachable': ceilings > threshold},
                        index=pd.Index(counts, name='size'))


def score_frame(data: np.ndarray = None, threshold: float = CONVENTIONAL_THRESHOLD) -> pd.DataFrame:
    """Tabulate the sample against its classical z-scores and the flag each one earns.

    Args:
        data: the sample, shape (n,).
        threshold: the cut-off on the absolute z-score.

    Returns:
        A pd.DataFrame indexed by 'position' (counted from 0), with columns 'value', 'classical_z'
        and 'flagged'.
    """
    values = np.asarray(data, dtype=float)
    scores = classical_z_scores(data=values)
    return pd.DataFrame({'value': values, 'classical_z': scores,
                         'flagged': np.abs(scores) > threshold},
                        index=pd.Index(np.arange(values.size), name='position'))


def plot_ceiling(data: np.ndarray = None, threshold: float = CONVENTIONAL_THRESHOLD,
                 largest_size: int = 60, output_path: pathlib.Path = None) -> pd.DataFrame:
    """Draw the ceiling against sample size, and the scores this sample reaches under it.

    Args:
        data: the sample, shape (n,).
        threshold: the cut-off a z-score rule would use.
        largest_size: the right end of the size axis in panel (a).
        output_path: where the figure is written.

    Returns:
        The points of panel (a), a pd.DataFrame with columns 'size' and 'ceiling'.
    """
    values = np.asarray(data, dtype=float)
    count = values.size
    scores = np.abs(classical_z_scores(data=values))
    ceiling = max_attainable_z(size=count)

    sizes = np.arange(2, largest_size + 1)
    ceilings = np.array([max_attainable_z(size=int(n)) for n in sizes])
    reachable_at = int(sizes[ceilings > threshold][0])

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    axis = axes[0]
    axis.plot(sizes, ceilings, color=COLOR_CEILING, linewidth=2.0, label='ceiling $(n-1)/\\sqrt{n}$')
    axis.axhline(threshold, color=COLOR_THRESHOLD, linewidth=2.0, linestyle='--',
                 label=f'threshold {threshold}')
    axis.axvline(reachable_at, color=COLOR_POINT, linewidth=1.2, linestyle=':')
    axis.scatter([count], [ceiling], s=90, color=COLOR_EXTREME, zorder=5,
                 label=f'this sample, n = {count}')
    axis.annotate(f'no rule at {threshold} can fire\nbelow n = {reachable_at}',
                  xy=(reachable_at, threshold), xytext=(reachable_at + 6, threshold - 1.5),
                  color=COLOR_POINT, fontsize=10,
                  arrowprops={'arrowstyle': '->', 'color': COLOR_POINT, 'linewidth': 1.0})
    axis.set_xlabel('sample size n')
    axis.set_ylabel('largest attainable absolute z')
    axis.set_title('(a) the ceiling is fixed by the sample size', fontsize=11, loc='left')
    axis.set_xlim(0, largest_size)
    axis.set_ylim(0, max(ceilings) * 1.08)
    axis.grid(alpha=0.25)
    axis.legend(loc='lower right', frameon=False)

    axis = axes[1]
    extreme = scores >= threshold
    positions = np.arange(count)
    axis.scatter(positions[~extreme], scores[~extreme], s=60, color=COLOR_POINT,
                 label='absolute z of an observation')
    axis.scatter(positions[extreme], scores[extreme], s=110, color=COLOR_EXTREME, zorder=5,
                 label='flagged observation')
    axis.axhline(ceiling, color=COLOR_CEILING, linewidth=2.0,
                 label=f'ceiling at n = {count}, {ceiling:.4f}')
    axis.axhline(threshold, color=COLOR_THRESHOLD, linewidth=2.0, linestyle='--',
                 label=f'threshold {threshold}')
    axis.set_xlabel('position in the sample')
    axis.set_ylabel('absolute classical z')
    axis.set_title('(b) the sample all but reaches its own ceiling', fontsize=11, loc='left')
    axis.set_ylim(0, ceiling * 1.35)
    axis.grid(alpha=0.25)
    axis.legend(loc='upper left', frameon=False)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"[2] Figure written to {output_path}")
    return pd.DataFrame({'size': sizes, 'ceiling': ceilings})


def parse_args() -> argparse.Namespace:
    """Parse and validate the command line options."""
    parser = argparse.ArgumentParser(
        description='Reproduce the worked example of the classical z-score ceiling document.')
    parser.add_argument('--input-csv', type=pathlib.Path, default=DEFAULT_SAMPLE_CSV,
                        help='CSV holding the sample (default: %(default)s)')
    parser.add_argument('--column', type=str, default='value',
                        help='column of --input-csv to read (default: %(default)s)')
    parser.add_argument('--threshold', type=float, default=CONVENTIONAL_THRESHOLD,
                        help='cut-off on the absolute z-score (default: %(default)s)')
    parser.add_argument('--save-figure', choices=['true', 'false'], default='true',
                        help='write the figure and the samples behind it (default: %(default)s)')
    parser.add_argument('--output-folder', type=pathlib.Path, default=None,
                        help='folder for the figure (default: z-score-ceiling_fig next to this script)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    args.save_figure = args.save_figure == 'true'
    if args.output_folder is None:
        # The figure is referenced from z-score-ceiling.md, whose images live in that folder.
        args.output_folder = pathlib.Path(__file__).resolve().parent / 'z-score-ceiling_fig'

    if not args.input_csv.is_file():
        parser.error(f"--input-csv is not a file: {args.input_csv}")
    if args.threshold <= 0.0:
        parser.error(f"--threshold must be positive, got {args.threshold}.")
    if args.save_figure:
        args.output_folder.mkdir(parents=True, exist_ok=True)

    return args


if __name__ == '__main__':
    options = parse_args()
    sample = load_sample(input_csv=options.input_csv, column=options.column)

    scores = np.abs(classical_z_scores(data=sample, ddof=DEFAULT_DDOF))
    ceiling = max_attainable_z(size=sample.size, ddof=DEFAULT_DDOF)
    print(f"[1] n = {sample.size}, largest |z| = {scores.max():.6f}, ceiling = {ceiling:.6f}, "
          f"which is {100 * scores.max() / ceiling:.2f}% of it; "
          f"the ceiling at n = {sample.size - 1} would be {max_attainable_z(size=sample.size - 1):.6f}")

    if options.save_figure:
        curve = plot_ceiling(data=sample, threshold=options.threshold,
                             output_path=options.output_folder / 'z_score_ceiling.png')
        # One observation per row for the sample, one sample size per row for the two curves.
        score_frame(data=sample, threshold=options.threshold).to_csv(
            options.output_folder / 'z_score_sample.csv')
        curve.to_csv(options.output_folder / 'z_score_curve.csv', index=False)
        ceiling_frame(sizes=np.array(TABULATED_SIZES), threshold=options.threshold).to_csv(
            options.output_folder / 'z_score_tabulated.csv')
        print(f"[3] Chart data written to {options.output_folder}")
