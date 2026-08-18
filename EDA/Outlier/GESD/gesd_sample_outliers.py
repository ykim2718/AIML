"""Chart the outliers of a 15-observation measurement sample with the generalized ESD procedure.

The sample spans a factor of 48.7, so the observations are drawn once on a log axis
to keep every point visible and once on a linear axis with the extreme value dropped, which is
where the borderline flag can actually be read. A third panel shows the decision itself, because
one of the two flags survives only at the looser of the two significance levels.

Changelog:
    0.0.0 - Initial release.
"""

__author__ = 'yRocket'
__version__ = "0.0.1.2026.8.17"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd

from gesd_outlier_detection import GesdResult, gesd_test

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

# The measurement sample under test. Only seven distinct values occur among the fifteen
# observations, so the sample is closer to a step function than to a normal one.
SAMPLE = np.array([0.0232, 0.0232, 0.0232, 0.0220, 0.0232, 0.0232, 0.6532, 0.0403,
                   0.0293, 0.0159, 0.0134, 0.0134, 0.0134, 0.0134, 0.0134])

# Verdict styles. Identity is carried by the marker shape as well as the colour, so the chart
# stays readable without colour; the two flagged points also carry a direct label.
VERDICT_STYLES = {
    'retained': (TABLEAU_COLORS['tab:blue'], 'o', 46),
    'borderline': (TABLEAU_COLORS['tab:orange'], 's', 96),
    'outlier': (TABLEAU_COLORS['tab:red'], 'D', 130),
}
COLOR_REFERENCE = TABLEAU_COLORS['tab:gray']

TUKEY_MULTIPLIER = 1.5


def classify(data: np.ndarray = None, main: GesdResult = None, strict: GesdResult = None) -> np.ndarray:
    """Label every observation as retained, borderline or outlier.

    An observation flagged at both significance levels is an outlier; one flagged only at the
    looser level is borderline, because the verdict then rests on a choice the data did not make.

    Args:
        data: the sample, shape (n,).
        main: the result at the looser significance level.
        strict: the result at the stricter significance level.

    Returns:
        An array of shape (n,) holding 'retained', 'borderline' or 'outlier'.
    """
    if not set(strict.positions).issubset(set(main.positions)):
        raise ValueError(f"the stricter level flagged {sorted(set(strict.positions) - set(main.positions))}, "
                         f"which the looser level did not; the two results are not nested as assumed.")
    verdict = np.full(data.size, 'retained', dtype=object)
    verdict[main.positions] = 'borderline'
    verdict[strict.positions] = 'outlier'
    return verdict


def tukey_upper_fence(data: np.ndarray = None, multiplier: float = TUKEY_MULTIPLIER) -> float:
    """Upper fence of the Tukey rule, drawn as an independent reference on the linear panel."""
    quartile_one, quartile_three = np.percentile(data, [25, 75])
    return float(quartile_three + multiplier * (quartile_three - quartile_one))


def plot_sample(data: np.ndarray = None, verdicts: np.ndarray = None, main: GesdResult = None,
                strict: GesdResult = None, output_path: pathlib.Path = None) -> None:
    """Draw the sample on a log axis, the sample without its extreme value, and the decision."""
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    positions = np.arange(1, data.size + 1)
    flagged = np.flatnonzero(verdicts != 'retained')
    labels = {'retained': 'retained',
              'borderline': f"borderline (alpha = {main.alpha} only)",
              'outlier': 'outlier (both levels)'}

    # Panel (a): every observation on a log axis, the only scale on which all of them stay legible.
    for name, (color, marker, size) in VERDICT_STYLES.items():
        picked = verdicts == name
        if picked.any():
            axes[0].scatter(positions[picked], data[picked], s=size, color=color, marker=marker,
                            edgecolors='white', linewidths=0.8, zorder=3, label=labels[name])
    axes[0].set_yscale('log')
    for index in flagged:
        axes[0].annotate(f"{data[index]:.4f}", (positions[index], data[index]),
                         textcoords='offset points', xytext=(9, 4), fontsize=10)
    axes[0].set_xlabel('observation number')
    axes[0].set_ylabel('value (log scale)')
    axes[0].set_title(f"(a) All {data.size} observations")
    axes[0].legend(loc='upper left', frameon=False)
    axes[0].grid(alpha=0.25, linewidth=0.6)

    # Panel (b): the extreme value removed, so the remaining spread and the borderline flag show.
    # The verdict legend of panel (a) covers this panel too; repeating it here would put a legend
    # swatch beside the real data point and the two would read as one thing.
    extreme = float(data.max())
    keep = data < extreme
    for name, (color, marker, size) in VERDICT_STYLES.items():
        picked = (verdicts == name) & keep
        if picked.any():
            axes[1].scatter(positions[picked], data[picked], s=size, color=color, marker=marker,
                            edgecolors='white', linewidths=0.8, zorder=3)
    fence = tukey_upper_fence(data=data)
    axes[1].axhline(fence, color=COLOR_REFERENCE, linestyle='--', linewidth=1.4,
                    label=f"Tukey upper fence = {fence:.4f}")
    for index in flagged:
        if keep[index]:
            axes[1].annotate(f"{data[index]:.4f}", (positions[index], data[index]),
                             textcoords='offset points', xytext=(10, -4), fontsize=10)
    axes[1].set_xlabel('observation number')
    axes[1].set_ylabel('value')
    axes[1].set_title(f"(b) The same sample without {extreme:g}")
    axes[1].legend(loc='lower left', frameon=False)
    axes[1].grid(alpha=0.25, linewidth=0.6)

    # Panel (c): the decision itself, so the reader can see how wide each margin was.
    main_frame, strict_frame = main.to_frame(), strict.to_frame()
    axes[2].plot(main_frame.index, main_frame['statistic'], marker='o', markersize=9, linewidth=2,
                 color=VERDICT_STYLES['retained'][0], label='statistic R_i')
    axes[2].plot(main_frame.index, main_frame['critical'], marker='s', markersize=8, linewidth=2,
                 linestyle='--', color=VERDICT_STYLES['outlier'][0],
                 label=f"critical value lambda_i (alpha = {main.alpha})")
    axes[2].plot(strict_frame.index, strict_frame['critical'], marker='^', markersize=8, linewidth=2,
                 linestyle=':', color=COLOR_REFERENCE,
                 label=f"critical value lambda_i (alpha = {strict.alpha})")
    axes[2].axvline(main.count, color=VERDICT_STYLES['borderline'][0], linewidth=1.4,
                    label=f"largest exceeding step = {main.count}")
    axes[2].set_xticks(main_frame.index)
    axes[2].set_xlabel('step i')
    axes[2].set_ylabel('value')
    axes[2].set_title('(c) Generalized ESD decision at each step')
    axes[2].legend(loc='upper right', frameon=False, fontsize=9)
    axes[2].grid(alpha=0.25, linewidth=0.6)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"[3] Figure written to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse and validate the command line options."""
    parser = argparse.ArgumentParser(
        description='Chart the outliers of a measurement sample with the generalized ESD procedure.')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='significance level a flag must reach to be drawn at all (default: %(default)s)')
    parser.add_argument('--strict-alpha', type=float, default=0.01,
                        help='significance level a flag must also reach to count as an outlier rather than '
                             'borderline (default: %(default)s)')
    parser.add_argument('--max-outliers', type=int, default=5,
                        help='upper bound r on the number of outliers (default: %(default)s)')
    parser.add_argument('--save-figure', choices=['true', 'false'], default='true',
                        help='write the figure and the sample behind it (default: %(default)s)')
    parser.add_argument('--output-folder', type=pathlib.Path, default=None,
                        help='folder for the figure (default: generalized-esd-outlier-detection_fig next to this script)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    args.save_figure = args.save_figure == 'true'
    if args.output_folder is None:
        # The figure is referenced from generalized-esd-outlier-detection.md, whose images live
        # in the folder named after that document.
        args.output_folder = (pathlib.Path(__file__).resolve().parent
                              / 'generalized-esd-outlier-detection_fig')

    for name, level in (('--alpha', args.alpha), ('--strict-alpha', args.strict_alpha)):
        if not 0.0 < level < 1.0:
            parser.error(f"{name} must lie strictly between 0 and 1, got {level}.")
    if args.strict_alpha >= args.alpha:
        parser.error(f"--strict-alpha must be below --alpha, got {args.strict_alpha} against {args.alpha}; "
                     f"otherwise the borderline band the chart draws would be empty by construction.")
    if not 1 <= args.max_outliers <= SAMPLE.size - 2:
        parser.error(f"--max-outliers must lie between 1 and n - 2 = {SAMPLE.size - 2}, got {args.max_outliers}.")
    if args.save_figure:
        args.output_folder.mkdir(parents=True, exist_ok=True)

    return args


if __name__ == '__main__':
    options = parse_args()

    result_main = gesd_test(data=SAMPLE, max_outliers=options.max_outliers, alpha=options.alpha)
    result_strict = gesd_test(data=SAMPLE, max_outliers=options.max_outliers, alpha=options.strict_alpha)
    sample_verdicts = classify(data=SAMPLE, main=result_main, strict=result_strict)

    print(f"[1] alpha = {options.alpha}: {result_main.count} flagged "
          f"{np.sort(SAMPLE[result_main.positions]).tolist()}")
    print(f"[2] alpha = {options.strict_alpha}: {result_strict.count} flagged "
          f"{np.sort(SAMPLE[result_strict.positions]).tolist()}")

    if options.save_figure:
        stem = pathlib.Path(__file__).stem
        plot_sample(data=SAMPLE, verdicts=sample_verdicts, main=result_main, strict=result_strict,
                    output_path=options.output_folder / f"{stem}.png")
        # The sample the chart was drawn from, one observation per row, unrounded and without summaries.
        pd.DataFrame({'value': SAMPLE, 'verdict': sample_verdicts},
                     index=pd.Index(np.arange(1, SAMPLE.size + 1), name='observation')
                     ).to_csv(options.output_folder / f"{stem}.csv")
        print(f"[4] Chart data written to {options.output_folder / f'{stem}.csv'}")
