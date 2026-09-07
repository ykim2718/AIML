"""Draw how long data overfits even though the columns are few.

Panel (a) grows the depth of one tree and shows the training error falling while the held-out error
turns back up. Panel (b) compares a random split, a group-aware split and a truly unseen set on data
whose rows come in groups. Panel (c) adds rows two ways, from new groups and inside the groups
already present, and follows the error on unseen groups.
"""
__author__ = 'yRocket'
__version__ = "0.0.2.2026.9.7"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor

__all__ = ['grouped_draw', 'capacity_curve', 'split_comparison', 'row_curve',
           'draw_long_data_overfitting']

FIGSIZE: tuple = (13.5, 4.0)
REFERENCE_WIDTH: float = 13.5        # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 10.0
COLORS: list = list(matplotlib.colors.TABLEAU_COLORS.values())
FULL_DEPTH: int = 40                 # the x position standing for an unlimited tree


def grouped_draw(n_group: int, rows_per_group: int, rng: np.random.Generator, n_column: int = 5,
                 group_sd: float = 1.5, noise_sd: float = 0.5) -> tuple:
    """Rows that come in groups, with one group-constant column and one group-level offset.

    Returns (x, y, group). The response follows the columns non-linearly and carries the offset.
    """
    if n_group < 2 or rows_per_group < 1:
        raise ValueError(f"need at least 2 groups and 1 row per group, got {n_group} and {rows_per_group}")
    group = np.repeat(np.arange(n_group), rows_per_group)
    offset = rng.normal(scale=group_sd, size=n_group)[group]
    setting = rng.normal(size=n_group)[group]
    columns = rng.normal(size=(group.size, n_column))
    x = np.column_stack([setting, columns])
    y = np.sin(2 * columns[:, 0]) + 0.5 * columns[:, 1] ** 2 + offset
    return x, y + rng.normal(scale=noise_sd, size=group.size), group


def capacity_curve(depths: list, n_group: int = 400, rows_per_group: int = 25,
                   seed: int = 0) -> pd.DataFrame:
    """Training and held-out error of one tree as its depth grows.

    Returns a DataFrame indexed by 'depth' with columns 'training' and 'held-out'.
    """
    rng = np.random.default_rng(seed)
    x_fit, y_fit, _ = grouped_draw(n_group=n_group, rows_per_group=rows_per_group, rng=rng)
    x_new, y_new, _ = grouped_draw(n_group=n_group, rows_per_group=rows_per_group, rng=rng)
    rows = []
    for depth in depths:
        tree = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(x_fit, y_fit)
        rows.append({'depth': FULL_DEPTH if depth is None else depth,
                     'training': float(root_mean_squared_error(y_fit, tree.predict(x_fit))),
                     'held-out': float(root_mean_squared_error(y_new, tree.predict(x_new)))})
    return pd.DataFrame(rows).set_index('depth')


def split_comparison(n_group: int = 200, rows_per_group: int = 25, n_fold: int = 5,
                     seed: int = 0) -> pd.Series:
    """Error reported by a random split, by a group-aware split, and on unseen groups.

    Returns a Series indexed by the name of the estimate.
    """
    rng = np.random.default_rng(seed)
    x_fit, y_fit, group = grouped_draw(n_group=n_group, rows_per_group=rows_per_group, rng=rng)
    x_new, y_new, _ = grouped_draw(n_group=n_group, rows_per_group=rows_per_group, rng=rng)
    model = HistGradientBoostingRegressor(random_state=0)
    scoring = 'neg_root_mean_squared_error'
    random_fold = -cross_val_score(model, x_fit, y_fit, scoring=scoring,
                                   cv=KFold(n_fold, shuffle=True, random_state=0)).mean()
    group_fold = -cross_val_score(model, x_fit, y_fit, groups=group, scoring=scoring,
                                  cv=GroupKFold(n_fold)).mean()
    model.fit(x_fit, y_fit)
    return pd.Series({'random split': float(random_fold), 'group split': float(group_fold),
                      'unseen groups': float(root_mean_squared_error(y_new, model.predict(x_new)))})


def row_curve(row_counts: list, rows_per_group: int = 25, fixed_group: int = 20,
              n_repeat: int = 15, seed: int = 0) -> pd.DataFrame:
    """Held-out error on unseen groups as rows are added in two ways, averaged over repeats.

    Returns a DataFrame indexed by 'rows' with columns 'new groups' and 'same groups'.
    """
    if n_repeat < 1:
        raise ValueError(f"n_repeat must be at least 1, got {n_repeat}")
    rng = np.random.default_rng(seed)
    x_new, y_new, _ = grouped_draw(n_group=200, rows_per_group=rows_per_group, rng=rng)
    rows = []
    for count in row_counts:
        if count % rows_per_group != 0 or count % fixed_group != 0:
            raise ValueError(f"row count {count} does not divide by {rows_per_group} and {fixed_group}")
        totals = {'new groups': 0.0, 'same groups': 0.0}
        for repeat in range(n_repeat):
            draws = {
                'new groups': grouped_draw(n_group=count // rows_per_group,
                                           rows_per_group=rows_per_group, rng=rng),
                'same groups': grouped_draw(n_group=fixed_group, rows_per_group=count // fixed_group,
                                            rng=rng),
            }
            for name, (x_fit, y_fit, _) in draws.items():
                model = HistGradientBoostingRegressor(random_state=0).fit(x_fit, y_fit)
                totals[name] += root_mean_squared_error(y_new, model.predict(x_new))
        rows.append({'rows': count, **{name: total / n_repeat for name, total in totals.items()}})
    return pd.DataFrame(rows).set_index('rows')


def draw_long_data_overfitting(capacity: pd.DataFrame, splits: pd.Series, rows: pd.DataFrame,
                               output_folder: pathlib.Path) -> pathlib.Path:
    """Write the three-panel figure and the series it was drawn from. Returns the figure path."""
    output_folder.mkdir(parents=True, exist_ok=True)
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    plt.rcParams.update({'font.size': font_size})

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=FIGSIZE)

    for name, color in (('training', '0.4'), ('held-out', COLORS[0])):
        axes[0].plot(capacity.index, capacity[name], marker='o', markersize=4, color=color,
                     label=name)
    axes[0].set_xlabel('tree depth (10000 rows, 6 columns)')
    axes[0].set_ylabel('RMSE')
    axes[0].legend(frameon=False, fontsize=font_size * 0.85)

    axes[1].bar(splits.index, splits.to_numpy(), width=0.6,
                color=[COLORS[3], COLORS[0], '0.4'])
    for index, value in enumerate(splits.to_numpy()):
        axes[1].text(index, value, f"{value:.2f}", ha='center', va='bottom',
                     fontsize=font_size * 0.9)
    axes[1].set_ylabel('RMSE reported')
    axes[1].set_xlabel('how the validation rows were chosen')
    axes[1].set_ylim(0, splits.max() * 1.25)

    for name, color in (('new groups', COLORS[0]), ('same groups', COLORS[1])):
        axes[2].plot(rows.index, rows[name], marker='o', markersize=4, color=color, label=name)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('rows used for fitting, added two ways')
    axes[2].set_ylabel('RMSE on unseen groups')
    axes[2].legend(frameon=False, fontsize=font_size * 0.85)

    fig.subplots_adjust(bottom=0.22, top=0.96, wspace=0.30)
    for axis, label in zip(axes, ('(a)', '(b)', '(c)')):
        position = axis.get_position()
        fig.text(position.x0 + position.width / 2, 0.04, label, ha='center', fontsize=font_size)

    figure_path = output_folder / 'long-data-overfitting.png'
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    capacity.to_csv(output_folder / 'capacity-curve.csv')
    splits.to_csv(output_folder / 'split-comparison.csv', header=['rmse'])
    rows.to_csv(output_folder / 'row-curve.csv')
    print(f"wrote {figure_path}")
    print(capacity.round(3).to_string())
    print(splits.round(3).to_string())
    print(rows.round(3).to_string())
    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f"{pathlib.Path(__file__).name} {__version__}\n"
                    f"Draw how long data overfits even though the columns are few.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--output-folder', type=pathlib.Path, required=True,
                        help="folder the figure and its series are written to")
    parser.add_argument('--n-fold', type=int, default=5, help="folds of the split comparison")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if args.n_fold < 2:
        parser.error(f"--n-fold must be at least 2, got {args.n_fold}")
    return args


if __name__ == '__main__':
    arguments = parse_args()
    capacity_frame = capacity_curve(depths=[2, 3, 4, 6, 8, 12, 16, 24, None])
    split_series = split_comparison(n_fold=arguments.n_fold)
    row_frame = row_curve(row_counts=[500, 1000, 2000, 5000, 10000, 20000])
    draw_long_data_overfitting(capacity=capacity_frame, splits=split_series, rows=row_frame,
                               output_folder=arguments.output_folder)
