"""Draw the Appendix B and Appendix C figures of inversion-problem.md."""
__author__ = 'yRocket'
__version__ = "0.4.0.2026.8.29"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from scipy.linalg import null_space
from scipy.optimize import minimize
from scipy.stats import chi2
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor

__all__ = ['build_appendix_b_model', 'draw_parity', 'draw_null_space', 'draw_constrained_inversion']

matplotlib.use('Agg')

DEFAULT_OUTPUT_FOLDER: pathlib.Path = pathlib.Path(__file__).parent / 'inversion-problem_fig'
FIGSIZE: tuple = (9.0, 4.0)
REFERENCE_WIDTH: float = 9.0     # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 9.0
DPI: int = 300
COLORS: list = list(TABLEAU_COLORS.values())


def font_size(scale: float = 1.0) -> float:
    return BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH * scale


def sub_caption(fig, axes_group: list, text: str, y: float = 0.02) -> None:
    """Put the panel label below the chart, centred on the axes the panel is made of."""
    boxes = [ax.get_position() for ax in axes_group]
    x = (min(box.x0 for box in boxes) + max(box.x1 for box in boxes)) / 2
    fig.text(x, y, text, ha='center', va='bottom', fontsize=font_size(1.05))


def build_appendix_b_model() -> tuple:
    """Rebuild the Appendix B data and PLS model. Returns (x_data, y_data, x_mean, y_mean, pls)."""
    rng = np.random.default_rng(0)
    n, n_comp = 100, 2
    # both inputs are centred at 0, x1 spread twice as wide as x2
    x_data = np.column_stack([rng.normal(0.0, 1.0, n), rng.normal(0.0, 0.5, n)])
    y_data = 1.5 * x_data[:, 0] - 0.8 * x_data[:, 1] + rng.normal(0, 0.85, n)

    x_mean, y_mean = x_data.mean(axis=0), y_data.mean()
    pls = PLSRegression(n_components=n_comp, scale=False).fit(x_data - x_mean, y_data - y_mean)
    return x_data, y_data, x_mean, y_mean, pls


def draw_parity(out_folder: pathlib.Path) -> pathlib.Path:
    """Appendix B: the sampled inputs and the forward model that the inversion runs backwards."""
    x_data, y_data, x_mean, y_mean, pls = build_appendix_b_model()
    y_pred = pls.predict(x_data - x_mean).ravel() + y_mean
    residual = y_data - y_pred
    r2 = 1.0 - float(np.sum(residual ** 2) / np.sum((y_data - y_data.mean()) ** 2))
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    fig = plt.figure(figsize=(13.0, 4.2))
    outer = fig.add_gridspec(1, 3, wspace=0.30)
    ax_series = fig.add_subplot(outer[0])
    middle = outer[1].subgridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                                  hspace=0.06, wspace=0.06)
    ax_main = fig.add_subplot(middle[1, 0])
    ax_top = fig.add_subplot(middle[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(middle[1, 1], sharey=ax_main)
    ax_parity = fig.add_subplot(outer[2])

    # the measured value in the order the samples arrive
    ax_series.plot(np.arange(len(y_data)), y_data, color=COLORS[0], linewidth=1.0,
                   marker='o', markersize=3, label='measured value y')
    ax_series.axhline(2.0, color=COLORS[3], linestyle=':', linewidth=1.2,
                      label='inversion target 2.0')
    ax_series.set_xlabel('sample order', fontsize=font_size())
    ax_series.set_ylabel('measured value y', fontsize=font_size())
    ax_series.tick_params(labelsize=font_size(0.9))
    ax_series.legend(fontsize=font_size(0.8), loc='upper left')

    # Mahalanobis distance from the sample centre, in units of standard deviation
    grid_x, grid_y = np.mgrid[x_data[:, 0].min() - 0.6:x_data[:, 0].max() + 0.6:200j,
                              x_data[:, 1].min() - 0.6:x_data[:, 1].max() + 0.6:200j]
    centre = x_data.mean(axis=0)
    cov_inv = np.linalg.inv(np.cov(x_data, rowvar=False))
    offset = np.stack([grid_x - centre[0], grid_y - centre[1]], axis=-1)
    distance = np.sqrt(np.einsum('...i,ij,...j->...', offset, cov_inv, offset))
    lines = ax_main.contour(grid_x, grid_y, distance, levels=[1.0, 2.0, 3.0],
                            colors=COLORS[0], linewidths=1.0)
    ax_main.clabel(lines, fmt='%.0f', fontsize=font_size(0.8))
    ax_main.scatter(x_data[:, 0], x_data[:, 1], s=14, color='0.45', label='100 samples')
    ax_main.set_xlabel('x1', fontsize=font_size())
    ax_main.set_ylabel('x2', fontsize=font_size())
    ax_main.tick_params(labelsize=font_size(0.9))
    ax_main.legend(fontsize=font_size(0.8), loc='upper left')

    ax_top.hist(x_data[:, 0], bins=12, color=COLORS[0])
    ax_top.tick_params(labelbottom=False, labelsize=font_size(0.8))
    ax_right.hist(x_data[:, 1], bins=12, orientation='horizontal', color=COLORS[0])
    ax_right.tick_params(labelleft=False, labelsize=font_size(0.8))

    span = [min(y_data.min(), y_pred.min()) - 0.2, max(y_data.max(), y_pred.max()) + 0.2]
    ax_parity.plot(span, span, color='0.5', linestyle='--', linewidth=1.0, label='1:1 line')
    ax_parity.scatter(y_data, y_pred, s=18, color=COLORS[0], label='100 samples')
    ax_parity.axhline(2.0, color=COLORS[3], linestyle=':', linewidth=1.2, label='inversion target 2.0')
    ax_parity.set_xlim(span)
    ax_parity.set_ylim(span)
    ax_parity.set_xlabel('measured value y', fontsize=font_size())
    ax_parity.set_ylabel('predicted value', fontsize=font_size())
    ax_parity.tick_params(labelsize=font_size(0.9))
    ax_parity.legend(fontsize=font_size(0.8), loc='upper left')

    fig.subplots_adjust(bottom=0.20)
    sub_caption(fig, [ax_series], '(a) the measured value in sample order')
    sub_caption(fig, [ax_main, ax_top, ax_right],
                '(b) the two sampled inputs, contoured by Mahalanobis distance')
    sub_caption(fig, [ax_parity], f'(c) parity plot, R2 = {r2:.3f}, RMSE = {rmse:.2f}')

    out_path = out_folder / 'appendix-b-parity.png'
    out_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    # the samples the histograms, the contours and the parity plot received
    sample_path = out_folder / 'appendix-b-samples.csv'
    np.savetxt(sample_path, np.column_stack([x_data, y_data, y_pred]), delimiter=',',
               header='x1,x2,y,y_pred', comments='', fmt='%.6f')
    print(f'wrote {out_path}, R2={r2:.4f}, RMSE={rmse:.4f}, '
          f'y range=[{y_data.min():.2f}, {y_data.max():.2f}]')
    print(f'wrote {sample_path}')
    return out_path


def draw_null_space(out_folder: pathlib.Path) -> pathlib.Path:
    """Appendix B: scores moved along the null space keep the predicted value."""
    x_data, y_data, x_mean, y_mean, pls = build_appendix_b_model()
    loadings, y_loadings = pls.x_loadings_, pls.y_loadings_

    y_des = 2.0
    t_star = np.linalg.pinv(y_loadings) @ np.array([y_des - y_mean])
    x_star = loadings @ t_star + x_mean
    null_dirs = null_space(y_loadings)

    def predict(x_vec: np.ndarray) -> float:
        return float(pls.predict((x_vec - x_mean)[None, :]).ravel()[0]) + y_mean

    alphas = np.linspace(-2.0, 2.0, 37)
    x_alt = np.array([loadings @ (t_star + null_dirs @ np.array([a])) + x_mean for a in alphas])
    preds = np.array([predict(x_vec) for x_vec in x_alt])
    distances = np.linalg.norm(x_alt - x_star, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    ax = axes[0]
    ax.scatter(distances, preds, s=22, color=COLORS[0], label='37 null space solutions')
    ax.axhline(y_des, color='0.4', linestyle='--', linewidth=1.0, label=f'target {y_des}')
    ax.set_ylim(y_des - 0.5, y_des + 0.5)
    ax.set_xlabel('input distance from the minimum-norm solution', fontsize=font_size())
    ax.set_ylabel('predicted value', fontsize=font_size())
    ax.tick_params(labelsize=font_size(0.9))
    ax.legend(fontsize=font_size(0.85), loc='upper left')

    ax = axes[1]
    ax.scatter(x_data[:, 0], x_data[:, 1], s=14, color='0.75', label='historical data')
    ax.plot(x_alt[:, 0], x_alt[:, 1], color=COLORS[0], linewidth=1.2, zorder=2)
    ax.scatter(x_alt[:, 0], x_alt[:, 1], s=22, color=COLORS[0], zorder=3,
               label='37 inputs that all predict 2.000')
    ax.scatter([x_star[0]], [x_star[1]], s=110, marker='*', color=COLORS[1], zorder=4,
               label='minimum-norm solution')
    ax.set_xlabel('x1', fontsize=font_size())
    ax.set_ylabel('x2', fontsize=font_size())
    ax.tick_params(labelsize=font_size(0.9))
    ax.legend(fontsize=font_size(0.85), loc='upper left')

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    sub_caption(fig, [axes[0]], '(a) the inputs move, the value does not')
    sub_caption(fig, [axes[1]], '(b) the solutions form a line in the input plane')
    out_path = out_folder / 'appendix-b-null-space.png'
    out_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}, predictions in [{preds.min():.4f}, {preds.max():.4f}], '
          f'distance up to {distances.max():.3f}')
    return out_path


def draw_constrained_inversion(out_folder: pathlib.Path) -> pathlib.Path:
    """Appendix C: the constrained solution hits the target inside the validity domain."""
    rng = np.random.default_rng(0)
    n = 400
    z = rng.normal(size=(n, 2))
    x_data = np.column_stack([z[:, 0], z[:, 1],
                              0.9 * z[:, 0] + 0.1 * rng.normal(size=n),
                              -0.7 * z[:, 1] + 0.1 * rng.normal(size=n)])
    y_data = np.sin(x_data[:, 0]) + 0.5 * x_data[:, 1] ** 2 + 0.2 * x_data[:, 2]

    forward = GradientBoostingRegressor(random_state=0).fit(x_data, y_data)
    pca = PCA(n_components=2).fit(x_data)
    t2_limit = chi2.ppf(0.95, df=2)
    spe_train = np.sum((x_data - pca.inverse_transform(pca.transform(x_data))) ** 2, axis=1)
    spe_limit = float(np.quantile(spe_train, 0.95))

    def t2(x_vec: np.ndarray) -> float:
        return float(np.sum(pca.transform(x_vec[None, :])[0] ** 2 / pca.explained_variance_))

    def spe(x_vec: np.ndarray) -> float:
        return float(np.sum((x_vec - pca.inverse_transform(pca.transform(x_vec[None, :]))[0]) ** 2))

    y_des = 1.0

    def objective(x_vec: np.ndarray) -> float:
        return float((forward.predict(x_vec[None, :])[0] - y_des) ** 2)

    result = minimize(objective, x0=x_data.mean(axis=0), method='COBYLA',
                      constraints=[{'type': 'ineq', 'fun': lambda x: t2_limit - t2(x)},
                                   {'type': 'ineq', 'fun': lambda x: spe_limit - spe(x)}],
                      options={'maxiter': 3000})
    x_sol = result.x
    # the same search without the SPE constraint, to show what the constraint buys
    loose = minimize(objective, x0=x_data.mean(axis=0), method='COBYLA',
                     constraints=[{'type': 'ineq', 'fun': lambda x: t2_limit - t2(x)}],
                     options={'maxiter': 3000})
    x_loose = loose.x

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    ax = axes[0]
    ax.scatter(x_data[:, 0], x_data[:, 2], s=8, color='0.75', label='historical data')
    ax.scatter([x_sol[0]], [x_sol[2]], s=90, marker='*', color=COLORS[2],
               label='solution with T2 and SPE')
    ax.scatter([x_loose[0]], [x_loose[2]], s=70, marker='X', color=COLORS[3],
               label='solution with T2 only')
    ax.set_xlabel('x1', fontsize=font_size())
    ax.set_ylabel('x3', fontsize=font_size())
    ax.tick_params(labelsize=font_size(0.9))
    ax.legend(fontsize=font_size(0.85), loc='upper left')

    ax = axes[1]
    labels = ['T2', 'SPE']
    ratios_sol = [t2(x_sol) / t2_limit, spe(x_sol) / spe_limit]
    ratios_loose = [t2(x_loose) / t2_limit, spe(x_loose) / spe_limit]
    idx = np.arange(len(labels))
    bars_sol = ax.bar(idx - 0.2, ratios_sol, width=0.4, color=COLORS[2], label='T2 and SPE')
    bars_loose = ax.bar(idx + 0.2, ratios_loose, width=0.4, color=COLORS[3], label='T2 only')
    ax.axhline(1.0, color=COLORS[4], linestyle='--', linewidth=1.2, label='limit')
    ax.set_yscale('log')                       # the loose SPE is two orders above the others
    ax.set_ylim(0.01, 300.0)
    for bars, ratios in ((bars_sol, ratios_sol), (bars_loose, ratios_loose)):
        for bar, ratio in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, ratio * 1.15, f'{ratio:.2f}',
                    ha='center', fontsize=font_size(0.85))
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=font_size(0.9))
    ax.set_ylabel('value / limit', fontsize=font_size())
    ax.tick_params(axis='y', labelsize=font_size(0.9))
    ax.legend(fontsize=font_size(0.85))

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    sub_caption(fig, [axes[0]], '(a) the constrained solution keeps the input correlation')
    sub_caption(fig, [axes[1]], f'(b) both hit the target {y_des:.1f}, only one stays valid')
    out_path = out_folder / 'appendix-c-constrained-inversion.png'
    out_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')
    print(f'  with SPE : pred={forward.predict(x_sol[None, :])[0]:.4f} '
          f'T2={t2(x_sol):.2f}/{t2_limit:.2f} SPE={spe(x_sol):.3f}/{spe_limit:.3f}')
    print(f'  T2 only  : pred={forward.predict(x_loose[None, :])[0]:.4f} '
          f'T2={t2(x_loose):.2f}/{t2_limit:.2f} SPE={spe(x_loose):.3f}/{spe_limit:.3f}')
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f'{pathlib.Path(__file__).name} {__version__}\n'
                    f'Draw the Appendix B and Appendix C figures of inversion-problem.md.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--output-folder', type=pathlib.Path, default=DEFAULT_OUTPUT_FOLDER,
                        help='folder every figure and sample file is written under '
                             f'(default: {DEFAULT_OUTPUT_FOLDER.name} beside this script)')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    if args.output_folder.exists() and not args.output_folder.is_dir():
        parser.error(f'--output-folder is not a folder: {args.output_folder}')
    return args


if __name__ == '__main__':
    arguments = parse_args()
    draw_parity(out_folder=arguments.output_folder)
    draw_null_space(out_folder=arguments.output_folder)
    draw_constrained_inversion(out_folder=arguments.output_folder)
