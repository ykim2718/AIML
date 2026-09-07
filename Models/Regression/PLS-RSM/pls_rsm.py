"""Fit a second-order response surface through PLS and locate its stationary point.

Two designs are run against the same true surface: a rotatable central composite design, where the
quadratic terms are nearly orthogonal, and a run of correlated operating data, where they are not.
The comparison against OLS on the same expanded matrix is the point of the script.

Changelog:
- 0.0.0.2026.9.6: initial release
"""

__author__ = 'yRocket'
__version__ = "0.0.0.2026.9.6"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import enum
import itertools
import pathlib
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from scipy import linalg
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict

__all__ = [
    'Design',
    'central_composite_design',
    'correlated_operating_data',
    'expand_quadratic',
    'true_response',
    'quadratic_from_coefficients',
    'stationary_point',
    'fit_pls_surface',
    'fit_ols_surface',
    'run',
]

FIGSIZE: tuple = (11.0, 5.0)
REFERENCE_WIDTH: float = 11.0    # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 10.0
FIGURE_DPI: int = 300
CONTOUR_GRID: int = 160          # samples per axis in the contour grid
COLORS: list = list(TABLEAU_COLORS.values())

N_FACTORS: int = 3
TRUE_INTERCEPT: float = 78.0
TRUE_LINEAR: np.ndarray = np.array([1.95, -0.55, 3.10])
TRUE_QUADRATIC: np.ndarray = np.array([
    [-2.00, -0.60, 0.45],
    [-0.60, -1.40, -0.35],
    [0.45, -0.35, -2.60],
])


DESIGN_LABELS: dict = {'ccd': 'CCD', 'correlated': 'corr.'}


class Design(enum.StrEnum):
    """Names of the two designs the script contrasts."""

    CCD = enum.auto()
    CORRELATED = enum.auto()


def true_response(x: np.ndarray, noise_sd: float = 0.0, rng: np.random.Generator = None) -> np.ndarray:
    """Evaluate the true quadratic surface at the coded settings in the rows of `x`."""
    if noise_sd < 0.0:
        raise ValueError(f"noise_sd must be non-negative, got {noise_sd}.")
    if noise_sd > 0.0 and rng is None:
        raise ValueError("rng is required when noise_sd is positive; pass a numpy Generator.")
    quadratic = np.einsum('ij,jk,ik->i', x, TRUE_QUADRATIC, x)
    y = TRUE_INTERCEPT + x @ TRUE_LINEAR + quadratic
    if noise_sd > 0.0:
        y = y + rng.normal(loc=0.0, scale=noise_sd, size=y.shape)
    return y


def central_composite_design(n_factors: int = N_FACTORS, n_center: int = 6) -> np.ndarray:
    """Build a rotatable central composite design: a full factorial, star points at alpha, and centers."""
    if n_factors < 2:
        raise ValueError(f"a response surface needs at least two factors, got {n_factors}.")
    if n_center < 1:
        raise ValueError(f"n_center must be at least one, got {n_center}.")
    factorial = np.array(list(itertools.product([-1.0, 1.0], repeat=n_factors)))
    alpha = float(len(factorial)) ** 0.25          # the value that makes the design rotatable
    star = np.zeros((2 * n_factors, n_factors))
    for i in range(n_factors):
        star[2 * i, i] = alpha
        star[2 * i + 1, i] = -alpha
    center = np.zeros((n_center, n_factors))
    return np.vstack([factorial, star, center])


def correlated_operating_data(n_runs: int, correlation: float, rng: np.random.Generator) -> np.ndarray:
    """Draw settings the way an unplanned process record supplies them, with the factors tied together."""
    if not 0.0 <= correlation < 1.0:
        raise ValueError(f"correlation must be in [0, 1), got {correlation}.")
    if n_runs < 1:
        raise ValueError(f"n_runs must be positive, got {n_runs}.")
    driver = rng.normal(size=n_runs)
    independent = rng.normal(size=(n_runs, N_FACTORS))
    x = correlation * driver[:, None] + (1.0 - correlation) * independent
    return x / np.std(x, axis=0, ddof=1)


def expand_quadratic(x: np.ndarray) -> tuple:
    """Append squares and two-factor products to `x`, returning the expanded matrix and its column names."""
    n_factors = x.shape[1]
    names = [f"x{i + 1}" for i in range(n_factors)]
    columns = [x]
    columns.append(x ** 2)
    names += [f"x{i + 1}^2" for i in range(n_factors)]
    pairs = list(itertools.combinations(range(n_factors), 2))
    if pairs:
        columns.append(np.column_stack([x[:, i] * x[:, j] for i, j in pairs]))
        names += [f"x{i + 1}x{j + 1}" for i, j in pairs]
    return np.hstack(columns), names


def quadratic_from_coefficients(coefficients: np.ndarray, names: list, n_factors: int) -> tuple:
    """Split expanded-space coefficients into the linear vector b and the symmetric matrix B of the surface."""
    if len(coefficients) != len(names):
        raise ValueError(f"got {len(coefficients)} coefficients for {len(names)} names.")
    index = {name: position for position, name in enumerate(names)}
    b = np.array([coefficients[index[f"x{i + 1}"]] for i in range(n_factors)])
    matrix = np.zeros((n_factors, n_factors))
    for i in range(n_factors):
        matrix[i, i] = coefficients[index[f"x{i + 1}^2"]]
    for i, j in itertools.combinations(range(n_factors), 2):
        half = 0.5 * coefficients[index[f"x{i + 1}x{j + 1}"]]
        matrix[i, j] = half
        matrix[j, i] = half
    return b, matrix


def stationary_point(b: np.ndarray, matrix: np.ndarray) -> tuple:
    """Solve for the stationary point of the fitted surface and return it with the eigenvalues that classify it."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.min(np.abs(eigenvalues)) < np.finfo(float).eps * max(1.0, np.max(np.abs(eigenvalues))):
        raise linalg.LinAlgError("the quadratic matrix is singular; the surface has no isolated stationary point.")
    return -0.5 * np.linalg.solve(matrix, b), eigenvalues


def fit_pls_surface(x_expanded: np.ndarray, y: np.ndarray, max_components: int, n_splits: int) -> tuple:
    """Choose the component count by cross-validated RMSE, refit on all rows, and return coefficients and the count."""
    # The rank caps the component count: past it a component carries no variance and PLS divides by zero.
    # A fold holds out rows, so the cap uses the rank of the smallest training set rather than of the whole.
    folds = KFold(n_splits=min(n_splits, x_expanded.shape[0]), shuffle=True, random_state=0)
    held_out = int(np.ceil(x_expanded.shape[0] / folds.get_n_splits()))
    rank = int(np.linalg.matrix_rank(x_expanded - x_expanded.mean(axis=0)))
    upper = min(max_components, rank, x_expanded.shape[0] - held_out - 1)
    if upper < 1:
        raise ValueError(f"no components are available for a {x_expanded.shape} design matrix of rank {rank}.")
    errors = []
    for n_components in range(1, upper + 1):
        with warnings.catch_warnings():
            # A fold can exhaust y before the cap, which leaves NaN predictions. That count and every larger
            # one are unusable, so the search stops there and says so instead of ranking a NaN error.
            warnings.simplefilter('ignore', category=RuntimeWarning)
            predicted = cross_val_predict(PLSRegression(n_components=n_components), x_expanded, y, cv=folds)
        if not np.all(np.isfinite(predicted)):
            print(f"PLS exhausted the response at {n_components} components; searching up to {n_components - 1}.",
                  file=sys.stderr)
            break
        errors.append(float(np.sqrt(np.mean((y - predicted.ravel()) ** 2))))
    if not errors:
        raise ValueError(f"no usable component count for a {x_expanded.shape} design matrix.")
    best = int(np.argmin(errors)) + 1
    model = PLSRegression(n_components=best).fit(x_expanded, y)
    return model.coef_.ravel(), best, errors


def fit_ols_surface(x_expanded: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve the normal equations for the same expanded matrix, which is what PLS is being compared against."""
    design = np.column_stack([np.ones(len(x_expanded)), x_expanded])
    gram = design.T @ design
    # The solve is deliberately not replaced by a pseudo-inverse: a singular gram matrix is the finding,
    # not something to paper over, and the caller reports it.
    coefficients = linalg.solve(gram, design.T @ y, assume_a='pos')
    return coefficients[1:]


def evaluate_design(x: np.ndarray, y: np.ndarray, max_components: int, n_splits: int) -> dict:
    """Fit both surfaces to one design and collect what the document compares."""
    x_expanded, names = expand_quadratic(x=x)
    condition = float(np.linalg.cond(np.column_stack([np.ones(len(x_expanded)), x_expanded])))
    pls_coefficients, n_components, cv_errors = fit_pls_surface(
        x_expanded=x_expanded, y=y, max_components=max_components, n_splits=n_splits)
    pls_b, pls_matrix = quadratic_from_coefficients(
        coefficients=pls_coefficients, names=names, n_factors=x.shape[1])
    pls_optimum, pls_eigenvalues = stationary_point(b=pls_b, matrix=pls_matrix)
    result = {
        'n_runs': len(x),
        'condition_number': condition,
        'n_components': n_components,
        'cv_rmse': min(cv_errors),
        'pls_optimum': pls_optimum,
        'pls_eigenvalues': pls_eigenvalues,
        'pls_matrix': pls_matrix,
        'pls_b': pls_b,
    }
    try:
        ols_coefficients = fit_ols_surface(x_expanded=x_expanded, y=y)
    except (linalg.LinAlgError, np.linalg.LinAlgError) as error:
        print(f"OLS failed on this design: {error}", file=sys.stderr)
        result['ols_optimum'] = None
        result['ols_coefficient_norm'] = float('nan')
        return result
    ols_b, ols_matrix = quadratic_from_coefficients(
        coefficients=ols_coefficients, names=names, n_factors=x.shape[1])
    result['ols_coefficient_norm'] = float(np.linalg.norm(ols_coefficients))
    try:
        result['ols_optimum'] = stationary_point(b=ols_b, matrix=ols_matrix)[0]
    except (linalg.LinAlgError, np.linalg.LinAlgError) as error:
        print(f"OLS surface has no isolated stationary point: {error}", file=sys.stderr)
        result['ols_optimum'] = None
    return result


def plot_results(results: dict, distances: pd.DataFrame, true_optimum: np.ndarray,
                 output_path: pathlib.Path) -> None:
    """Draw the fitted surface of each design and the spread of the located optimum over the replicates."""
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    matplotlib.rcParams.update({'font.size': font_size})
    grid = np.linspace(-2.0, 2.0, CONTOUR_GRID)
    mesh_1, mesh_2 = np.meshgrid(grid, grid)
    figure, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    for axis, (design, result) in zip(axes[:2], results.items()):
        held = result['pls_optimum'][2]
        flat = np.column_stack([mesh_1.ravel(), mesh_2.ravel(), np.full(mesh_1.size, held)])
        surface = flat @ result['pls_b'] + np.einsum('ij,jk,ik->i', flat, result['pls_matrix'], flat)
        contour = axis.contourf(mesh_1, mesh_2, surface.reshape(mesh_1.shape), levels=18, cmap='viridis')
        figure.colorbar(contour, ax=axis, shrink=0.82)
        axis.plot(result['pls_optimum'][0], result['pls_optimum'][1], marker='o', markersize=7,
                  color=COLORS[2], linestyle='none', label='PLS')
        if result['ols_optimum'] is not None:
            # An open marker, drawn last, so that a PLS point underneath it still shows when the two coincide.
            axis.plot(result['ols_optimum'][0], result['ols_optimum'][1], marker='s', markersize=11,
                      markerfacecolor='none', markeredgewidth=2.0, color=COLORS[3], linestyle='none', label='OLS')
        # Drawn last so it stays readable where all three points coincide, as they do on the orthogonal design.
        axis.plot(true_optimum[0], true_optimum[1], marker='+', markersize=13, markeredgewidth=2.2,
                  color='white', linestyle='none', label='true')
        axis.set_xlabel('$x_1$')
        axis.set_ylabel('$x_2$')
        axis.set_xlim(grid[0], grid[-1])
        axis.set_ylim(grid[0], grid[-1])
        axis.legend(loc='lower left', framealpha=0.85, fontsize=font_size * 0.85)
    groups, positions, tick_labels = [], [], []
    for position, (design, method) in enumerate(itertools.product([str(d) for d in Design], ['pls', 'ols'])):
        selected = distances.loc[(distances['design'] == design) & (distances['method'] == method), 'distance']
        groups.append(selected.dropna().to_numpy())
        positions.append(position + 1)
        tick_labels.append(f"{DESIGN_LABELS[design]}\n{method.upper()}")
    box = axes[2].boxplot(groups, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], [COLORS[2], COLORS[3], COLORS[2], COLORS[3]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for median in box['medians']:
        median.set_color('black')
    axes[2].set_yscale('log')
    axes[2].set_xticks(positions, tick_labels, fontsize=font_size * 0.9)
    axes[2].set_ylabel('distance to the true optimum')
    axes[2].grid(axis='y', alpha=0.3)
    labels = ['(a) central composite design', '(b) correlated operating data',
              f"(c) over {distances['replicate'].nunique()} replicates"]
    for axis, label in zip(axes, labels):
        position = axis.get_position()
        figure.text(position.x0 + position.width / 2.0, 0.02, label, ha='center', va='bottom')
    figure.subplots_adjust(bottom=0.24, wspace=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def run(output_folder: pathlib.Path, n_runs: int, correlation: float, noise_sd: float, max_components: int,
        n_splits: int, seed: int, n_replicates: int) -> pd.DataFrame:
    """Run both designs over `n_replicates` draws and write the response table, the distances and the figure.

    Returns a pd.DataFrame indexed by (design, method), with columns median_distance, q25, q75,
    median_condition, median_components and n_failed.
    """
    if n_replicates < 1:
        raise ValueError(f"n_replicates must be positive, got {n_replicates}.")
    true_optimum = -0.5 * np.linalg.solve(TRUE_QUADRATIC, TRUE_LINEAR)
    ccd = central_composite_design()
    records, first_results, first_responses = [], {}, []
    for replicate in range(n_replicates):
        rng = np.random.default_rng(seed + replicate)
        designs = {
            Design.CCD: ccd,
            Design.CORRELATED: correlated_operating_data(n_runs=n_runs, correlation=correlation, rng=rng),
        }
        for design, x in designs.items():
            y = true_response(x=x, noise_sd=noise_sd, rng=rng)
            result = evaluate_design(x=x, y=y, max_components=max_components, n_splits=n_splits)
            for method, optimum in (('pls', result['pls_optimum']), ('ols', result['ols_optimum'])):
                records.append({
                    'replicate': replicate,
                    'design': str(design),
                    'method': method,
                    'condition_number': result['condition_number'],
                    'n_components': result['n_components'] if method == 'pls' else np.nan,
                    'distance': float('nan') if optimum is None else float(np.linalg.norm(optimum - true_optimum)),
                })
            if replicate == 0:
                first_results[design] = result
                frame = pd.DataFrame(x, columns=[f"x{i + 1}" for i in range(x.shape[1])])
                frame.insert(0, 'design', str(design))
                frame['y'] = y
                first_responses.append(frame)
    distances = pd.DataFrame.from_records(records)
    output_folder.mkdir(parents=True, exist_ok=True)
    pd.concat(first_responses, ignore_index=True).to_csv(output_folder / 'responses.csv', index=False)
    distances.to_csv(output_folder / 'distances.csv', index=False)
    grouped = distances.groupby(['design', 'method'])
    summary = pd.DataFrame({
        'median_distance': grouped['distance'].median(),
        'q25': grouped['distance'].quantile(0.25),
        'q75': grouped['distance'].quantile(0.75),
        'median_condition': grouped['condition_number'].median(),
        'median_components': grouped['n_components'].median(),
        'n_failed': grouped['distance'].apply(lambda column: int(column.isna().sum())),
    })
    summary.to_csv(output_folder / 'summary.csv')
    plot_results(results=first_results, distances=distances, true_optimum=true_optimum,
                 output_path=output_folder / 'README_fig' / 'pls-rsm-surfaces.png')
    print(f"true optimum: {np.array2string(true_optimum, precision=3)}")
    print(summary.to_string(float_format=lambda value: f"{value:.4g}"))
    return summary


def parse_args() -> argparse.Namespace:
    """Parse the command line, validate the combinations, and return the namespace."""
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f"{pathlib.Path(__file__).name} {__version__}\n"
                    "Fit a second-order response surface through PLS and locate its stationary point.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=f"{pathlib.Path(__file__).name} {__version__}")
    parser.add_argument('--output-folder', type=pathlib.Path, default=pathlib.Path(__file__).parent,
                        help='root folder for the response table, the summary and the figure')
    parser.add_argument('--n-runs', type=int, default=20, help='runs in the correlated operating data')
    parser.add_argument('--correlation', type=float, default=0.85,
                        help='how strongly the factors of the correlated design move together, in [0, 1)')
    parser.add_argument('--noise-sd', type=float, default=0.35, help='standard deviation of the response noise')
    parser.add_argument('--max-components', type=int, default=9, help='largest PLS component count to consider')
    parser.add_argument('--n-splits', type=int, default=5, help='folds used to choose the component count')
    parser.add_argument('--n-replicates', type=int, default=200, help='independent draws the comparison is made over')
    parser.add_argument('--seed', type=int, default=20260906, help='seed for the response noise and the settings')
    args = parser.parse_args()
    if not 0.0 <= args.correlation < 1.0:
        parser.error(f"--correlation must be in [0, 1), got {args.correlation}")
    if args.noise_sd < 0.0:
        parser.error(f"--noise-sd must be non-negative, got {args.noise_sd}")
    if args.n_runs < N_FACTORS + 1:
        parser.error(f"--n-runs must exceed the factor count, got {args.n_runs}")
    if args.max_components < 1:
        parser.error(f"--max-components must be positive, got {args.max_components}")
    if args.n_splits < 2:
        parser.error(f"--n-splits must be at least two, got {args.n_splits}")
    if args.n_replicates < 1:
        parser.error(f"--n-replicates must be positive, got {args.n_replicates}")
    if args.output_folder.exists() and not args.output_folder.is_dir():
        parser.error(f"--output-folder is not a folder: {args.output_folder}")
    return args


if __name__ == '__main__':
    arguments = parse_args()
    run(output_folder=arguments.output_folder, n_runs=arguments.n_runs, correlation=arguments.correlation,
        noise_sd=arguments.noise_sd, max_components=arguments.max_components, n_splits=arguments.n_splits,
        seed=arguments.seed, n_replicates=arguments.n_replicates)
