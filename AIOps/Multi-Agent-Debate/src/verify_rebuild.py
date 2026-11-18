"""Check the rebuilt PLS response surface against the acceptance criteria of the final spec.

AC-1 to AC-3 test the rebuild alone. AC-4 fits the original model (pls_rsm.py) and the rebuild to the same data,
a rotatable central composite design and correlated operating data over many replicates, and compares the results.
With --spec draft, the rebuild from the draft spec v0 (pls_rsm_rebuilt_v0.py) is compared instead, on the three
outputs v0 names, since v0 has no acceptance criterion that can be run.

Changelog:
- 0.1.0+20261118: add --spec draft to compare the rebuild from the draft spec v0
- 0.0.0+20261118: initial release
"""

__author__ = 'yRocket'
__version__ = "0.1.0+20261118"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib
import sys
from typing import Final, Literal, TypeAlias, get_args

import numpy as np
import pandas as pd

import pls_rsm
import pls_rsm_rebuilt
import pls_rsm_rebuilt_v0

__all__ = [
    'check_column_order',
    'check_too_few_rows',
    'check_singular_matrix',
    'compare_with_original',
]

EXPECTED_COLUMNS: list = ['x1', 'x2', 'x3', 'x1^2', 'x2^2', 'x3^2', 'x1x2', 'x1x3', 'x2x3']
Spec: TypeAlias = Literal['final', 'draft']
SPECS: Final[tuple] = get_args(Spec)
COMPARED_KEYS: dict = {
    'final': ('coefficients', 'cv_rmse', 'stationary_point', 'eigenvalues'),
    'draft': ('stationary_point', 'eigenvalues'),
}


def check_column_order() -> bool:
    """AC-1: for k = 3 the expanded columns come in the order the spec fixes."""
    _, names = pls_rsm_rebuilt.expand_quadratic(x=np.zeros((4, 3)))
    return names == EXPECTED_COLUMNS


def check_too_few_rows() -> bool:
    """AC-2: two rows leave no component count, which raises ValueError."""
    x = np.array([[-1.0, 0.5, 0.0], [1.0, -0.5, 0.3]])
    try:
        pls_rsm_rebuilt.fit_surface(x=x, y=np.array([1.0, 2.0]))
    except ValueError:
        return True
    return False


def check_singular_matrix() -> bool:
    """AC-3: a singular B raises numpy.linalg.LinAlgError."""
    try:
        pls_rsm_rebuilt.stationary_point(b=np.array([1.0, 1.0]), matrix=np.diag([1.0, 0.0]))
    except np.linalg.LinAlgError:
        return True
    return False


def _fit_original(x: np.ndarray, y: np.ndarray, max_components: int, n_splits: int) -> dict:
    """Fit the original model through its own functions, in the order its evaluate_design calls them."""
    x_expanded, names = pls_rsm.expand_quadratic(x=x)
    coefficients, n_components, cv_rmse = pls_rsm.fit_pls_surface(
        x_expanded=x_expanded, y=y, max_components=max_components, n_splits=n_splits)
    b, matrix = pls_rsm.quadratic_from_coefficients(coefficients=coefficients, names=names, n_factors=x.shape[1])
    point, eigenvalues = pls_rsm.stationary_point(b=b, matrix=matrix)
    return {'coefficients': coefficients, 'n_components': n_components, 'cv_rmse': cv_rmse,
            'stationary_point': point, 'eigenvalues': eigenvalues}


def _fit_rebuilt(spec: Spec, x: np.ndarray, y: np.ndarray, max_components: int, n_splits: int) -> dict:
    """Fit the rebuild made from the given spec; the draft rebuild takes no search settings, as v0 names none."""
    if spec not in SPECS:
        raise ValueError(f"spec must be one of {SPECS}, got {spec!r}.")
    if spec == 'draft':
        return pls_rsm_rebuilt_v0.fit_surface(x=x, y=y)
    return pls_rsm_rebuilt.fit_surface(x=x, y=y, max_components=max_components, n_splits=n_splits)


def compare_with_original(spec: Spec, n_replicates: int, seed: int, n_runs: int, correlation: float,
                          noise_sd: float, max_components: int, n_splits: int, rtol: float,
                          atol: float) -> pd.DataFrame:
    """AC-4: fit both models to the same data and record, per replicate and design, whether they agree.

    Returns a pd.DataFrame with a default RangeIndex and columns replicate, design, n_components_original,
    n_components_rebuilt, one max_abs_diff_<key> column per key of COMPARED_KEYS[spec], and match.
    """
    records = []
    compared: tuple = COMPARED_KEYS[spec]
    ccd = pls_rsm.central_composite_design()
    for replicate in range(n_replicates):
        rng = np.random.default_rng(seed + replicate)
        designs = {
            str(pls_rsm.Design.CCD): ccd,
            str(pls_rsm.Design.CORRELATED): pls_rsm.correlated_operating_data(
                n_runs=n_runs, correlation=correlation, rng=rng),
        }
        for design, x in designs.items():
            y = pls_rsm.true_response(x=x, noise_sd=noise_sd, rng=rng)
            original = _fit_original(x=x, y=y, max_components=max_components, n_splits=n_splits)
            rebuilt = _fit_rebuilt(spec=spec, x=x, y=y, max_components=max_components, n_splits=n_splits)
            same_count = original['n_components'] == rebuilt['n_components']
            same_length = spec == 'draft' or len(original['cv_rmse']) == len(rebuilt['cv_rmse'])
            record = {
                'replicate': replicate,
                'design': design,
                'n_components_original': original['n_components'],
                'n_components_rebuilt': rebuilt['n_components'],
            }
            match = same_count and same_length
            for key in compared:
                left = np.asarray(original[key], dtype=float)
                right = np.asarray(rebuilt[key], dtype=float)
                if left.shape != right.shape:
                    record[f"max_abs_diff_{key}"] = float('nan')
                    match = False
                    continue
                record[f"max_abs_diff_{key}"] = float(np.max(np.abs(left - right)))
                match = match and bool(np.allclose(left, right, rtol=rtol, atol=atol))
            record['match'] = match
            records.append(record)
    return pd.DataFrame.from_records(records)


def parse_args() -> argparse.Namespace:
    """Parse the command line, validate the values, and return the namespace."""
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f"{pathlib.Path(__file__).name} {__version__}\n"
                    "Check the rebuilt PLS response surface against the acceptance criteria of the final spec.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=f"{pathlib.Path(__file__).name} {__version__}")
    parser.add_argument('--spec', choices=list(SPECS), default='final',
                        help='final: rebuild from the final spec, AC-1 to AC-4; draft: rebuild from v0, AC-4 only')
    parser.add_argument('--output-folder', type=pathlib.Path, required=True,
                        help='folder for the per-replicate comparison table')
    parser.add_argument('--n-replicates', type=int, default=200, help='independent draws compared in AC-4')
    parser.add_argument('--seed', type=int, default=20260906, help='seed of the first replicate')
    parser.add_argument('--n-runs', type=int, default=20, help='runs in the correlated operating data')
    parser.add_argument('--correlation', type=float, default=0.85, help='factor correlation, in [0, 1)')
    parser.add_argument('--noise-sd', type=float, default=0.35, help='standard deviation of the response noise')
    parser.add_argument('--max-components', type=int, default=9, help='largest PLS component count')
    parser.add_argument('--n-splits', type=int, default=5, help='folds used to choose the component count')
    parser.add_argument('--rtol', type=float, default=1e-9, help='relative tolerance of AC-4')
    parser.add_argument('--atol', type=float, default=1e-9, help='absolute tolerance of AC-4')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if args.n_replicates < 1:
        parser.error(f"--n-replicates must be positive, got {args.n_replicates}")
    if not 0.0 <= args.correlation < 1.0:
        parser.error(f"--correlation must be in [0, 1), got {args.correlation}")
    if args.noise_sd < 0.0:
        parser.error(f"--noise-sd must be non-negative, got {args.noise_sd}")
    if args.rtol < 0.0 or args.atol < 0.0:
        parser.error(f"--rtol and --atol must be non-negative, got {args.rtol} and {args.atol}")
    if args.output_folder.exists() and not args.output_folder.is_dir():
        parser.error(f"--output-folder is not a folder: {args.output_folder}")
    return args


if __name__ == '__main__':
    arguments = parse_args()
    checks = {}
    if arguments.spec == 'final':
        checks = {'AC-1': check_column_order(), 'AC-2': check_too_few_rows(), 'AC-3': check_singular_matrix()}
    comparison = compare_with_original(
        spec=arguments.spec, n_replicates=arguments.n_replicates, seed=arguments.seed, n_runs=arguments.n_runs,
        correlation=arguments.correlation, noise_sd=arguments.noise_sd, max_components=arguments.max_components,
        n_splits=arguments.n_splits, rtol=arguments.rtol, atol=arguments.atol)
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(arguments.output_folder / f"rebuild_comparison_{arguments.spec}.csv", index=False)
    checks['AC-4'] = bool(comparison['match'].all())
    for name, passed in checks.items():
        print(f"{name}: {'pass' if passed else 'FAIL'}")
    same_count = int((comparison['n_components_original'] == comparison['n_components_rebuilt']).sum())
    print(f"AC-4 matched {int(comparison['match'].sum())} of {len(comparison)} fits; "
          f"same component count in {same_count}; "
          f"largest stationary point difference {comparison['max_abs_diff_stationary_point'].max():.3g}; "
          f"median stationary point difference {comparison['max_abs_diff_stationary_point'].median():.3g}")
    if not all(checks.values()):
        sys.exit(1)
