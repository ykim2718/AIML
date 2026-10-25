#!/usr/bin/env python3
"""Compute wafer uniformity indices and draw two wafer maps that share the same index value.

The figure and the numbers the accompanying document quotes are produced here, so the picture and
the text come from one run.
"""
__author__ = 'yRocket'
__version__ = "0.0.0.2026.9.4"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

__all__ = ['measurement_points', 'radial_thickness', 'uniformity_indices', 'range_bias_frame',
           'draw_wafer_maps']

FIGSIZE: tuple = (12.0, 4.4)
REFERENCE_WIDTH: float = 12.0    # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 10.0
DPI: int = 300
WAFER_RADIUS_MM: float = 150.0
EDGE_EXCLUSION_MM: float = 3.0
RING_RADII: tuple = (0.0, 0.25, 0.50, 0.70, 0.85, 0.95)   # as a fraction of the wafer radius
RING_COUNTS: tuple = (1, 6, 8, 10, 12, 12)                # 49 points in total
NOMINAL_THICKNESS_A: float = 1000.0
CURVATURE: float = 0.06          # amplitude of the radial term, as a fraction of the nominal
SAMPLE_SIZES: tuple = (5, 9, 13, 17, 21, 25, 49, 121)
RANGE_TRIALS: int = 2_000_000
CURVE_COLORS: tuple = tuple(TABLEAU_COLORS.values())


def measurement_points(ring_radii: tuple = RING_RADII, ring_counts: tuple = RING_COUNTS,
                       wafer_radius: float = WAFER_RADIUS_MM) -> pd.DataFrame:
    """Concentric-ring measurement pattern.

    Returns a pd.DataFrame with a RangeIndex and columns 'x_mm', 'y_mm' and 'r_mm'.
    """
    if len(ring_radii) != len(ring_counts):
        raise ValueError(f"ring_radii and ring_counts must be the same length; "
                         f"got {len(ring_radii)} and {len(ring_counts)}")
    if max(ring_radii) * wafer_radius > wafer_radius - EDGE_EXCLUSION_MM:
        raise ValueError(f"outermost ring at {max(ring_radii) * wafer_radius:.1f} mm falls inside the "
                         f"{EDGE_EXCLUSION_MM} mm edge exclusion zone")
    x, y = [], []
    for fraction, count in zip(ring_radii, ring_counts):
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        x.extend(fraction * wafer_radius * np.cos(angles))
        y.extend(fraction * wafer_radius * np.sin(angles))
    frame = pd.DataFrame({'x_mm': x, 'y_mm': y})
    frame['r_mm'] = np.hypot(frame['x_mm'], frame['y_mm'])
    return frame


def radial_thickness(r_mm: np.ndarray, centre_high: bool, site_radii: np.ndarray,
                     nominal: float = NOMINAL_THICKNESS_A, curvature: float = CURVATURE,
                     wafer_radius: float = WAFER_RADIUS_MM) -> np.ndarray:
    """Thickness of a purely radial profile, thick at the centre or thick at the edge.

    The radial term is centred on site_radii, the radii of the measurement pattern, so that both
    signs put the mean measured thickness at the nominal. The two profiles then carry byte-identical
    uniformity indices while looking nothing alike, which is what makes them worth comparing.
    """
    site_radii = np.asarray(site_radii, dtype=float)
    if site_radii.size < 2:
        raise ValueError(f"site_radii must hold the measurement pattern; got {site_radii.size} radii")
    shape = lambda r: 1.0 - 2.0 * (r / wafer_radius) ** 2  # noqa: E731 - one expression, used twice
    centred = shape(r_mm) - shape(site_radii).mean()
    return nominal * (1.0 + (curvature if centre_high else -curvature) * centred)


def uniformity_indices(values: np.ndarray) -> pd.Series:
    """The three standard indices of one wafer.

    Returns a pd.Series indexed by 'mean', 'std', 'range_pct', 'sigma1_pct' and 'sigma3_pct', where
    the percentages are the half-range, the one-sigma and the three-sigma non-uniformity.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError(f"at least 2 measurements are needed; got {values.size}")
    mean = values.mean()
    if mean <= 0:
        raise ValueError(f"the mean must be positive to normalise by it; got {mean}")
    std = values.std(ddof=1)
    return pd.Series({'mean': mean,
                      'std': std,
                      'range_pct': (values.max() - values.min()) / (2.0 * mean) * 100.0,
                      'sigma1_pct': std / mean * 100.0,
                      'sigma3_pct': 3.0 * std / mean * 100.0})


def range_bias_frame(sample_sizes: tuple = SAMPLE_SIZES, trials: int = RANGE_TRIALS,
                     seed: int = 0) -> pd.DataFrame:
    """How the half-range index grows with the point count on a wafer that has no spatial signature.

    Returns a pd.DataFrame indexed by 'n_points' with columns 'expected_range_over_sigma' and
    'range_pct_at_cv_1pct', the latter being the half-range index of a wafer whose one-sigma index
    is exactly 1 percent.
    """
    rng = np.random.default_rng(seed)
    rows = {}
    for n in sample_sizes:
        draws = rng.standard_normal(size=(trials, n))
        d2 = float((draws.max(axis=1) - draws.min(axis=1)).mean())
        rows[n] = {'expected_range_over_sigma': d2, 'range_pct_at_cv_1pct': d2 / 2.0}
    frame = pd.DataFrame(rows).T
    frame.index.name = 'n_points'
    return frame


def draw_wafer_maps(points: pd.DataFrame, output_path: pathlib.Path,
                    wafer_radius: float = WAFER_RADIUS_MM) -> pathlib.Path:
    """Draw the two wafer maps and their radial profiles side by side and save the figure."""
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    grid = np.linspace(-wafer_radius, wafer_radius, 400)
    mesh_x, mesh_y = np.meshgrid(grid, grid)
    mesh_r = np.hypot(mesh_x, mesh_y)
    outside = mesh_r > wafer_radius - EDGE_EXCLUSION_MM
    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=FIGSIZE)
    figure.subplots_adjust(bottom=0.24, wspace=0.55)
    site_radii = points['r_mm'].to_numpy()
    fields = {high: np.ma.masked_where(outside, radial_thickness(mesh_r, centre_high=high, site_radii=site_radii))
              for high in (True, False)}
    levels = np.linspace(min(f.min() for f in fields.values()), max(f.max() for f in fields.values()), 25)
    for axis, centre_high, name in zip(axes[:2], (True, False), ('Centre thick', 'Edge thick')):
        field = fields[centre_high]
        contour = axis.contourf(mesh_x, mesh_y, field, levels=levels, cmap='viridis')
        axis.scatter(points['x_mm'], points['y_mm'], s=6, c='white', edgecolors='black', linewidths=0.3)
        axis.set_aspect('equal')
        axis.set_xlabel(f'{name}  |  x (mm)', fontsize=font_size)
        axis.set_ylabel('y (mm)', fontsize=font_size)
        axis.tick_params(labelsize=font_size * 0.85)
        bar = figure.colorbar(contour, ax=axis, fraction=0.046, pad=0.04)
        bar.ax.tick_params(labelsize=font_size * 0.8)
    radii = np.linspace(0.0, wafer_radius - EDGE_EXCLUSION_MM, 200)
    for centre_high, label, color in zip((True, False), ('Centre thick', 'Edge thick'), CURVE_COLORS):
        axes[2].plot(radii, radial_thickness(radii, centre_high=centre_high, site_radii=site_radii),
                     color=color, linewidth=1.6, label=label)
    axes[2].set_xlabel('Radius (mm)', fontsize=font_size)
    axes[2].set_ylabel('Thickness (A)', fontsize=font_size)
    axes[2].tick_params(labelsize=font_size * 0.85)
    axes[2].legend(fontsize=font_size * 0.9, frameon=False)
    axes[2].grid(visible=True, alpha=0.25)
    for axis, label in zip(axes, ('(a)', '(b)', '(c)')):
        position = axis.get_position()
        figure.text(position.x0 + position.width / 2.0, 0.045, label, ha='center', va='center',
                    fontsize=font_size)
    figure.savefig(output_path, dpi=DPI)
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f'{pathlib.Path(__file__).name} {__version__}\n'
                    'Compute the wafer uniformity indices and draw the two wafer maps that share them.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--output-folder', type=pathlib.Path, required=True,
                        help='folder that receives the figure and the csv tables; created if absent')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    arguments = parser.parse_args()
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    return arguments


if __name__ == '__main__':
    args = parse_args()
    sites = measurement_points()
    site_radii = sites['r_mm'].to_numpy()
    sites['centre_thick_A'] = radial_thickness(site_radii, centre_high=True, site_radii=site_radii)
    sites['edge_thick_A'] = radial_thickness(site_radii, centre_high=False, site_radii=site_radii)
    indices = pd.DataFrame({'centre_thick': uniformity_indices(sites['centre_thick_A'].to_numpy()),
                            'edge_thick': uniformity_indices(sites['edge_thick_A'].to_numpy())}).T
    bias = range_bias_frame()
    sites.to_csv(args.output_folder / 'wafer_measurements.csv', index=False)
    indices.to_csv(args.output_folder / 'uniformity_indices.csv')
    bias.to_csv(args.output_folder / 'range_bias.csv')
    figure_path = draw_wafer_maps(points=sites, output_path=args.output_folder / 'wafer_uniformity_index.png')
    print(f'figure  {figure_path}')
    print(f'measurement points: {len(sites)}')
    print(indices.round(4).to_string())
    print(bias.round(3).to_string())
