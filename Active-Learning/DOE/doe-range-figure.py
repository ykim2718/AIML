"""Draw the process response against one parameter, marking the window, the cliff and the two DOE designs.

Changelog:
    0.1.0.2026.8.27 Magnify the narrow design in an inset, since its points overlap at full scale.
    0.0.0.2026.8.27 Initial release.
"""

import argparse
import csv
import enum
import pathlib
import sys

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

__author__ = 'yRocket'
__version__ = "0.1.0.2026.8.27"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

matplotlib.use('Agg')


class Design(enum.StrEnum):
    """DOE design names used as figure and file keys."""

    WIDE = enum.auto()
    NARROW = enum.auto()


PLOT_COLOR: dict = {
    Design.WIDE: TABLEAU_COLORS['tab:red'],
    Design.NARROW: TABLEAU_COLORS['tab:blue'],
}

MARKER: dict = {
    Design.WIDE: 'o',
    Design.NARROW: 's',
}

# Text size before the scale option is applied. One size is used throughout, since the axes carry no ticks.
BASE_FONT_SIZE: float = 9.0

# Header of the file that keeps the sampled points, so the numbers quoted elsewhere can be recomputed.
POINT_FIELD: list = ['design', 'parameter', 'response']


def true_response(parameter: np.ndarray = None, center: float = None, half_width: float = None,
                  sharpness: float = None) -> np.ndarray:
    """Return a response that is flat near the center and collapses on both shoulders.

    A super gaussian is used because a plain gaussian has no flat top. The flat top is the part a process is run
    on, and the steep shoulders are the cliff, so the exponent sets how abruptly the result gives way.
    """
    return 100.0 * np.exp(-(np.abs(parameter - center) / half_width) ** sharpness)


def level_edges(center: float = None, half_width: float = None, sharpness: float = None,
                level: float = None) -> tuple:
    """Solve the response for the two parameter values where it crosses the given level.

    The response is symmetric about the center, so one edge is solved and the other is mirrored. Crossing the
    specification gives the window edges, and crossing a collapsed level gives the outer end of the cliff.
    """
    if not 0.0 < level < 100.0:
        raise ValueError(f"level must lie between 0 and 100 for the edges to exist: {level}")

    offset = half_width * (-np.log(level / 100.0)) ** (1.0 / sharpness)

    return center - offset, center + offset


def sample_design(design: Design = None, center: float = None, span: float = None, n_points: int = None,
                  noise: float = None, rng: np.random.Generator = None, **response_kwargs) -> tuple:
    """Place points evenly across the design span and read the response with measurement noise added."""
    parameter = np.linspace(center - span, center + span, n_points)
    response = true_response(parameter=parameter, center=center, **response_kwargs)

    return parameter, response + rng.normal(loc=0.0, scale=noise, size=n_points)


def write_points(points: dict = None, out_path: pathlib.Path = None) -> None:
    """Write one row per sampled point so that the figure can be redrawn from values rather than read off."""
    with out_path.open(mode='w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=POINT_FIELD)
        writer.writeheader()
        for design, (parameter, response) in points.items():
            for one_parameter, one_response in zip(parameter, response):
                writer.writerow({
                    'design': design.value,
                    'parameter': f"{one_parameter:.4f}",
                    'response': f"{one_response:.4f}",
                })


def add_narrow_inset(axis: plt.Axes = None, points: dict = None, center: float = None,
                     half_width: float = None, sharpness: float = None, zoom_factor: float = None,
                     font_size: float = None) -> None:
    """Magnify the narrow design, whose points land on top of one another at the scale of the wide one.

    The inset keeps the same response curve and the same center line, so the reader sees that the whole inset
    is the single mark the parent axes shows, rather than a second and unrelated picture.
    """
    parameter, response = points[Design.NARROW]
    zoom_span = zoom_factor * 0.5 * (parameter.max() - parameter.min())
    if zoom_span <= 0.0:
        raise ValueError("the narrow design has no span, so there is nothing for the inset to magnify")

    inset = axis.inset_axes([0.33, 0.22, 0.34, 0.30])
    grid = np.linspace(center - zoom_span, center + zoom_span, 400)
    inset.plot(grid, true_response(parameter=grid, center=center, half_width=half_width, sharpness=sharpness),
               color=TABLEAU_COLORS['tab:gray'], linewidth=1.6)
    inset.axvline(center, color=TABLEAU_COLORS['tab:purple'], linestyle=':', linewidth=1.4)
    inset.plot(parameter, response, linestyle='none', marker=MARKER[Design.NARROW], markersize=8.0,
               color=PLOT_COLOR[Design.NARROW], markeredgecolor='black', markeredgewidth=0.6)

    spread = response.max() - response.min()
    inset.set_xlim(center - zoom_span, center + zoom_span)
    inset.set_ylim(response.min() - 0.7 * spread, response.max() + 0.7 * spread)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title('narrow DOE magnified', fontsize=font_size)

    axis.indicate_inset_zoom(inset, edgecolor='black', linewidth=1.0, alpha=0.8)


def plot_range(points: dict = None, center: float = None, half_width: float = None, sharpness: float = None,
               specification: float = None, collapse: float = None, zoom_factor: float = None,
               font_scale: float = None, out_path: pathlib.Path = None, dpi: int = None) -> tuple:
    """Draw the response curve with the window shaded, the cliff shaded beside it and both designs overlaid.

    The cliff is shaded only over the steep shoulder, between the window edge and the parameter at which the
    response has collapsed to the given level. Beyond that the response is flat again and nothing is at stake,
    so shading it as cliff would overstate where the result is still giving way.
    """
    lower, upper = level_edges(center=center, half_width=half_width, sharpness=sharpness, level=specification)
    cliff_low, cliff_high = level_edges(center=center, half_width=half_width, sharpness=sharpness,
                                        level=collapse)

    wide_parameter = points[Design.WIDE][0]
    axis_low, axis_high = wide_parameter.min(), wide_parameter.max()
    grid = np.linspace(axis_low, axis_high, 2000)
    curve = true_response(parameter=grid, center=center, half_width=half_width, sharpness=sharpness)

    font_size = BASE_FONT_SIZE * font_scale
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(9.5, 6.4))

    axis.axvspan(lower, upper, color=TABLEAU_COLORS['tab:green'], alpha=0.15, label='process window')
    axis.axvspan(cliff_low, lower, color=TABLEAU_COLORS['tab:orange'], alpha=0.20, label='process cliff')
    axis.axvspan(upper, cliff_high, color=TABLEAU_COLORS['tab:orange'], alpha=0.20)

    axis.plot(grid, curve, color=TABLEAU_COLORS['tab:gray'], linewidth=1.8, label='true response')
    axis.axhline(specification, color=TABLEAU_COLORS['tab:gray'], linestyle='--', linewidth=1.1,
                 label='specification')
    axis.axvline(center, color=TABLEAU_COLORS['tab:purple'], linestyle=':', linewidth=1.4,
                 label='POR center')

    for design, (parameter, response) in points.items():
        axis.plot(parameter, response, linestyle='none', marker=MARKER[design], markersize=7.5,
                  color=PLOT_COLOR[design], markeredgecolor='black', markeredgewidth=0.6,
                  label=f"{design.value} DOE  ({parameter.size} points)")

    add_narrow_inset(axis=axis, points=points, center=center, half_width=half_width, sharpness=sharpness,
                     zoom_factor=zoom_factor, font_size=font_size)

    axis.annotate('narrow DOE piles up at the center', xy=(center, points[Design.NARROW][1].max()),
                  xytext=(center - 98.0, 118.0), fontsize=font_size,
                  arrowprops={'arrowstyle': '->', 'color': 'black', 'linewidth': 0.9})
    axis.annotate('window edge is set by the specification', xy=(upper, specification),
                  xytext=(upper + 6.0, specification + 16.0), fontsize=font_size,
                  arrowprops={'arrowstyle': '->', 'color': 'black', 'linewidth': 0.9})
    axis.annotate('cliff is set by the response', xy=(upper + 20.0, true_response(
        parameter=np.array([upper + 20.0]), center=center, half_width=half_width, sharpness=sharpness)[0]),
        xytext=(upper + 24.0, 55.0), fontsize=font_size,
        arrowprops={'arrowstyle': '->', 'color': 'black', 'linewidth': 0.9})

    axis.set_xlabel('Process parameter', fontsize=font_size)
    axis.set_ylabel('Response', fontsize=font_size)
    axis.set_xlim(axis_low, axis_high)
    axis.set_ylim(-6.0, 132.0)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fontsize=font_size, ncol=3,
                frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return lower, upper


def build_figure(out_folder: pathlib.Path = None, center: float = None, half_width: float = None,
                 sharpness: float = None, specification: float = None, collapse: float = None,
                 wide_span: float = None,
                 narrow_span: float = None, n_wide: int = None, n_narrow: int = None, noise: float = None,
                 zoom_factor: float = None, font_scale: float = None, seed: int = None, dpi: int = None) -> None:
    """Sample both designs from the same response and write the figure and the sampled points."""
    rng = np.random.default_rng(seed)
    response_kwargs = {'half_width': half_width, 'sharpness': sharpness}

    points: dict = {
        Design.WIDE: sample_design(design=Design.WIDE, center=center, span=wide_span, n_points=n_wide,
                                   noise=noise, rng=rng, **response_kwargs),
        Design.NARROW: sample_design(design=Design.NARROW, center=center, span=narrow_span, n_points=n_narrow,
                                     noise=noise, rng=rng, **response_kwargs),
    }

    lower, upper = plot_range(points=points, center=center, half_width=half_width, sharpness=sharpness,
                              specification=specification, collapse=collapse, zoom_factor=zoom_factor,
                              font_scale=font_scale,
                              out_path=out_folder / 'fig1_doe_range.png', dpi=dpi)
    write_points(points=points, out_path=out_folder / 'fig1_doe_range_points.csv')

    print(f"process window: {lower:.1f} to {upper:.1f}")


def parse_args() -> argparse.Namespace:
    """Read the command line options and reject a geometry that leaves the figure meaningless."""
    parser = argparse.ArgumentParser(description="Draw the DOE range figure for the wide and narrow DOE note.")
    parser.add_argument('--out-folder', type=str, required=True,
                        help="folder the figure is written into; it is created when missing")
    parser.add_argument('--center', type=float, default=400.0,
                        help="parameter value the process is centered on")
    parser.add_argument('--half-width', type=float, default=45.0,
                        help="parameter offset at which the response falls to 1/e of its peak")
    parser.add_argument('--sharpness', type=float, default=6.0,
                        help="super gaussian exponent; larger gives a flatter top and a steeper cliff")
    parser.add_argument('--specification', type=float, default=90.0,
                        help="response the result must reach, which is what fixes the window edges")
    parser.add_argument('--collapse', type=float, default=10.0,
                        help="response level treated as collapsed, which fixes the outer end of the cliff")
    parser.add_argument('--wide-span', type=float, default=100.0,
                        help="half span of the wide design, measured from the center")
    parser.add_argument('--narrow-span', type=float, default=2.0,
                        help="half span of the narrow design, measured from the center")
    parser.add_argument('--n-wide', type=int, default=9,
                        help="number of points in the wide design")
    parser.add_argument('--n-narrow', type=int, default=9,
                        help="number of points in the narrow design")
    parser.add_argument('--noise', type=float, default=1.2,
                        help="standard deviation of the measurement noise added to the response")
    parser.add_argument('--zoom-factor', type=float, default=3.0,
                        help="how many narrow half spans the magnified inset reaches on each side of the center")
    parser.add_argument('--font-scale', type=float, default=1.5,
                        help="multiplier applied to every text size in the figure")
    parser.add_argument('--seed', type=int, default=0,
                        help="seed of the random generator, so the figure is reproducible")
    parser.add_argument('--dpi', type=int, default=300,
                        help="resolution of the saved figure")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    out_folder = pathlib.Path(args.out_folder)
    if out_folder.exists() and not out_folder.is_dir():
        parser.error(f"--out-folder is not a folder: {out_folder}")
    out_folder.mkdir(parents=True, exist_ok=True)
    args.out_folder = out_folder

    if args.half_width <= 0.0:
        parser.error(f"--half-width must be positive: {args.half_width}")
    if args.sharpness <= 1.0:
        parser.error(f"--sharpness must exceed 1, otherwise the response has no flat top: {args.sharpness}")
    if not 0.0 < args.specification < 100.0:
        parser.error(f"--specification must lie between 0 and 100 to cross the response: {args.specification}")
    if not 0.0 < args.collapse < args.specification:
        parser.error(f"--collapse must lie between 0 and --specification, since the cliff runs below the "
                     f"window edge: {args.collapse}, {args.specification}")
    if args.narrow_span <= 0.0 or args.wide_span <= args.narrow_span:
        parser.error(f"--wide-span must exceed --narrow-span, and both must be positive: "
                     f"{args.wide_span}, {args.narrow_span}")
    if args.n_wide < 3 or args.n_narrow < 3:
        parser.error(f"each design needs at least 3 points: {args.n_wide}, {args.n_narrow}")
    if args.noise < 0.0:
        parser.error(f"--noise must not be negative: {args.noise}")
    if args.zoom_factor <= 1.0:
        parser.error(f"--zoom-factor must exceed 1, otherwise the inset cuts the design off: {args.zoom_factor}")
    if args.font_scale <= 0.0:
        parser.error(f"--font-scale must be positive: {args.font_scale}")
    if args.dpi <= 0:
        parser.error(f"--dpi must be positive: {args.dpi}")

    return args


if __name__ == '__main__':
    cli = parse_args()
    build_figure(out_folder=cli.out_folder, center=cli.center, half_width=cli.half_width,
                 sharpness=cli.sharpness, specification=cli.specification, collapse=cli.collapse,
                 wide_span=cli.wide_span,
                 narrow_span=cli.narrow_span, n_wide=cli.n_wide, n_narrow=cli.n_narrow, noise=cli.noise,
                 zoom_factor=cli.zoom_factor, font_scale=cli.font_scale, seed=cli.seed, dpi=cli.dpi)
