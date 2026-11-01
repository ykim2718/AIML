# Math/Linear-Algebra/Null-Space/src/null_space_rank_nullity.py
"""Draw the rank-nullity theorem for the 3 x 3 matrix of the null space primer.

Fig 1 (a) shows input points in R^3 and the null space line of A. Fig 1 (b) shows where A sends the same points:
every image lies on the column space plane, and every point of the null space line goes to the origin.
"""

import argparse
import pathlib
import sys
from typing import Final

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TABLEAU_COLORS

matplotlib.use("Agg")

__author__ = 'yRocket'
__version__ = "0.0.2+20261031"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD
__all__ = ["RankNullity"]

FIGURE_FOLDER_NAME: Final[str] = "null-space-primer-ko_fig"


class RankNullity:
    """Map a grid of input points and the null space line through A and plot the input and output spaces."""

    MATRIX: Final[tuple[tuple[float, ...], ...]] = ((1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 2.0))
    NULL_VECTOR: Final[tuple[float, ...]] = (-1.0, -1.0, 1.0)  # n of equation (6)
    FIGSIZE: Final[tuple[float, float]] = (11.0, 5.4)
    REFERENCE_WIDTH: Final[float] = 11.0
    BASE_FONT_SIZE: Final[float] = 10.0
    INPUT_LIMIT: Final[float] = 1.6
    OUTPUT_LIMIT: Final[float] = 4.2  # A sends the corner (1, 1, 1) to (2, 2, 4)
    LABEL_Y: Final[float] = 0.02  # panel label height in figure coordinates, below both panels
    TOLERANCE: Final[float] = 1e-10
    ELEVATION: Final[float] = 25.0
    INPUT_AZIMUTH: Final[float] = -58.0  # shows the null space line across the cube
    OUTPUT_AZIMUTH: Final[float] = -130.0  # looks along the plane normal (-1, -1, 1) so the plane shows its face

    def __init__(self, output_folder: pathlib.Path, n_grid: int = 5, dpi: int = 300) -> None:
        if n_grid < 2:
            raise ValueError(f"n_grid must be at least 2 to span the cube: {n_grid=}")
        self.output_folder = output_folder
        self.n_grid = n_grid
        self.dpi = dpi
        self.font_size = self.BASE_FONT_SIZE * self.FIGSIZE[0] / self.REFERENCE_WIDTH
        self.colors = list(TABLEAU_COLORS.values())
        self.a = np.array(self.MATRIX)
        self.n = np.array(self.NULL_VECTOR)
        axis = np.linspace(-1.0, 1.0, self.n_grid)
        self.grid = np.array(np.meshgrid(axis, axis, axis)).reshape(3, -1)  # 3 x N input points
        self.line = np.outer(self.n, np.linspace(-1.2, 1.2, 25))  # 3 x 25 points on Null(A)

    def report(self) -> str:
        """Check A n = 0, the rank and nullity, and that every image lies on the column space plane."""
        if not np.allclose(self.a @ self.n, 0.0, atol=self.TOLERANCE):
            raise RuntimeError(f"A n is not the zero vector: {self.a @ self.n}")
        rank = int(np.linalg.matrix_rank(self.a))
        nullity = self.a.shape[1] - rank
        normal = np.cross(self.a[:, 0], self.a[:, 1])  # normal of the plane spanned by the first two columns
        off_plane = float(np.abs(normal @ (self.a @ self.grid)).max())
        if off_plane > self.TOLERANCE:
            raise RuntimeError(f"an image leaves the column space plane by {off_plane}")
        return (f"rank: {rank}, nullity: {nullity}, rank + nullity: {rank + nullity} = n = {self.a.shape[1]}\n"
                f"largest |A x| on the null space line: {np.abs(self.a @ self.line).max():.3g}\n"
                f"normal of the column space plane: {normal.tolist()}, largest distance of an image: {off_plane:.3g}")

    def _frame(self, ax: plt.Axes, limit: float, symbol: str, azim: float) -> None:
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.set_xlabel(symbol.format(k=1), fontsize=self.font_size)
        ax.set_ylabel(symbol.format(k=2), fontsize=self.font_size)
        ax.set_zlabel(symbol.format(k=3), fontsize=self.font_size)
        ax.tick_params(labelsize=self.font_size * 0.8)
        ax.view_init(elev=self.ELEVATION, azim=azim)

    def _plane(self, ax: plt.Axes, first: np.ndarray, second: np.ndarray, extent: float, color: str) -> None:
        s, t = np.meshgrid(np.linspace(-extent, extent, 2), np.linspace(-extent, extent, 2))
        points = [first[k] * s + second[k] * t for k in range(3)]
        ax.plot_surface(*points, color=color, alpha=0.18, linewidth=0.0)

    def draw(self) -> pathlib.Path:
        """Fig 1: (a) input space with the null space line, (b) output space with the column space plane."""
        images = self.a @ self.grid
        fig = plt.figure(figsize=self.FIGSIZE)
        ax_in = fig.add_subplot(1, 2, 1, projection="3d")
        ax_out = fig.add_subplot(1, 2, 2, projection="3d")
        fig.subplots_adjust(left=0.0, right=0.97, bottom=0.1, top=1.0, wspace=0.05)

        ax_in.scatter(*self.grid, s=6.0, color=self.colors[7], alpha=0.6, label="input points x")
        ax_in.plot(*self.line, color=self.colors[1], linewidth=2.5, label=r"Null($\mathbf{A}$), nullity = 1")
        ax_in.quiver(0.0, 0.0, 0.0, *self.n, color=self.colors[3], linewidth=2.0, arrow_length_ratio=0.15)
        ax_in.text(*(self.n * 1.15), r"$\mathbf{n}$", color=self.colors[3], fontsize=self.font_size)
        self._frame(ax=ax_in, limit=self.INPUT_LIMIT, symbol=r"$x_{k}$", azim=self.INPUT_AZIMUTH)
        ax_in.legend(loc="upper left", fontsize=self.font_size * 0.85, frameon=False)

        self._plane(ax=ax_out, first=self.a[:, 0], second=self.a[:, 1], extent=2.0, color=self.colors[0])
        ax_out.scatter(*images, s=6.0, color=self.colors[7], alpha=0.6, label=r"images $\mathbf{A}\mathbf{x}$")
        ax_out.scatter(0.0, 0.0, 0.0, s=60.0, color=self.colors[1], zorder=5,
                       label=r"image of Null($\mathbf{A}$): origin")
        ax_out.plot([], [], color=self.colors[0], linewidth=6.0, alpha=0.4,
                    label=r"column space, rank = 2")  # legend proxy for the translucent plane
        self._frame(ax=ax_out, limit=self.OUTPUT_LIMIT, symbol=r"$(\mathbf{{A}}\mathbf{{x}})_{k}$",
                    azim=self.OUTPUT_AZIMUTH)
        ax_out.legend(loc="upper left", fontsize=self.font_size * 0.85, frameon=False)

        for ax, label in zip((ax_in, ax_out), (r"(a) Input space $\mathbb{R}^3$", r"(b) Output space $\mathbb{R}^3$")):
            box = ax.get_position()
            fig.text((box.x0 + box.x1) / 2.0, self.LABEL_Y, label, ha="center", va="bottom", fontsize=self.font_size)
        return self._save(fig=fig, name="fig1.png")

    def _save(self, fig: plt.Figure, name: str) -> pathlib.Path:
        path = self.output_folder / FIGURE_FOLDER_NAME / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f"{pathlib.Path(__file__).name} {__version__}\n\n{__doc__}",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=__version__)
    parser.add_argument("--output-folder", type=pathlib.Path, required=True,
                        help=f"folder that receives {FIGURE_FOLDER_NAME}/fig1.png")
    parser.add_argument("--n-grid", type=int, default=5, help="input points per axis of the cube (default: 5)")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    if not args.output_folder.is_dir():
        parser.print_help()
        sys.exit(f"\n--output-folder is not an existing folder: {args.output_folder}")
    return args


if __name__ == "__main__":
    options = parse_args()
    figure = RankNullity(output_folder=options.output_folder, n_grid=options.n_grid)
    print(figure.report())
    print(figure.draw())
