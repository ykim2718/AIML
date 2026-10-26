# Math/Linear-Algebra/Singular-Value-Decomposition/src/svd_circle_to_ellipse.py
"""Draw how the points of a unit circle move to an ellipse under the three factors of an SVD.

Fig 1 follows the points through V^T, Sigma and U of one 2 x 2 matrix. Fig 2 keeps U and V fixed and
changes the second singular value, so that the ellipse runs from a circle to a line segment.
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
__version__ = "0.0.1+20261026"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD
__all__ = ["CircleToEllipse"]

FIGURE_FOLDER_NAME: Final[str] = "singular-value-decomposition-primer-ko_fig"


class CircleToEllipse:
    """Map a matrix of unit circle points through the SVD factors of a 2 x 2 matrix and plot each stage."""

    MATRIX: Final[tuple[tuple[float, float], ...]] = ((3.0, 0.0), (4.0, 5.0))
    FIGSIZE: Final[tuple[float, float]] = (12.0, 3.9)
    REFERENCE_WIDTH: Final[float] = 12.0
    BASE_FONT_SIZE: Final[float] = 10.0
    AXIS_LIMIT: Final[float] = 8.5  # leaves room for the arrow labels at sigma1 = 6.708
    UNIT_AXIS_LIMIT: Final[float] = 1.6  # panels that still hold the unit circle
    LABEL_Y: Final[float] = -0.22  # panel label height in axes coordinates, below the tick labels
    SECOND_SINGULAR_VALUES: Final[tuple[str, ...]] = ("sigma1", "sqrt5", "1", "0")

    def __init__(self, output_folder: pathlib.Path, n_points: int = 360, dpi: int = 300) -> None:
        if n_points < 8:
            raise ValueError(f"n_points must be at least 8 to draw a closed curve: {n_points=}")
        self.output_folder = output_folder
        self.n_points = n_points
        self.dpi = dpi
        self.font_size = self.BASE_FONT_SIZE * self.FIGSIZE[0] / self.REFERENCE_WIDTH
        self.colors = list(TABLEAU_COLORS.values())
        self.a = np.array(self.MATRIX)
        u, s, vt = np.linalg.svd(self.a)
        # numpy returns both singular vector pairs with flipped signs; flip back to match equation (7)
        sign = np.sign(u[0, :])
        self.u, self.s, self.vt = u * sign, s, (vt.T * sign).T
        theta = 2.0 * np.pi * np.arange(self.n_points) / self.n_points
        self.points = np.vstack([np.cos(theta), np.sin(theta)])  # 2 x N matrix P

    def stages(self) -> list[np.ndarray]:
        """Return P, V^T P, Sigma V^T P and U Sigma V^T P, each a 2 x N matrix."""
        rotated = self.vt @ self.points
        scaled = np.diag(self.s) @ rotated
        return [self.points, rotated, scaled, self.u @ scaled]

    def report(self) -> str:
        """Print the singular values and the semi-axis lengths measured on the mapped points."""
        mapped = self.stages()[-1]
        if not np.allclose(mapped, self.a @ self.points):
            raise RuntimeError("U Sigma V^T P does not reproduce A P; the sign flip broke the factorisation")
        lengths = np.linalg.norm(mapped, axis=0)
        farthest = mapped[:, int(np.argmax(lengths))] / lengths.max()
        farthest = farthest * np.sign(farthest[0])  # the two ends of the major axis are equally far
        return (f"singular values: {self.s.round(3).tolist()}\n"
                f"longest and shortest image of {self.n_points} points: "
                f"{lengths.max():.3f}, {lengths.min():.3f}\n"
                f"direction of the longest image: {farthest.round(3).tolist()} (u1 = {self.u[:, 0].round(3).tolist()})")

    def _frame(self, ax: plt.Axes, label: str, limit: float) -> None:
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.axhline(0.0, color="#b0bec5", linewidth=0.8)
        ax.axvline(0.0, color="#b0bec5", linewidth=0.8)
        ax.tick_params(labelsize=self.font_size * 0.85)
        ax.text(0.5, self.LABEL_Y, label, transform=ax.transAxes, ha="center", va="top", fontsize=self.font_size)

    def _arrow(self, ax: plt.Axes, vector: np.ndarray, color: str, text: str) -> None:
        ax.annotate("", xy=tuple(vector), xytext=(0.0, 0.0),
                    arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.8})
        ax.text(vector[0] * 1.08, vector[1] * 1.08, text, color=color, fontsize=self.font_size,
                ha="left" if vector[0] >= 0 else "right", va="bottom" if vector[1] >= 0 else "top")

    def draw_stages(self) -> pathlib.Path:
        """Fig 1: the unit circle points before and after each factor, with the tracked axes."""
        titles = [r"(a) $\mathbf{P}$", r"(b) $\mathbf{V}^\top \mathbf{P}$", r"(c) $\Sigma\mathbf{V}^\top \mathbf{P}$",
                  r"(d) $\mathbf{U}\Sigma\mathbf{V}^\top \mathbf{P} = \mathbf{A}\mathbf{P}$"]
        start = self.points[:, :1]  # the point at theta = 0, tracked to show the reflections
        tracked = [start, self.vt @ start, np.diag(self.s) @ self.vt @ start,
                   self.u @ np.diag(self.s) @ self.vt @ start]
        axes_by_stage = [(self.vt.T[:, 0], self.vt.T[:, 1], r"$\mathbf{v}_1$", r"$\mathbf{v}_2$"),
                         (np.array([1.0, 0.0]), np.array([0.0, 1.0]), r"$\mathbf{e}_1$", r"$\mathbf{e}_2$"),
                         (np.array([self.s[0], 0.0]), np.array([0.0, self.s[1]]), r"$\sigma_1$", r"$\sigma_2$"),
                         (self.s[0] * self.u[:, 0], self.s[1] * self.u[:, 1],
                          r"$\sigma_1\mathbf{u}_1$", r"$\sigma_2\mathbf{u}_2$")]
        fig, axes = plt.subplots(1, 4, figsize=self.FIGSIZE)
        fig.subplots_adjust(bottom=0.2, wspace=0.3)
        for ax, stage, title, point, (first, second, first_text, second_text) in zip(
                axes, self.stages(), titles, tracked, axes_by_stage):
            ax.scatter(stage[0], stage[1], s=2.0, color=self.colors[7])
            self._arrow(ax=ax, vector=first, color=self.colors[0], text=first_text)
            self._arrow(ax=ax, vector=second, color=self.colors[1], text=second_text)
            ax.scatter(point[0], point[1], s=30.0, color=self.colors[2], zorder=3)
            self._frame(ax=ax, label=title,
                        limit=self.UNIT_AXIS_LIMIT if np.abs(stage).max() <= 1.0 + 1e-9 else self.AXIS_LIMIT)
        return self._save(fig=fig, name="fig1.png")

    def draw_second_value(self) -> pathlib.Path:
        """Fig 2: keep U, V and sigma1, and shrink sigma2 from sigma1 to zero."""
        values = {"sigma1": self.s[0], "sqrt5": self.s[1], "1": 1.0, "0": 0.0}
        fig, axes = plt.subplots(1, 4, figsize=self.FIGSIZE)
        fig.subplots_adjust(bottom=0.2, wspace=0.3)
        for index, (ax, key) in enumerate(zip(axes, self.SECOND_SINGULAR_VALUES)):
            sigma = np.array([self.s[0], values[key]])
            mapped = self.u @ np.diag(sigma) @ self.vt @ self.points
            ax.scatter(self.points[0], self.points[1], s=1.0, color=self.colors[7], alpha=0.5)
            ax.scatter(mapped[0], mapped[1], s=2.0, color=self.colors[0])
            self._arrow(ax=ax, vector=sigma[0] * self.u[:, 0], color=self.colors[0], text=r"$\sigma_1\mathbf{u}_1$")
            if sigma[1] > 0.0:
                self._arrow(ax=ax, vector=sigma[1] * self.u[:, 1], color=self.colors[1],
                            text=r"$\sigma_2\mathbf{u}_2$")
            self._frame(ax=ax, label=f"({'abcd'[index]}) " + rf"$\sigma_2 = {sigma[1]:.3f}$", limit=self.AXIS_LIMIT)
        return self._save(fig=fig, name="fig2.png")

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
                        help=f"folder that receives {FIGURE_FOLDER_NAME}/fig1.png and fig2.png")
    parser.add_argument("--n-points", type=int, default=360, help="number of unit circle points (default: 360)")
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
    figure = CircleToEllipse(output_folder=options.output_folder, n_points=options.n_points)
    print(figure.report())
    print(figure.draw_stages())
    print(figure.draw_second_value())
