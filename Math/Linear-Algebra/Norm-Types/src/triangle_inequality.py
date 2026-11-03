# Math/Linear-Algebra/Norm-Types/src/triangle_inequality.py
"""Draw the triangle inequality of the types of norms document.

Fig 3 (a) puts y at the tip of x so that x, y and x + y are the three sides of a triangle, where the L2 norm of x + y
is smaller than the sum of the two L2 norms. Fig 3 (b) points y in the same direction as x, where the triangle lies on
one line and the two sides are equal; x + y is drawn shifted below that line so that it stays visible.
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
__version__ = "0.0.0+20261103"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD
__all__ = ["TriangleInequality"]

FIGURE_FOLDER_NAME: Final[str] = "norm-types-ko_fig"


class TriangleInequality:
    """Draw x, y and x + y head to tail for a triangle and for two vectors in the same direction."""

    X_TRIANGLE: Final[tuple[float, float]] = (2.0, 2.0)
    Y_TRIANGLE: Final[tuple[float, float]] = (2.5, -2.0)
    X_LINE: Final[tuple[float, float]] = (2.0, 1.0)
    Y_LINE: Final[tuple[float, float]] = (2.4, 1.2)  # same direction as X_LINE
    FIGSIZE: Final[tuple[float, float]] = (11.0, 4.6)
    REFERENCE_WIDTH: Final[float] = 11.0
    BASE_FONT_SIZE: Final[float] = 12.0
    LINE_WIDTH: Final[float] = 3.0
    LABEL_Y: Final[float] = 0.02  # panel label height in figure coordinates, below both panels
    RELATION_Y: Final[float] = 0.14  # height of the norm relation in figure coordinates, above the panel labels
    MARGIN: Final[float] = 0.6
    TOLERANCE: Final[float] = 1e-12
    SUM_SHIFT_LINE: Final[tuple[float, float]] = (0.25, -0.5)  # moves x + y off the line of x and y in panel (b)

    def __init__(self, output_folder: pathlib.Path, dpi: int = 300) -> None:
        self.output_folder = output_folder
        self.dpi = dpi
        self.font_size = self.BASE_FONT_SIZE * self.FIGSIZE[0] / self.REFERENCE_WIDTH
        self.colors = list(TABLEAU_COLORS.values())

    def report(self) -> str:
        """Check the strict inequality of panel (a) and the equality of panel (b) with the L2 norm."""
        lines = []
        for name, x, y, strict in (("(a)", self.X_TRIANGLE, self.Y_TRIANGLE, True),
                                   ("(b)", self.X_LINE, self.Y_LINE, False)):
            x, y = np.array(x), np.array(y)
            left, right = float(np.linalg.norm(x + y)), float(np.linalg.norm(x) + np.linalg.norm(y))
            if strict and not left < right - self.TOLERANCE:
                raise RuntimeError(f"panel {name} is not a strict inequality: {left=}, {right=}")
            if not strict and abs(left - right) > self.TOLERANCE:
                raise RuntimeError(f"panel {name} is not an equality: {left=}, {right=}")
            lines.append(f"{name} ||x + y|| = {left:.3f}, ||x|| + ||y|| = {right:.3f}")
        return "\n".join(lines)

    def _arrow(self, ax: plt.Axes, start: np.ndarray, vector: np.ndarray, color: str, label: str,
               offset: tuple[float, float]) -> None:
        ax.annotate("", xy=start + vector, xytext=start,
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": self.LINE_WIDTH, "mutation_scale": 22})
        middle = start + vector / 2.0
        ax.text(middle[0] + offset[0], middle[1] + offset[1], label, color=color, fontsize=self.font_size,
                ha="center", va="center")

    def _panel(self, ax: plt.Axes, x: np.ndarray, y: np.ndarray, x_offset: tuple[float, float],
               y_offset: tuple[float, float], sum_offset: tuple[float, float],
               sum_shift: tuple[float, float] = (0.0, 0.0)) -> None:
        origin = np.zeros(2)
        ax.fill([0.0, x[0], x[0] + y[0]], [0.0, x[1], x[1] + y[1]], color=self.colors[2], alpha=0.08)
        self._arrow(ax=ax, start=origin + np.array(sum_shift), vector=x + y, color=self.colors[3], label=r"$\mathbf{x} + \mathbf{y}$",
                    offset=sum_offset)
        self._arrow(ax=ax, start=origin, vector=x, color=self.colors[2], label=r"$\mathbf{x}$", offset=x_offset)
        self._arrow(ax=ax, start=x, vector=y, color=self.colors[0], label=r"$\mathbf{y}$", offset=y_offset)
        points = np.array([origin, x, x + y, origin + np.array(sum_shift), x + y + np.array(sum_shift)])
        ax.set_xlim(points[:, 0].min() - self.MARGIN, points[:, 0].max() + self.MARGIN)
        ax.set_ylim(points[:, 1].min() - self.MARGIN * 2.0, points[:, 1].max() + self.MARGIN)
        ax.set_aspect("equal")
        ax.axis("off")

    def draw(self) -> pathlib.Path:
        """Fig 3: (a) x, y and x + y as a triangle, (b) x and y in the same direction."""
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=self.FIGSIZE)
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.24, top=0.98, wspace=0.1)
        self._panel(ax=ax_a, x=np.array(self.X_TRIANGLE), y=np.array(self.Y_TRIANGLE),
                    x_offset=(-0.35, 0.3), y_offset=(0.35, 0.3), sum_offset=(0.0, -0.35))
        self._panel(ax=ax_b, x=np.array(self.X_LINE), y=np.array(self.Y_LINE),
                    x_offset=(-0.25, 0.4), y_offset=(-0.25, 0.4), sum_offset=(0.3, -0.45),
                    sum_shift=self.SUM_SHIFT_LINE)
        relations = (r"$\Vert \mathbf{x} + \mathbf{y} \Vert < \Vert \mathbf{x} \Vert + \Vert \mathbf{y} \Vert$",
                     r"$\Vert \mathbf{x} + \mathbf{y} \Vert = \Vert \mathbf{x} \Vert + \Vert \mathbf{y} \Vert$")
        for ax, label, relation in zip((ax_a, ax_b), ("(a) Different directions", "(b) Same direction"), relations):
            box = ax.get_position()
            fig.text((box.x0 + box.x1) / 2.0, self.RELATION_Y, relation, ha="center", va="bottom",
                     fontsize=self.font_size)
            fig.text((box.x0 + box.x1) / 2.0, self.LABEL_Y, label, ha="center", va="bottom", fontsize=self.font_size)
        return self._save(fig=fig, name="fig3.png")

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
                        help=f"folder that receives {FIGURE_FOLDER_NAME}/fig3.png")
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
    figure = TriangleInequality(output_folder=options.output_folder)
    print(figure.report())
    print(figure.draw())
