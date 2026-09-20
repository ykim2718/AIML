# Feature-Engineering/Feature-Selection/BIC/src/bic_cartoon.py
"""Draw the cartoon that puts the AIC and the BIC formula side by side.

Version: 0.1.0+20260920
"""

import pathlib
from typing import Final, Literal, TypeAlias, get_args

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.patches import FancyBboxPatch

matplotlib.use("Agg")

__all__ = ["CriterionCartoon"]

FIGURE_PATH: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parent.parent / "bic_fig/fig1.png"


class CriterionCartoon:
    """Render one panel per information criterion, each formula with its terms called out."""

    Criterion: TypeAlias = Literal["aic", "bic"]
    CRITERIA: Final[tuple[Criterion, ...]] = get_args(Criterion)

    FIGSIZE: Final[tuple[float, float]] = (9.0, 5.8)
    REFERENCE_WIDTH: Final[float] = 9.0
    BASE_FONT_SIZE: Final[float] = 11.0
    PANEL_TOP: Final[tuple[float, float]] = (0.80, 0.42)  # the top edge of each panel box

    SEGMENTS: Final[dict[Criterion, tuple[str, ...]]] = {
        "aic": (r"$\mathrm{AIC}$", r"$\,=\,$", r"$-2\,\ln(L)$", r"$\,+\,$", r"$2k$"),
        "bic": (r"$\mathrm{BIC}$", r"$\,=\,$", r"$-2\,\ln(L)$", r"$\,+\,$", r"$k\,\ln(n)$"),
    }
    HEADINGS: Final[dict[Criterion, str]] = {
        "aic": "Akaike Information Criterion",
        "bic": "Bayesian Information Criterion",
    }
    CALLOUTS: Final[dict[Criterion, tuple[tuple[int, float, str], ...]]] = {
        "aic": ((2, 0.20, "$L$: maximized likelihood"), (4, 0.82, "$k$: number of parameters")),
        "bic": ((2, 0.20, "the fit term of AIC, unchanged"), (4, 0.82, "$n$: number of observations")),
    }
    FOOTERS: Final[dict[Criterion, str]] = {
        "aic": "Every parameter costs 2, whatever the sample size",
        "bic": "Every parameter costs $\\ln(n)$, so a larger sample buys fewer parameters",
    }

    def __init__(self, figure_path: pathlib.Path = FIGURE_PATH, dpi: int = 300) -> None:
        self.figure_path = figure_path
        self.dpi = dpi
        self.colors: dict[CriterionCartoon.Criterion, str] = {
            "aic": TABLEAU_COLORS["tab:green"],
            "bic": TABLEAU_COLORS["tab:orange"],
        }
        self.font_size = self.BASE_FONT_SIZE * self.FIGSIZE[0] / self.REFERENCE_WIDTH

    def draw(self) -> pathlib.Path:
        """Write the cartoon to `figure_path` and return that path."""
        fig, ax = plt.subplots(figsize=self.FIGSIZE)
        ax.set_axis_off()
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.text(0.5, 0.95, "Choose the best model", ha="center", va="center",
                fontsize=self.font_size * 1.9, fontweight="bold", color="#263238")
        for criterion, top in zip(self.CRITERIA, self.PANEL_TOP):
            self._panel(fig=fig, ax=ax, criterion=criterion, top=top)
        ax.text(0.5, 0.02, "Lower value wins. $\\ln(n)$ passes 2 at $n = 8$, "
                            "where BIC starts dropping parameters that AIC keeps.",
                ha="center", va="center", fontsize=self.font_size, color="#263238")
        self.figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.figure_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return self.figure_path

    def _panel(self, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes,
               criterion: "CriterionCartoon.Criterion", top: float) -> None:
        """Draw one criterion: its heading, its boxed formula, the callouts and the footer."""
        if criterion not in self.CRITERIA:
            raise ValueError(f"{criterion!r} not in {self.CRITERIA}")
        color = self.colors[criterion]
        ax.text(0.06, top, criterion.upper(), ha="left", va="center", color=color,
                fontsize=self.font_size * 1.5, fontweight="bold")
        ax.text(0.17, top, self.HEADINGS[criterion], ha="left", va="center", color=color,
                fontsize=self.font_size * 1.2)
        centers = self._formula(fig=fig, ax=ax, criterion=criterion, y=top - 0.11, color=color)
        for index, x_label, text in self.CALLOUTS[criterion]:
            ax.annotate(text, xy=(centers[index], top - 0.155), xytext=(x_label, top - 0.245),
                        ha="center", va="center", fontsize=self.font_size * 0.95, color="#455a64",
                        arrowprops=dict(arrowstyle="->", color="#90a4ae",
                                        connectionstyle="arc3,rad=0.25"))
        ax.text(0.5, top - 0.305, self.FOOTERS[criterion], ha="center", va="center",
                fontsize=self.font_size * 0.95, color=color)

    def _formula(self, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes,
                 criterion: "CriterionCartoon.Criterion", y: float, color: str) -> list[float]:
        """Lay the segments out left to right around x = 0.5 and return the center of each."""
        size = self.font_size * 1.7
        widths = [self._width(fig=fig, ax=ax, text=segment, size=size)
                  for segment in self.SEGMENTS[criterion]]
        left = 0.5 - sum(widths) / 2.0
        centers: list[float] = []
        for segment, width in zip(self.SEGMENTS[criterion], widths):
            ax.text(left, y, segment, ha="left", va="center", fontsize=size, color="#263238")
            centers.append(left + width / 2.0)
            left += width
        pad = 0.025
        ax.add_patch(FancyBboxPatch((0.5 - sum(widths) / 2.0 - pad, y - 0.055), sum(widths) + 2 * pad, 0.11,
                                    boxstyle="round,pad=0.012", linewidth=1.4,
                                    edgecolor=color, facecolor=color + "18" if False else "white",
                                    zorder=0))
        return centers

    def _width(self, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes,
               text: str, size: float) -> float:
        """Measure the width of `text` in axes coordinates by drawing and removing it."""
        handle = ax.text(0.0, 0.0, text, fontsize=size)
        fig.canvas.draw()
        extent = handle.get_window_extent(renderer=fig.canvas.get_renderer())
        handle.remove()
        return extent.transformed(ax.transAxes.inverted()).width


if __name__ == "__main__":
    print(CriterionCartoon().draw())
