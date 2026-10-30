# Math/Graph-Theory/Bipartite-Graph/src/bipartite_layouts.py
"""Draw one complete bipartite graph K(4, 4) in two layouts: eight vertices on a circle, and two columns.

The circle hides the partition; the two columns show that every edge runs between the odd and the even vertices.
"""

import argparse
import pathlib
import sys
from typing import Final

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.patches import FancyBboxPatch

matplotlib.use("Agg")

__author__ = 'yRocket'
__version__ = "0.0.0+20261030"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD
__all__ = ["BipartiteLayouts"]

FIGURE_FOLDER_NAME: Final[str] = "bipartite-graph-quick-start-ko_fig"


class BipartiteLayouts:
    """Build K(n, n) on vertices 1..2n with odd and even vertices as the two sets, and draw it twice."""

    FIGSIZE: Final[tuple[float, float]] = (10.0, 4.6)
    REFERENCE_WIDTH: Final[float] = 10.0
    BASE_FONT_SIZE: Final[float] = 11.0
    NODE_SIZE: Final[float] = 520.0
    LABEL_Y: Final[float] = -0.06  # panel label height in axes coordinates
    EDGE_COLOR: Final[str] = "#1f3a5f"

    def __init__(self, output_folder: pathlib.Path, n_per_set: int = 4, dpi: int = 300) -> None:
        if not 2 <= n_per_set <= len(TABLEAU_COLORS):
            raise ValueError(f"n_per_set must be in [2, {len(TABLEAU_COLORS)}] to give each P vertex a color: "
                             f"{n_per_set=}")
        self.output_folder = output_folder
        self.n_per_set = n_per_set
        self.dpi = dpi
        self.font_size = self.BASE_FONT_SIZE * self.FIGSIZE[0] / self.REFERENCE_WIDTH
        self.colors = list(TABLEAU_COLORS.values())
        self.p_set = list(range(1, 2 * n_per_set + 1, 2))  # odd vertices
        self.q_set = list(range(2, 2 * n_per_set + 1, 2))  # even vertices
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(1, 2 * n_per_set + 1))
        self.graph.add_edges_from((p, q) for p in self.p_set for q in self.q_set)

    def report(self) -> str:
        """Confirm the graph is bipartite with the odd and even vertices as its two sets."""
        if not nx.is_bipartite(self.graph):
            raise RuntimeError("the graph built from odd-even edges is not bipartite")
        left, right = nx.algorithms.bipartite.sets(self.graph)
        if sorted([sorted(left), sorted(right)]) != sorted([self.p_set, self.q_set]):
            raise RuntimeError(f"bipartite sets {sorted(left)}, {sorted(right)} differ from the odd and even vertices")
        return (f"vertices: {self.graph.number_of_nodes()}, edges: {self.graph.number_of_edges()}\n"
                f"P = {self.p_set}, Q = {self.q_set}, bipartite: True")

    def _circle_positions(self) -> dict[int, np.ndarray]:
        count = self.graph.number_of_nodes()
        angles = np.pi / 2.0 - 2.0 * np.pi * np.arange(count) / count  # vertex 1 at the top, clockwise
        return {node: np.array([np.cos(angle), np.sin(angle)]) for node, angle in zip(range(1, count + 1), angles)}

    def _column_positions(self) -> dict[int, np.ndarray]:
        heights = np.linspace(1.0, -1.0, self.n_per_set)
        positions = {p: np.array([-1.0, h]) for p, h in zip(self.p_set, heights)}
        positions.update({q: np.array([1.0, h]) for q, h in zip(self.q_set, heights)})
        return positions

    def _nodes(self, ax: plt.Axes, positions: dict[int, np.ndarray], fill: dict[int, str]) -> None:
        nx.draw_networkx_nodes(self.graph, pos=positions, ax=ax, node_size=self.NODE_SIZE,
                               node_color=[fill[node] for node in self.graph.nodes], edgecolors=self.EDGE_COLOR,
                               linewidths=1.5)
        nx.draw_networkx_labels(self.graph, pos=positions, ax=ax, font_size=self.font_size)

    def _set_box(self, ax: plt.Axes, x: float, text: str) -> None:
        ax.add_patch(FancyBboxPatch((x - 0.22, -1.25), 0.44, 2.5, boxstyle="round,pad=0.02", fill=False,
                                    linestyle="--", edgecolor="#607d8b", linewidth=1.0))
        ax.text(x, -1.42, text, ha="center", va="top", fontsize=self.font_size)

    def draw(self) -> pathlib.Path:
        """Fig 2: (a) the circle layout with one edge color, (b) the two-column layout colored by P vertex."""
        fig, axes = plt.subplots(1, 2, figsize=self.FIGSIZE)
        fig.subplots_adjust(bottom=0.14, wspace=0.15)
        circle = self._circle_positions()
        nx.draw_networkx_edges(self.graph, pos=circle, ax=axes[0], edge_color=self.EDGE_COLOR, width=1.2)
        self._nodes(ax=axes[0], positions=circle, fill={node: "white" for node in self.graph.nodes})
        columns = self._column_positions()
        for index, p in enumerate(self.p_set):
            nx.draw_networkx_edges(self.graph, pos=columns, ax=axes[1], edgelist=[(p, q) for q in self.q_set],
                                   edge_color=self.colors[index], width=1.4)
        fill = {p: "#dbe9f6" for p in self.p_set} | {q: "#fde5cf" for q in self.q_set}
        self._nodes(ax=axes[1], positions=columns, fill=fill)
        self._set_box(ax=axes[1], x=-1.0, text="Set P (odd)")
        self._set_box(ax=axes[1], x=1.0, text="Set Q (even)")
        labels = [f"(a) Circle layout of {self.graph.number_of_nodes()} vertices",
                  r"(b) Two-column layout, $P \cap Q = \emptyset$"]
        for ax, label in zip(axes, labels):
            ax.set_xlim(-1.45, 1.45)
            ax.set_ylim(-1.75, 1.3)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.text(0.5, self.LABEL_Y, label, transform=ax.transAxes, ha="center", va="top", fontsize=self.font_size)
        path = self.output_folder / FIGURE_FOLDER_NAME / "fig2.png"
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
                        help=f"folder that receives {FIGURE_FOLDER_NAME}/fig2.png")
    parser.add_argument("--n-per-set", type=int, default=4, help="vertices in each of the two sets (default: 4)")
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
    layouts = BipartiteLayouts(output_folder=options.output_folder, n_per_set=options.n_per_set)
    print(layouts.report())
    print(layouts.draw())
