# Math/Graph-Theory/Bipartite-Graph/src/bipartite_layouts.py
"""Draw the bipartite graph figures of the quick start.

Fig 2 puts a general bipartite graph next to the complete bipartite graph K(3, 2) on the same five vertices.
Fig 3 draws one complete bipartite graph K(4, 4) in two layouts: eight vertices on a circle, and two columns.
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
__version__ = "0.1.0+20261030"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD
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
    P_FILL: Final[str] = "#dbe9f6"
    Q_FILL: Final[str] = "#fde5cf"
    SMALL_P: Final[tuple[str, ...]] = ("a", "b", "c")
    SMALL_Q: Final[tuple[str, ...]] = ("d", "e")
    SMALL_EDGES: Final[tuple[tuple[str, str], ...]] = (("a", "d"), ("a", "e"), ("b", "e"), ("c", "d"))

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

    def draw_general_and_complete(self) -> pathlib.Path:
        """Fig 2: (a) a bipartite graph with 4 of the 6 possible edges, (b) the complete bipartite graph K(3, 2)."""
        complete = [(p, q) for p in self.SMALL_P for q in self.SMALL_Q]
        if not set(self.SMALL_EDGES) < set(complete):
            raise RuntimeError(f"the general graph must use a strict subset of the {len(complete)} P-Q pairs")
        heights = {p: h for p, h in zip(self.SMALL_P, np.linspace(1.0, -1.0, len(self.SMALL_P)))}
        heights |= {q: h for q, h in zip(self.SMALL_Q, np.linspace(0.5, -0.5, len(self.SMALL_Q)))}
        positions = {v: np.array([-1.0 if v in self.SMALL_P else 1.0, heights[v]]) for v in heights}
        fill = {p: self.P_FILL for p in self.SMALL_P} | {q: self.Q_FILL for q in self.SMALL_Q}
        labels = [f"(a) Bipartite, {len(self.SMALL_EDGES)} of {len(complete)} possible edges",
                  rf"(b) Complete bipartite $K({len(self.SMALL_P)}, {len(self.SMALL_Q)})$, "
                  rf"$m n = {len(complete)}$ edges"]
        fig, axes = plt.subplots(1, 2, figsize=self.FIGSIZE)
        fig.subplots_adjust(bottom=0.14, wspace=0.15)
        for ax, edges, label in zip(axes, (list(self.SMALL_EDGES), complete), labels):
            graph = nx.Graph(edges)
            nx.draw_networkx_edges(graph, pos=positions, ax=ax, edge_color=self.EDGE_COLOR, width=1.4)
            nx.draw_networkx_nodes(graph, pos=positions, ax=ax, nodelist=list(heights), node_size=self.NODE_SIZE,
                                   node_color=[fill[v] for v in heights], edgecolors=self.EDGE_COLOR, linewidths=1.5)
            nx.draw_networkx_labels(graph, pos=positions, ax=ax, font_size=self.font_size)
            self._set_box(ax=ax, x=-1.0, text=r"$P = \{a, b, c\}$")
            self._set_box(ax=ax, x=1.0, text=r"$Q = \{d, e\}$")
            self._finish(ax=ax, label=label)
        return self._save(fig=fig, name="fig2.png")

    def _finish(self, ax: plt.Axes, label: str) -> None:
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.75, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.5, self.LABEL_Y, label, transform=ax.transAxes, ha="center", va="top", fontsize=self.font_size)

    def _save(self, fig: plt.Figure, name: str) -> pathlib.Path:
        path = self.output_folder / FIGURE_FOLDER_NAME / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def draw_layouts(self) -> pathlib.Path:
        """Fig 3: (a) the circle layout with one edge color, (b) the two-column layout colored by P vertex."""
        fig, axes = plt.subplots(1, 2, figsize=self.FIGSIZE)
        fig.subplots_adjust(bottom=0.14, wspace=0.15)
        circle = self._circle_positions()
        nx.draw_networkx_edges(self.graph, pos=circle, ax=axes[0], edge_color=self.EDGE_COLOR, width=1.2)
        self._nodes(ax=axes[0], positions=circle, fill={node: "white" for node in self.graph.nodes})
        columns = self._column_positions()
        for index, p in enumerate(self.p_set):
            nx.draw_networkx_edges(self.graph, pos=columns, ax=axes[1], edgelist=[(p, q) for q in self.q_set],
                                   edge_color=self.colors[index], width=1.4)
        fill = {p: self.P_FILL for p in self.p_set} | {q: self.Q_FILL for q in self.q_set}
        self._nodes(ax=axes[1], positions=columns, fill=fill)
        self._set_box(ax=axes[1], x=-1.0, text="Set P (odd)")
        self._set_box(ax=axes[1], x=1.0, text="Set Q (even)")
        labels = [f"(a) Circle layout of {self.graph.number_of_nodes()} vertices",
                  r"(b) Two-column layout, $P \cap Q = \emptyset$"]
        for ax, label in zip(axes, labels):
            self._finish(ax=ax, label=label)
        return self._save(fig=fig, name="fig3.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f"{pathlib.Path(__file__).name} {__version__}\n\n{__doc__}",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=__version__)
    parser.add_argument("--output-folder", type=pathlib.Path, required=True,
                        help=f"folder that receives {FIGURE_FOLDER_NAME}/fig2.png and fig3.png")
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
    print(layouts.draw_general_and_complete())
    print(layouts.draw_layouts())
