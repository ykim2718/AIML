# Feature-Engineering/Feature-Selection/BIC/src/bic_model_selection.py
"""Pick the size of a linear model with AIC and with BIC on a dataset whose support is known.

Version: 0.1.0+20260920
"""

import pathlib
from typing import Final, Literal, TypeAlias, get_args

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TABLEAU_COLORS

matplotlib.use("Agg")

__all__ = ["BICSubsetSelection"]

FIGURE_PATH: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parent.parent / "bic_fig/fig2.png"


class BICSubsetSelection:
    """Grow a linear model one feature at a time and score every size with both criteria."""

    Criterion: TypeAlias = Literal["aic", "bic"]
    CRITERIA: Final[tuple[Criterion, ...]] = get_args(Criterion)

    FIGSIZE: Final[tuple[float, float]] = (9.0, 5.0)
    REFERENCE_WIDTH: Final[float] = 9.0
    BASE_FONT_SIZE: Final[float] = 11.0
    EXTRA_PARAMETERS: Final[int] = 2  # the intercept and the noise variance

    def __init__(self, n_samples: int = 200, n_features: int = 10,
                 coefficients: tuple[float, ...] = (3.0, -2.0, 1.5),
                 noise_sigma: float = 1.0, seed: int = 2,
                 figure_path: pathlib.Path = FIGURE_PATH, dpi: int = 300) -> None:
        self.n_samples = n_samples
        self.n_features = n_features
        self.coefficients = coefficients
        self.noise_sigma = noise_sigma
        self.seed = seed
        self.figure_path = figure_path
        self.dpi = dpi
        self.colors: dict[BICSubsetSelection.Criterion, str] = {
            "aic": TABLEAU_COLORS["tab:green"],
            "bic": TABLEAU_COLORS["tab:orange"],
        }
        self.font_size = self.BASE_FONT_SIZE * self.FIGSIZE[0] / self.REFERENCE_WIDTH
        self.x, self.y = self._make_data()

    def _make_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Draw X from a standard normal and build y from the first `len(coefficients)` columns."""
        rng = np.random.default_rng(seed=self.seed)
        x = rng.standard_normal(size=(self.n_samples, self.n_features))
        signal = x[:, :len(self.coefficients)] @ np.asarray(self.coefficients)
        y = signal + rng.normal(loc=0.0, scale=self.noise_sigma, size=self.n_samples)
        return x, y

    def _residual_sum_of_squares(self, columns: list[int]) -> float:
        """Fit an ordinary least squares model on `columns` plus an intercept and return its RSS."""
        design = np.column_stack([np.ones(self.n_samples)] + [self.x[:, c] for c in columns])
        coefficients, _, _, _ = np.linalg.lstsq(design, self.y, rcond=None)
        residual = self.y - design @ coefficients
        return float(residual @ residual)

    def _minus_two_log_likelihood(self, rss: float) -> float:
        """Return the Gaussian deviance with the noise variance at its maximum likelihood value."""
        return self.n_samples * (np.log(2.0 * np.pi) + np.log(rss / self.n_samples) + 1.0)

    def _by_aic(self, rss: float, n_parameters: int) -> float:
        """Score one model with the Akaike information criterion."""
        return self._minus_two_log_likelihood(rss=rss) + 2.0 * n_parameters

    def _by_bic(self, rss: float, n_parameters: int) -> float:
        """Score one model with the Bayesian information criterion."""
        return self._minus_two_log_likelihood(rss=rss) + n_parameters * np.log(self.n_samples)

    def score(self, criterion: "BICSubsetSelection.Criterion", rss: float, n_parameters: int) -> float:
        """Dispatch to the criterion named by `criterion`."""
        if criterion not in self.CRITERIA:
            raise ValueError(f"{criterion!r} not in {self.CRITERIA}")
        return getattr(self, f"_by_{criterion}")(rss=rss, n_parameters=n_parameters)

    def forward_path(self) -> list[dict[str, float]]:
        """Add the feature that lowers the RSS most, and score every model along the path.

        Returns one record per model size with the keys `size`, `added`, `rss`, `deviance`,
        `aic` and `bic`.
        """
        remaining = list(range(self.n_features))
        chosen: list[int] = []
        path: list[dict[str, float]] = []
        while remaining:
            best = min(remaining, key=lambda c: self._residual_sum_of_squares(columns=chosen + [c]))
            chosen.append(best)
            remaining.remove(best)
            rss = self._residual_sum_of_squares(columns=chosen)
            n_parameters = len(chosen) + self.EXTRA_PARAMETERS
            path.append({
                "size": len(chosen),
                "added": best,
                "rss": rss,
                "deviance": self._minus_two_log_likelihood(rss=rss),
                "aic": self.score(criterion="aic", rss=rss, n_parameters=n_parameters),
                "bic": self.score(criterion="bic", rss=rss, n_parameters=n_parameters),
                "chosen": tuple(chosen),
            })
        return path

    def recovery_rate(self, n_repeats: int = 200) -> dict["BICSubsetSelection.Criterion", int]:
        """Count how often each criterion stops on exactly the true support over `n_repeats` draws."""
        support = set(range(len(self.coefficients)))
        hits = {criterion: 0 for criterion in self.CRITERIA}
        for offset in range(n_repeats):
            trial = BICSubsetSelection(n_samples=self.n_samples, n_features=self.n_features,
                                       coefficients=self.coefficients, noise_sigma=self.noise_sigma,
                                       seed=self.seed + offset, figure_path=self.figure_path, dpi=self.dpi)
            path = trial.forward_path()
            for criterion in self.CRITERIA:
                best = min(path, key=lambda record: record[criterion])
                hits[criterion] += int(set(best["chosen"]) == support)
        return hits

    def report(self, path: list[dict[str, float]]) -> str:
        """Lay the path out as one row per model size, marking the minimum of each criterion."""
        best = {criterion: min(path, key=lambda r: r[criterion])["size"] for criterion in self.CRITERIA}
        lines = [f"{'k':>2} {'added':>7} {'RSS':>10} {'-2lnL':>10} {'AIC':>10} {'BIC':>10}  minimum"]
        for record in path:
            marks = [criterion.upper() for criterion in self.CRITERIA if best[criterion] == record["size"]]
            lines.append(f"{int(record['size']):>2} {'x' + str(int(record['added'])):>7} "
                         f"{record['rss']:>10.1f} {record['deviance']:>10.1f} "
                         f"{record['aic']:>10.1f} {record['bic']:>10.1f}  {' '.join(marks)}")
        return "\n".join(lines)

    def draw(self, path: list[dict[str, float]]) -> pathlib.Path:
        """Plot both criteria against the model size and mark where each one takes its minimum."""
        fig, ax = plt.subplots(figsize=self.FIGSIZE)
        sizes = [record["size"] for record in path]
        for criterion in self.CRITERIA:
            values = [record[criterion] for record in path]
            ax.plot(sizes, values, marker="o", color=self.colors[criterion], label=criterion.upper())
            argmin = int(np.argmin(values))
            ax.scatter([sizes[argmin]], [values[argmin]], s=180, facecolors="none",
                       edgecolors=self.colors[criterion], linewidths=2.0, zorder=3)
            ax.annotate(f"{criterion.upper()} minimum at {sizes[argmin]} features",
                        xy=(sizes[argmin], values[argmin]),
                        xytext=(sizes[argmin] + 0.35, values[argmin] + (-26.0 if criterion == "aic" else 22.0)),
                        color=self.colors[criterion], fontsize=self.font_size)
        ax.axvline(len(self.coefficients), color="#90a4ae", linestyle="--", linewidth=1.2)
        ax.text(len(self.coefficients) + 0.1, ax.get_ylim()[1], " true support", va="top",
                color="#455a64", fontsize=self.font_size)
        ax.set_xlabel("Number of features in the model", fontsize=self.font_size)
        ax.set_ylabel("Criterion value", fontsize=self.font_size)
        ax.set_xticks(sizes)
        ax.tick_params(labelsize=self.font_size * 0.9)
        ax.legend(fontsize=self.font_size)
        ax.grid(alpha=0.3)
        self.figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.figure_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return self.figure_path


if __name__ == "__main__":
    selection = BICSubsetSelection()
    forward = selection.forward_path()
    print(selection.report(path=forward))
    print(selection.recovery_rate())
    print(selection.draw(path=forward))
