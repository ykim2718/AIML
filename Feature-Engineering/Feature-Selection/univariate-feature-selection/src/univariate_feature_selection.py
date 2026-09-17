__author__ = "yRocket"
__version__ = "0.1.0+20260913"

import textwrap
from typing import Literal

import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import (SelectKBest, chi2, f_classif, mutual_info_classif,
                                       r_regression)


class UnivariateFeatureSelector:
    """Score every feature alone against the target and keep the k best ones.

    The scoring method name is a Literal that lists the metrics of section 4, so the code says which
    one is applied. A name outside that list raises ValueError.

    Args:
        k: number of features the cut-off keeps.
        random_state: seed of the mutual information estimate.
    """

    def __init__(self, k: int = 2, random_state: int = 0) -> None:
        if k < 1:
            raise ValueError(f"k must be at least 1: {k=}")
        self.k = k
        self.random_state = random_state

    def drop_constant(self, X: np.ndarray) -> np.ndarray:
        """Return the columns whose value changes across the samples, dropping the constant ones."""
        varying = np.where(np.ptp(X, axis=0) > 0.0)[0]
        if len(varying) == 0:
            raise ValueError(f"every feature holds one value: {X.shape=}")
        return varying

    def score(self, X: np.ndarray, y: np.ndarray,
              method: Literal["chi2", "anova", "pearson", "mutual_info"] = "chi2") -> np.ndarray:
        """Return the score of every feature, in column order, under the named metric."""
        if method == "chi2":
            return chi2(X, y)[0]
        if method == "anova":
            return f_classif(X, y)[0]
        if method == "pearson":
            return np.abs(r_regression(X, y))
        if method == "mutual_info":
            return mutual_info_classif(X, y, random_state=self.random_state)
        raise ValueError(f"unknown scoring method: {method=}")

    def run(self, X: np.ndarray, y: np.ndarray,
            method: Literal["chi2", "anova", "pearson", "mutual_info"] = "chi2") -> dict:
        """Return the varying columns, their scores, and the columns left after the cut-off.

        The constant features go first, before any test is run: their score is undefined and they
        carry nothing the target can be told apart by.
        """
        varying = self.drop_constant(X=X)
        if self.k > len(varying):
            raise ValueError(f"k asks for more features than vary: {self.k=}, {len(varying)=}")
        selector = SelectKBest(score_func=lambda features, target: self.score(X=features, y=target, method=method),
                               k=self.k).fit(X[:, varying], y)
        return {"varying": varying, "scores": selector.scores_, "columns": varying[selector.get_support()]}


if __name__ == "__main__":
    data = load_iris()
    # Append a column that never changes, to show the first filter removing it
    X = np.column_stack([data.data, np.full(len(data.data), 3.0)])
    names = np.asarray(list(data.feature_names) + ["constant probe"])
    selector = UnivariateFeatureSelector(k=2)

    def show(label: str, columns: np.ndarray) -> None:
        """Print the label with the count, then the feature names in alphabetical order."""
        print(f"\n{label} ({len(columns)} features)")
        print(textwrap.fill(", ".join(sorted(names[columns])), width=100,
                            initial_indent="  ", subsequent_indent="  "))

    print(f"input: {X.shape[0]} samples, {X.shape[1]} features")
    show(label="input", columns=np.arange(X.shape[1]))
    show(label="left by the constant filter", columns=selector.drop_constant(X=X))

    for scoring_method in ("chi2", "anova", "pearson", "mutual_info"):
        result = selector.run(X=X, y=data.target, method=scoring_method)
        show(label=f"selected by {scoring_method}", columns=result["columns"])
        scores = ", ".join(f"{name} {score:.2f}"
                           for name, score in sorted(zip(names[result["varying"]], result["scores"])))
        print(textwrap.fill(scores, width=100, initial_indent="  scores: ", subsequent_indent="  "))
