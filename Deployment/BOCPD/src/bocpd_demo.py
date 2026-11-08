"""Bayesian Online Change Point Detection (BOCPD) demo on a synthetic VM residual stream.

Gaussian observations with unknown mean and variance, Normal-Gamma conjugate prior,
constant hazard H = 1 / lambda. Prints P(r_t = 0), P(r_t <= w) and the MAP run length.
"""
__author__ = 'yRocket'
__version__ = "0.0.0+20261108"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD
__all__ = ['NormalGammaBocpd', 'make_stream']

import argparse
import math
import pathlib
import sys

import numpy as np


class NormalGammaBocpd:
    """BOCPD with a Normal-Gamma prior (mu0, kappa0, alpha0, beta0) and constant hazard 1 / hazard_lambda."""

    def __init__(self, hazard_lambda: float = 100.0, mu0: float = 0.0, kappa0: float = 1.0,
                 alpha0: float = 1.0, beta0: float = 1.0):
        if hazard_lambda <= 1.0:
            raise ValueError(f"hazard_lambda must be > 1, got {hazard_lambda=}")
        if min(kappa0, alpha0, beta0) <= 0.0:
            raise ValueError(f"kappa0, alpha0, beta0 must be > 0, got {kappa0=}, {alpha0=}, {beta0=}")
        self.hazard = 1.0 / hazard_lambda
        self.prior = (mu0, kappa0, alpha0, beta0)
        self.run_prob = np.array([1.0])  # P(r_0 = 0) = 1
        self.mu = np.array([mu0])
        self.kappa = np.array([kappa0])
        self.alpha = np.array([alpha0])
        self.beta = np.array([beta0])

    def _predictive(self, x: float) -> np.ndarray:
        """Student-t predictive density of x for every run length."""
        nu = 2.0 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa))
        z = (x - self.mu) / scale
        log_norm = np.array([math.lgamma((n + 1.0) / 2.0) - math.lgamma(n / 2.0) for n in nu])
        return np.exp(log_norm) / (np.sqrt(nu * np.pi) * scale) * (1.0 + z ** 2 / nu) ** (-(nu + 1.0) / 2.0)

    def update(self, x: float) -> np.ndarray:
        """Absorb one observation and return the run length posterior P(r_t | x_1:t)."""
        pred = self._predictive(x=x)
        growth = self.run_prob * pred * (1.0 - self.hazard)
        change = float(np.sum(self.run_prob * pred * self.hazard))
        joint = np.concatenate(([change], growth))
        evidence = float(np.sum(joint))
        if not evidence > 0.0:
            raise FloatingPointError(f"evidence underflow at x={x}")
        self.run_prob = joint / evidence
        mu0, kappa0, alpha0, beta0 = self.prior
        new_beta = self.beta + self.kappa * (x - self.mu) ** 2 / (2.0 * (self.kappa + 1.0))
        self.beta = np.concatenate(([beta0], new_beta))
        self.mu = np.concatenate(([mu0], (self.kappa * self.mu + x) / (self.kappa + 1.0)))
        self.kappa = np.concatenate(([kappa0], self.kappa + 1.0))
        self.alpha = np.concatenate(([alpha0], self.alpha + 0.5))
        return self.run_prob


def make_stream(seed: int, n_before: int, n_after: int, shift: float, outlier_t: int,
                outlier_value: float) -> np.ndarray:
    """N(0, 1) for n_before samples, N(shift, 1) for n_after samples, one outlier at 1-based index outlier_t."""
    rng = np.random.default_rng(seed)
    stream = np.concatenate((rng.normal(loc=0.0, scale=1.0, size=n_before),
                             rng.normal(loc=shift, scale=1.0, size=n_after)))
    if not 1 <= outlier_t <= stream.size:
        raise ValueError(f"outlier_t must be in [1, {stream.size}], got {outlier_t=}")
    stream[outlier_t - 1] = outlier_value
    return stream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{pathlib.Path(__file__).name} {__version__}: BOCPD demo with an outlier and a mean shift.")
    parser.add_argument('-v', '--version', action='version', version=f"%(prog)s {__version__}")
    parser.add_argument('--seed', type=int, required=True, help="random seed of the synthetic stream")
    parser.add_argument('--hazard-lambda', type=float, default=100.0, help="expected run length (default 100)")
    parser.add_argument('--window', type=int, default=3, help="w in P(r_t <= w) (default 3)")
    parser.add_argument('--n-before', type=int, default=60, help="samples before the shift (default 60)")
    parser.add_argument('--n-after', type=int, default=40, help="samples after the shift (default 40)")
    parser.add_argument('--shift', type=float, default=3.0, help="mean shift in sigma units (default 3)")
    parser.add_argument('--outlier-t', type=int, default=31, help="1-based index of the outlier (default 31)")
    parser.add_argument('--outlier-value', type=float, default=6.0, help="outlier value (default 6)")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if args.window < 0:
        parser.error(f"--window must be >= 0, got {args.window}")
    if args.n_before < 1 or args.n_after < 1:
        parser.error("--n-before and --n-after must be >= 1")
    return args


if __name__ == '__main__':
    args = parse_args()
    x = make_stream(seed=args.seed, n_before=args.n_before, n_after=args.n_after, shift=args.shift,
                    outlier_t=args.outlier_t, outlier_value=args.outlier_value)
    detector = NormalGammaBocpd(hazard_lambda=args.hazard_lambda)
    print(f"{'t':>4} {'x_t':>7} {'P(r_t=0)':>9} {'P(r_t<=w)':>10} {'MAP r_t':>8}")
    for t, x_t in enumerate(x, start=1):
        post = detector.update(x=float(x_t))
        print(f"{t:>4} {x_t:>7.3f} {post[0]:>9.4f} {post[:args.window + 1].sum():>10.3f} {int(post.argmax()):>8}")
