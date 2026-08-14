# Feature-Engineering/PCA/fpca-worked-example.py
# Builds the worked FPCA example: 4 curves on an 11-point grid, planted
# mean + 2 orthonormal eigenfunctions, then runs the estimation pipeline
# (mean -> centering -> covariance -> eigenproblem -> scores -> reconstruction)
# and saves the figures used by fpca-worked-example.md.

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

np.set_printoptions(precision=4, suppress=True)

# ---------- palette (light mode) ----------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # slots 1-4
BLUE = SERIES[0]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "axes.axisbelow": True,
    }
)


def strip_spines(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------- construction ----------
t = np.round(np.arange(0.0, 1.01, 0.1), 10)  # 11 grid points
K = len(t)
w = np.full(K, 0.1)  # trapezoid quadrature weights
w[0] = w[-1] = 0.05

mu = 1.0 + 2.0 * t  # mean function
phi1 = np.sqrt(2.0) * np.sin(np.pi * t)  # eigenfunction 1
phi2 = np.sqrt(2.0) * np.cos(np.pi * t)  # eigenfunction 2

xi1 = np.array([2.0, 1.0, -1.0, -2.0])  # planted PC1 scores
xi2 = np.array([1.0, -1.0, -1.0, 1.0])  # planted PC2 scores
n = len(xi1)

X = mu + np.outer(xi1, phi1) + np.outer(xi2, phi2)  # 4 x 11 data matrix

print("== grid t ==\n", t)
print("== weights w ==\n", w)
print("== mu(t) ==\n", mu)
print("== phi1(t) ==\n", phi1)
print("== phi2(t) ==\n", phi2)
print("== data X (rows = curves) ==\n", np.round(X, 4))

# quadrature check of orthonormality
print("<phi1,phi1> =", np.sum(w * phi1 * phi1))
print("<phi2,phi2> =", np.sum(w * phi2 * phi2))
print("<phi1,phi2> =", np.round(np.sum(w * phi1 * phi2), 12))

# ---------- step 1: mean and centering ----------
mu_hat = X.mean(axis=0)
Xc = X - mu_hat
print("== mu_hat ==\n", np.round(mu_hat, 4))
print("== centered Xc ==\n", np.round(Xc, 4))

# ---------- step 2: covariance surface ----------
G = Xc.T @ Xc / (n - 1)  # 11 x 11
print("== G corners / center ==")
print("G(0,0) =", round(G[0, 0], 4), " G(0,1) =", round(G[0, -1], 4))
print("G(0.5,0.5) =", round(G[5, 5], 4), " G(0.5,0) =", round(G[5, 0], 4))
print("== G full ==\n", np.round(G, 3))

# ---------- step 3: weighted eigenproblem ----------
Whalf = np.diag(np.sqrt(w))
A = Whalf @ G @ Whalf  # symmetric problem
lam, U = np.linalg.eigh(A)
order = np.argsort(lam)[::-1]
lam = lam[order]
U = U[:, order]
Phi = np.diag(1.0 / np.sqrt(w)) @ U  # back-transform to functions

# sign convention: make the first nonzero value of each eigenfunction positive
for j in range(Phi.shape[1]):
    k = np.argmax(np.abs(Phi[:, j]) > 1e-8)
    if Phi[k, j] < 0:
        Phi[:, j] *= -1.0

print("== eigenvalues ==\n", np.round(lam, 6))
fve = lam / lam.sum()
print("== FVE ==\n", np.round(fve, 6))
print("== phi1_hat ==\n", np.round(Phi[:, 0], 4))
print("== phi2_hat ==\n", np.round(Phi[:, 1], 4))
print("max|phi1_hat - phi1| =", np.max(np.abs(Phi[:, 0] - phi1)))
print("max|phi2_hat - phi2| =", np.max(np.abs(Phi[:, 1] - phi2)))

# ---------- step 4: scores ----------
S = Xc @ (Phi[:, :2] * w[:, None])  # 4 x 2, quadrature of Xc * phi
print("== scores ==\n", np.round(S, 4))

# worked sum for curve 1, component 1
terms = w * Xc[0] * Phi[:, 0]
print("== score(1,1) term-by-term ==")
for k in range(K):
    print(
        f"t={t[k]:.1f}  Xc={Xc[0, k]: .4f}  phi1={Phi[k, 0]: .4f}"
        f"  w={w[k]:.2f}  term={terms[k]: .5f}"
    )
print("sum =", round(terms.sum(), 6))

# ---------- step 5: reconstruction ----------
X1_k0 = mu_hat
X1_k1 = mu_hat + S[0, 0] * Phi[:, 0]
X1_k2 = mu_hat + S[0, 0] * Phi[:, 0] + S[0, 1] * Phi[:, 1]
for name, rec in (("K=0", X1_k0), ("K=1", X1_k1), ("K=2", X1_k2)):
    ise = np.sum(w * (X[0] - rec) ** 2)
    print(f"ISE curve1 {name}: {ise:.6f}")

# ---------- figures ----------
labels = [f"$X_{i + 1}$" for i in range(n)]

# Fig 1: observed curves + mean, centered curves
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
# label anchors (t index, vertical offset in points) chosen where curves separate
obs_anchor = [(4, 8), (9, 8), (1, -14), (6, -14)]
cen_anchor = [(4, 8), (7, 8), (2, -14), (6, -14)]
for i in range(n):
    axes[0].plot(t, X[i], color=SERIES[i], label=labels[i])
    k, dy = obs_anchor[i]
    axes[0].annotate(
        labels[i], (t[k], X[i, k]), xytext=(0, dy),
        textcoords="offset points", ha="center", fontsize=9, color=INK,
    )
axes[0].plot(t, mu_hat, color=INK, linestyle="--", linewidth=1.6,
             label=r"$\hat{\mu}$")
axes[0].annotate(r"$\hat{\mu}$", (0.42, mu_hat[4]), xytext=(0, -14),
                 textcoords="offset points", ha="center", fontsize=9, color=INK)
axes[0].set_title("(a) Observed curves and mean")
axes[0].set_xlabel("t")
axes[0].set_ylabel("X(t)")
axes[0].set_ylim(-1.6, 5.6)
for i in range(n):
    axes[1].plot(t, Xc[i], color=SERIES[i])
    k, dy = cen_anchor[i]
    axes[1].annotate(
        labels[i], (t[k], Xc[i, k]), xytext=(0, dy),
        textcoords="offset points", ha="center", fontsize=9, color=INK,
    )
axes[1].axhline(0, color=BASELINE, linewidth=1)
axes[1].set_title("(b) Centered curves")
axes[1].set_xlabel("t")
axes[1].set_ylabel(r"$X(t)-\hat{\mu}(t)$")
axes[1].set_ylim(-3.8, 3.8)
for ax in axes:
    strip_spines(ax)
    ax.set_xlim(-0.02, 1.02)
fig.legend(loc="outside lower center", ncol=5, fontsize=9)
fig.savefig("fpca-worked-example-fig1.png", dpi=150)
plt.close(fig)

# Fig 2: covariance surface heatmap (diverging, zero at gray midpoint)
div = LinearSegmentedColormap.from_list(
    "bwr_brand", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#7c1f1f"]
)
vmax = np.abs(G).max()
fig, ax = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
im = ax.imshow(
    G, origin="lower", extent=[0, 1, 0, 1], cmap=div,
    norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
)
ax.set_title(r"$\hat{G}(s,t)$")
ax.set_xlabel("t")
ax.set_ylabel("s")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.outline.set_edgecolor(BASELINE)
fig.savefig("fpca-worked-example-fig2.png", dpi=150)
plt.close(fig)

# Fig 3: eigenfunctions + eigenvalues
fig, axes = plt.subplots(
    1, 2, figsize=(9.6, 3.8), width_ratios=[1.6, 1], constrained_layout=True
)
axes[0].plot(t, Phi[:, 0], color=SERIES[0], label=r"$\hat{\varphi}_1$")
axes[0].plot(t, Phi[:, 1], color=SERIES[1], label=r"$\hat{\varphi}_2$")
axes[0].axhline(0, color=BASELINE, linewidth=1)
axes[0].annotate(r"$\hat{\varphi}_1$", (0.5, Phi[5, 0]), xytext=(0, 6),
                 textcoords="offset points", ha="center", color=INK)
axes[0].annotate(r"$\hat{\varphi}_2$", (0.08, Phi[1, 1]), xytext=(0, 8),
                 textcoords="offset points", ha="center", color=INK)
axes[0].set_title("(a) Estimated eigenfunctions")
axes[0].set_xlabel("t")
axes[0].legend(loc="lower left", fontsize=8)
strip_spines(axes[0])

bars = axes[1].bar(
    [1, 2], lam[:2], width=0.55, color=BLUE, edgecolor=SURFACE, linewidth=2
)
for x, (v, f) in enumerate(zip(lam[:2], fve[:2]), start=1):
    axes[1].annotate(
        f"{v:.3f}\n({f * 100:.1f}%)", (x, v), xytext=(0, 4),
        textcoords="offset points", ha="center", fontsize=9, color=INK,
    )
axes[1].set_title("(b) Eigenvalues and FVE")
axes[1].set_xticks([1, 2], [r"$\hat{\lambda}_1$", r"$\hat{\lambda}_2$"])
axes[1].set_ylim(0, 4.2)
strip_spines(axes[1])
fig.savefig("fpca-worked-example-fig3.png", dpi=150)
plt.close(fig)

# Fig 4: score plane + reconstruction of curve 1
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
axes[0].axhline(0, color=BASELINE, linewidth=1)
axes[0].axvline(0, color=BASELINE, linewidth=1)
axes[0].scatter(S[:, 0], S[:, 1], s=70, color=BLUE, zorder=3)
for i in range(n):
    axes[0].annotate(
        labels[i], (S[i, 0], S[i, 1]), xytext=(8, 6),
        textcoords="offset points", color=INK,
    )
axes[0].set_title("(a) Score plane")
axes[0].set_xlabel(r"$\hat{\xi}_{i1}$")
axes[0].set_ylabel(r"$\hat{\xi}_{i2}$")
axes[0].set_xlim(-2.6, 2.6)
axes[0].set_ylim(-1.6, 1.6)
strip_spines(axes[0])

axes[1].plot(t, X[0], color=INK, linewidth=2.4, label="$X_1$ observed")
axes[1].plot(t, X1_k0, color=SERIES[2], linestyle=":", label=r"$K=0$: $\hat{\mu}$")
axes[1].plot(t, X1_k1, color=SERIES[1], linestyle="--", label="$K=1$")
axes[1].plot(t, X1_k2, color=SERIES[0], linestyle="-", linewidth=1.4, label="$K=2$")
axes[1].set_title("(b) Reconstruction of $X_1$")
axes[1].set_xlabel("t")
axes[1].set_ylabel("X(t)")
axes[1].legend(loc="lower center", fontsize=8)
strip_spines(axes[1])
fig.savefig("fpca-worked-example-fig4.png", dpi=150)
plt.close(fig)

print("figures saved")
