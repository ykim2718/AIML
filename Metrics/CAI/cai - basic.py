"""
Center Alignment Index (CAI)
=============================
1:1 line에서 데이터 중심(mean)의 일치 정도를 0~1 scale로 평가하는 지표

수학적 정의
-----------
두 변수 X, Y의 쌍대 데이터 (x_i, y_i)가 있을 때:

  μ_x = mean(X),  μ_y = mean(Y)
  σ_x = std(X),   σ_y = std(Y)

1:1 line (y = x) 위에서 데이터 중심이 완벽히 일치하면 μ_x = μ_y 이다.

■ 편향 크기 (Location Shift):
  u = (μ_x - μ_y) / √(σ_x · σ_y)

  → μ_x - μ_y (절대 편향)를 데이터의 고유 스케일(기하평균 표준편차)로
    정규화하여 단위에 무관한 무차원 지표를 만든다.

■ Center Alignment Index:
  CAI = 1 / (1 + u²)

  성질:
  - μ_x = μ_y  →  u = 0  →  CAI = 1  (완벽한 중심 일치)
  - |μ_x - μ_y| 증가  →  u² 증가  →  CAI → 0  (중심 불일치)
  - 항상 0 < CAI ≤ 1
  - σ_x, σ_y로 정규화되어 측정 단위에 무관
  - Lin's CCC 분해에서 bias correction factor (Cb)의 위치 성분과 직접 관련

■ 참고: Lin's CCC와의 관계
  Lin's CCC = r × Cb  에서
  Cb = 2 / (v + 1/v + u²),  v = σ_x / σ_y

  CAI는 Cb에서 스케일 차이(v)를 분리하고 순수한 위치 편향만 평가한다.
  즉, v=1 (스케일 동일)일 때 Cb = 1/(1 + u²/2) ≈ CAI 와 유사.

사용 예시
---------
  from center_alignment_index import calc_cai
  cai, details = calc_cai(x, y)
  print(f"CAI = {cai:.4f}")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys


# ─────────────────────────────────────────────
# Core function
# ─────────────────────────────────────────────
def calc_cai(x, y):
    """
    Center Alignment Index (CAI) 계산

    Parameters
    ----------
    x, y : array-like
        쌍대 데이터 (같은 길이)

    Returns
    -------
    cai : float
        0~1 사이의 중심 일치 지표 (1 = 완벽 일치)
    details : dict
        구성 요소 (mean_x, mean_y, std_x, std_y, bias, u, u_squared)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert len(x) == len(y), "x, y 길이가 같아야 합니다."

    mu_x, mu_y = np.mean(x), np.mean(y)
    sigma_x, sigma_y = np.std(x, ddof=1), np.std(y, ddof=1)

    bias = mu_x - mu_y
    scale = np.sqrt(sigma_x * sigma_y)

    if scale == 0:
        # 두 변수 모두 상수일 경우
        cai = 1.0 if bias == 0 else 0.0
        u = 0.0 if bias == 0 else np.inf
    else:
        u = bias / scale
        cai = 1.0 / (1.0 + u ** 2)

    details = {
        "mean_x": mu_x,
        "mean_y": mu_y,
        "std_x": sigma_x,
        "std_y": sigma_y,
        "bias": bias,
        "u (location shift)": u,
        "u_squared": u ** 2,
        "CAI": cai,
    }
    return cai, details


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def plot_cai(x, y, title="Center Alignment Index (CAI)", ax=None):
    """CAI를 시각적으로 보여주는 scatter plot"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cai, info = calc_cai(x, y)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.figure

    # scatter
    ax.scatter(x, y, alpha=0.5, s=30, color="#4C72B0", edgecolors="white", linewidth=0.5)

    # 1:1 line
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    margin = (hi - lo) * 0.05
    lims = [lo - margin, hi + margin]
    ax.plot(lims, lims, "k--", linewidth=1, label="1:1 line")

    # data center
    mx, my = info["mean_x"], info["mean_y"]
    ax.plot(mx, my, "r*", markersize=16, label=f"Center ({mx:.2f}, {my:.2f})")

    # 1:1 line 위 투영점
    proj = (mx + my) / 2
    ax.plot(proj, proj, "g^", markersize=12, label=f"1:1 projection ({proj:.2f}, {proj:.2f})")

    # 편향 화살표
    ax.annotate(
        "", xy=(mx, my), xytext=(proj, proj),
        arrowprops=dict(arrowstyle="<->", color="red", lw=1.5),
    )

    # 텍스트 박스
    textstr = (
        f"CAI = {cai:.4f}\n"
        f"Bias = {info['bias']:+.4f}\n"
        f"u = {info['u (location shift)']:.4f}"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", bbox=props)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    return fig, ax, cai


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Case 1: Perfect center alignment (CAI ≈ 1)
    n = 100
    x1 = np.random.normal(50, 10, n)
    y1 = x1 + np.random.normal(0, 3, n)
    plot_cai(x1, y1, "Case 1: No Bias (CAI ~ 1.0)", ax=axes[0, 0])

    # Case 2: Small bias (CAI ~ 0.7-0.9)
    x2 = np.random.normal(50, 10, n)
    y2 = x2 + np.random.normal(5, 3, n)
    plot_cai(x2, y2, "Case 2: Small Bias (CAI ~ 0.8)", ax=axes[0, 1])

    # Case 3: Large bias (CAI ~ 0.2-0.5)
    x3 = np.random.normal(50, 10, n)
    y3 = x3 + np.random.normal(15, 3, n)
    plot_cai(x3, y3, "Case 3: Large Bias (CAI ~ 0.3)", ax=axes[1, 0])

    # Case 4: Very large bias (CAI → 0)
    x4 = np.random.normal(50, 10, n)
    y4 = x4 + np.random.normal(40, 3, n)
    plot_cai(x4, y4, "Case 4: Very Large Bias (CAI ~ 0)", ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(f"{sys.argv[0]}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 결과 출력
    print("=" * 60)
    print("Center Alignment Index (CAI) - Demo Results")
    print("=" * 60)
    for i, (x, y, label) in enumerate([
        (x1, y1, "Case 1: 중심 일치"),
        (x2, y2, "Case 2: 약간의 편향"),
        (x3, y3, "Case 3: 큰 편향"),
        (x4, y4, "Case 4: 매우 큰 편향"),
    ]):
        cai, info = calc_cai(x, y)
        print(f"\n{label}")
        print(f"  Mean X = {info['mean_x']:.2f},  Mean Y = {info['mean_y']:.2f}")
        print(f"  Bias   = {info['bias']:+.4f}")
        print(f"  u      = {info['u (location shift)']:.4f}")
        print(f"  CAI    = {cai:.4f}")
    print("\n" + "=" * 60)
