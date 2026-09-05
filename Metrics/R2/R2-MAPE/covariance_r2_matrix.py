"""
공분산 행렬을 변동시켜 R²(=상관²)를 제어하고, 모양별 산점도 행렬 + xy 상관 차트를 그린다.

설계
----
이변량 정규분포 (X, Y) 를
    Σ = s² · [[1, r],
              [r, 1]]
로 생성한다.
  · s (scale)  : 구름의 '크기' → wide(넓음, s 큼) / narrow(좁음, s 작음)
  · r (상관)   : 타원의 '기울기/납작함' → R² = r²

  - circle  : r = 0   → R² = 0  (등방, 기울기 없음)
  - ellipse : r ≠ 0   → R² = r² (45° 기울어진 타원, |r| 클수록 납작)

요청 케이스 (총 9개)
  wide  circle  : R²=0
  narrow circle : R²=0
  wide  ellipse : R²=0.2, 0.5, 0.7
  narrow ellipse: R²=0.2, 0.5, 0.7, 0.9
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False
rng = np.random.default_rng(7)

N = 600
WIDE, NARROW = 3.0, 1.0          # scale s
AX_LIM = 11.0                    # 모든 패널 공통 축범위(→ wide/narrow 대비가 보이게)
MU = 50.0                        # MAPE 정의용 Y 평균 이동(Y>0 보장). μy = MU

# (이름, scale, R²_target) ------------------------------------------------
cases = [
    ("Wide circle",      WIDE,   0.0),
    ("Narrow circle",    NARROW, 0.0),
    ("Wide ellipse",     WIDE,   0.2),
    ("Wide ellipse",     WIDE,   0.5),
    ("Wide ellipse",     WIDE,   0.7),
    ("Narrow ellipse",   NARROW, 0.2),
    ("Narrow ellipse",   NARROW, 0.5),
    ("Narrow ellipse",   NARROW, 0.7),
    ("Narrow ellipse",   NARROW, 0.9),
]


def make_cov(s, r2):
    """Σ = s² [[1, r],[r, 1]],  r = +sqrt(R²)"""
    r = np.sqrt(r2)
    return (s ** 2) * np.array([[1.0, r], [r, 1.0]]), r


# ----------------------------------------------------------------------
# 1) 데이터 생성 + 측정
# ----------------------------------------------------------------------
data = []
hdr = (f"{'case':16s}{'s':>5s}{'r(target)':>11s}{'R2(target)':>12s}"
       f"{'r(meas)':>10s}{'R2(meas)':>11s}{'MAPE%':>9s}")
print(hdr)
for name, s, r2t in cases:
    cov, r = make_cov(s, r2t)
    xy = rng.multivariate_normal([0.0, 0.0], cov, size=N)
    x, y = xy[:, 0], xy[:, 1]
    r_meas = np.corrcoef(x, y)[0, 1]
    R2_meas = r_meas ** 2

    # OLS Y~X 잔차로 MAPE 계산 (Y는 +MU 이동해 양수화 → 분모 안정)
    b = np.cov(x, y)[0, 1] / np.var(x)
    a = y.mean() - b * x.mean()
    resid = y - (a + b * x)
    mape = np.mean(np.abs(resid / (y + MU))) * 100.0

    data.append(dict(name=name, s=s, r2t=r2t, r=r, cov=cov,
                     x=x, y=y, r_meas=r_meas, R2_meas=R2_meas, mape=mape))
    print(f"{name:16s}{s:5.1f}{r:11.3f}{r2t:12.2f}{r_meas:10.3f}"
          f"{R2_meas:11.3f}{mape:9.3f}")


# ----------------------------------------------------------------------
# 2) 산점도 '행렬' 차트 (3 x 3)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(13, 13))
for ax, d in zip(axes.ravel(), data):
    x, y = d["x"], d["y"]
    ax.scatter(x, y, s=10, alpha=0.45, color="#1f77b4", edgecolors="none")

    # OLS 회귀선 (y ~ x)
    b = np.cov(x, y)[0, 1] / np.var(x)
    a = y.mean() - b * x.mean()
    xs = np.array([-AX_LIM, AX_LIM])
    ax.plot(xs, a + b * xs, "r-", lw=1.8)

    ax.set_xlim(-AX_LIM, AX_LIM)
    ax.set_ylim(-AX_LIM, AX_LIM)
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.5)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)

    shape = "circle" if d["r2t"] == 0 else "ellipse"
    ax.set_title(f"{d['name']}\n"
                 f"R2(target)={d['r2t']:.1f}  R2(meas)={d['R2_meas']:.3f}",
                 fontsize=11, fontweight="bold")
    # 공분산 행렬 주석
    c = d["cov"]
    txt = (f"$\\Sigma$=[[{c[0,0]:.1f}, {c[0,1]:.1f}],\n"
           f"        [{c[1,0]:.1f}, {c[1,1]:.1f}]]\n"
           f"r={d['r']:.3f}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=8.5,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.75, ec="gray"))

fig.suptitle("Covariance-varied scatter matrix:  scale = wide/narrow,  tilt(r) -> R2",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig("covariance_r2_matrix.png", dpi=140)
print("\n저장: covariance_r2_matrix.png")


# ----------------------------------------------------------------------
# 3) xy 상관 차트 (요약)
# ----------------------------------------------------------------------
fig2, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6))

# (A) 케이스별 측정 R² / |r|  막대+점
labels = [f"{d['name'].split()[0][0]}{d['name'].split()[1][0]}\nR2t={d['r2t']:.1f}"
          for d in data]
idx = np.arange(len(data))
R2_meas = [d["R2_meas"] for d in data]
R2_tgt = [d["r2t"] for d in data]
colors = ["#d62728" if d["s"] == WIDE else "#2ca02c" for d in data]

axA.bar(idx, R2_meas, color=colors, alpha=0.7, label="R2 (measured)")
axA.plot(idx, R2_tgt, "ko--", lw=1.3, ms=6, label="R2 (target)")
axA.set_xticks(idx)
axA.set_xticklabels(labels, fontsize=8)
axA.set_ylabel("R2")
axA.set_title("Measured vs target R2 per case\n(red=wide, green=narrow)")
axA.grid(True, axis="y", ls=":", alpha=0.6)
axA.legend()

# (B) target R² vs measured R²  (대각선 = 완벽 일치)
axB.scatter(R2_tgt, R2_meas, s=90, c=colors, edgecolors="k", zorder=3)
for d, xt, ym in zip(data, R2_tgt, R2_meas):
    axB.annotate(f"r={d['r']:.2f}", (xt, ym), textcoords="offset points",
                 xytext=(6, 6), fontsize=8)
axB.plot([0, 1], [0, 1], "k--", lw=1.2, label="y = x (perfect)")
axB.set_xlabel("R2 (target, from covariance)")
axB.set_ylabel("R2 (measured, from data)")
axB.set_title("Covariance design -> realized correlation")
axB.set_xlim(-0.05, 1.0)
axB.set_ylim(-0.05, 1.0)
axB.set_aspect("equal")
axB.grid(True, ls=":", alpha=0.6)
axB.legend(loc="upper left")

fig2.tight_layout()
fig2.savefig("covariance_r2_correlation.png", dpi=140)
print("저장: covariance_r2_correlation.png")


# ----------------------------------------------------------------------
# 4) R² - MAPE 차트
#    이 설계는 총분산 s² 고정, 상관 r 만 변동 → σe = s·sqrt(1-R²)
#    ⇒ MAPE ≈ C·sqrt(1-R²),  C = sqrt(2/π)·100·s/μy   (스케일별 다른 곡선)
# ----------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(9, 6.2))

R2_arr = np.array([d["R2_meas"] for d in data])
MAPE_arr = np.array([d["mape"] for d in data])
col = ["#d62728" if d["s"] == WIDE else "#2ca02c" for d in data]

ax3.scatter(100 * R2_arr, MAPE_arr, s=110, c=col, edgecolors="k", zorder=3)
for d in data:
    ax3.annotate(f"r={d['r']:.2f}", (100 * d["R2_meas"], d["mape"]),
                 textcoords="offset points", xytext=(7, 5), fontsize=8)

# 이론 곡선 (스케일별): MAPE = C·sqrt(1-R²)
rg = np.linspace(0, 0.97, 200)
for s, c, lab in [(WIDE, "#d62728", "wide"), (NARROW, "#2ca02c", "narrow")]:
    C = np.sqrt(2 / np.pi) * 100 * s / MU
    ax3.plot(100 * rg, C * np.sqrt(1 - rg), color=c, lw=2,
             label=f"theory {lab}:  {C:.2f}*sqrt(1-R2)  (s={s:.0f})")

ax3.set_xlabel("100 x R^2", fontsize=12)
ax3.set_ylabel("MAPE [%]", fontsize=12)
ax3.set_title("R^2 vs MAPE for covariance cases\n"
              "(fixed total variance, vary r -> MAPE ~ sqrt(1-R^2))",
              fontsize=12, fontweight="bold")
ax3.grid(True, ls=":", alpha=0.6)
ax3.legend(loc="upper right", fontsize=10)
fig3.tight_layout()
fig3.savefig("covariance_r2_mape.png", dpi=140)
print("저장: covariance_r2_mape.png")
