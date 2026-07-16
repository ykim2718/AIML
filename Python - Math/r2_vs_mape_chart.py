"""
100 x R2 (x축) vs MAPE (y축) 산점도 차트

- x축: 100 * R2  (결정계수를 백분율 스케일로 표현, 0~100)
- y축: MAPE [%]   (평균 절대 백분율 오차)

일반적으로 모델 성능이 좋을수록(R2가 1에 가까울수록) MAPE는 작아지는
음의 상관관계를 가진다. 이 경향을 반영하여 데이터를 생성한다.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1) 데이터 생성
# ----------------------------------------------------------------------
rng = np.random.default_rng(42)  # 재현성을 위한 시드

n = 120

# R2 는 0.5 ~ 0.99 범위로 생성 (현실적인 모델 성능 범위)
r2 = rng.uniform(0.5, 0.99, size=n)

# MAPE 는 R2가 클수록 작아지도록 생성 (음의 상관관계 + 약간의 노이즈)
#   기본 추세: MAPE ≈ 30 * (1 - R2)
#   노이즈   : 정규분포로 산포 추가
noise = rng.normal(0.0, 1.5, size=n)
mape = 30.0 * (1.0 - r2) + noise
mape = np.clip(mape, 0.1, None)  # 음수 방지

x = 100.0 * r2  # x축 값

# ----------------------------------------------------------------------
# 2) 차트 그리기
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(
    x, mape,
    c=mape, cmap="viridis_r",
    s=50, alpha=0.8, edgecolors="k", linewidths=0.4,
)

# 추세선 (1차 회귀)
coef = np.polyfit(x, mape, 1)
trend = np.poly1d(coef)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, trend(x_line), "r--", linewidth=1.8,
        label=f"Trend: y = {coef[0]:.3f}x + {coef[1]:.2f}")

ax.set_xlabel("100 × R²", fontsize=12)
ax.set_ylabel("MAPE [%]", fontsize=12)
ax.set_title("100 × R²  vs  MAPE", fontsize=14, fontweight="bold")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper right")

cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("MAPE [%]", fontsize=11)

fig.tight_layout()

# ----------------------------------------------------------------------
# 3) 저장 및 표시
# ----------------------------------------------------------------------
out_path = "r2_vs_mape_chart.png"
fig.savefig(out_path, dpi=150)
print(f"차트를 저장했습니다: {out_path}")

plt.show()
