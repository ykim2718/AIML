"""
실제 회귀모델에서 R²와 MAPE의 '진짜' 관계를 실험으로 보이는 스크립트.

핵심 아이디어
-------------
신호(설명 가능한 부분)는 고정하고 '잡음 크기'만 바꿔 가며
보통최소제곱(OLS) 회귀모델을 수백 개 학습시킨다.
각 모델에서 R²와 MAPE를 '정의대로' 직접 계산해 (100·R², MAPE) 산점도를 그린다.

이론 유도 (왜 직선이 아니라 곡선인가)
-------------------------------------
잔차 표준편차를 σe, 신호 표준편차를 σs, y의 평균을 μy 라 하자.
  · R²  = 1 - SSE/SST = 1 - σe² / (σs² + σe²)
        ⇒ 1 - R² = σe² / (σs² + σe²)
        ⇒ σe = σs · sqrt( (1-R²) / R² )                ... (★)
  · MAPE = mean|(y-ŷ)/y|·100 ≈ sqrt(2/π)·100·σe/μy     (가우시안 잔차)
(★)를 대입하면

        MAPE = K · sqrt( (1-R²) / R² ),
        K   = sqrt(2/π) · 100 · σs / μy   (상수: 신호크기/평균에만 의존)

즉 R²와 MAPE의 진짜 관계는 '직선'이 아니라
x=100R² 로 쓰면  MAPE = K·sqrt((100-x)/x)  형태의 '곡선'이다.
R²→1 에서 0으로 수렴하고, R²→0 에서 급격히 발산한다.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


def ols_fit(X, y):
    """절편 포함 보통최소제곱(OLS) 정규방정식 해. (실제 회귀모델)"""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def predict(X, beta):
    return beta[0] + X @ beta[1:]


def r2_score(y, yhat):
    sse = np.sum((y - yhat) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return 1.0 - sse / sst


def mape(y, yhat):
    return np.mean(np.abs((y - yhat) / y)) * 100.0


# ----------------------------------------------------------------------
# 1) 데이터 생성 설계 (신호는 고정, 잡음만 변화)
# ----------------------------------------------------------------------
BETA = np.array([8.0, -6.0, 5.0])   # 진짜 회귀계수 (신호)
BASELINE = 1000.0                   # y 평균을 크게 잡아 y>0 보장 (MAPE 안정)
N = 600                             # 표본 크기
TEST_FRAC = 0.4

# 신호 통계(σs, μy)를 큰 표본으로 사전 추정 → 이론 상수 K 계산용
big = rng.uniform(0, 10, size=(200_000, 3))
signal = big @ BETA
sigma_s = signal.std()
mu_y = BASELINE + signal.mean()
K_theory = np.sqrt(2 / np.pi) * 100.0 * sigma_s / mu_y
print(f"신호 표준편차 σs = {sigma_s:.3f},  y 평균 μy = {mu_y:.3f}")
print(f"이론 상수 K = sqrt(2/pi)*100*σs/μy = {K_theory:.4f}")

# ----------------------------------------------------------------------
# 2) 잡음 수준을 바꿔 가며 실제 회귀모델 학습 → R², MAPE 측정
#    각 잡음 수준에서 REPEAT 번 반복 학습 후 평균 → 측정잡음 제거, 참 곡선 노출
# ----------------------------------------------------------------------
noise_levels = np.linspace(3.0, 115.0, 60)
REPEAT = 40
R2, MAPE = [], []
for sigma_e in noise_levels:
    r2s, mapes = [], []
    for _ in range(REPEAT):
        X = rng.uniform(0, 10, size=(N, 3))
        y = BASELINE + X @ BETA + rng.normal(0, sigma_e, N)

        # train/test 분할
        idx = rng.permutation(N)
        cut = int(N * (1 - TEST_FRAC))
        tr, te = idx[:cut], idx[cut:]

        beta_hat = ols_fit(X[tr], y[tr])    # 실제 회귀 학습
        yp = predict(X[te], beta_hat)       # 테스트셋 예측

        r2s.append(r2_score(y[te], yp))
        mapes.append(mape(y[te], yp))
    R2.append(np.mean(r2s))                 # 반복 평균
    MAPE.append(np.mean(mapes))

R2 = np.array(R2)
MAPE = np.array(MAPE)

# 잡음이 과해 R²<0 이 된 비정상 모델은 제외 (회귀가 평균보다 나쁜 경우)
valid = R2 > 0.01
R2, MAPE = R2[valid], MAPE[valid]
x = 100.0 * R2

# ----------------------------------------------------------------------
# 3) 두 가지 모델로 적합 → 누가 진짜 관계인지 비교
# ----------------------------------------------------------------------
# (A) 순진한 직선:  MAPE = intercept + slope·(100R²)
#     주의: np.polyfit 은 [기울기, 절편] 순서로 반환한다.
slope, intercept = np.polyfit(x, MAPE, 1)
mape_lin = intercept + slope * x
r2_lin = r2_score(MAPE, mape_lin)

# (B) 구조식 곡선:  MAPE = K·sqrt((1-R²)/R²),  K는 최소제곱으로 추정
g = np.sqrt((1 - R2) / R2)
K_fit = np.sum(MAPE * g) / np.sum(g * g)     # 1-파라미터 최소제곱
mape_curve = K_fit * g
r2_curve = r2_score(MAPE, mape_curve)

print(f"\n[Linear] MAPE = {intercept:.4f} + ({slope:.4f})*100R2     fit-R2 = {r2_lin:.4f}")
print(f"[Curve ] MAPE = {K_fit:.4f}*sqrt((1-R2)/R2)               fit-R2 = {r2_curve:.4f}")
print(f"         (fitted K={K_fit:.4f}  vs  theory K={K_theory:.4f})")

# ----------------------------------------------------------------------
# 4) 시각화
# ----------------------------------------------------------------------
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(9, 6.2))
ax.scatter(x, MAPE, s=40, alpha=0.8, color="#2c7fb8",
           edgecolors="k", linewidths=0.4,
           label=f"Real OLS models ({len(x)} levels x {REPEAT} reps, averaged)")

xs = np.linspace(max(x.min(), 1e-6), x.max(), 400)
# 직선 모델
ax.plot(xs, intercept + slope * xs, "r--", lw=2,
        label=f"Naive linear fit  (fit-R2={r2_lin:.3f})")
# 구조식 곡선 (적합)
gs = np.sqrt((100 - xs) / xs)               # = sqrt((1-R²)/R²)
ax.plot(xs, K_fit * gs, "g-", lw=2.2,
        label=f"Structural curve  K*sqrt((1-R2)/R2)  (fit-R2={r2_curve:.3f})")
# 이론 곡선 (계수 추정 없이 K_theory)
ax.plot(xs, K_theory * gs, color="orange", ls=":", lw=2.2,
        label=f"Theory curve  (K={K_theory:.2f}, no fitting)")

ax.set_xlabel("100 x R^2", fontsize=12)
ax.set_ylabel("MAPE [%]", fontsize=12)
ax.set_title("Real regression: R^2 vs MAPE  --  the true relation is a CURVE",
             fontsize=13, fontweight="bold")
ax.grid(True, ls=":", alpha=0.6)
ax.legend(loc="upper right", fontsize=10)
fig.tight_layout()

out = "real_r2_vs_mape_experiment.png"
fig.savefig(out, dpi=150)
print(f"\n차트 저장: {out}")
