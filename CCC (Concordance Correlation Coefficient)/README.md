# 📊 CCC & Bland-Altman Analysis Guide

> **Concordance Correlation Coefficient (CCC)** + **Bland-Altman Plot** 완전 가이드  
> 두 측정 방법의 일치도 평가를 위한 통계 분석 도구 모음

---

## 목차

- [CCC란 무엇인가](#ccc란-무엇인가)
- [CCC 구성 요소](#ccc-구성-요소)
- [해석 기준](#해석-기준)
- [Bland-Altman Plot](#bland-altman-plot)
- [CCC + Bland-Altman 조합 분석](#ccc--bland-altman-조합-분석)
- [데이터 집중 문제와 대안 지표](#데이터-집중-문제와-대안-지표)
- [AI/ML에서의 활용](#aiml에서의-활용)
- [Python 코드 예시](#python-코드-예시)
- [지표 선택 가이드](#지표-선택-가이드)

---

## CCC란 무엇인가

**Lin's Concordance Correlation Coefficient** (1989, Lawrence Lin)는 두 측정 방법이 얼마나 일치하는지를 평가하는 지표입니다.

단순 Pearson r이 "선형 관계의 강도"만 측정하는 것과 달리, CCC는 **정확도(accuracy) × 정밀도(precision)** 를 동시에 측정합니다.

### 수식

$$\rho_c = \frac{2\sigma_{xy}}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2} = r \times C_b$$

| 기호 | 의미 |
|------|------|
| $\rho_c$ | Concordance Correlation Coefficient |
| $r$ | Pearson 상관계수 (정밀도) |
| $C_b$ | 편향 보정 계수 (정확도, 0~1) |
| $\mu_x, \mu_y$ | 두 방법의 평균 |
| $\sigma_x^2, \sigma_y^2$ | 두 방법의 분산 |
| $\sigma_{xy}$ | 공분산 |

### Pearson r과의 핵심 차이

```
r = 0.98 이어도 CCC = 0.73 이 될 수 있다 — 체계적 편향이 있을 때
```

- **r**: y=x 가 아닌 임의의 직선을 기준으로 선형 관계만 측정
- **CCC**: y=x (완벽한 일치선) 으로부터의 이탈을 측정

---

## CCC 구성 요소

```
CCC (ρc)
├── Pearson r        ← 정밀도 (Precision): 데이터가 직선을 따르는가
└── Cb (편향 보정)   ← 정확도 (Accuracy): 그 직선이 y=x 인가
    ├── 평균 편향 (μx − μy)
    └── 분산 편향 (σx vs σy)
```

- `Cb = 1.0` → 완벽한 정확도 (회귀선이 y=x와 일치)
- `Cb < 1.0` → 편향 존재 (평균 차이 또는 스케일 차이)

---

## 해석 기준

| CCC 값 | 해석 (McBride 2005) |
|--------|---------------------|
| > 0.99 | Almost perfect |
| 0.95 ~ 0.99 | Substantial |
| 0.90 ~ 0.95 | Moderate |
| < 0.90 | Poor |

### 5가지 편향 시나리오

| 시나리오 | CCC | Pearson r | 원인 |
|----------|-----|-----------|------|
| 좋은 일치 | 높음 | 높음 | — |
| 평균 편향 | **낮음** | 높음 ⚠ | 방법B가 일정하게 높거나 낮음 |
| 스케일 편향 | **낮음** | 높음 ⚠ | 방법B의 분산이 다름 |
| 넓은 산포 | 낮음 | 낮음 | 측정 정밀도 부족 |
| 비례 편향 | 중간 | 높음 ⚠ | 큰 값일수록 오차 증가 |

> ⚠ r만 보고 "일치한다"고 판단하는 것은 잘못된 결론으로 이어질 수 있습니다.

---

## Bland-Altman Plot

1986년 Bland & Altman이 제안한 방법 비교의 표준 시각화입니다.

- **X축**: 두 측정값의 평균 `(A + B) / 2`
- **Y축**: 두 측정값의 차이 `B − A`

### 핵심 지표

| 지표 | 계산 | 의미 |
|------|------|------|
| Mean Difference (MD) | `mean(B − A)` | 체계적 편향. 0이면 이상적 |
| LoA (상한) | `MD + 1.96 × SD` | 95% 차이값의 상한 |
| LoA (하한) | `MD − 1.96 × SD` | 95% 차이값의 하한 |
| LoA 폭 | `3.92 × SD` | 좁을수록 두 방법이 일치 |

### 비례 편향 감지

Bland-Altman 플롯에서 점들이 **좌→우로 경사진 패턴**을 보이면 비례 편향(proportional bias)이 존재합니다. 이는 CCC 플롯만으로는 감지하기 어렵습니다.

---

## CCC + Bland-Altman 조합 분석

두 플롯은 서로 다른 각도에서 같은 문제를 봅니다.

```
CCC 플롯         → "전반적으로 얼마나 일치하나" (요약 숫자)
Bland-Altman     → "왜, 어떻게 안 일치하나"   (편향의 성질 진단)
```

| 패턴 | CCC 플롯에서 | Bland-Altman에서 | 진단 |
|------|-------------|-----------------|------|
| 좋은 일치 | 점들이 y=x 위 | MD≈0, 좁은 LoA | 두 방법 교체 가능 |
| 평균 편향 | y=x에서 평행 이동 | MD ≠ 0 (일정) | 보정(calibration) 필요 |
| 비례 편향 | 기울기 차이 | 우상향 패턴 | 구간별 보정 필요 |
| 넓은 산포 | 넓게 퍼짐 | 넓은 LoA | 측정 정밀도 개선 필요 |

---

## 데이터 집중 문제와 대안 지표

### 문제: 좁은 범위에 집중된 데이터

데이터가 좁은 범위에 뭉쳐 있으면 CCC/r이 **분산 부족으로 불안정**해집니다.

```python
# 범위가 좁으면 → 분모가 0에 수렴 → CCC 불안정
ccc = 2 * cov / (var_x + var_y + (mean_x - mean_y)**2)
#                ↑ 이 값이 거의 0이 됨
```

또한 **두 군집** 데이터는 반대 문제를 일으킵니다: CCC/r이 실제보다 인위적으로 높게 나타납니다 (레인지 효과).

### 대안 지표 (범위 무관)

| 지표 | 수식 | 특징 |
|------|------|------|
| **RMSD₁:₁** | `sqrt(mean((y-x)²)) / sqrt(2)` | 1:1선까지 수직 거리, 정규 오차에 최적 |
| **MAD₁:₁** | `mean(|y - x|)` | 이상값에 강건, 단위 = 측정 단위 |
| **TDI (90%)** | 90번째 백분위수의 `|y - x|` | "90%가 이 오차 이내"로 해석 |
| **CP** | `count(|y-x| ≤ δ) / n × 100` | 허용 기준 δ 대비 통과율 (%) |

### 진단 체크리스트

```
□ Concordance plot에서 데이터가 y=x 선 위에 넓게 퍼져 있는가?
□ Bland-Altman x축이 임상적으로 의미 있는 전 범위를 커버하는가?
□ 데이터 범위(span) / LoA 폭 비율이 충분히 큰가?
□ 두 군집이 혼합되어 레인지 효과가 발생하지 않는가?
```

### 권장 조합

- 데이터 범위 충분 → **CCC + Bland-Altman**
- 데이터 집중 / 범위 좁음 → **RMSD₁:₁ + TDI(90%) + CP**

---

## AI/ML에서의 활용

### 1. 회귀 모델 평가

RMSE나 R²만으로는 모델 예측값의 체계적 편향을 감지하기 어렵습니다.

```python
from sklearn.metrics import make_scorer

def ccc_score(y_true, y_pred):
    mu_t, mu_p = np.mean(y_true), np.mean(y_pred)
    var_t, var_p = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mu_t) * (y_pred - mu_p))
    return 2 * cov / (var_t + var_p + (mu_t - mu_p)**2)

# GridSearchCV의 최적화 목표로 CCC 사용
ccc_scorer = make_scorer(ccc_score, greater_is_better=True)
grid = GridSearchCV(model, param_grid, scoring=ccc_scorer)
```

### 2. 주요 적용 분야

| 분야 | 활용 |
|------|------|
| 의료 영상 AI | AI vs 전문의 병변 측정 일치도 (FDA 권장) |
| 자연어 처리 | 자동 채점 vs human annotator 점수 일치도 |
| 도메인 적응 | 소스 → 타겟 전이 후 예측 일관성 평가 |
| 앙상블 모델 | 서브모델 간 일치도로 다양성 정량화 |
| 생체신호 예측 | 혈압, 혈당, SpO₂ 예측의 임상 허용 오차 평가 |

---

## Python 코드 예시

### 기본 CCC 구현

```python
import numpy as np

def concordance_correlation_coefficient(y_true, y_pred):
    """
    Lin's Concordance Correlation Coefficient (CCC)
    Returns: dict with ccc, r, Cb, bias, var_ratio
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mu_t, mu_p = np.mean(y_true), np.mean(y_pred)
    var_t = np.var(y_true)
    var_p = np.var(y_pred)
    cov   = np.mean((y_true - mu_t) * (y_pred - mu_p))

    r   = cov / np.sqrt(var_t * var_p)
    ccc = 2 * cov / (var_t + var_p + (mu_t - mu_p)**2)
    cb  = ccc / r if r != 0 else 0

    return {
        'ccc':       round(ccc, 4),
        'r':         round(r,   4),
        'Cb':        round(cb,  4),
        'bias':      round(mu_p - mu_t, 4),
        'var_ratio': round(np.sqrt(var_p / var_t), 4),
    }
```

### 신뢰구간 포함 (Fisher's Z + Bootstrap)

```python
def ccc_with_ci(y_true, y_pred, n_boot=1000):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)

    mu_t, mu_p = np.mean(y_true), np.mean(y_pred)
    var_t, var_p = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mu_t) * (y_pred - mu_p))
    ccc = 2 * cov / (var_t + var_p + (mu_t - mu_p)**2)

    # Fisher's Z-transform 95% CI
    z      = np.arctanh(ccc)
    se     = 1 / np.sqrt(n - 3)
    ci_low = np.tanh(z - 1.96 * se)
    ci_up  = np.tanh(z + 1.96 * se)

    # Bootstrap CI
    boot = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        bt, bp = y_true[idx], y_pred[idx]
        m1, m2 = np.mean(bt), np.mean(bp)
        v1, v2 = np.var(bt), np.var(bp)
        cv = np.mean((bt - m1) * (bp - m2))
        d  = v1 + v2 + (m1 - m2)**2
        boot.append(2 * cv / d if d > 0 else 0)

    boot = np.array(boot)
    return {
        'ccc':          round(ccc, 4),
        'ci_fisher':    (round(ci_low, 4), round(ci_up, 4)),
        'ci_bootstrap': (round(np.percentile(boot, 2.5), 4),
                         round(np.percentile(boot, 97.5), 4)),
    }
```

### 집중 데이터용 대안 지표

```python
def robust_agreement_metrics(y_true, y_pred, delta=5.0, tdi_pct=0.90):
    """
    범위에 무관한 1:1선 이탈 측정 지표
    - RMSD₁:₁ : 1:1선까지 수직 RMSD
    - MAD₁:₁  : 평균 절대 편차
    - TDI      : Total Deviation Index
    - CP       : Coverage Probability
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diffs     = y_pred - y_true
    abs_diffs = np.abs(diffs)

    rmsd_11 = np.sqrt(np.mean(diffs**2)) / np.sqrt(2)
    mad_11  = np.mean(abs_diffs)
    tdi     = np.quantile(abs_diffs, tdi_pct)
    cp      = np.mean(abs_diffs <= delta) * 100

    return {
        'RMSD_1_1': round(rmsd_11, 4),
        'MAD_1_1':  round(mad_11,  4),
        f'TDI_{int(tdi_pct*100)}pct': round(tdi, 4),
        f'CP_delta_{delta}': round(cp, 2),
    }
```

### CCC + Bland-Altman 통합 리포트

```python
import matplotlib.pyplot as plt

def full_agreement_report(y_true, y_pred,
                          method_names=("Reference", "Model"),
                          delta=5.0):
    y_true, y_pred = np.array(y_true, dtype=float), \
                     np.array(y_pred, dtype=float)

    # CCC
    res_ccc = concordance_correlation_coefficient(y_true, y_pred)

    # Bland-Altman
    diff   = y_pred - y_true
    mean_  = (y_pred + y_true) / 2
    md     = np.mean(diff)
    sd     = np.std(diff, ddof=1)
    loa_u  = md + 1.96 * sd
    loa_l  = md - 1.96 * sd

    # 대안 지표
    res_rob = robust_agreement_metrics(y_true, y_pred, delta=delta)

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Concordance plot
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, color='#534AB7')
    lim = [min(y_true.min(), y_pred.min()) - 2,
           max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lim, lim, '--', color='#1D9E75', lw=1.5, label='y=x')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(method_names[0]); ax.set_ylabel(method_names[1])
    ax.set_title(f"Concordance Plot\n"
                 f"CCC={res_ccc['ccc']}, r={res_ccc['r']}, "
                 f"Cb={res_ccc['Cb']}")
    ax.legend(fontsize=9)

    # Bland-Altman plot
    ax = axes[1]
    ax.scatter(mean_, diff, alpha=0.6, s=30, color='#D85A30')
    ax.axhline(md,    color='#D85A30', lw=2,   label=f'Bias={md:.2f}')
    ax.axhline(loa_u, color='gray',    lw=1.2, ls='--',
               label=f'+1.96SD={loa_u:.2f}')
    ax.axhline(loa_l, color='gray',    lw=1.2, ls='--',
               label=f'-1.96SD={loa_l:.2f}')
    ax.axhline(0,     color='black',   lw=0.5, ls=':')
    ax.set_xlabel('Mean of two methods')
    ax.set_ylabel('Difference (Model − Reference)')
    ax.set_title(f"Bland-Altman Plot\n"
                 f"LoA: [{loa_l:.2f}, {loa_u:.2f}]")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('agreement_report.png', dpi=150, bbox_inches='tight')
    plt.show()

    return {**res_ccc, **res_rob,
            'bias_MD': round(md, 4),
            'LoA': (round(loa_l, 4), round(loa_u, 4))}
```

---

## 지표 선택 가이드

```
데이터가 넓은 범위에 고루 분포?
├── YES → CCC + Bland-Altman (표준 조합)
│         └── 비례 편향 의심 시 BA 플롯 경사 확인
└── NO  → 데이터가 좁은 범위에 집중?
          ├── YES → RMSD₁:₁ + TDI(90%) + CP
          │         (CCC/r은 신뢰 불가)
          └── 두 군집 혼합?
              └── YES → 군집별 분리 분석 후 각각 CCC 적용
                        (전체 합산 CCC는 레인지 효과로 과대 추정)
```

### 최종 권장 보고 항목

```python
report = {
    # 필수
    'CCC':      ...,   # 전반적 일치도
    'r':        ...,   # 정밀도
    'Cb':       ...,   # 정확도 (편향 계수)
    'CI_95':    ...,   # 신뢰구간

    # Bland-Altman
    'Bias_MD':  ...,   # 평균 편향
    'LoA':      ...,   # Limits of Agreement

    # 집중 데이터 또는 임상 검증 시 추가
    'RMSD_1_1': ...,
    'TDI_90':   ...,
    'CP':       ...,   # 허용 한계 δ 명시 필요
}
```

---

## 참고 문헌

- Lin, L.I. (1989). *A concordance correlation coefficient to evaluate reproducibility.* Biometrics, 45(1), 255-268.
- Bland, J.M. & Altman, D.G. (1986). *Statistical methods for assessing agreement between two methods of clinical measurement.* Lancet, 327(8476), 307-310.
- McBride, G.B. (2005). *A proposal for strength-of-agreement criteria for Lin's Concordance Correlation Coefficient.* NIWA Client Report.
- Barnhart, H.X. et al. (2007). *Overview on assessing agreement with continuous measurements.* Journal of Biopharmaceutical Statistics.

---

## 라이선스

MIT License

---

*이 가이드는 측정 방법 비교(Method Comparison Study)를 위한 통계 분석 레퍼런스입니다.*
