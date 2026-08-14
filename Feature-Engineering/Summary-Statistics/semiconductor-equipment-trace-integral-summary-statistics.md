# Semiconductor Equipment Trace Integral Summary Statistics
Rev. 18 | Created: 2026-07-29 | Updated: 2026-08-14 16:19 CDT

장비 trace 시계열을 wafer당 고정 길이 vector로 변환하는 특징 가운데, 적분 연산자로 정의되는 AUC 계열 특징의 정의, 수학적 성질, 실패 모드, 차원 통제, 검증 규약을 정리한다.

Data는 [wafer, feature, trace] 의 dimension을 갖는다. 여기서 trace의 예로는 sensor 신호나 FDC data가 있다.

핵심 제약은 결과가 wide data, 즉 $p \gg n$ 이 되지 않아야 한다.

## 1. Integral Statistics

단일 wafer trace를 $x(t)$, $t \in [0, T]$ 로 두고, $\bar{x}$ 를 그 산술평균, $\tau = t - T/2$ 를 중심화한 시간축으로 쓴다. 이하의 모든 적분은 실제 timestamp 차분에 의한 사다리꼴 적분으로 계산한다.

AUC는 summary statistics의 부분집합이다. trace 전체를 스칼라 하나로 요약한다는 점에서 평균이나 표준편차와 동일한 범주에 속하며, 적분 연산자를 쓴다는 점만 다르다.

### 1.1 Plain AUC Carries No Information

균등 sampling에서

$$\text{AUC} = \int_0^T x(t)\,dt \approx \bar{x}\,T$$

이므로 단순 AUC는 평균 × 길이다. 평균 $\bar{x}$ 와 길이 $T$ 를 이미 특징으로 갖고 있다면 단순 AUC는 완전 공선이며, VIF만 올리고 정확도를 떨어뜨린다. AUC의 가치는 평균 × 길이로 분해되지 않는 변형형에서 나온다.

### 1.2 AUC Variants That Add Information

$x_0$ 는 초기 구간의 중앙값 (pre-trace baseline) 이고, $x_{\text{ref}}(t)$ 는 golden reference trace다.

Table 1. Informative AUC variants

| ID | Feature | Definition | Physical meaning |
|---|---|---|---|
| A1 | `auc_base` | $\int (x - x_0)\,dt$ | 초기 baseline 대비 누적량 |
| A2 | `auc_res_abs` | $\int \lvert x - x_{\text{ref}}\rvert\,dt$ | Golden 대비 L1 형상 이탈량 |
| A3 | `auc_res_over` / `auc_res_under` | $\int \max(x - x_{\text{ref}},0)\,dt$ / $\int \max(x_{\text{ref}} - x,0)\,dt$ | 초과와 미달의 분리 (부호 상쇄 방지) |
| A4 | `pauc_k` | trace를 등간격 K block 분할 후 block별 적분 비율 | 형상 정보 보존 |
| A5 | `tv` | $\int \lvert dx/dt \rvert\,dt$ | 진동과 불안정성 (ringing, hunting) |
| A6 | `auc_energy` | $\int (x - \bar{x})^2\,dt$ | 산포의 시간 누적 |

A2/A3가 핵심이다. 스칼라 하나로 파형 전체의 이탈을 요약하므로 정보 대비 차원 비율이 가장 좋다.

### 1.3 Phase Features from Cumulative AUC Quantiles

$$F(t) = \frac{\int_0^t \lvert x - x_0\rvert\,d\tau}{\int_0^T \lvert x - x_0\rvert\,d\tau}, \qquad t_q = F^{-1}(q)$$

- `t25`, `t50`, `t75` 는 신호의 무게중심이 시간축 앞쪽인지 뒤쪽인지를 나타낸다.
- Ramp 속도, 정착 지연, valve와 MFC의 응답 지연이 여기에 반영된다.
- 0–1 무차원이므로 chamber와 recipe 간 직접 비교가 가능하다.

### 1.4 Domain Physics AUC

물리량과 직결되므로 응답변수와의 선형성이 높다.

Table 2. Physics-based integrals

| Trace | Integral form | Physical quantity |
|---|---|---|
| 가스 유량 (sccm) | $\int Q\,dt$ | 총 투입 가스량 (sccm·s) |
| RF power | $\int P\,dt$ | 총 이온/plasma dose (J) |
| 압력 | $\int p\,dt$ | 압력–시간 적분 |
| 온도 | $\int \exp(-E_a / k_B T(t))\,dt$ | Arrhenius thermal budget |

마지막 항이 특히 유효하다. 증착과 확산의 두께는 온도의 산술평균보다 Arrhenius 가중 적분에 훨씬 선형적으로 반응한다. $E_a$ 는 문헌값을 쓰거나 소수 후보값 중 CV로 선택한다.

### 1.5 Golden Reference Construction

A2/A3에 필요한 참조 trace $x_{\text{ref}}$ 는 다음과 같이 만든다.

1. trace 길이가 wafer마다 다르면 0–1 상대시간으로 resampling한다.
2. Train fold 내부 wafer들의 시점별 중앙값 trace를 $x_{\text{ref}}$ 로 사용한다.
3. Chamber가 여러 대라면 chamber별로 별도 산출한다.

전체 데이터의 평균 또는 중앙값 trace를 $x_{\text{ref}}$ 로 쓰면 test 정보가 특징에 침투한다. 반드시 fold 내부에서만 산출한다.

### 1.6 Slope of the Cumulative AUC Curve

누적곡선을 $C(t) = \int_0^t x\,d\tau$ 로 두고, 전체 구간에 OLS 직선을 적합했을 때의 기울기 $b$ 를 구한다. 직관적으로는 $\bar{x}$ 가 나올 것 같지만 그렇지 않다. $\bar{x}$ 가 나오는 것은 secant 기울기 $C(T)/T$ 이고, OLS 기울기는 다르다.

#### Derivation

$$b = \frac{\mathrm{Cov}(t, C)}{\mathrm{Var}(t)}, \qquad \mathrm{Var}(t) = \frac{T^2}{12}$$

분자를 부분적분한다. 양 끝에서 0이 되는 $v(\tau) = \tau^2/2 - T^2/8$ 을 쓰면

$$\int \tau\,C\,dt = \underbrace{[vC]_0^T}_{=\,0} - \int v\,C'\,dt = -\int_{-T/2}^{T/2}\left(\frac{\tau^2}{2}-\frac{T^2}{8}\right)x\,d\tau$$

이고, 정리하면 다음을 얻는다.

$$\boxed{\;b = \int_{-T/2}^{T/2} w(\tau)\,x(\tau)\,d\tau, \qquad w(\tau) = \frac{6}{T^{3}}\left(\frac{T^{2}}{4}-\tau^{2}\right)\;}$$

$w$ 는 $\int w\,d\tau = 1$ 인 정규화 포물선 (Epanechnikov) kernel로, 중앙에서 최대이고 양 끝에서 정확히 0이다.

#### The Slope Is a Weighted Average, Not a Trend

$w \ge 0$ 이고 적분이 1이므로 $b$ 는 원신호의 가중평균이다. 추세를 재는 양이 전혀 아니며, 실제로는 양 끝단을 버리고 중앙부에 가중치를 준 중심경향 추정량이다.

#### The Real Information Is the Second Time Moment

중점 기준 2차 moment를

$$m_2 = \frac{\int \tau^2 x\,d\tau}{\int x\,d\tau}$$

로 두면 다음이 성립한다.

$$\boxed{\;b = \bar{x}\left(\frac{3}{2} - \frac{6\,m_2}{T^{2}}\right)\;}$$

검산으로 $x$ 가 상수면 $m_2 = T^2/12$ 이므로 $b = \bar{x}(3/2 - 1/2) = \bar{x}$ 가 되어 일치한다.

즉 $b$ 는 평균과 시간축 분산 함수의 곱으로 완전히 분해된다. 평균 $\bar{x}$ 를 이미 특징으로 갖고 있다면 $b$ 를 그대로 넣는 것은 심한 공선성만 만든다.

#### Use the Dimensionless Ratio R

$$R = \frac{b}{\bar{x}} = \frac{3}{2} - \frac{6\,m_2}{T^{2}}$$

Table 3. Interpretation of R

| R | Waveform |
|---|---|
| $R = 1$ | 평탄하거나 시간축 대칭 균일 |
| $R \gt 1$ | 신호가 중앙에 집중 (단봉, 중앙 볼록) |
| $R \lt 1$ | 신호가 양 끝단에 집중 (U자형, 중앙 함몰) |

`R` 은 trace의 물리 단위와 무관하고 곱셈적 scale 변화에 불변이다. Gain drift나 chamber 간 절대 offset에 강건하며, 총량인 AUC와 구조적으로 독립적인 정보를 담는다.

#### Noise Properties

$$\int w^2\,d\tau = \frac{36}{T^{6}}\cdot\frac{T^{5}}{30} = \frac{1.2}{T} \quad\Longrightarrow\quad \mathrm{Var}(b) = 1.2\,\mathrm{Var}(\bar{x})$$

평균 대비 분산이 20%만 증가한다. 양의 kernel 평균이기 때문이며, 원신호에 직접 적합한 OLS 기울기가 잡음을 증폭하는 것과 정반대다.

### 1.7 Block AUC Slope

진짜 추세를 재되 잡음에 강건하게 하려면, trace를 폭 $w$ 의 등간격 연속 block $K$ 개로 나눠 block별 AUC 수열의 기울기를 취한다. Block 간 간격은 $h$ 로 둔다.

$$A_k = \int_{t_k}^{t_k + w} x\,dt, \qquad \text{slope}_{\text{blk}} = \mathrm{OLS}(A_k \sim k)$$

#### Variance Scale versus Raw Differencing

Table 4. Variance comparison

| Method | Variance |
|---|---|
| 2점 차분 | $\approx 2\sigma^2/h^2$ (증폭) |
| Block AUC 값 | $\sigma^2 / w$ (감소) |
| Block AUC의 OLS 기울기 | $\dfrac{12\sigma^2}{h^2 K(K^2-1)}$ |

적분 후 미분은 Savitzky–Golay 평활 미분과 같은 역할을 한다. 미분이 증폭할 고주파 성분을 적분이 먼저 제거한다.

$w$ 는 잡음 상관시간보다 크게 하되 $K \ge 10$ 을 유지한다. $K \lt 8$ 이면 기울기 추정이 불안정해진다.

#### Frequency View

Table 5. Frequency characteristics per feature

| Feature | Frequency characteristic | Strong domain |
|---|---|---|
| AUC | 저역통과 (DC) | 총 dose / 총량 |
| 원신호 OLS 기울기 | 고역통과 | 순간 변화율 (잡음 취약) |
| Block AUC slope | 대역통과 | 추세 성분만 선택적 추출 |
| 누적 AUC moment (`R`, `t50`) | 위상 | 시간축 상 분포 위치 |

AUC와 기울기를 함께 쓰는 근거는 서로 다른 주파수 대역을 보기 때문이다. 선형 신호 $x = a + bt$ 에서는 $\text{AUC} = aT + bT^2/2$ 이므로 둘이 강하게 공선이고, 파형이 비선형일 때만 분리된다.

### 1.8 Run-to-Run AUC Slope on the Wafer Axis

Wafer index $i$ 에 대한 AUC의 기울기는 wafer 내부 특징이 아니라 장비 상태 특징이다.

$$s = \mathrm{OLS}\big(\text{AUC}_i \sim i\big) \quad \text{over the last } m \text{ wafers (strictly causal)}$$

잡는 물리량은 chamber seasoning 진행도, 벽면 증착물 축적, target과 전극 소모, MFC와 압력계의 calibration drift, PM 이후 회복 궤적이다. AUC가 drift를 누적하는 성질이 여기서는 단점이 아니라 신호가 된다. R2R 제어 지표 및 covariate shift 진단에 그대로 사용한다.

반드시 strictly causal window, 즉 현재 wafer 이전 $m$ 장만 사용한다. 중심 이동창이나 전체 구간 회귀를 쓰면 미래 정보가 들어가고, 시계열 CV로는 잡히지 않는 종류의 누출이 생긴다.

Trace나 특징 단위가 아니라 chamber 단위로 소수만 추가한다.

## 2. Failure Modes and Mitigations

### 2.1 Baseline Drift

적분은 drift를 누적한다. Trace에 느린 offset $\delta$ 가 있으면 AUC는 $\delta T$ 만큼 통째로 이동하므로, 공정 변화가 아니라 계측 calibration 이력이 AUC 변동의 지배 성분이 되어 model을 완전히 오염시킨다. 필수 대응은 다음과 같다.

1. A1/A2 형태만 사용한다. Baseline 또는 golden reference를 빼고 적분하면 drift가 상쇄된다.
2. Pre-trace baseline을 차감한다. 초기 구간 중앙값 $x_0$ 기준으로 $\int (x - x_0)\,dt$ 를 쓴다.
3. Chamber가 여러 대라면 chamber 내 중심화로 절대 offset을 제거한다.
4. Drift 자체를 chamber 누적 RF hour 등 별도 공변량으로 명시한다.

### 2.2 Timestamp Accuracy

1 sample의 오배치만으로도 trace 길이에 반비례하는 적분 오차가 생기고, AUC는 이 오차에 선형으로 민감하다.

- 명목 $\Delta t$ 대신 실제 timestamp 차분을 사용하여 sampling jitter를 흡수한다.
- 결측은 적분 전에 선형보간한다. NaN 하나가 적분 전체를 무효화한다.
- trace 시작과 종료 경계의 transient 구간은 baseline 산출에서 제외한다.

### 2.3 Value of Dimensionless Features

`R`, `t50`, `pauc_k` 는 모두 무차원이라 장비와 chamber 간 이식성이 있다. 절대 AUC 위주의 특징 세트는 chamber 교차 검증에서 무너지고, 무차원 특징 위주는 유지된다. 이것이 drift 오염의 가장 빠른 진단법이다.

## 3. Dimensionality Control

### 3.1 Channel Pruning

다음에 해당하는 channel을 제거한다.

- 분산이 0에 가까운 상수 channel
- Setpoint 복제 channel
- 상호상관 $\lvert\rho\rvert \gt 0.98$ 인 중복 channel

Chamber가 여러 대라면 이 판정은 chamber별로 수행한다.

### 3.2 Taxonomy Group Pooling

AUC는 가법적 (additive) 이므로 그룹 합산이 통계적 편법이 아니라 물리량이다.

- 개별 가스 AUC의 합은 총 가스 투입량으로, 특징 1개가 된다.
- RF forward AUC의 합은 총 전력 dose로, 특징 1개가 된다.
- 복수의 압력계 channel은 대표 1개와 channel 간 AUC 비율로 줄인다.

### 3.3 Supervised Final Reduction

- Taxonomy를 group으로 하는 sparse group lasso는 물리적으로 해석 가능한 희소성을 준다. 개별 channel 전부를 lasso에 던지는 것보다 표본이 작을 때 선택 변동성이 훨씬 작다.
- PLS latent 변수를 쓴다.

표본이 작으면 이 단계 이후 규제 선형 model이나 PLS가 GBM보다 안정적이다.

## 4. Recommended Feature Set

Trace당 소수의 특징만 유지한다.

Table 6. Recommended per-trace features

| Feature | Type | Captures |
|---|---|---|
| `auc_base` | Integral | 총량 / dose |
| `t50` | Integral (위상) | 무게중심 |
| `R` | Integral (2차 moment) | 중앙집중도 |
| `auc_res_abs` | Integral (참조 기반) | Golden 대비 형상 이탈 |

선택적 추가는 `slope_blk` (대역통과 추세) 와 `tv` (진동) 다. 평균 $\bar{x}$ 와 단순 AUC는 $T$ 가 wafer마다 일정하다면 상수배 관계이므로 둘 중 하나만 사용한다.

여기에 §1.8의 chamber 단위 R2R 특징을 별도로 소수 추가한다. 특징별 수학적 정체는 [Appendix B](#appendix-b-feature-moment-correspondence) 에 정리한다.

## 5. Implementation

```python
# Python
import numpy as np

def integral_features(x, t, x_ref=None, n_sub=3, block_w=20):
    """
    x, t : (N,) single-wafer trace. t holds real timestamps.
    x_ref: golden reference (computed inside the train fold only, aligned to x).
    """
    dt   = np.diff(t)                                  # real differences, not nominal
    trap = lambda y: np.sum((y[:-1] + y[1:]) / 2 * dt)
    T    = t[-1] - t[0]
    tau  = (t - t[0]) - T / 2                          # centered time axis
    f    = {}

    x0 = np.median(x[:5])                              # pre-trace baseline
    d0 = x - x0

    f['auc_base']   = trap(d0)                              # A1
    f['tv']         = np.sum(np.abs(np.diff(x)))            # A5
    f['auc_energy'] = trap((x - x.mean()) ** 2)             # A6

    if x_ref is not None:                                   # A2 / A3
        d = x - x_ref
        f['auc_res_abs']   = trap(np.abs(d))
        f['auc_res_over']  = trap(np.maximum(d, 0))
        f['auc_res_under'] = trap(np.maximum(-d, 0))

    # A4: partial AUC -- equal-width block ratios (keeps shape only)
    tot = trap(np.abs(d0)) + 1e-12
    for k, ix in enumerate(np.array_split(np.arange(len(x)), n_sub)):
        seg = np.abs(d0)[ix]
        f[f'pauc_{k}'] = np.sum((seg[:-1] + seg[1:]) / 2 * dt[ix[:-1]]) / tot

    # phase: cumulative AUC quantiles (dimensionless)
    a = np.abs(d0)
    c = np.concatenate([[0.0], np.cumsum((a[:-1] + a[1:]) / 2 * dt)])
    c /= (c[-1] + 1e-12)
    for q in (0.25, 0.50, 0.75):
        f[f't{int(q*100)}'] = np.interp(q, c, t - t[0]) / T

    # R: dimensionless form of the cumulative-AUC OLS slope = 3/2 - 6*m2/T^2
    m2 = trap(tau**2 * d0) / (trap(d0) + 1e-12)
    f['R'] = 1.5 - 6.0 * m2 / T**2

    # block AUC slope: band-pass trend
    nb = max(len(x) // block_w, 3)
    A  = np.array([np.sum((d0[ix][:-1] + d0[ix][1:]) / 2 * dt[ix[:-1]])
                   for ix in np.array_split(np.arange(len(x)), nb)])
    k  = np.arange(len(A))
    f['slope_blk'] = np.polyfit(k, A, 1)[0]

    return f
```

## 6. Validation Protocol

### 6.1 Redundancy Checks before Adding Features

Table 7. Redundancy checks

| Check | Threshold | Action |
|---|---|---|
| $\mathrm{corr}(\text{auc\_base},\ \bar{x} T)$ | $\gt 0.99$ | AUC 폐기, 평균만 유지 |
| $\mathrm{corr}(b,\ \bar{x})$ | $\gt 0.95$ | $b$ 대신 무차원 `R` 사용 |
| $\mathrm{corr}(\text{slope\_blk},\ \text{auc\_base})$ | $\gt 0.95$ | 파형이 거의 선형이므로 하나 폐기 |
| $\mathrm{Var}(R)$ (wafer 간) | $\approx 0$ | 형상 변동 없음, 폐기 |

### 6.2 Performance Criteria

- Test $R^2$ 증분만 근거로 사용한다. Train $R^2$ 개선은 근거가 되지 않는다. 표본이 작으면 특징을 늘릴수록 train $R^2$ 는 거의 항상 올라간다.
- Nested time-series CV를 쓴다. Golden reference, pruning 기준, scaler, 그룹 정의를 모두 train fold 내부에서만 산출한다.
- $R^2_{\max}$ 대비로 평가한다. 계측 반복성 $\sigma$ 로부터 잡음 천장을 계산하고, 그 대비 몇 %에 도달했는지로 판단한다. 천장의 80%를 넘으면 특징 추가를 멈추는 것이 합리적이다.
- Chamber 교차 검증을 한다. 한 chamber로 학습해 다른 chamber로 평가하며, 판정 기준은 §2.3과 같다.

### 6.3 Dimensionality Gate

최종 model 입력 특징 수 $p_{\text{final}}$ 이 $n/5$ 이하인지 확인한다.

## Appendix A. Terminology

- **AUC (Area Under the Curve)**: trace를 시간에 대해 적분한 값이다.
- **Channel**: trace 하나가 기록되는 개별 측정 계열이다.
- **Covariate shift**: 입력 변수의 분포가 학습 시점과 예측 시점 사이에 달라지는 dataset shift의 한 형태다.
- **CV (Cross-Validation)**: 데이터를 여러 fold로 나눠 학습과 평가를 반복하는 model 검증 방법이다.
- **DC (Direct Current)**: 주파수 0 성분, 즉 신호의 평균 성분을 가리킨다.
- **Epanechnikov kernel**: $1 - u^2$ 에 비례하는 포물선 형태의 kernel 함수다.
- **FDC (Fault Detection and Classification)**: 장비 신호로 공정 이상을 탐지하고 분류하는 체계다.
- **GBM (Gradient Boosting Machine)**: 얕은 결정 나무를 순차적으로 더해 가는 ensemble 학습 방법이다.
- **Golden reference**: 정상 상태를 대표하는 기준 trace다.
- **L1**: 절댓값의 합 또는 적분으로 정의되는 norm이다.
- **MFC (Mass Flow Controller)**: 가스 유량을 제어하는 장치다.
- **OLS (Ordinary Least Squares)**: 잔차 제곱합을 최소화하는 선형 회귀 적합 방법이다.
- **PLS (Partial Least Squares)**: 예측변수와 응답변수의 공분산을 최대화하는 latent 변수 회귀 방법이다.
- **PM (Preventive Maintenance)**: 장비의 예방 정비 작업이다.
- **R2R (Run-to-Run)**: wafer 또는 lot 단위 실행 간의 보정 제어를 가리킨다.
- **RF (Radio Frequency)**: plasma 생성에 쓰이는 고주파 전력을 가리킨다.
- **Savitzky–Golay filter**: 이동 구간에 다항식을 적합해 평활과 미분을 동시에 수행하는 filter다.
- **sccm (standard cubic centimeters per minute)**: 표준 상태 기준 분당 세제곱센티미터로 나타낸 가스 유량 단위다.
- **Sparse group lasso**: 그룹 단위와 개별 계수 단위의 희소성을 동시에 유도하는 규제 회귀 방법이다.
- **Summary statistics**: trace 전체를 소수의 스칼라로 요약한 값이다.
- **Taxonomy**: trace를 가스, 전력, 압력, 온도 등 물리 계통으로 묶는 분류 체계다.
- **Thermal budget**: 공정 중 온도와 시간의 누적 효과를 나타내는 양이다.
- **Trace**: 장비가 주기적으로 송출하는 시계열 기록이다.
- **VIF (Variance Inflation Factor)**: 다중공선성이 회귀 계수의 분산을 키우는 정도를 나타내는 지표다.
- **Wide data**: 표본 수 $n$ 보다 변수 수 $p$ 가 많은 ($p \gt n$) 데이터를 가리킨다.

## Appendix B. Feature-Moment Correspondence

Table 8. Feature-moment correspondence

| Feature | Mathematical identity | Time moment order |
|---|---|---|
| 산술평균, 단순 AUC | 균등가중 평균 | 0차 |
| 원신호 OLS 기울기 | $\int \tau x\,d\tau$ 정규화 | 1차 |
| $b$ / `R` | $\int \tau^2 x\,d\tau$ 를 통한 분해 | 2차 |
| `t50` | 누적분포 중앙값 | 위상 (분위) |
| `skew` | $\int \tau^3 x\,d\tau$ 정규화 | 3차 |
| `slope_blk` | 대역통과 filter 후 1차 | 1차 (평활) |

0차만 쓰면 형상 정보가 전부 소실되고, 차수를 무한정 올리면 잡음만 증폭된다. 0–2차와 위상 하나의 조합이 정보 대비 분산의 절충점이다.
