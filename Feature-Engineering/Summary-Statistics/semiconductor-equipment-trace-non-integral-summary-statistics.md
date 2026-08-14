# Semiconductor Equipment Trace Non-Integral Summary Statistics
Rev. 13 | Created: 2026-07-29 | Updated: 2026-08-14 18:06 CDT

장비 trace 시계열을 wafer당 고정 길이 vector로 변환하는 특징 가운데, 적분 연산자를 쓰지 않는 non-integral 특징의 정의, 실패 모드, 차원 통제, 검증 규약을 정리한다.

Data는 [wafer, feature, trace] 의 dimension을 갖는다. 여기서 trace의 예로는 sensor 신호나 FDC data가 있다.

핵심 제약은 결과가 wide data, 즉 $p \gg n$ 이 되지 않아야 한다.

## 1. Non-Integral Statistics

단일 wafer trace를 $x(t)$, $t \in [0, T]$ 로 두고, $\bar{x}$ 를 그 산술평균으로 쓴다.

적분을 쓰지 않는 기본 특징으로, 해석성이 높고 계산 비용이 낮다. 특징은 잡는 축으로 묶이며, 축마다 core를 하나 두고 대체 가능한 특징이 있으면 option으로 함께 둔다. Option이 없는 축도 있다. Option은 core와 같은 축을 재므로 둘을 함께 넣으면 공선성이 생긴다. 축당 하나만 고르고, 어느 쪽을 고를지는 §5.1의 검사로 정한다.

Table 1. Non-integral features

| Feature | Axis | Captures | Role | Redundant with |
|---|---|---|---|---|
| `mean` | 위치 | 산술 중심 | Core | — |
| `median` | 위치 | outlier에 강건한 중심 | Option | `mean` |
| `std` | 산포 | 전체 구간 변동 폭 | Core | — |
| `range` | 산포 | 최대 진폭 | Option | `std` |
| `iqr` | 산포 | outlier에 강건한 변동 폭 | Option | `std` |
| `slope` | 추세 | 1차 시간 moment | Core | — |
| `sigma_st` | 단기 변동 | 이웃 sample 간 변동, SPC short-term sigma | Core | — |
| `max_delta` | 단기 변동 | 단발 jump 진폭 | Option | `sigma_st` |
| `sigma_ratio` | 거칠기 | 단기 변동과 전체 변동의 비 | Core | — |
| `zcr` | 거칠기 | 중심선 교차 빈도 | Option | `sigma_ratio` |
| `peak_time_norm` | 시각 | 최대점의 상대 시각 | Core | — |
| `duration` | 길이 | 처리 시간 | Core | — |

수식 정의는 [Appendix B](#appendix-b-feature-definitions) 에, 계산 절차는 [Appendix C](#appendix-c-implementation) 에 정리한다.

### 1.1 Short-Term Sigma and Roughness

`sigma_st` 는 MSSD를 2로 나눈 값의 제곱근이며, SPC의 I-MR chart가 쓰는 short-term sigma와 같은 양이다. `std` 가 전체 구간의 변동을 재는 데 반해 이쪽은 이웃 sample 사이의 변동만 재므로, 둘은 서로 다른 시간 scale을 본다.

두 양의 비 `sigma_ratio` 는 lag-1 autocorrelation $\rho_1$ 의 직접적인 추정량이다.

$$\text{sigma\_ratio} = \frac{\sigma_{\text{st}}}{s} = \sqrt{1 - \hat{\rho}_1}$$

백색잡음이면 $\mathrm{MSSD} = 2s^2$ 이므로 비가 정확히 1이 된다. 임의로 문턱을 정할 필요 없이 이론값 1을 기준으로 읽으면 된다.

- 비가 1보다 작으면 신호가 매끄럽다. 이웃 sample이 서로 닮아 있다.
- 비가 1이면 백색잡음이다.
- 비가 1보다 크면 sample 단위로 진동한다.

`sigma_ratio` 는 분자와 분모가 같은 단위라 gain drift와 chamber 간 scale 차이에 불변이다. 같은 축의 option인 `zcr` 은 중심선 교차를 세는 방식이라 진폭 outlier에 강건한 대신, ramp가 있는 trace에서는 중심선을 한 번만 지나므로 진동을 과소평가한다. 추세가 뚜렷한 trace에는 `sigma_ratio`, 진폭이 튀는 trace에는 `zcr` 이 맞다.

## 2. Failure Modes and Mitigations

### 2.1 Baseline Drift

Trace에 느린 offset $\delta$ 가 있으면 위치 특징인 `mean` 과 `median` 은 그만큼 통째로 이동한다. 반면 나머지 특징은 산포·차분·순위로 정의되어 offset에 불변이므로 drift의 영향을 받지 않는다. 위치 특징만 오염된다는 이 비대칭이 진단의 출발점이다.

- 초기 구간 중앙값을 pre-trace baseline으로 잡아 차감한 뒤 위치 특징을 계산한다.
- Chamber가 여러 대라면 chamber 내 중심화로 절대 offset을 제거한다.
- Drift 자체를 chamber 누적 가동 시간 등 별도 공변량으로 명시한다.

### 2.2 Timestamp and Missing Samples

`slope`, `sigma_st`, `max_delta` 는 sampling 간격에 직접 의존하므로 timestamp 품질에 민감하다.

- 명목 $\Delta t$ 대신 실제 timestamp 차분을 사용하여 sampling jitter를 흡수한다.
- 결측은 특징 계산 전에 선형보간한다. 보간하지 않으면 결측 위치의 큰 차분이 `sigma_st` 와 `max_delta` 를 부풀린다.
- trace 시작과 종료 경계의 transient 구간은 baseline 산출에서 제외한다.

`sigma_ratio` 도 sampling 주기에 의존한다. 매끄러운 신호에서는 $\Delta x \approx x' \Delta t$ 이므로 비가 $\Delta t$ 에 비례한다. 진폭 단위로는 무차원이지만 시간 단위로는 아니므로, 주기가 다른 recipe나 chamber 사이에서는 직접 비교하지 않는다.

### 2.3 Unit Dependence

`mean`, `median`, `std`, `range`, `iqr`, `sigma_st`, `max_delta` 는 모두 trace의 물리 단위를 그대로 갖는다. 따라서 gain drift나 chamber 간 절대 offset에 취약하고, chamber 교차 검증에서 먼저 무너지는 쪽이 된다.

`sigma_ratio`, `zcr`, `peak_time_norm` 은 무차원이라 곱셈적 scale 변화에 불변이므로 이식성이 높다. 변동계수 $s/\bar{x}$ 처럼 비율로 만든 특징도 같은 성질을 갖는다. 다만 `sigma_ratio` 와 `zcr` 의 chamber 간 비교는 §2.2의 sampling 주기 조건을 만족할 때만 유효하다. 단위를 갖는 특징과 무차원 특징을 함께 두고, chamber 교차 검증에서 어느 쪽이 살아남는지로 오염 여부를 판정한다.

## 3. Dimensionality Control

### 3.1 Channel Pruning

다음에 해당하는 channel을 제거한다.

- 분산이 0에 가까운 상수 channel
- Setpoint 복제 channel
- 상호상관 $\lvert\rho\rvert \gt 0.98$ 인 중복 channel

Chamber가 여러 대라면 이 판정은 chamber별로 수행한다. 상수 channel은 `sigma_ratio` 의 분모를 0으로 만들므로 이 단계에서 반드시 걸러야 한다.

### 3.2 Taxonomy Group Pooling

평균과 산포는 가법적이지 않으므로, 그룹에 속한 channel의 특징을 그냥 더해도 물리량이 되지 않는다. 대신 다음과 같이 줄인다.

- 그룹 대표 channel 1개의 특징을 남긴다.
- 그룹 내 channel 간 비율이나 산포를 특징 1개로 요약한다.

### 3.3 Supervised Final Reduction

- Taxonomy를 group으로 하는 sparse group lasso는 물리적으로 해석 가능한 희소성을 준다. 개별 channel 전부를 lasso에 던지는 것보다 표본이 작을 때 선택 변동성이 훨씬 작다.
- PLS latent 변수를 쓴다.

표본이 작으면 이 단계 이후 규제 선형 model이나 PLS가 GBM보다 안정적이다.

## 4. Feature Selection

Table 1의 core 7개를 기본 세트로 삼는다. 일곱 축을 하나씩만 덮으므로 축 사이의 공선성이 구조적으로 없다.

Option은 core를 보태는 것이 아니라 **교체**하는 용도다. 축당 두 개를 함께 넣으면 §5.1의 검사에 걸린다. 교체 기준은 다음과 같다.

- 위치가 outlier에 흔들리면 `mean` 대신 `median` 을 쓴다.
- 산포가 outlier에 흔들리면 `std` 대신 `iqr` 을, 절대 진폭 자체가 규격 항목이면 `range` 를 쓴다.
- 단발 spike가 관심사면 `sigma_st` 대신 `max_delta` 를 쓴다. 전자는 spike 하나를 전체 구간에 평균해 버린다.
- 진폭이 크게 튀는 trace라면 `sigma_ratio` 대신 `zcr` 을 쓴다 (§1.1).

`duration` 은 trace 길이가 wafer마다 달라질 때만 정보를 가지므로, 길이가 고정된 recipe에서는 core에서 뺀다.

## 5. Validation Protocol

### 5.1 Redundancy Checks before Adding Features

Core와 option을 함께 넣었는지, 또는 core끼리 우연히 겹쳤는지를 다음으로 판정한다.

Table 2. Redundancy checks

| Check | Threshold | Action |
|---|---|---|
| $\mathrm{corr}(\text{median},\ \text{mean})$ | $\gt 0.98$ | 분포가 대칭, `median` 폐기 |
| $\mathrm{corr}(\text{iqr},\ \text{std})$ | $\gt 0.98$ | outlier 없음, `iqr` 폐기 |
| $\mathrm{corr}(\text{range},\ \text{std})$ | $\gt 0.95$ | `range` 폐기 |
| $\mathrm{corr}(\text{max\_delta},\ \text{sigma\_st})$ | $\gt 0.95$ | 단발 spike 없음, `max_delta` 폐기 |
| $\mathrm{corr}(\text{zcr},\ \text{sigma\_ratio})$ | $\gt 0.95$ | 같은 거칠기를 잼, `zcr` 폐기 |
| $\mathrm{corr}(\text{sigma\_st},\ \text{std})$ | $\gt 0.95$ | 두 시간 scale이 구분되지 않음, `sigma_ratio` 도 무정보 |
| $\mathrm{Var}(\text{sigma\_ratio})$ (wafer 간) | $\approx 0$ | 거칠기가 일정, `sigma_ratio` 와 `zcr` 폐기 |
| $\mathrm{Var}(\text{duration})$ (wafer 간) | $\approx 0$ | 길이가 일정, 폐기 |

### 5.2 Performance Criteria

- Test $R^2$ 증분만 근거로 사용한다. Train $R^2$ 개선은 근거가 되지 않는다. 표본이 작으면 특징을 늘릴수록 train $R^2$ 는 거의 항상 올라간다.
- Nested time-series CV를 쓴다. Pruning 기준, scaler, 그룹 정의를 모두 train fold 내부에서만 산출한다.
- $R^2_{\max}$ 대비로 평가한다. 계측 반복성 $\sigma$ 로부터 잡음 천장을 계산하고, 그 대비 몇 %에 도달했는지로 판단한다. 천장의 80%를 넘으면 특징 추가를 멈추는 것이 합리적이다.
- Chamber 교차 검증을 한다. 한 chamber로 학습해 다른 chamber로 평가하며, 판정 기준은 §2.3과 같다.

### 5.3 Dimensionality Gate

최종 model 입력 특징 수 $p_{\text{final}}$ 이 $n/5$ 이하인지 확인한다.

## Appendix A. Terminology

- **Channel**: trace 하나가 기록되는 개별 측정 계열이다.
- **CV (Cross-Validation)**: 데이터를 여러 fold로 나눠 학습과 평가를 반복하는 model 검증 방법이다.
- **FDC (Fault Detection and Classification)**: 장비 신호로 공정 이상을 탐지하고 분류하는 체계다.
- **GBM (Gradient Boosting Machine)**: 얕은 결정 나무를 순차적으로 더해 가는 ensemble 학습 방법이다.
- **I-MR chart**: 개별값과 이동범위를 함께 보는 SPC 관리도다.
- **MSSD (Mean Square Successive Difference)**: 이웃 sample 차분의 제곱평균이다.
- **OLS (Ordinary Least Squares)**: 잔차 제곱합을 최소화하는 선형 회귀 적합 방법이다.
- **PLS (Partial Least Squares)**: 예측변수와 응답변수의 공분산을 최대화하는 latent 변수 회귀 방법이다.
- **Sparse group lasso**: 그룹 단위와 개별 계수 단위의 희소성을 동시에 유도하는 규제 회귀 방법이다.
- **SPC (Statistical Process Control)**: 공정 변동을 통계적 관리한계로 감시하는 체계다.
- **Summary statistics**: trace 전체를 소수의 스칼라로 요약한 값이다.
- **Taxonomy**: trace를 가스, 전력, 압력, 온도 등 물리 계통으로 묶는 분류 체계다.
- **Trace**: 장비가 주기적으로 송출하는 시계열 기록이다.
- **Wide data**: 표본 수 $n$ 보다 변수 수 $p$ 가 많은 ($p \gt n$) 데이터를 가리킨다.
- **ZCR (Zero-Crossing Rate)**: 신호가 기준선을 가로지르는 빈도다.

## Appendix B. Feature Definitions

Trace를 $x_1, \dots, x_N$ 으로, timestamp를 $t_1, \dots, t_N$ 으로 두고 $T = t_N - t_1$ 로 쓴다. 차분은 $\Delta x_i = x_{i+1} - x_i$ 이며, $i$ 는 sample index, $N$ 은 sample 수다.

Variable 열은 그 행에서 새로 쓰는 기호만 밝힌다. 표시가 없는 행은 위의 공통 표기만 쓴다.

Table 3. Feature definitions

| Feature | Definition | Variable |
|---|---|---|
| `mean` | $\bar{x} = \dfrac{1}{N}\sum_{i=1}^{N} x_i$ | $\bar{x}$ 는 산술평균이며, 이후 행에서 그대로 쓴다 |
| `median` | 정렬한 $x$ 의 중앙값 | — |
| `std` | $s = \sqrt{\dfrac{1}{N-1}\sum_{i=1}^{N}(x_i - \bar{x})^2}$ | $s$ 는 표본표준편차이며, 이후 행에서 그대로 쓴다. 분모 $N-1$ 은 불편추정을 위한 자유도다 |
| `range` | $\max_i x_i - \min_i x_i$ | — |
| `iqr` | $Q_3 - Q_1$ | $Q_1$ 과 $Q_3$ 은 $x$ 의 제1·제3 사분위수다 |
| `slope` | $\dfrac{\sum_i (t_i - \bar{t})(x_i - \bar{x})}{\sum_i (t_i - \bar{t})^2}$ | $\bar{t}$ 는 timestamp의 산술평균이다 |
| `sigma_st` | $\sigma_{\text{st}} = \sqrt{\mathrm{MSSD}/2}$ | $\mathrm{MSSD} = \dfrac{1}{N-1}\sum_{i=1}^{N-1}(\Delta x_i)^2$ 는 이웃 차분 제곱의 평균이다. 2로 나누는 근거는 §1.1에 있다 |
| `max_delta` | $\max_i \lvert \Delta x_i \rvert$ | — |
| `sigma_ratio` | $\sigma_{\text{st}} / s$ | 두 값 모두 위 행에서 정의한 것이다 |
| `zcr` | $\dfrac{1}{N-1}\sum_{i=1}^{N-1} \mathbb{1}\big[\mathrm{sign}(x_i - \bar{x}) \neq \mathrm{sign}(x_{i+1} - \bar{x})\big]$ | $\mathbb{1}[\cdot]$ 은 조건이 참이면 1, 아니면 0인 지시함수다. $\mathrm{sign}$ 의 기준선은 $\bar{x}$ 다 |
| `peak_time_norm` | $(t_{i^*} - t_1) / T$ | $i^* = \arg\max_i x_i$ 는 최댓값이 나오는 sample index다 |
| `duration` | $T$ | — |

## Appendix C. Implementation

```python
# Python
import numpy as np

def non_integral_summary(x, t):
    """
    x, t : (N,) single-wafer trace. t holds real timestamps.
    """
    T        = t[-1] - t[0]
    q1, q3   = np.percentile(x, [25, 75])
    xc       = x - x.mean()                          # centered, for zero crossings
    dx       = np.diff(x)
    std      = x.std(ddof=1)
    mssd     = np.mean(dx ** 2)                      # mean square successive difference
    sigma_st = np.sqrt(mssd / 2)                     # SPC short-term sigma

    if std == 0:
        raise ValueError("constant trace: prune the channel before summarizing")

    return {
        'mean':           x.mean(),
        'median':         np.median(x),
        'std':            std,
        'range':          x.max() - x.min(),
        'iqr':            q3 - q1,
        'slope':          np.polyfit(t - t[0], x, 1)[0],   # OLS on the raw signal
        'sigma_st':       sigma_st,
        'max_delta':      np.max(np.abs(dx)),
        'sigma_ratio':    sigma_st / std,
        'zcr':            np.mean(np.signbit(xc[:-1]) != np.signbit(xc[1:])),
        'peak_time_norm': (t[np.argmax(x)] - t[0]) / T,
        'duration':       T,
    }
```
