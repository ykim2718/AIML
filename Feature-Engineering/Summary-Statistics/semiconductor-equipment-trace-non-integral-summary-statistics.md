# Semiconductor Equipment Trace Non-Integral Summary Statistics
Rev. 11 | Created: 2026-07-29 | Updated: 2026-08-14 16:52 CDT

장비 trace 시계열을 wafer당 고정 길이 vector로 변환하는 특징 가운데, 적분 연산자를 쓰지 않는 non-integral 특징의 정의, 실패 모드, 차원 통제, 검증 규약을 정리한다.

Data는 [wafer, feature, trace] 의 dimension을 갖는다. 여기서 trace의 예로는 sensor 신호나 FDC data가 있다.

핵심 제약은 결과가 wide data, 즉 $p \gg n$ 이 되지 않아야 한다.

## 1. Non-Integral Statistics

단일 wafer trace를 $x(t)$, $t \in [0, T]$ 로 두고, $\bar{x}$ 를 그 산술평균으로 쓴다.

적분을 쓰지 않는 기본 특징으로, 해석성이 높고 계산 비용이 낮다.

Table 1. Non-integral features

| Feature | Definition | Captures |
|---|---|---|
| `mean` | $\bar{x}$ | 위치 |
| `median` | 표본중앙값 | 강건 위치 |
| `std` | 표본표준편차 | 산포 |
| `range` | $\max - \min$ | 진폭 |
| `iqr` | $Q_3 - Q_1$ | 강건 산포 |
| `slope` | $\mathrm{OLS}(x \sim t)$ 의 기울기 | 1차 시간 moment, 즉 추세 |
| `noise` | $\mathrm{std}(\Delta x)$ | 고주파 잡음 수준 |
| `max_delta` | $\max \lvert \Delta x \rvert$ | 단발 jump 진폭 |
| `zcr` | 중심화한 신호의 zero-crossing rate | 진동 빈도 |
| `peak_time_norm` | $\arg\max x / T$ | 최대점 시각 |
| `duration` | $T$ | 처리 시간 |

각 특징이 잡는 축이 서로 다르다. `mean` 과 `median` 은 위치, `std` 와 `iqr` 은 산포, `slope` 는 시간축 추세, `noise` 와 `max_delta` 는 고주파 성분의 크기, `zcr` 은 그 빈도, `peak_time_norm` 은 최대점의 시각이며, 이 축들이 겹치는 만큼만 특징 사이에 공선성이 생긴다. 계산 절차는 [Appendix B](#appendix-b-implementation) 에 정리한다.

`mean` 과 `median` 은 짝으로 둘 때 값을 한다. 둘의 차이가 곧 진폭 분포의 비대칭이므로, 3차 moment를 따로 계산하는 것보다 강건하게 같은 정보를 준다. `noise` 와 `max_delta` 도 같은 관계다. `noise` 는 차분의 표준편차라 단발 spike 하나를 평균해 버리지만, `max_delta` 는 그 하나를 그대로 잡는다.

## 2. Failure Modes and Mitigations

### 2.1 Baseline Drift

Trace에 느린 offset $\delta$ 가 있으면 위치 특징인 `mean` 과 `median` 은 그만큼 통째로 이동한다. 반면 나머지 특징은 산포·차분·순위로 정의되어 offset에 불변이므로 drift의 영향을 받지 않는다. 위치 특징만 오염된다는 이 비대칭이 진단의 출발점이다.

- 초기 구간 중앙값을 pre-trace baseline으로 잡아 차감한 뒤 위치 특징을 계산한다.
- Chamber가 여러 대라면 chamber 내 중심화로 절대 offset을 제거한다.
- Drift 자체를 chamber 누적 가동 시간 등 별도 공변량으로 명시한다.

### 2.2 Timestamp and Missing Samples

`slope` 와 `noise` 는 sampling 간격에 직접 의존하므로 timestamp 품질에 민감하다.

- 명목 $\Delta t$ 대신 실제 timestamp 차분을 사용하여 sampling jitter를 흡수한다.
- 결측은 특징 계산 전에 선형보간한다. 보간하지 않으면 `noise` 가 결측 위치의 큰 차분을 잡음으로 오인한다.
- trace 시작과 종료 경계의 transient 구간은 baseline 산출에서 제외한다.

### 2.3 Unit Dependence

`mean`, `std`, `range`, `iqr` 은 모두 trace의 물리 단위를 그대로 갖는다. 따라서 gain drift나 chamber 간 절대 offset에 취약하고, chamber 교차 검증에서 먼저 무너지는 쪽이 된다.

변동계수 $\mathrm{std}/\bar{x}$ 처럼 비율로 만든 특징과 시간축을 정규화한 `peak_time_norm` 은 곱셈적 scale 변화에 불변이므로 이식성이 높다. 단위를 갖는 특징과 무차원 특징을 함께 두고, chamber 교차 검증에서 어느 쪽이 살아남는지로 오염 여부를 판정한다.

## 3. Dimensionality Control

### 3.1 Channel Pruning

다음에 해당하는 channel을 제거한다.

- 분산이 0에 가까운 상수 channel
- Setpoint 복제 channel
- 상호상관 $\lvert\rho\rvert \gt 0.98$ 인 중복 channel

Chamber가 여러 대라면 이 판정은 chamber별로 수행한다.

### 3.2 Taxonomy Group Pooling

평균과 산포는 가법적이지 않으므로, 그룹에 속한 channel의 특징을 그냥 더해도 물리량이 되지 않는다. 대신 다음과 같이 줄인다.

- 그룹 대표 channel 1개의 특징을 남긴다.
- 그룹 내 channel 간 비율이나 산포를 특징 1개로 요약한다.

### 3.3 Supervised Final Reduction

- Taxonomy를 group으로 하는 sparse group lasso는 물리적으로 해석 가능한 희소성을 준다. 개별 channel 전부를 lasso에 던지는 것보다 표본이 작을 때 선택 변동성이 훨씬 작다.
- PLS latent 변수를 쓴다.

표본이 작으면 이 단계 이후 규제 선형 model이나 PLS가 GBM보다 안정적이다.

## 4. Recommended Feature Set

Trace당 소수의 특징만 유지한다.

Table 2. Recommended per-trace features

| Feature | Captures |
|---|---|
| `mean` | 위치 |
| `std` | 산포 |
| `slope` | 시간축 추세 |
| `noise` | 고주파 잡음 |

`iqr` 은 outlier가 잦은 trace에서 `std` 를 대체하고, `median` 은 위치가 outlier에 흔들릴 때 `mean` 을 보완한다. 단발 spike가 문제라면 `range` 보다 `max_delta` 를 쓴다. 전자는 절대 진폭이라 drift에 딸려 움직이지만 후자는 차분이라 그렇지 않다. `zcr` 과 `peak_time_norm` 은 각각 진동과 timing이 응답변수와 관련될 때만 추가하며, `duration` 은 trace 길이가 wafer마다 달라질 때만 정보를 갖는다.

## 5. Validation Protocol

### 5.1 Redundancy Checks before Adding Features

Table 3. Redundancy checks

| Check | Threshold | Action |
|---|---|---|
| $\mathrm{corr}(\text{median},\ \text{mean})$ | $\gt 0.98$ | 분포가 대칭, `median` 폐기 |
| $\mathrm{corr}(\text{std},\ \text{iqr})$ | $\gt 0.98$ | 하나만 유지, outlier가 잦으면 `iqr` |
| $\mathrm{corr}(\text{range},\ \text{std})$ | $\gt 0.95$ | `range` 폐기 |
| $\mathrm{corr}(\text{noise},\ \text{std})$ | $\gt 0.95$ | 잡음이 산포를 지배, `noise` 폐기 |
| $\mathrm{corr}(\text{max\_delta},\ \text{noise})$ | $\gt 0.95$ | 단발 spike가 없음, `max_delta` 폐기 |
| $\mathrm{Var}(\text{zcr})$ (wafer 간) | $\approx 0$ | 진동 양상이 일정, 폐기 |
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
- **OLS (Ordinary Least Squares)**: 잔차 제곱합을 최소화하는 선형 회귀 적합 방법이다.
- **PLS (Partial Least Squares)**: 예측변수와 응답변수의 공분산을 최대화하는 latent 변수 회귀 방법이다.
- **Sparse group lasso**: 그룹 단위와 개별 계수 단위의 희소성을 동시에 유도하는 규제 회귀 방법이다.
- **Summary statistics**: trace 전체를 소수의 스칼라로 요약한 값이다.
- **Taxonomy**: trace를 가스, 전력, 압력, 온도 등 물리 계통으로 묶는 분류 체계다.
- **Trace**: 장비가 주기적으로 송출하는 시계열 기록이다.
- **Wide data**: 표본 수 $n$ 보다 변수 수 $p$ 가 많은 ($p \gt n$) 데이터를 가리킨다.

## Appendix B. Implementation

```python
# Python
import numpy as np

def non_integral_summary(x, t):
    """
    x, t : (N,) single-wafer trace. t holds real timestamps.
    """
    T      = t[-1] - t[0]
    q1, q3 = np.percentile(x, [25, 75])
    xc     = x - x.mean()                            # centered, for zero crossings
    dx     = np.diff(x)

    return {
        'mean':           x.mean(),
        'median':         np.median(x),
        'std':            x.std(ddof=1),
        'range':          x.max() - x.min(),
        'iqr':            q3 - q1,
        'slope':          np.polyfit(t - t[0], x, 1)[0],   # OLS on the raw signal
        'noise':          np.std(dx, ddof=1),
        'max_delta':      np.max(np.abs(dx)),
        'zcr':            np.mean(np.signbit(xc[:-1]) != np.signbit(xc[1:])),
        'peak_time_norm': (t[np.argmax(x)] - t[0]) / T,
        'duration':       T,
    }
```
