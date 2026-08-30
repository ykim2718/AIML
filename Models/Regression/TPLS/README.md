# TPLS (Temporal Partial Least Squares)
Rev. 0 | Created: 2026-08-30 | Updated: 2026-08-30 15:56 CDT

PLS 는 예측변수 $X$ 를 응답 $y$ 와의 공분산이 큰 방향으로 투영하여, 변수 수가 관측 수를 넘는 자료에서도 회귀와 축약을 한 번에 끝내는 방법이다 [[1](#ref-1)]. 이 문서가 다루는 temporal PLS 는 그 자료에 시간이 개입할 때 무엇이 달라지는지를 정리한 것이다. 시간은 세 가지 서로 다른 자리로 들어오며, 어느 자리인지에 따라 쓰는 방법도 검증 방식도 갈리므로, 방법을 고르기 전에 그 자리를 먼저 정해야 한다.

## 1. Scope

- 다루는 범위: 시간 구조를 가진 자료에 대한 PLS 계열 방법, 성분 수 선택과 검증, 잠재공간 기반 monitoring.
- 다루지 않는 범위: 정적 PLS 알고리즘의 유도, 비선형 회귀 일반, PLS 가 아닌 시계열 예측 model.
- 전제: 응답 $y$ 가 존재하고, 변수 수 $p$ 가 관측 수 $n$ 에 비해 크거나 비슷한 상황.

## 2. Where Time Enters

시간이 개입하는 자리는 세 곳이고, 그것이 곧 temporal PLS 의 분류축이다. Fig 1 은 그 세 축과 각 축에 속하는 방법이며, Table 1 은 축마다 무엇이 달라지는지를 적은 것이다.

```text
Temporal PLS
|
+-- Within a sample (lag structure)
|   +-- Lagged augmentation ............... DPLS
|   +-- Dynamic inner relation ............ DiPLS
|
+-- Across a sample (trajectory as a mode)
|   +-- Batch-wise unfolding .............. MPLS
|   +-- Variable-wise unfolding ........... MPLS (variable-wise)
|   +-- Trilinear decomposition ........... N-PLS
|
+-- Across arrival (model adaptation)
    +-- Recursive update .................. RPLS
    +-- Moving window ..................... moving window PLS
```

Fig 1. Three places time enters a PLS model

Table 1. What each axis changes

| Axis | One row of `X` | Time in the model | Typical data |
|------|----------------|-------------------|--------------|
| Within a sample | One time point, with its own past appended | Lag order $s$, a model parameter | Continuous process, soft sensor |
| Across a sample | One whole trajectory | Time index, a column position | Batch process, equipment trace |
| Across arrival | One time point or one batch | Arrival order, a fitting schedule | Drifting process, long campaign |

세 축은 배타적이지 않다. Batch 자료를 MPLS 로 적합해 두고 batch 가 쌓일 때마다 RPLS 로 갱신하는 구성처럼, 서로 다른 축의 방법을 겹쳐 쓰는 것이 오히려 보통이다.

## 3. Static PLS As The Baseline

세 축 모두 정적 PLS 를 출발점으로 삼으므로, 비교의 기준이 되는 형태를 먼저 고정한다. PLS 는 아래 목적함수로 성분 방향 $w$ 를 하나씩 찾고, 찾은 방향으로 얻은 score $t = Xw$ 를 뺀 잔차에서 다음 방향을 다시 찾는다.

$$\max_{\lVert w \rVert = 1} \mathrm{Cov}(Xw,\, y)^2$$

PCA 가 $\mathrm{Var}(Xw)$ 를 최대화하는 것과 달리 PLS 는 응답과의 공분산을 최대화하므로, 분산은 크지만 응답과 무관한 방향에 성분을 쓰지 않는다. 성분 수 $A$ 는 유일한 hyperparameter 이자 유일한 정칙화 강도이며, $A$ 를 키우면 PLS 는 OLS 로 수렴한다.

시간이 개입하면 이 기준이 두 곳에서 깨진다. 첫째, 행이 서로 독립이라는 가정이 깨지므로 무작위 분할 교차검증이 성능을 낙관 편향시킨다. 둘째, 자료를 낳는 공정이 바뀌면 한 번 적합한 $w$ 가 계속 옳다는 보장이 사라진다. 앞의 것은 7 장에서, 뒤의 것은 6 장에서 다룬다.

## 4. Lag Structure Within A Sample

관측이 시점마다 하나씩 도착하고 현재 응답이 과거 입력에 반응하는 경우이다. 응답이 입력에 늦게 반응하는 공정에서 정적 PLS 는 그 지연을 표현할 수 없어 잔차에 자기상관을 남긴다.

### 4.1 Lagged Augmentation

가장 단순한 처방은 자료 행렬 자체를 늘리는 것이다. 시점 $t$ 의 행에 과거 $s$ 개 시점의 입력을 붙여 $[x_t,\, x_{t-1},\, \ldots,\, x_{t-s}]$ 를 한 행으로 삼고, 늘어난 행렬에 정적 PLS 를 그대로 적용한다 [[2](#ref-2)]. 동적 구조가 model 이 아니라 자료 쪽에 들어가므로 구현이 가장 간단하다.

```python
# Python
import numpy as np

def lag_matrix(X: np.ndarray, s: int) -> np.ndarray:
    """Append s past time points to each row; drop the first s rows."""
    if s < 0:
        raise ValueError(f"s must be non-negative, got {s}")
    n = X.shape[0]
    if n <= s:
        raise ValueError(f"need more than {s} rows, got {n}")
    return np.hstack([X[s - k: n - k] for k in range(s + 1)])
```

대가는 차원이다. 변수 $p$ 개에 lag $s$ 를 주면 열이 $p(s+1)$ 개로 늘고, 그 열들은 서로 강하게 상관되어 있다. 성분 수를 제대로 통제하지 못하면 늘어난 열이 그대로 과적합으로 돌아온다.

### 4.2 Dynamic Inner Relation

DiPLS 는 자료 대신 model 을 고친다. 외부 model 은 정적 PLS 와 같이 두되, score 와 응답을 잇는 내부 관계를 $u_t = \beta_0 t_t + \beta_1 t_{t-1} + \cdots + \beta_s t_{t-s}$ 처럼 동적으로 두고, 가중치 $w$ 와 동적 계수 $\beta$ 를 함께 최적화한다 [[3](#ref-3)]. 잠재변수가 애초에 동적으로 예측력이 큰 방향으로 뽑히므로, 같은 성분 수에서 lagged augmentation 보다 잔차 자기상관이 작다.

두 방법의 선택 기준은 단순하다. 지연이 몇 시점인지 대략 알고 $p$ 가 작으면 lagged augmentation 으로 충분하고, 지연이 불분명하거나 $p$ 가 커서 열 확장을 감당하기 어려우면 DiPLS 가 낫다.

## 5. Trajectory As A Mode

Batch 하나가 표본 하나이고, 그 batch 안에서 여러 변수가 시간을 따라 궤적을 그리는 경우이다. 자료는 (batch $I$) $\times$ (variable $J$) $\times$ (time $K$) 의 3-way 배열이 되고, 응답은 batch 당 하나의 품질값이다. PLS 는 2-way 방법이므로 이 배열을 어떻게 다룰지가 곧 방법의 이름이 된다.

### 5.1 Batch-Wise Unfolding

3-way 배열을 $I \times JK$ 로 펴서, batch 하나를 한 행으로 만든 뒤 정적 PLS 를 적용한다. 이것이 MPLS 이며, batch 공정 monitoring 의 표준 형태이다 [[4](#ref-4)]. 각 열이 특정 변수의 특정 시점이므로 성분의 loading 을 시간축에 그대로 그릴 수 있고, 어느 구간이 품질을 좌우했는지가 그림 하나로 읽힌다.

전제가 둘 있다. Batch 마다 궤적의 길이가 같아야 하고, 시점이 서로 맞추어져 있어야 한다. 길이가 다르면 dynamic time warping 이나 지시변수 기반 정렬로 먼저 맞춘다. 열이 $JK$ 개로 늘어나므로 $p \gg n$ 이 극단으로 가지만, 인접 시점끼리 강하게 상관되어 유효 차원은 훨씬 작다.

### 5.2 Variable-Wise Unfolding

같은 MPLS 틀에서 배열을 $IK \times J$ 로 펴면, 시점 하나가 한 행이 된다 [[5](#ref-5)]. 행 수가 늘고 열 수가 $J$ 로 유지되므로 $p \gg n$ 문제가 사라지고 길이가 다른 batch 도 그대로 쓸 수 있다. 대신 batch 하나가 표본이라는 성질이 사라지므로, batch 단위 응답을 예측하는 자리에는 맞지 않고 시점 단위의 이상 감지에 쓴다.

### 5.3 Trilinear Decomposition

N-PLS 는 배열을 펴지 않고 3-way 구조를 유지한 채 각 mode 에 대한 loading 을 동시에 찾는다 [[6](#ref-6)]. 시간 mode 의 loading 이 하나의 매끄러운 곡선으로 나오므로 해석이 간명하고, 추정할 parameter 가 unfolding 보다 훨씬 적어 batch 수가 적을 때 안정적이다. 대신 궤적이 각 mode 의 곱으로 분해된다는 삼선형 가정을 자료가 만족해야 하며, 공정 조건마다 궤적의 모양 자체가 달라지면 이 가정이 깨져 MPLS 보다 나빠진다.

## 6. Adaptation Across Arrival

자료가 계속 도착하고 공정이 서서히 변하는 경우이다. 앞의 두 축이 model 안에 시간을 넣는 문제였다면, 이 축은 model 자체를 언제 어떻게 갱신할지의 문제이다.

### 6.1 Recursive PLS

RPLS 는 새 자료가 도착할 때마다 전체 이력을 다시 적합하지 않고 기존 model 을 갱신한다 [[7](#ref-7)]. 이전 model 의 loading 과 score 를 가중된 가상 관측처럼 다루어 새 block 과 함께 다시 분해하므로, 이력 전체를 보관하지 않고도 이력을 반영한 model 이 유지된다. 망각인자를 두면 오래된 자료의 가중치가 지수적으로 줄어들어, 공정이 변할 때 model 이 따라간다.

### 6.2 Moving Window

최근 $N$ 개 관측만 남기고 그 window 로 매번 다시 적합하는 방법이다. 구현이 단순하고 급격한 변화에 가장 빨리 반응하지만, window 밖으로 나간 조건은 완전히 잊으므로 드물게 재현되는 조건에서 성능이 떨어진다.

### 6.3 What Adaptation Costs

갱신에는 대가가 따른다. Model 이 자료를 따라가면 잔차도 함께 줄어들므로, 정말로 이상인 변화까지 정상으로 학습해 버릴 수 있다. Monitoring 을 겸하는 model 이라면 8 장의 통계량이 관리 한계 안에 있을 때만 갱신하도록 조건을 걸어, 이상 구간의 자료가 model 에 흡수되지 않게 한다.

## 7. Component Selection And Validation

$A$ 를 고르는 절차가 곧 model 의 복잡도를 정하는 절차이다. 시간이 개입한 자료에서는 무작위 $k$-fold 를 쓰면 안 된다. 인접 시점이 강하게 상관되어 있어, 무작위 분할은 학습 구간과 사실상 같은 자료로 검증하게 만들고 성능을 낙관 편향시킨다.

Table 2. Validation split by where time enters

| Axis | Split unit | Rule |
|------|------------|------|
| Within a sample | Contiguous time block | Validation block strictly after the training block, with a gap of at least the lag order |
| Across a sample | Whole batch | Batches of one lot never split across folds |
| Across arrival | Arrival order | Fit on the past, score on the next block, then roll forward |

어느 축이든 원칙은 하나다. 검증 자료는 학습 시점 이후에 도착한 것이어야 한다. 전처리의 중심과 척도, lag 길이, 궤적 정렬 기준도 모두 학습 구간만으로 정한 뒤 검증 구간에 적용한다. 이 값들을 전체 자료로 먼저 계산하면 검증 구간의 정보가 학습으로 새어 들어간다.

## 8. Monitoring In The Latent Space

적합한 PLS model 은 예측만 내놓는 것이 아니라, 새 관측이 학습 때 본 것과 같은 자료인지도 함께 판정한다. 두 통계량이 쓰인다. Hotelling $T^2$ 는 score 가 잠재공간 안에서 중심에서 얼마나 떨어졌는지를, SPE 는 관측이 잠재공간에서 얼마나 벗어났는지를 잰다. 앞의 것은 아는 방향에서의 이상, 뒤의 것은 model 이 설명하지 못하는 새로운 이상에 반응한다.

두 통계량 중 하나가 한계를 넘으면 기여도 분해로 어느 변수가 그 값을 밀어 올렸는지를 본다. MPLS 라면 기여도가 (변수, 시점) 단위로 나오므로 batch 의 어느 구간에서 무엇이 어긋났는지가 바로 지목된다. 이것이 predictor 로서의 PLS 와 monitoring 도구로서의 PLS 가 하나의 적합으로 함께 얻어지는 지점이다.

## 9. Choosing Among Them

Table 3. Method by question

| Question | Method | Why |
|----------|--------|-----|
| Response lags the input, dimension is modest | Lagged augmentation | Simplest form, no new algorithm |
| Response lags the input, lag unknown or $p$ large | DiPLS | Dynamics in the model, no column blow-up |
| Batch quality from an aligned trajectory | MPLS | Loading readable on the time axis |
| Batch quality, few batches or ragged length | N-PLS or variable-wise unfolding | Fewer parameters, alignment relaxed |
| Process drifts over a long campaign | RPLS with a forgetting factor | History kept without storing it |
| Abrupt change, recent behavior only | Moving window PLS | Fastest response, oldest conditions dropped |

선택의 순서는 축이 먼저이고 방법이 나중이다. 시간이 표본 안에 있는지, 표본 자체가 궤적인지, 아니면 표본이 도착하는 순서에 있는지를 먼저 정하면 후보는 두세 개로 줄어든다. 잔차에 자기상관이 남았는지, 궤적의 길이가 고른지, 공정이 변하는지 — 세 가지 확인이 그 판단에 필요한 전부이다.

## References

<a id="ref-1"></a>[1] Wold, S., Sjöström, M. and Eriksson, L., "[PLS-regression: a basic tool of chemometrics](https://doi.org/10.1016/S0169-7439(01)00155-1)", Chemometrics and Intelligent Laboratory Systems, 58(2), 109-130, 2001.

<a id="ref-2"></a>[2] Kaspar, M. H. and Ray, W. H., "[Dynamic PLS modelling for process control](https://doi.org/10.1016/0009-2509(93)85001-6)", Chemical Engineering Science, 48(20), 3447-3461, 1993.

<a id="ref-3"></a>[3] Dong, Y. and Qin, S. J., "[Dynamic-Inner Partial Least Squares for Dynamic Data Modeling](https://doi.org/10.1016/j.ifacol.2015.08.167)", IFAC-PapersOnLine, 48(8), 117-122, 2015.

<a id="ref-4"></a>[4] Nomikos, P. and MacGregor, J. F., "[Multi-way partial least squares in monitoring batch processes](https://doi.org/10.1016/0169-7439(95)00043-7)", Chemometrics and Intelligent Laboratory Systems, 30(1), 97-108, 1995.

<a id="ref-5"></a>[5] Wold, S., Kettaneh, N., Fridén, H. and Holmberg, A., "[Modelling and diagnostics of batch processes and analogous kinetic experiments](https://doi.org/10.1016/S0169-7439(98)00162-2)", Chemometrics and Intelligent Laboratory Systems, 44(1-2), 331-340, 1998.

<a id="ref-6"></a>[6] Bro, R., "[Multiway calibration. Multilinear PLS](https://doi.org/10.1002/%28SICI%291099-128X%28199601%2910%3A1%3C47%3A%3AAID-CEM400%3E3.0.CO%3B2-C)", Journal of Chemometrics, 10(1), 47-61, 1996.

<a id="ref-7"></a>[7] Qin, S. J., "[Recursive PLS algorithms for adaptive data modeling](https://doi.org/10.1016/S0098-1354(97)00262-7)", Computers & Chemical Engineering, 22(4-5), 503-514, 1998.

---

## Appendix A. Terminology

- **batch process**: 원료를 넣고 정해진 순서를 거쳐 산물을 꺼내는, 시작과 끝이 있는 공정. 한 번의 실행이 batch 하나이다.
- **DiPLS**: Dynamic-inner PLS. 내부 관계를 동적으로 두어 score 의 과거 값이 현재 응답을 설명하게 한 PLS.
- **DPLS**: Dynamic PLS. 과거 시점의 입력을 열로 붙인 자료 행렬에 PLS 를 적용하는 방법.
- **dynamic time warping**: 두 궤적의 시간축을 늘이고 줄여 대응하는 시점끼리 맞추는 정렬 방법.
- **forgetting factor**: 오래된 자료의 가중치를 지수적으로 줄이는 계수. 망각인자.
- **Hotelling $T^2$**: Score 가 잠재공간의 중심에서 떨어진 정도를 재는 통계량.
- **loading**: 성분 하나가 각 변수에 부여한 가중치. 성분의 의미를 읽는 자리.
- **MPLS**: Multi-way PLS. 3-way 배열을 batch 하나가 한 행이 되도록 펴서 적용하는 PLS.
- **N-PLS**: Multilinear PLS. 배열을 펴지 않고 각 mode 의 loading 을 동시에 찾는 PLS.
- **OLS**: Ordinary Least Squares. 잔차 제곱합을 최소화하는 회귀.
- **PCA**: Principal Component Analysis. 분산이 큰 방향으로 투영하는 비지도 축약.
- **PLS**: Partial Least Squares. 응답과의 공분산이 큰 방향으로 투영하는 지도 학습형 축약.
- **RPLS**: Recursive PLS. 새 자료로 기존 model 을 갱신하는 PLS.
- **score**: 관측을 성분 방향으로 투영한 값. 잠재공간에서의 좌표.
- **soft sensor**: 측정하기 어려운 값을 측정하기 쉬운 값들로부터 추정하는 model.
- **SPE**: Squared Prediction Error. 관측이 잠재공간에서 벗어난 정도를 재는 잔차 통계량.
