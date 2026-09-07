# PLS-RSM (Partial Least Squares Response Surface Methodology) (Korean)
Rev. 2 | Created: 2026-09-06 | Updated: 2026-09-07 00:27 CDT

RSM 은 공정 조건을 2차 다항식으로 근사하고 그 곡면의 정류점에서 최적 조건을 읽는 방법이며, Box 와 Wilson 이 세운 뒤로 공정 최적화의 기본형으로 남아 있다 [[1](#ref-1)]. PLS 는 예측변수를 응답과의 공분산이 큰 방향으로 투영하여, 변수가 서로 얽혀 있거나 관측보다 많을 때에도 회귀와 축약을 한 번에 끝낸다 [[3](#ref-3)]. 이 문서가 다루는 PLS-RSM 은 그 둘을 겹쳐 쓰는 방식, 곧 2차로 확장한 입력 행렬에 OLS 대신 PLS 를 적합하는 구성이다.

먼저 이름부터 밝혀 둔다. PLS-RSM 은 문헌에 확립된 고유 method 이름이 아니다. 공정 최적화 논문들이 두 기법을 함께 쓰면서 붙인 통칭이며, 그래서 이 문서는 그것을 하나의 알고리즘이 아니라 **조합의 설계 선택**으로 다룬다. 어느 자리에서 무엇이 달라지는지가 곧 이 문서의 내용이다.

## 1. Scope

- 다루는 범위: 2차 반응표면을 PLS 로 적합하는 절차, 성분 수 선택, 곡면 복원과 정류점 해석, 그 조합이 실제로 무엇을 사는지.
- 다루지 않는 범위: PLS 알고리즘의 유도, 실험계획 자체의 최적성 이론, 비선형 PLS 일반 [[5](#ref-5)].
- 전제: 응답 $y$ 가 하나 이상 있고, 입력이 연속형 공정 인자이며, 그 인자들이 서로 얽혀 있거나 수가 많은 상황.

## 2. Why The Two Are Combined

RSM 의 표준 절차는 2차 모형을 OLS 로 적합하는 것이다.

$$\hat{y} = b_0 + \mathbf{x}^{\top}\mathbf{b} + \mathbf{x}^{\top}\mathbf{B}\mathbf{x}$$

여기서 $\mathbf{b}$ 는 1차 계수 벡터이고 $\mathbf{B}$ 는 2차항과 교호작용항을 담은 대칭행렬이다. 인자가 $k$ 개면 추정할 계수가 $1 + k + k(k+1)/2$ 개로 늘어, $k = 6$ 이면 28 개, $k = 10$ 이면 66 개가 된다. OLS 는 $(\mathbf{X}^{\top}\mathbf{X})^{-1}$ 을 요구하므로 두 곳에서 무너진다. 관측이 계수보다 적으면 역행렬이 아예 없고, 열이 서로 얽혀 있으면 역행렬은 있으되 계수의 분산이 부풀어 정류점이 자료의 작은 흔들림에 크게 움직인다.

PLS 는 그 역행렬을 요구하지 않는다. 성분 방향 $\mathbf{w}$ 를 아래 목적함수로 하나씩 찾고, 찾은 방향의 score 를 뺀 잔차에서 다음 방향을 다시 찾는다 [[2](#ref-2)].

$$\max_{\lVert \mathbf{w} \rVert = 1} \mathrm{Cov}(\mathbf{X}\mathbf{w},\, y)^2$$

성분 수 $A$ 가 유일한 hyperparameter 이자 유일한 정칙화 강도이며, $A$ 를 키우면 PLS 는 OLS 로 수렴한다. 이것이 이 조합의 성격을 정한다 — PLS-RSM 은 OLS 를 대체하는 것이 아니라, $A$ 를 통해 OLS 와 축약된 해 사이 어디쯤에 설 것인지를 고르는 장치다.

## 3. The Expanded Input Matrix

절차의 첫 단계는 입력을 2차로 넓히는 것이다. 인자 $k$ 개의 행 $\mathbf{x}$ 는 아래 열들로 확장된다.

Table 1. Columns of the expanded matrix

| Block | Columns | Count |
|-------|---------|-------|
| Linear | $x_1, \ldots, x_k$ | $k$ |
| Square | $x_1^2, \ldots, x_k^2$ | $k$ |
| Interaction | $x_i x_j,\ i \lt j$ | $k(k-1)/2$ |

이 확장이 조합의 성격을 두 번째로 정한다. 원래 인자들이 서로 독립이더라도 $x_i$ 와 $x_i^2$ 는 상관되고, 중심화하지 않으면 그 상관이 매우 크다. 즉 **확장 자체가 다중공선성을 만든다.** 그래서 인자를 부호화 (coded) 하여 중심을 0 으로 옮기는 것이 RSM 의 관행이며, PLS 를 쓰더라도 이 관행은 그대로 지킨다.

## 4. Fitting And Choosing The Component Count

확장 행렬에 PLS 를 적합하고 성분 수를 고른다. 성분 수는 이 방법의 유일한 조절 손잡이이므로, 고르는 방식이 곧 모형의 성격이다.

- 교차검증 RMSE 를 최소화하는 $A$ 를 고른다. 실험계획 자료는 행이 적어 fold 하나가 크게 흔들리므로, fold 를 나누는 난수에 따라 고른 $A$ 가 바뀔 수 있다.
- $A$ 의 상한은 확장 행렬의 rank 이며, 그것을 넘어서면 성분이 설명할 분산이 남지 않아 알고리즘이 0 으로 나눈다.
- $A$ 를 rank 까지 올리면 PLS 는 OLS 와 같아진다. 성분 수가 최대로 뽑히는 상황이라면 PLS 를 쓰는 이유가 사라졌다는 뜻이므로, 그 자체가 신호다.

변수의 기여를 읽어야 하면 VIP 를 함께 본다. VIP 는 성분 공간에서 각 변수가 응답 설명에 기여한 몫을 모은 값이며, 관례상 1 을 넘는 변수를 중요한 것으로 본다 [[6](#ref-6)]. 다만 확장 행렬에서는 $x_i$, $x_i^2$, $x_i x_j$ 가 따로 채점되므로, 인자 하나의 중요도를 보려면 그 인자가 관여한 열을 묶어 읽어야 한다.

## 5. Recovering The Surface

PLS 가 돌려주는 것은 성분 공간의 적재가 아니라 확장 변수 공간의 회귀계수이므로, 곡면 복원은 그 계수를 제자리에 꽂는 일이다. 열 이름과 계수를 짝지어 $\mathbf{b}$ 와 $\mathbf{B}$ 를 세운다.

$$b_i = \hat{\beta}_{x_i}, \qquad B_{ii} = \hat{\beta}_{x_i^2}, \qquad B_{ij} = B_{ji} = \tfrac{1}{2}\hat{\beta}_{x_i x_j}$$

교호작용 계수를 절반으로 나누는 것은 $\mathbf{x}^{\top}\mathbf{B}\mathbf{x}$ 가 $B_{ij}$ 와 $B_{ji}$ 를 두 번 세기 때문이다. 이 한 줄을 빠뜨리면 정류점이 조용히 틀린 자리에 놓인다.

정류점은 기울기를 0 으로 두어 얻는다.

$$\mathbf{x}^{\ast} = -\tfrac{1}{2}\mathbf{B}^{-1}\mathbf{b}$$

그 점이 무엇인지는 $\mathbf{B}$ 의 고윳값이 정한다 [[8](#ref-8)]. 모두 음수면 극대, 모두 양수면 극소, 부호가 섞이면 안장점이다. 안장점이 나왔다는 것은 실험 영역 안에 최적이 없다는 뜻이므로, 그때는 정류점을 보고할 것이 아니라 능선 (ridge) 을 따라 영역 밖으로 나가는 방향을 보고해야 한다.

응답이 여럿이면 각 응답의 곡면을 따로 세운 뒤 desirability 로 묶어 하나의 목적함수로 만든다 [[7](#ref-7)]. PLS 는 응답이 여럿인 경우를 한 모형으로도 다루지만, 최적점을 찾는 단계에서는 응답마다 목표가 다르므로 묶는 규칙이 따로 필요하다.

## 6. What The Combination Buys

이 조합이 실제로 무엇을 사는지는 확인해 볼 수 있다. [Appendix B](#appendix-b-python-example) 의 script 는 같은 참 곡면 하나를 두고 두 가지 자료 수집 방식을 200 회씩 복제하여, PLS 와 OLS 가 찾은 정류점이 참 최적에서 얼마나 떨어지는지를 잰다.

- **Central composite design**: 요인배치, 축점, 중심점으로 이루어진 회전가능 설계이며 [[1](#ref-1)], 확장 행렬의 조건수 중앙값이 3.6 이다. 2차 모형을 위한 표준 설계로는 그 밖에 Box-Behnken 이 있으며, 수준을 세 개만 쓴다는 점이 다르다 [[4](#ref-4)].
- **Correlated operating data**: 인자들이 함께 움직이는 관측 자료이며, 같은 확장을 거친 뒤 조건수 중앙값이 337.2 이다.

<img src="pls-rsm_fig/pls-rsm-surfaces.png" width="1000" style="max-width: 100%;" alt="Fig 1">
<p>Fig 1. Fitted surfaces and the spread of the located optimum over 200 replicates</p>

Table 2. Distance from the located optimum to the true optimum

| Design | Method | Median | Q25 | Q75 |
|--------|--------|--------|-----|-----|
| CCD | PLS | 0.107 | 0.072 | 0.164 |
| CCD | OLS | 0.107 | 0.072 | 0.164 |
| Correlated | PLS | 1.265 | 0.818 | 1.790 |
| Correlated | OLS | 1.470 | 1.087 | 1.942 |

읽을 것이 셋 있고, 셋 다 이 조합을 과장하지 않는 쪽이다.

첫째, **설계가 직교하면 PLS 는 아무것도 사지 않는다.** CCD 에서 두 방법의 거리 분포는 소수점 아래까지 같고, PLS 가 더 가까웠던 복제의 비율은 200 회 중 49.5% 로 동전 던지기와 구분되지 않는다. 설계가 좋으면 교차검증이 성분 수를 rank 근처까지 올려 PLS 가 OLS 로 수렴하기 때문이다.

둘째, **얽힌 자료에서는 이득이 있으나 크지 않다.** 거리 중앙값이 1.470 에서 1.265 로 줄고 PLS 가 더 가까운 복제가 57.5% 이다. 방향은 분명하지만, 이것으로 나쁜 자료가 좋은 자료가 되지는 않는다.

셋째, **두 자료의 차이가 두 방법의 차이보다 훨씬 크다.** 거리 중앙값이 0.107 에서 1.265 로 열 배 이상 벌어진다. 즉 PLS-RSM 이 답하는 물음은 "어떤 회귀를 쓸 것인가" 이고, 최적점의 정확도를 정하는 것은 "어떻게 자료를 얻을 것인가" 이다. 앞의 것으로 뒤의 것을 대신할 수 없다.

## 7. Where It Is Used

- **바이오·제약 공정**: 온도, pH, 교반 속도, 영양소 농도처럼 서로 얽힌 인자가 많은 배양·합성 조건 최적화. 인자 수가 많아 완전한 설계를 감당하기 어려운 자리다.
- **화학·재료 공정**: NIR, Raman 같은 분광 자료가 입력으로 들어오는 수율 최적화. 변수가 관측보다 많은 전형적인 자리이며, PLS 가 원래 자란 자리이기도 하다 [[3](#ref-3)].
- **품질 공학**: 이미 쌓인 운전 기록에서 개선 방향을 읽어야 하는 경우. 설계된 자료가 아니므로 section 6 의 두 번째 자료에 해당한다.

## 8. Cautions

- **확장이 공선성을 만든다.** 원래 인자가 직교해도 2차항과 교호작용항은 그렇지 않다. 부호화와 중심화를 거치지 않으면 PLS 를 써도 얻는 것이 줄어든다.
- **정류점은 외삽일 수 있다.** $-\frac{1}{2}\mathbf{B}^{-1}\mathbf{b}$ 는 실험 영역 밖에 놓일 수 있고, 그 자리의 곡면은 자료가 뒷받침하지 않는다. 정류점이 영역 밖이면 값을 보고하지 말고 영역 경계에서 제약 최적화를 다시 푼다.
- **성분 수가 곧 모형이다.** $A$ 를 크게 잡으면 OLS 의 불안정을 그대로 되찾고, 작게 잡으면 곡률을 표현하지 못해 최적점이 중심 쪽으로 끌린다. 교차검증 없이 고정된 $A$ 를 쓰지 않는다.
- **PLS 는 설계를 대신하지 않는다.** section 6 의 세 번째 관찰이 그것이다. 인자를 움직일 수 있다면 설계를 하고, 움직일 수 없을 때 PLS 를 꺼낸다.
- **응답이 여럿이면 최적화 규칙을 따로 정한다.** 곡면을 여러 개 세우는 것과 그중 하나를 고르는 것은 다른 일이다.

## References

<a id="ref-1"></a>
[1] Box, G. E. P. and Wilson, K. B. (1951). [On the Experimental Attainment of Optimum Conditions](https://doi.org/10.1111/j.2517-6161.1951.tb00067.x). *Journal of the Royal Statistical Society: Series B*, 13(1), 1–38.<br>
<a id="ref-2"></a>
[2] Wold, S., Ruhe, A., Wold, H. and Dunn, W. J. (1984). [The Collinearity Problem in Linear Regression. The Partial Least Squares (PLS) Approach to Generalized Inverses](https://doi.org/10.1137/0905052). *SIAM Journal on Scientific and Statistical Computing*, 5(3), 735–743.<br>
<a id="ref-3"></a>
[3] Wold, S., Sjöström, M. and Eriksson, L. (2001). [PLS-regression: a basic tool of chemometrics](https://doi.org/10.1016/S0169-7439(01)00155-1). *Chemometrics and Intelligent Laboratory Systems*, 58(2), 109–130.<br>
<a id="ref-4"></a>
[4] Box, G. E. P. and Behnken, D. W. (1960). [Some New Three Level Designs for the Study of Quantitative Variables](https://doi.org/10.1080/00401706.1960.10489912). *Technometrics*, 2(4), 455–475.<br>
<a id="ref-5"></a>
[5] Wold, S., Kettaneh-Wold, N. and Skagerberg, B. (1989). [Nonlinear PLS modeling](https://doi.org/10.1016/0169-7439(89)80111-X). *Chemometrics and Intelligent Laboratory Systems*, 7(1–2), 53–65.<br>
<a id="ref-6"></a>
[6] Chong, I.-G. and Jun, C.-H. (2005). [Performance of some variable selection methods when multicollinearity is present](https://doi.org/10.1016/j.chemolab.2004.12.011). *Chemometrics and Intelligent Laboratory Systems*, 78(1–2), 103–112.<br>
<a id="ref-7"></a>
[7] Derringer, G. and Suich, R. (1980). [Simultaneous Optimization of Several Response Variables](https://doi.org/10.1080/00224065.1980.11980968). *Journal of Quality Technology*, 12(4), 214–219.<br>
<a id="ref-8"></a>
[8] Myers, R. H., Montgomery, D. C. and Anderson-Cook, C. M. (2016). *Response Surface Methodology: Process and Product Optimization Using Designed Experiments* (4th ed.). [Wiley](https://www.wiley.com/en-us/Response+Surface+Methodology:+Process+and+Product+Optimization+Using+Designed+Experiments,+4th+Edition-p-9781118916018). ISBN 978-1-118-91601-8.

---

## Appendix A. Terminology

- **canonical analysis**: 정류점에서 $\mathbf{B}$ 의 고윳값을 보고 그 점이 극대인지 극소인지 안장점인지 가리는 절차이다.
- **CCD**: central composite design 이며, 요인배치점과 축점과 중심점으로 2차 모형을 추정하는 설계이다.
- **coded variable**: 실험 영역의 중심을 0, 요인배치 수준을 ±1 로 옮긴 무차원 인자이다.
- **condition number**: 행렬의 최대 특이값과 최소 특이값의 비이며, 클수록 계수 추정이 자료의 흔들림에 민감하다.
- **desirability**: 여러 응답을 각각 0 과 1 사이로 옮긴 뒤 하나로 묶어 최적화하는 함수이다.
- **latent variable**: PLS 가 찾은 성분 방향으로 자료를 투영해 얻은 새 변수이며, score 라고도 한다.
- **OLS**: ordinary least squares 이며, 잔차 제곱합을 최소화하는 회귀이다.
- **rank**: 행렬의 선형독립인 열의 수이며, PLS 성분 수의 상한이다.
- **RMSE**: root mean squared error 이며, 잔차 제곱 평균의 제곱근이다.
- **rotatable design**: 중심에서 같은 거리에 있는 모든 점에서 예측 분산이 같은 설계이다.
- **stationary point**: 적합된 곡면의 기울기가 0 이 되는 점이다.
- **VIP**: variable importance in the projection 이며, 성분 공간에서 각 변수가 응답 설명에 기여한 몫이다.

## Appendix B. Python Example

Section 6 의 Fig 1 과 Table 2 를 낸 script 이다. 두 설계를 같은 참 곡면에 대해 복제하고, 확장, 적합, 곡면 복원, 정류점 계산을 차례로 거쳐 거리 분포를 남긴다. 인자 없이 실행하면 문서와 같은 결과가 나오며, `-h` 로 조절할 수 있는 값을 볼 수 있다.

```python
# Applied-Statistics/DOE/PLS-RSM/pls_rsm.py
__author__ = 'yRocket'
__version__ = "0.0.0.2026.9.6"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import enum
import itertools
import pathlib
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from scipy import linalg
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict

__all__ = [
    'Design',
    'central_composite_design',
    'correlated_operating_data',
    'expand_quadratic',
    'true_response',
    'quadratic_from_coefficients',
    'stationary_point',
    'fit_pls_surface',
    'fit_ols_surface',
    'run',
]

FIGSIZE: tuple = (11.0, 5.0)
REFERENCE_WIDTH: float = 11.0    # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 10.0
FIGURE_DPI: int = 300
CONTOUR_GRID: int = 160          # samples per axis in the contour grid
COLORS: list = list(TABLEAU_COLORS.values())

N_FACTORS: int = 3
TRUE_INTERCEPT: float = 78.0
TRUE_LINEAR: np.ndarray = np.array([1.95, -0.55, 3.10])
TRUE_QUADRATIC: np.ndarray = np.array([
    [-2.00, -0.60, 0.45],
    [-0.60, -1.40, -0.35],
    [0.45, -0.35, -2.60],
])


DESIGN_LABELS: dict = {'ccd': 'CCD', 'correlated': 'corr.'}


class Design(enum.StrEnum):
    """Names of the two designs the script contrasts."""

    CCD = enum.auto()
    CORRELATED = enum.auto()


def true_response(x: np.ndarray, noise_sd: float = 0.0, rng: np.random.Generator = None) -> np.ndarray:
    """Evaluate the true quadratic surface at the coded settings in the rows of `x`."""
    if noise_sd < 0.0:
        raise ValueError(f"noise_sd must be non-negative, got {noise_sd}.")
    if noise_sd > 0.0 and rng is None:
        raise ValueError("rng is required when noise_sd is positive; pass a numpy Generator.")
    quadratic = np.einsum('ij,jk,ik->i', x, TRUE_QUADRATIC, x)
    y = TRUE_INTERCEPT + x @ TRUE_LINEAR + quadratic
    if noise_sd > 0.0:
        y = y + rng.normal(loc=0.0, scale=noise_sd, size=y.shape)
    return y


def central_composite_design(n_factors: int = N_FACTORS, n_center: int = 6) -> np.ndarray:
    """Build a rotatable central composite design: a full factorial, star points at alpha, and centers."""
    if n_factors < 2:
        raise ValueError(f"a response surface needs at least two factors, got {n_factors}.")
    if n_center < 1:
        raise ValueError(f"n_center must be at least one, got {n_center}.")
    factorial = np.array(list(itertools.product([-1.0, 1.0], repeat=n_factors)))
    alpha = float(len(factorial)) ** 0.25          # the value that makes the design rotatable
    star = np.zeros((2 * n_factors, n_factors))
    for i in range(n_factors):
        star[2 * i, i] = alpha
        star[2 * i + 1, i] = -alpha
    center = np.zeros((n_center, n_factors))
    return np.vstack([factorial, star, center])


def correlated_operating_data(n_runs: int, correlation: float, rng: np.random.Generator) -> np.ndarray:
    """Draw settings the way an unplanned process record supplies them, with the factors tied together."""
    if not 0.0 <= correlation < 1.0:
        raise ValueError(f"correlation must be in [0, 1), got {correlation}.")
    if n_runs < 1:
        raise ValueError(f"n_runs must be positive, got {n_runs}.")
    driver = rng.normal(size=n_runs)
    independent = rng.normal(size=(n_runs, N_FACTORS))
    x = correlation * driver[:, None] + (1.0 - correlation) * independent
    return x / np.std(x, axis=0, ddof=1)


def expand_quadratic(x: np.ndarray) -> tuple:
    """Append squares and two-factor products to `x`, returning the expanded matrix and its column names."""
    n_factors = x.shape[1]
    names = [f"x{i + 1}" for i in range(n_factors)]
    columns = [x]
    columns.append(x ** 2)
    names += [f"x{i + 1}^2" for i in range(n_factors)]
    pairs = list(itertools.combinations(range(n_factors), 2))
    if pairs:
        columns.append(np.column_stack([x[:, i] * x[:, j] for i, j in pairs]))
        names += [f"x{i + 1}x{j + 1}" for i, j in pairs]
    return np.hstack(columns), names


def quadratic_from_coefficients(coefficients: np.ndarray, names: list, n_factors: int) -> tuple:
    """Split expanded-space coefficients into the linear vector b and the symmetric matrix B of the surface."""
    if len(coefficients) != len(names):
        raise ValueError(f"got {len(coefficients)} coefficients for {len(names)} names.")
    index = {name: position for position, name in enumerate(names)}
    b = np.array([coefficients[index[f"x{i + 1}"]] for i in range(n_factors)])
    matrix = np.zeros((n_factors, n_factors))
    for i in range(n_factors):
        matrix[i, i] = coefficients[index[f"x{i + 1}^2"]]
    for i, j in itertools.combinations(range(n_factors), 2):
        half = 0.5 * coefficients[index[f"x{i + 1}x{j + 1}"]]
        matrix[i, j] = half
        matrix[j, i] = half
    return b, matrix


def stationary_point(b: np.ndarray, matrix: np.ndarray) -> tuple:
    """Solve for the stationary point of the fitted surface and return it with the eigenvalues that classify it."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.min(np.abs(eigenvalues)) < np.finfo(float).eps * max(1.0, np.max(np.abs(eigenvalues))):
        raise linalg.LinAlgError("the quadratic matrix is singular; the surface has no isolated stationary point.")
    return -0.5 * np.linalg.solve(matrix, b), eigenvalues


def fit_pls_surface(x_expanded: np.ndarray, y: np.ndarray, max_components: int, n_splits: int) -> tuple:
    """Choose the component count by cross-validated RMSE, refit on all rows, and return coefficients and the count."""
    # The rank caps the component count: past it a component carries no variance and PLS divides by zero.
    # A fold holds out rows, so the cap uses the rank of the smallest training set rather than of the whole.
    folds = KFold(n_splits=min(n_splits, x_expanded.shape[0]), shuffle=True, random_state=0)
    held_out = int(np.ceil(x_expanded.shape[0] / folds.get_n_splits()))
    rank = int(np.linalg.matrix_rank(x_expanded - x_expanded.mean(axis=0)))
    upper = min(max_components, rank, x_expanded.shape[0] - held_out - 1)
    if upper < 1:
        raise ValueError(f"no components are available for a {x_expanded.shape} design matrix of rank {rank}.")
    errors = []
    for n_components in range(1, upper + 1):
        with warnings.catch_warnings():
            # A fold can exhaust y before the cap, which leaves NaN predictions. That count and every larger
            # one are unusable, so the search stops there and says so instead of ranking a NaN error.
            warnings.simplefilter('ignore', category=RuntimeWarning)
            predicted = cross_val_predict(PLSRegression(n_components=n_components), x_expanded, y, cv=folds)
        if not np.all(np.isfinite(predicted)):
            print(f"PLS exhausted the response at {n_components} components; searching up to {n_components - 1}.",
                  file=sys.stderr)
            break
        errors.append(float(np.sqrt(np.mean((y - predicted.ravel()) ** 2))))
    if not errors:
        raise ValueError(f"no usable component count for a {x_expanded.shape} design matrix.")
    best = int(np.argmin(errors)) + 1
    model = PLSRegression(n_components=best).fit(x_expanded, y)
    return model.coef_.ravel(), best, errors


def fit_ols_surface(x_expanded: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve the normal equations for the same expanded matrix, which is what PLS is being compared against."""
    design = np.column_stack([np.ones(len(x_expanded)), x_expanded])
    gram = design.T @ design
    # The solve is deliberately not replaced by a pseudo-inverse: a singular gram matrix is the finding,
    # not something to paper over, and the caller reports it.
    coefficients = linalg.solve(gram, design.T @ y, assume_a='pos')
    return coefficients[1:]


def evaluate_design(x: np.ndarray, y: np.ndarray, max_components: int, n_splits: int) -> dict:
    """Fit both surfaces to one design and collect what the document compares."""
    x_expanded, names = expand_quadratic(x=x)
    condition = float(np.linalg.cond(np.column_stack([np.ones(len(x_expanded)), x_expanded])))
    pls_coefficients, n_components, cv_errors = fit_pls_surface(
        x_expanded=x_expanded, y=y, max_components=max_components, n_splits=n_splits)
    pls_b, pls_matrix = quadratic_from_coefficients(
        coefficients=pls_coefficients, names=names, n_factors=x.shape[1])
    pls_optimum, pls_eigenvalues = stationary_point(b=pls_b, matrix=pls_matrix)
    result = {
        'n_runs': len(x),
        'condition_number': condition,
        'n_components': n_components,
        'cv_rmse': min(cv_errors),
        'pls_optimum': pls_optimum,
        'pls_eigenvalues': pls_eigenvalues,
        'pls_matrix': pls_matrix,
        'pls_b': pls_b,
    }
    try:
        ols_coefficients = fit_ols_surface(x_expanded=x_expanded, y=y)
    except (linalg.LinAlgError, np.linalg.LinAlgError) as error:
        print(f"OLS failed on this design: {error}", file=sys.stderr)
        result['ols_optimum'] = None
        result['ols_coefficient_norm'] = float('nan')
        return result
    ols_b, ols_matrix = quadratic_from_coefficients(
        coefficients=ols_coefficients, names=names, n_factors=x.shape[1])
    result['ols_coefficient_norm'] = float(np.linalg.norm(ols_coefficients))
    try:
        result['ols_optimum'] = stationary_point(b=ols_b, matrix=ols_matrix)[0]
    except (linalg.LinAlgError, np.linalg.LinAlgError) as error:
        print(f"OLS surface has no isolated stationary point: {error}", file=sys.stderr)
        result['ols_optimum'] = None
    return result


def plot_results(results: dict, distances: pd.DataFrame, true_optimum: np.ndarray,
                 output_path: pathlib.Path) -> None:
    """Draw the fitted surface of each design and the spread of the located optimum over the replicates."""
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    matplotlib.rcParams.update({'font.size': font_size})
    grid = np.linspace(-2.0, 2.0, CONTOUR_GRID)
    mesh_1, mesh_2 = np.meshgrid(grid, grid)
    figure, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    for axis, (design, result) in zip(axes[:2], results.items()):
        held = result['pls_optimum'][2]
        flat = np.column_stack([mesh_1.ravel(), mesh_2.ravel(), np.full(mesh_1.size, held)])
        surface = flat @ result['pls_b'] + np.einsum('ij,jk,ik->i', flat, result['pls_matrix'], flat)
        contour = axis.contourf(mesh_1, mesh_2, surface.reshape(mesh_1.shape), levels=18, cmap='viridis')
        figure.colorbar(contour, ax=axis, shrink=0.82)
        axis.plot(result['pls_optimum'][0], result['pls_optimum'][1], marker='o', markersize=7,
                  color=COLORS[2], linestyle='none', label='PLS')
        if result['ols_optimum'] is not None:
            # An open marker, drawn last, so that a PLS point underneath it still shows when the two coincide.
            axis.plot(result['ols_optimum'][0], result['ols_optimum'][1], marker='s', markersize=11,
                      markerfacecolor='none', markeredgewidth=2.0, color=COLORS[3], linestyle='none', label='OLS')
        # Drawn last so it stays readable where all three points coincide, as they do on the orthogonal design.
        axis.plot(true_optimum[0], true_optimum[1], marker='+', markersize=13, markeredgewidth=2.2,
                  color='white', linestyle='none', label='true')
        axis.set_xlabel('$x_1$')
        axis.set_ylabel('$x_2$')
        axis.set_xlim(grid[0], grid[-1])
        axis.set_ylim(grid[0], grid[-1])
        axis.legend(loc='lower left', framealpha=0.85, fontsize=font_size * 0.85)
    groups, positions, tick_labels = [], [], []
    for position, (design, method) in enumerate(itertools.product([str(d) for d in Design], ['pls', 'ols'])):
        selected = distances.loc[(distances['design'] == design) & (distances['method'] == method), 'distance']
        groups.append(selected.dropna().to_numpy())
        positions.append(position + 1)
        tick_labels.append(f"{DESIGN_LABELS[design]}\n{method.upper()}")
    box = axes[2].boxplot(groups, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
    for patch, color in zip(box['boxes'], [COLORS[2], COLORS[3], COLORS[2], COLORS[3]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for median in box['medians']:
        median.set_color('black')
    axes[2].set_yscale('log')
    axes[2].set_xticks(positions, tick_labels, fontsize=font_size * 0.9)
    axes[2].set_ylabel('distance to the true optimum')
    axes[2].grid(axis='y', alpha=0.3)
    labels = ['(a) central composite design', '(b) correlated operating data',
              f"(c) over {distances['replicate'].nunique()} replicates"]
    for axis, label in zip(axes, labels):
        position = axis.get_position()
        figure.text(position.x0 + position.width / 2.0, 0.02, label, ha='center', va='bottom')
    figure.subplots_adjust(bottom=0.24, wspace=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)


def run(output_folder: pathlib.Path, n_runs: int, correlation: float, noise_sd: float, max_components: int,
        n_splits: int, seed: int, n_replicates: int) -> pd.DataFrame:
    """Run both designs over `n_replicates` draws and write the response table, the distances and the figure.

    Returns a pd.DataFrame indexed by (design, method), with columns median_distance, q25, q75,
    median_condition, median_components and n_failed.
    """
    if n_replicates < 1:
        raise ValueError(f"n_replicates must be positive, got {n_replicates}.")
    true_optimum = -0.5 * np.linalg.solve(TRUE_QUADRATIC, TRUE_LINEAR)
    ccd = central_composite_design()
    records, first_results, first_responses = [], {}, []
    for replicate in range(n_replicates):
        rng = np.random.default_rng(seed + replicate)
        designs = {
            Design.CCD: ccd,
            Design.CORRELATED: correlated_operating_data(n_runs=n_runs, correlation=correlation, rng=rng),
        }
        for design, x in designs.items():
            y = true_response(x=x, noise_sd=noise_sd, rng=rng)
            result = evaluate_design(x=x, y=y, max_components=max_components, n_splits=n_splits)
            for method, optimum in (('pls', result['pls_optimum']), ('ols', result['ols_optimum'])):
                records.append({
                    'replicate': replicate,
                    'design': str(design),
                    'method': method,
                    'condition_number': result['condition_number'],
                    'n_components': result['n_components'] if method == 'pls' else np.nan,
                    'distance': float('nan') if optimum is None else float(np.linalg.norm(optimum - true_optimum)),
                })
            if replicate == 0:
                first_results[design] = result
                frame = pd.DataFrame(x, columns=[f"x{i + 1}" for i in range(x.shape[1])])
                frame.insert(0, 'design', str(design))
                frame['y'] = y
                first_responses.append(frame)
    distances = pd.DataFrame.from_records(records)
    output_folder.mkdir(parents=True, exist_ok=True)
    pd.concat(first_responses, ignore_index=True).to_csv(output_folder / 'responses.csv', index=False)
    distances.to_csv(output_folder / 'distances.csv', index=False)
    grouped = distances.groupby(['design', 'method'])
    summary = pd.DataFrame({
        'median_distance': grouped['distance'].median(),
        'q25': grouped['distance'].quantile(0.25),
        'q75': grouped['distance'].quantile(0.75),
        'median_condition': grouped['condition_number'].median(),
        'median_components': grouped['n_components'].median(),
        'n_failed': grouped['distance'].apply(lambda column: int(column.isna().sum())),
    })
    summary.to_csv(output_folder / 'summary.csv')
    plot_results(results=first_results, distances=distances, true_optimum=true_optimum,
                 output_path=output_folder / 'pls-rsm_fig' / 'pls-rsm-surfaces.png')
    print(f"true optimum: {np.array2string(true_optimum, precision=3)}")
    print(summary.to_string(float_format=lambda value: f"{value:.4g}"))
    return summary


def parse_args() -> argparse.Namespace:
    """Parse the command line, validate the combinations, and return the namespace."""
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f"{pathlib.Path(__file__).name} {__version__}\n"
                    "Fit a second-order response surface through PLS and locate its stationary point.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=f"{pathlib.Path(__file__).name} {__version__}")
    parser.add_argument('--output-folder', type=pathlib.Path, default=pathlib.Path(__file__).parent,
                        help='root folder for the response table, the summary and the figure')
    parser.add_argument('--n-runs', type=int, default=20, help='runs in the correlated operating data')
    parser.add_argument('--correlation', type=float, default=0.85,
                        help='how strongly the factors of the correlated design move together, in [0, 1)')
    parser.add_argument('--noise-sd', type=float, default=0.35, help='standard deviation of the response noise')
    parser.add_argument('--max-components', type=int, default=9, help='largest PLS component count to consider')
    parser.add_argument('--n-splits', type=int, default=5, help='folds used to choose the component count')
    parser.add_argument('--n-replicates', type=int, default=200, help='independent draws the comparison is made over')
    parser.add_argument('--seed', type=int, default=20260906, help='seed for the response noise and the settings')
    args = parser.parse_args()
    if not 0.0 <= args.correlation < 1.0:
        parser.error(f"--correlation must be in [0, 1), got {args.correlation}")
    if args.noise_sd < 0.0:
        parser.error(f"--noise-sd must be non-negative, got {args.noise_sd}")
    if args.n_runs < N_FACTORS + 1:
        parser.error(f"--n-runs must exceed the factor count, got {args.n_runs}")
    if args.max_components < 1:
        parser.error(f"--max-components must be positive, got {args.max_components}")
    if args.n_splits < 2:
        parser.error(f"--n-splits must be at least two, got {args.n_splits}")
    if args.n_replicates < 1:
        parser.error(f"--n-replicates must be positive, got {args.n_replicates}")
    if args.output_folder.exists() and not args.output_folder.is_dir():
        parser.error(f"--output-folder is not a folder: {args.output_folder}")
    return args
```
