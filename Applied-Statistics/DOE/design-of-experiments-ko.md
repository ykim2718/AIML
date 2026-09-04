# Design of Experiments (Korean)
Rev. 1 | Created: 2026-09-04 | Updated: 2026-09-04 19:26 UTC

> MathWorks Statistics and Machine Learning Toolbox 의 design of experiments 장이 다루는 design
> 계열들 [[1](#ref-1)] 을 같은 구성으로 옮기고, 모든 구성법을 numpy 와 scipy 위에서 Python 으로
> 구현한 기록.

## 1. Design of Experiments

수동적으로 모은 자료에는 어려움이 따른다. 반응에서 저절로 일어난 같은 변화를 여러 factor 가 동시에
설명할 수 있고, 그런 자료에 맞춘 model 은 그것들을 갈라내지 못한다. 계획된 실험은 factor 값을 의도적으로
정해 그 어려움을 없앤다. factor 들이 서로 독립으로 움직이게 되므로 반응에 대한 각각의 효과를 따로
추정할 수 있다.

측정한 $n$ 개의 반응을 모은 vector 를 $\mathbf{y}$, factor 설정에서 만든 model matrix 를
$\mathbf{X}$, 계수를 $\boldsymbol{\beta}$ 라 하자. $\sigma^2$ 이 정해지면 최소제곱추정량과 그
공분산은 design 만으로 정해진다.

$$\hat{\boldsymbol{\beta}} = \left( \mathbf{X}^{\top}\mathbf{X} \right)^{-1} \mathbf{X}^{\top}\mathbf{y}, \qquad \mathrm{Cov}\left[ \hat{\boldsymbol{\beta}} \right] = \sigma^{2} \left( \mathbf{X}^{\top}\mathbf{X} \right)^{-1}$$

그러므로 실험을 계획한다는 것은 run 개수라는 예산 아래에서 이 공분산이 작아지도록 $\mathbf{X}$ 의
행을 고르는 일이다. 아래의 계열들은 어떤 model 을 겨냥하는지, 그리고 그 예산을 어떻게 쓰는지에서
서로 다르다.

Table 1. Design families and the model each one serves.

| Family | Model aimed at | Typical use |
|---|---|---|
| Full factorial | 모든 효과와 교호작용 | factor 가 적고 모든 조합을 감당할 수 있을 때 |
| Fractional factorial | 주효과와 일부 교호작용 | 많은 factor 의 screening |
| Response surface | 완전 이차 model | 알려진 작업점 근처에서의 최적화 |
| D-optimal | 명시한 임의의 model | 불규칙한 예산, 제약, covariate |

이 문서의 Python block 은 순서대로 읽도록 되어 있다. 각 block 은 하나의 header 위에서 앞선 block 이
정의한 이름을 쓴다. 그 뒤에 오는 block 은 출력이다.

```python
# Python
import itertools
from typing import Sequence

import numpy as np
from scipy.linalg import hadamard

np.set_printoptions(linewidth=100, suppress=True, precision=4)
```

이 문서 전체에서 두 수준 factor 는 $-1$ 과 $+1$ 로 부호화하고, 연속 factor 는 작업 범위가 $[-1, 1]$ 이
되도록 척도를 맞춘다. 본문에서 정의 없이 쓴 용어는 [Appendix A](#appendix-a-terminology) 에 모았다.

## 2. Full Factorial Designs

Full factorial design 은 factor 수준의 모든 조합에서 반응을 측정한다. 수준이
$N_1, \ldots, N_k$ 이면 treatment 마다 하나씩 $N_1 \times \cdots \times N_k$ 개의 run 이 필요하고,
factor 들이 만들어낼 수 있는 모든 효과와 교호작용을 받쳐 준다.

### 2.1. Multilevel Designs

Factor 마다 수준 개수가 같을 필요는 없다. Design 은 수준 집합들의 곱집합이며, 첫 열이 가장 빨리
변하도록 나열한다.

```python
# Python
def full_factorial(levels: Sequence[int]) -> np.ndarray:
    """Full factorial design. Column i takes levels 0..levels[i]-1; the first column varies fastest."""
    levels = list(levels)
    if any(v < 2 for v in levels):
        raise ValueError(f"every factor needs at least 2 levels; got {levels}")
    grids = np.meshgrid(*[np.arange(v) for v in levels], indexing='ij')
    return np.column_stack([g.ravel(order='F') for g in grids])


print(full_factorial([2, 3]))
```

두 수준 factor 와 세 수준 factor 는 여섯 개의 run 을 준다.

```text
[[0 0]
 [1 0]
 [0 1]
 [1 1]
 [0 2]
 [1 2]]
```

### 2.2. Two-Level Designs

모든 factor 가 두 수준이면 design 은 $2^k$ 개의 run 을 가지며, 아래의 fractional 계열과 response
surface 계열의 출발점이 된다.

```python
# Python
def two_level_factorial(n_factors: int) -> np.ndarray:
    """Two-level full factorial design coded as -1 and +1."""
    if n_factors < 1:
        raise ValueError(f"n_factors must be at least 1; got {n_factors}")
    return 2.0 * full_factorial([2] * n_factors) - 1.0


print(two_level_factorial(3))
```

```text
[[-1. -1. -1.]
 [ 1. -1. -1.]
 [-1.  1. -1.]
 [ 1.  1. -1.]
 [-1. -1.  1.]
 [ 1. -1.  1.]
 [-1.  1.  1.]
 [ 1.  1.  1.]]
```

열들이 직교이고 균형을 이루는데, 그것이 모든 효과를 서로 독립으로, 그리고 그 run 개수가 허용하는
가장 작은 분산으로 추정할 수 있게 하는 것이다.

## 3. Fractional Factorial Designs

### 3.1. Introduction

Full factorial 의 run 개수는 $2^k$ 으로 늘고, 그 run 대부분은 좀처럼 크지 않은 고차 교호작용을 사는 데
쓰인다. Fractional factorial design 은 treatment 의 부분집합만 남기되, 중요하다고 믿는 효과가 추정
가능하게 남도록 고른다. 그 대가가 confounding 이다. 남은 각 열은 여러 효과의 합을 지니며, 그 design
으로 한 실험은 그것들을 갈라내지 못한다.

Resolution 은 confounding 이 얼마나 나쁜지를 이름 붙인 것으로, defining relation 에서 가장 짧은
word 의 길이이다.

Table 2. Design resolution [[5](#ref-5)].

| Resolution | Main effects confounded with | Two-way interactions confounded with |
|---|---|---|
| III | 2차 교호작용 | 서로 |
| IV | 3차 교호작용 | 서로 |
| V | 4차 교호작용 | 3차 교호작용 |

### 3.2. Plackett-Burman Designs

주효과만 유의하다고 볼 때, Plackett-Burman design 은 run 개수가 2 의 거듭제곱이 아니라 4 의 배수인
resolution III design 을 준다 [[2](#ref-2)]. 그래서 두 수준 factorial 사이의 빈틈을 메운다. factor
11 개가 run 16 개가 아니라 12 개로 처리된다.

Design 은 Hadamard matrix 로 만든다. run 개수가 2 의 거듭제곱이면 scipy 가 그것을 바로 주고, 그렇지
않으면 알려진 generator 행의 순환행렬 아래에 $-1$ 로 채운 행 하나를 붙인 것이 design 이다.

```python
# Python
PB_GENERATORS = {
    12: '++-+++---+-',
    20: '++--++++-+-+----++-',
    24: '+++++-+-++--++--+-+----',
}


def plackett_burman(n_factors: int) -> np.ndarray:
    """Plackett-Burman design with the smallest run count that is a multiple of 4 and exceeds n_factors."""
    if n_factors < 1:
        raise ValueError(f"n_factors must be at least 1; got {n_factors}")
    n_runs = 4 * int(np.ceil((n_factors + 1) / 4))
    if n_runs & (n_runs - 1) == 0:
        design = hadamard(n_runs)[:, 1:] * 1.0
    elif n_runs in PB_GENERATORS:
        row = np.array([1.0 if c == '+' else -1.0 for c in PB_GENERATORS[n_runs]])
        design = np.vstack([np.roll(row, k) for k in range(n_runs - 1)] + [-np.ones(n_runs - 1)])
    else:
        raise ValueError(f"no Plackett-Burman construction available for {n_runs} runs")
    return design[:, :n_factors]


design = plackett_burman(11)
print(design.shape, np.allclose(design.T @ design, 12 * np.eye(11)))
```

```text
(12, 11) True
```

factor 11 개를 run 12 개로 screening 하며, 확인 결과 열들이 직교임이 드러난다. run 개수가 2 의
거듭제곱도 아니고 generator 가 있는 것도 아니면, 다른 design 으로 물러서는 대신 오류를 낸다.

### 3.3. General Fractional Designs

General fractional design 은 basic factor 들로 이루어진 full factorial 에서 출발해, 남은 factor 를
그것들의 곱으로 정의한다. 한 factor 를 정의하는 그 곱이 그 factor 의 generator 이며, generator 가
design 과 그 confounding 을 함께 정한다.

```python
# Python
def fractional_factorial(generators: str) -> np.ndarray:
    """Fractional factorial design from generators such as 'a b c abc'. Single letters are the basic factors."""
    terms = generators.split()
    basic = ''.join(t for t in terms if len(t) == 1)
    if not basic:
        raise ValueError(f"generators must contain at least one single-letter basic factor; got {generators!r}")
    if len(set(basic)) != len(basic):
        raise ValueError(f"basic factors must be distinct; got {basic!r}")
    base = two_level_factorial(len(basic))
    columns = []
    for term in terms:
        unknown = set(term) - set(basic)
        if unknown:
            raise ValueError(f"term {term!r} uses factors {sorted(unknown)} that are not basic factors")
        columns.append(np.prod(base[:, [basic.index(c) for c in term]], axis=1))
    return np.column_stack(columns)


print(fractional_factorial('a b c abc'))
```

factor 4 개를 run 8 개로 다루며, 이는 full factorial 이 필요로 할 16 개의 절반이다.

```text
[[-1. -1. -1. -1.]
 [ 1. -1. -1.  1.]
 [-1.  1. -1.  1.]
 [ 1.  1. -1. -1.]
 [-1. -1.  1.  1.]
 [ 1. -1.  1. -1.]
 [-1.  1.  1. -1.]
 [ 1.  1.  1.  1.]]
```

Defining relation 은 generator 가 함의하는 word 들이 생성하며, resolution 은 그 word 들이 생성하는
부분군에서 가장 짧은 word 이다.

```python
# Python
def design_resolution(generators: str) -> int:
    """Resolution of a fractional factorial design: the length of the shortest word in the defining relation."""
    terms = generators.split()
    basic = ''.join(t for t in terms if len(t) == 1)
    names = [chr(ord('A') + i) for i in range(len(terms))]
    words = []
    for name, term in zip(names, terms):
        if len(term) == 1:
            continue
        words.append(frozenset([names[basic.index(c)] for c in term] + [name]))
    if not words:
        return len(terms) + 1  # a full factorial has no defining relation, so no effect is confounded
    subgroup = set()
    for size in range(1, len(words) + 1):
        for combination in itertools.combinations(words, size):
            word = frozenset()
            for w in combination:
                word = word.symmetric_difference(w)
            if word:
                subgroup.add(word)
    return min(len(w) for w in subgroup)


for spec in ['a b ab', 'a b c ab', 'a b c abc', 'a b c d abcd']:
    print(f'{spec:16s} runs={len(fractional_factorial(spec)):2d} resolution={design_resolution(spec)}')
```

```text
a b ab           runs= 4 resolution=3
a b c ab         runs= 8 resolution=3
a b c abc        runs= 8 resolution=4
a b c d abcd     runs=16 resolution=5
```

run 8 개짜리 두 design 은 비용이 같고 generator 만 다른데, 하나는 주효과를 2차 교호작용과 confound
시키고 다른 하나는 그러지 않는다. Generator 가 선택의 전부이다.

## 4. Response Surface Designs

### 4.1. Introduction

중요한 factor 가 무엇인지 알고 목표가 최적화라면 model 이 곡률을 지녀야 한다. 최적점은 정류점인데
일차 model 에는 정류점이 없기 때문이다. factor 가 $k$ 개인 완전 이차 model 의 계수는
$(k+1)(k+2)/2$ 개이다.

$$y = \beta_0 + \sum_{i=1}^{k} \beta_i x_i + \sum_{i \lt j} \beta_{ij} x_i x_j + \sum_{i=1}^{k} \beta_{ii} x_i^{2} + \varepsilon$$

두 수준 design 으로는 이 model 을 맞출 수 없다. 제곱항이 두 수준에서 같은 값을 갖기 때문이다. 아래
두 design 은 서로 다른 방식으로 세 번째 수준을 더한다.

### 4.2. Central Composite Designs

Central composite design 은 정육면체의 꼭짓점에 두 수준 factorial 을, factor 축 위 중심에서 거리
$\alpha$ 인 곳에 star point 를, 그리고 하나 이상의 centre run 을 둔다. $\alpha = (2^k)^{1/4}$ 로 잡으면
design 이 rotatable 이 되어 예측분산이 방향이 아니라 중심으로부터의 거리에만 의존한다.

세 변형은 star point 가 어디에 놓이는지에서 다르다. Circumscribed design 은 그것을 정육면체 바깥에
두므로 두 수준 범위를 넘는 factor 설정이 필요하다. Faced design 은 $\alpha = 1$ 로 하여 정육면체의 면
위에 둔다. Inscribed design 은 circumscribed 의 모양을 유지하되 꼭짓점이 아니라 star point 가
$\pm 1$ 에 오도록 척도를 다시 맞춘다.

```python
# Python
def central_composite(n_factors: int, n_center: int = 1, kind: str = 'circumscribed') -> np.ndarray:
    """Central composite design. kind is 'circumscribed', 'inscribed' or 'faced'."""
    if n_factors < 2:
        raise ValueError(f"a central composite design needs at least 2 factors; got {n_factors}")
    if n_center < 1:
        raise ValueError(f"n_center must be at least 1; got {n_center}")
    if kind not in ('circumscribed', 'inscribed', 'faced'):
        raise ValueError(f"kind must be 'circumscribed', 'inscribed' or 'faced'; got {kind!r}")
    cube = two_level_factorial(n_factors)
    alpha = 1.0 if kind == 'faced' else (2.0 ** n_factors) ** 0.25
    star = np.zeros((2 * n_factors, n_factors))
    for i in range(n_factors):
        star[2 * i, i] = alpha
        star[2 * i + 1, i] = -alpha
    if kind == 'inscribed':
        cube, star = cube / alpha, star / alpha
    return np.vstack([cube, star, np.zeros((n_center, n_factors))])


print(central_composite(2, n_center=1))
```

```text
[[-1.     -1.    ]
 [ 1.     -1.    ]
 [-1.      1.    ]
 [ 1.      1.    ]
 [ 1.4142  0.    ]
 [-1.4142  0.    ]
 [ 0.      1.4142]
 [ 0.     -1.4142]
 [ 0.      0.    ]]
```

잘못된 `kind` 는 기본값으로 물러서는 대신 오류를 낸다. 그래서 철자를 틀린 변형이 호출자가 요청하지
않은 design 을 조용히 만들어내는 일이 없다.

### 4.3. Box-Behnken Designs

Box-Behnken design 도 완전 이차 model 을 맞추지만 두 factor 를 동시에 극단으로 두는 일이 없다
[[3](#ref-3)]. 점들이 design 공간의 모서리 중점과 중심에 놓이므로 정육면체의 꼭짓점을 피하게 되고,
모든 factor 의 극단을 한꺼번에 묶는 run 이 없다. 안에 박힌 factorial design 도 없다.

```python
# Python
def box_behnken(n_factors: int, n_center: int = 3) -> np.ndarray:
    """Box-Behnken design: a two-level factorial in each pair of factors with the rest held at the centre."""
    if n_factors < 3:
        raise ValueError(f"a Box-Behnken design needs at least 3 factors; got {n_factors}")
    if n_center < 1:
        raise ValueError(f"n_center must be at least 1; got {n_center}")
    block = two_level_factorial(2)
    rows = []
    for i, j in itertools.combinations(range(n_factors), 2):
        edge = np.zeros((len(block), n_factors))
        edge[:, [i, j]] = block
        rows.append(edge)
    return np.vstack(rows + [np.zeros((n_center, n_factors))])


print(box_behnken(3, n_center=3))
```

```text
[[-1. -1.  0.]
 [ 1. -1.  0.]
 [-1.  1.  0.]
 [ 1.  1.  0.]
 [-1.  0. -1.]
 [ 1.  0. -1.]
 [-1.  0.  1.]
 [ 1.  0.  1.]
 [ 0. -1. -1.]
 [ 0.  1. -1.]
 [ 0. -1.  1.]
 [ 0.  1.  1.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]]
```

Table 3. Run counts of the two response surface designs, three centre runs each.

| Factors | Quadratic terms | Box-Behnken | Central composite |
|---:|---:|---:|---:|
| 3 | 10 | 15 | 17 |
| 4 | 15 | 27 | 27 |
| 5 | 21 | 43 | 45 |

위의 짝 단위 구성법은 factor 3 개에서 5 개까지 발표된 Box-Behnken design 을 그대로 재현한다. 5 개를
넘으면 발표된 design 은 balanced incomplete block design 을 쓰며 모든 짝을 차례로 도는 것보다 작다.
따라서 이 함수는 여전히 타당하지만 최소는 아니게 된다.

## 5. D-Optimal Designs

### 5.1. Introduction

위의 계열들은 규칙적인 상황을 위해 만들어졌다. factor 설정이 정육면체를 이루고, run 개수가 2 의
거듭제곱이거나 4 의 배수이며, 모든 factor 가 자유롭게 움직이는 상황이다. D-optimal design 은 그것을
모두 버린다. model 과 run 예산을 주어진 것으로 받고, 정보행렬 $\mathbf{X}^{\top}\mathbf{X}$ 의
행렬식을 최대로 하여 계수의 공분산을 최소로 하는 run 집합을 찾는다.

D-efficiency 는 그 행렬식을 정규화하여 크기가 다른 design 끼리 비교할 수 있게 한다. 직교 design 에서
1 이고 그 밖에서는 그보다 작다. 여기서 $p$ 는 model 항의 개수이다.

$$D = \frac{\left| \mathbf{X}^{\top}\mathbf{X} \right|^{1/p}}{n}$$

```python
# Python
def model_matrix(design: np.ndarray, model: str = 'linear') -> np.ndarray:
    """Model matrix with an intercept. model is 'linear', 'interaction' or 'quadratic'."""
    if model not in ('linear', 'interaction', 'quadratic'):
        raise ValueError(f"model must be 'linear', 'interaction' or 'quadratic'; got {model!r}")
    design = np.atleast_2d(design)
    n_runs, n_factors = design.shape
    columns = [np.ones(n_runs), *design.T]
    if model in ('interaction', 'quadratic'):
        for i, j in itertools.combinations(range(n_factors), 2):
            columns.append(design[:, i] * design[:, j])
    if model == 'quadratic':
        for i in range(n_factors):
            columns.append(design[:, i] ** 2)
    return np.column_stack(columns)


def d_efficiency(design: np.ndarray, model: str = 'linear') -> float:
    """D-efficiency, the normalised determinant of the information matrix. Larger is better; 1 is the maximum."""
    x = model_matrix(design, model=model)
    n_runs, n_terms = x.shape
    if n_runs < n_terms:
        raise ValueError(f"{n_runs} runs cannot fit {n_terms} model terms")
    determinant = np.linalg.det(x.T @ x)
    if determinant <= 0:
        return 0.0
    return determinant ** (1.0 / n_terms) / n_runs


print(round(d_efficiency(two_level_factorial(2), model='linear'), 4))
```

```text
1.0
```

두 수준 full factorial 은 linear model 에서 최댓값에 이르는데, 이것이 이 척도가 의도대로 맞춰졌는지를
보는 확인이다. 직교 design 이 존재한다면 D-optimal 탐색이 그보다 나을 수는 없다.

### 5.2. Generating D-Optimal Designs

탐색은 반복적이다. Coordinate-exchange algorithm 은 무작위 design 에서 출발해 더 나아지지 않을 때까지
한 가지 이동을 되풀이한다. run 하나의 factor 하나를 잡아 grid 위의 모든 값을 시도하고 가장 좋은 것을
남기는 것이다 [[4](#ref-4)]. 결과가 어디에서 출발했는지에 달려 있으므로, 무작위 출발점 여러 개에서
전체를 되풀이하고 가장 좋은 design 을 남긴다.

```python
# Python
def coordinate_exchange(n_runs: int, n_factors: int, model: str = 'linear', n_levels: int = 3,
                        n_tries: int = 5, seed: int = None) -> np.ndarray:
    """D-optimal design by coordinate exchange over a grid of n_levels values spanning [-1, 1]."""
    if n_levels < 2:
        raise ValueError(f"n_levels must be at least 2; got {n_levels}")
    rng = np.random.default_rng(seed)
    grid = np.linspace(-1.0, 1.0, n_levels)
    best, best_score = None, -np.inf
    for _ in range(n_tries):
        design = rng.choice(grid, size=(n_runs, n_factors))
        score = d_efficiency(design, model=model)
        improved = True
        while improved:
            improved = False
            for run in range(n_runs):
                for factor in range(n_factors):
                    current = design[run, factor]
                    for value in grid:
                        design[run, factor] = value
                        trial = d_efficiency(design, model=model)
                        if trial > score + 1e-12:
                            score, current, improved = trial, value, True
                    design[run, factor] = current
        if score > best_score:
            best, best_score = design.copy(), score
    return best


design = coordinate_exchange(n_runs=12, n_factors=3, model='quadratic', n_levels=3, n_tries=5, seed=1)
print(round(d_efficiency(design, model='quadratic'), 4))
print(design)
```

factor 3 개의 quadratic model 에 run 12 개로, Table 3 의 가장 작은 Box-Behnken design 보다 3 개 적고
model 이 가진 계수 10 개보다 2 개 많다.

```text
0.4498
[[ 1.  1.  1.]
 [-1. -1. -1.]
 [ 1.  1. -1.]
 [-1.  1.  1.]
 [-1.  1. -1.]
 [-1.  0.  0.]
 [ 0.  0. -1.]
 [ 0.  1.  0.]
 [ 1. -1.  1.]
 [ 1. -1. -1.]
 [ 1.  0.  1.]
 [-1. -1.  1.]]
```

model 의 항 개수보다 적은 run 을 요청하면 `d_efficiency` 에서 오류가 난다. 그 크기의 design 으로는
model 을 추정할 수 없고, 그런 것을 돌려주면 실수를 감추게 되기 때문이다.

### 5.3. Augmenting D-Optimal Designs

실험은 흔히 단계를 나누어 진행되고, 이미 수행한 run 은 다시 고를 수 있는 대상이 아니다. Augmentation
은 그 행들을 고정한 채 새 행만 탐색하므로, 더해지는 run 은 기존 design 에 없는 것을 기준으로 골라진다.

여기서의 탐색은 좌표 하나가 아니라 행 전체를 candidate set 과 맞바꾼다. 행을 고정된 목록에서 뽑는
경우에는 그것이 자연스러운 이동이다.

```python
# Python
def row_exchange(candidates: np.ndarray, n_runs: int, model: str = 'linear', n_tries: int = 5,
                 seed: int = None, fixed: np.ndarray = None) -> np.ndarray:
    """D-optimal design chosen from a candidate set by row exchange. Rows in fixed are kept and not exchanged."""
    if n_runs < 1:
        raise ValueError(f"n_runs must be at least 1; got {n_runs}")
    rng = np.random.default_rng(seed)
    fixed = np.empty((0, candidates.shape[1])) if fixed is None else np.atleast_2d(fixed)
    best, best_score = None, -np.inf
    for _ in range(n_tries):
        chosen = candidates[rng.choice(len(candidates), size=n_runs, replace=True)]
        score = d_efficiency(np.vstack([fixed, chosen]), model=model)
        improved = True
        while improved:
            improved = False
            for run in range(n_runs):
                current = chosen[run].copy()
                for row in candidates:
                    chosen[run] = row
                    trial = d_efficiency(np.vstack([fixed, chosen]), model=model)
                    if trial > score + 1e-12:
                        score, current, improved = trial, row.copy(), True
                chosen[run] = current
        if score > best_score:
            best, best_score = chosen.copy(), score
    return best


existing = two_level_factorial(2)
allowed = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=2)))[:, ::-1]
added = row_exchange(allowed, n_runs=2, model='quadratic', n_tries=5, seed=3, fixed=existing)
print(added)
print(round(d_efficiency(np.vstack([existing, added]), model='quadratic'), 4))
```

꼭짓점 run 4 개로는 항이 6 개인 quadratic model 을 아예 맞출 수 없다. run 두 개를 더하면 추정할 수
있게 되는데, 탐색은 그 둘을 모두 꼭짓점이 아닌 자리, 곧 기존 design 에 아무것도 없는 자리에 둔다.

```text
[[ 0.  1.]
 [-1.  0.]]
0.42
```

### 5.4. Specifying Fixed Covariate Factors

어떤 factor 는 설정되는 것이 아니라 기록된다. 주위 온도, 근무 중인 작업자, batch 의 경과 시간 같은
것이다. 그 값은 run 마다 미리 알려져 있으나 고를 수는 없다. 그러면 design 문제는 covariate 열이 주어진
상태에서 통제 가능한 factor 를 골라, model 이 통제 효과와 covariate 효과를 갈라내게 하는 일이 된다.

```python
# Python
def covariate_exchange(covariates: np.ndarray, n_factors: int, model: str = 'linear', n_levels: int = 3,
                       n_tries: int = 5, seed: int = None) -> np.ndarray:
    """D-optimal design whose last columns are the given fixed covariates, one run per covariate row."""
    covariates = np.atleast_2d(covariates)
    rng = np.random.default_rng(seed)
    grid = np.linspace(-1.0, 1.0, n_levels)
    n_runs = len(covariates)
    best, best_score = None, -np.inf
    for _ in range(n_tries):
        controlled = rng.choice(grid, size=(n_runs, n_factors))
        score = d_efficiency(np.hstack([controlled, covariates]), model=model)
        improved = True
        while improved:
            improved = False
            for run in range(n_runs):
                for factor in range(n_factors):
                    current = controlled[run, factor]
                    for value in grid:
                        controlled[run, factor] = value
                        trial = d_efficiency(np.hstack([controlled, covariates]), model=model)
                        if trial > score + 1e-12:
                            score, current, improved = trial, value, True
                    controlled[run, factor] = current
        if score > best_score:
            best, best_score = controlled.copy(), score
    return np.hstack([best, covariates])


drift = np.linspace(-1.0, 1.0, 8).reshape(-1, 1)
print(covariate_exchange(drift, n_factors=2, model='linear', n_levels=3, n_tries=3, seed=4))
```

Covariate 는 run 8 개에 걸쳐 꾸준히 흘러간다. 탐색은 그 흐름에 대해 균형을 이루는 무늬를 통제 가능한
두 열에 놓아 답하며, 그래서 어느 통제 효과도 그 흐름과 confound 되지 않는다.

```text
[[-1.     -1.     -1.    ]
 [ 1.     -1.     -0.7143]
 [ 1.      1.     -0.4286]
 [-1.      1.     -0.1429]
 [-1.      1.      0.1429]
 [ 1.      1.      0.4286]
 [ 1.     -1.      0.7143]
 [-1.     -1.      1.    ]]
```

### 5.5. Specifying Categorical Factors

Categorical factor 에는 수치 척도가 없으므로 그 수준을 $\pm 1$ 쪽으로 밀 수 없다. 수준이 $L$ 개인
factor 는 $L - 1$ 개의 부호화된 열로 model 에 들어가고, 탐색은 그 열들 위에서 이루어진다.

```python
# Python
def effect_code(levels: np.ndarray, n_levels: int) -> np.ndarray:
    """Effect coding of one categorical factor into n_levels - 1 columns."""
    levels = np.asarray(levels, dtype=int)
    if levels.min() < 0 or levels.max() >= n_levels:
        raise ValueError(f"levels must lie in 0..{n_levels - 1}; got range {levels.min()}..{levels.max()}")
    coded = np.zeros((len(levels), n_levels - 1))
    for column in range(n_levels - 1):
        coded[levels == column, column] = 1.0
    coded[levels == n_levels - 1, :] = -1.0
    return coded


levels = np.array(list(itertools.product(range(3), repeat=3)))[:, ::-1]
coded = np.hstack([effect_code(levels[:, k], 3) for k in range(3)])
selected = row_exchange(coded, n_runs=9, model='linear', n_tries=5, seed=7)
chosen = np.array([levels[int(np.where((coded == r).all(axis=1))[0][0])] for r in selected])
print(chosen[np.lexsort((chosen[:, 2], chosen[:, 1], chosen[:, 0]))])
```

수준이 세 개인 categorical factor 세 개는 27 개의 후보 treatment 를 준다. run 9 개를 요청하면, 탐색은
모든 factor 의 모든 수준이 세 번씩 나타나고 두 factor 에서 뽑은 수준의 모든 짝이 정확히 한 번씩
나타나는 design 을 돌려준다.

```text
[[0 0 1]
 [0 1 0]
 [0 2 2]
 [1 0 0]
 [1 1 2]
 [1 2 1]
 [2 0 2]
 [2 1 1]
 [2 2 0]]
```

### 5.6. Specifying Candidate Sets

section 5.3 의 row exchange 는 고를 수 있는 run 의 목록을 필요로 한다. factor 공간이 단순한
정육면체라면 그 목록은 그 위의 grid 이고, 그것을 만드는 일은 기계적이다.

```python
# Python
def candidate_set(n_factors: int, n_levels: int = 3) -> np.ndarray:
    """Candidate set: the full factorial of n_levels values spanning [-1, 1] in every factor."""
    if n_levels < 2:
        raise ValueError(f"n_levels must be at least 2; got {n_levels}")
    grid = np.linspace(-1.0, 1.0, n_levels)
    return np.array(list(itertools.product(grid, repeat=n_factors)))[:, ::-1]


selected = row_exchange(candidate_set(2, n_levels=3), n_runs=6, model='quadratic', n_tries=5, seed=2)
print(selected)
```

```text
[[ 1.  1.]
 [-1.  1.]
 [ 0.  1.]
 [-1. -1.]
 [ 1.  0.]
 [ 1. -1.]]
```

Candidate set 을 만들게 두지 않고 명시적으로 건네는 이유는 factor 공간이 정육면체가 아닌 경우가 흔하기
때문이다. 성분의 합이 1 이어야 하는 mixture, 함께 높을 수 없는 두 설정, 차가운 상태로 빠르게 돌 수 없는
기계가 모두 grid 에서 행을 덜어내는 제약이다. Candidate set 에서 그 행들을 지우는 것으로 충분하다.
탐색은 목록에 없는 run 을 결코 제안하지 않기 때문이다.

## References

<a id="ref-1"></a>
[1] MathWorks. [Design of Experiments — Statistics and Machine Learning Toolbox Documentation](https://kr.mathworks.com/help/stats/design-of-experiments.html).

<a id="ref-2"></a>
[2] Plackett, R. L., & Burman, J. P. (1946). [The Design of Optimum Multifactorial Experiments](https://doi.org/10.1093/biomet/33.4.305).
*Biometrika*, 33(4), 305–325.

<a id="ref-3"></a>
[3] Box, G. E. P., & Behnken, D. W. (1960). [Some New Three Level Designs for the Study of
Quantitative Variables](https://doi.org/10.1080/00401706.1960.10489912). *Technometrics*, 2(4), 455–475.

<a id="ref-4"></a>
[4] Meyer, R. K., & Nachtsheim, C. J. (1995). [The Coordinate-Exchange Algorithm for Constructing
Exact Optimal Experimental Designs](https://doi.org/10.1080/00401706.1995.10485889). *Technometrics*, 37(1), 60–69.

<a id="ref-5"></a>
[5] Montgomery, D. C. (2019). *Design and Analysis of Experiments* (10th ed.). Wiley.
ISBN 978-1-119-49249-8.

---

## Appendix A. Terminology

- **Basic factor**: generator 가 아니라 fractional design 이 출발점으로 삼는 full factorial 에서 열이
  나오는 factor.
- **Centre run**: 모든 연속 factor 가 자기 범위의 가운데에 있는 run.
- **Confounding**: 두 효과가 같은 design 열에 실려, 그 실험의 어떤 분석으로도 갈라낼 수 없는 상태.
- **Defining relation**: fractional factorial design 에서 1 로 채워진 열과 같아지는 factor 열들의 곱의
  모임이며, 그중 가장 짧은 word 가 resolution 을 준다.
- **Factor**: 실험자가 값을 정하거나 기록하는 입력.
- **Generator**: fractional factorial design 에서 더해지는 factor 를 정의하는 basic factor 들의 곱.
- **Level**: 한 design 에서 factor 가 갖는 값 가운데 하나.
- **Response**: run 의 측정된 출력.
- **Rotatable**: 예측분산이 factor 공간 중심으로부터의 거리에만 의존하는 design.
- **Run**: factor 수준의 한 조합에서 실험을 한 번 수행하는 것이며, design 의 한 행.
- **Star point**: central composite design 에서 factor 하나를 중심 밖으로 옮기고 나머지는 중심에 두는
  run.
- **Treatment**: factor 수준의 한 조합.
