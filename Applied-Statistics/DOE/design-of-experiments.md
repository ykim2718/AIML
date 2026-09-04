# Design of Experiments
Rev. 3 | Created: 2026-08-30 | Updated: 2026-09-04 20:10 UTC

> The design families of the MathWorks Statistics and Machine Learning Toolbox chapter on design of
> experiments [[1](#ref-1)], with the same organisation and with every construction implemented in
> Python on top of numpy and scipy.

## 1. Design of Experiments

Passive data collection leads to a difficulty. The same naturally occurring change in a response
may be explained by several factors at once, and a model fitted to such data cannot separate them.
Designed experiments remove the difficulty by setting the factor values deliberately, so that the
factors move independently of each other and their effects on the response can be estimated apart.

Write $\mathbf{y}$ for the vector of $n$ measured responses, $\mathbf{X}$ for the model matrix
built from the factor settings, and $\boldsymbol{\beta}$ for the coefficients. The least squares
estimate and its covariance follow from the design alone once $\sigma^2$ is fixed.

$$\hat{\boldsymbol{\beta}} = \left( \mathbf{X}^{\top}\mathbf{X} \right)^{-1} \mathbf{X}^{\top}\mathbf{y}, \qquad \mathrm{Cov}\left[ \hat{\boldsymbol{\beta}} \right] = \sigma^{2} \left( \mathbf{X}^{\top}\mathbf{X} \right)^{-1}$$

Designing an experiment is therefore choosing the rows of $\mathbf{X}$ so that this covariance is
small, subject to a budget on the number of runs. The families below differ in which model they aim
at and in how they spend the budget.

Table 1. Design families and the model each one serves.

| Family | Model aimed at | Typical use |
|---|---|---|
| Full factorial | All effects and interactions | Few factors, every combination affordable |
| Fractional factorial | Main effects, some interactions | Screening many factors |
| Response surface | Full quadratic | Optimisation near a known operating point |
| D-optimal | Any stated model | Irregular budgets, constraints, covariates |

The Python blocks in this document are meant to be read in order; each one uses the names defined
before it, on top of a single header. The blocks that follow them hold the printed output.

```python
# Python
import itertools
from typing import Sequence

import numpy as np
from scipy.linalg import hadamard

np.set_printoptions(linewidth=100, suppress=True, precision=4)
```

Two-level factors are coded as $-1$ and $+1$ throughout, and continuous factors are scaled so that
their working range is $[-1, 1]$. Terms used without definition in the body are collected in
[Appendix A](#appendix-a-terminology).

## 2. Full Factorial Designs

A full factorial design measures the response at every combination of the factor levels. With
$N_1, \ldots, N_k$ levels it needs $N_1 \times \cdots \times N_k$ runs, one per treatment, and it
supports every effect and interaction the factors can produce.

### 2.1. Multilevel Designs

Factors need not share a level count. The design is the cartesian product of the level sets, listed
so that the first column varies fastest.

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

The two-level factor and the three-level factor give six runs.

```text
[[0 0]
 [1 0]
 [0 1]
 [1 1]
 [0 2]
 [1 2]]
```

### 2.2. Two-Level Designs

When every factor has two levels the design has $2^k$ runs and is the starting point for the
fractional and response surface families below.

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

The columns are orthogonal and balanced, which is what makes every effect estimable independently
of the others and with the smallest variance the run count allows.

## 3. Fractional Factorial Designs

### 3.1. Introduction

The run count of a full factorial grows as $2^k$, and most of those runs buy high-order
interactions that are rarely large. A fractional factorial design keeps a subset of the treatments,
chosen so that the effects believed to matter stay estimable. The price is confounding: each
retained column carries the sum of several effects, and no experiment run on that design can tell
them apart.

The resolution names how bad the confounding is, and is the length of the shortest word in the
defining relation.

Table 2. Design resolution [[5](#ref-5)].

| Resolution | Main effects confounded with | Two-way interactions confounded with |
|---|---|---|
| III | Two-way interactions | Each other |
| IV | Three-way interactions | Each other |
| V | Four-way interactions | Three-way interactions |

### 3.2. Plackett-Burman Designs

When only main effects are considered significant, a Plackett-Burman design gives a resolution III
design whose run count is a multiple of 4 rather than a power of 2 [[2](#ref-2)]. It therefore
fills the gaps between the two-level factorials: 11 factors take 12 runs instead of 16.

The design is built from a Hadamard matrix. Where the run count is a power of 2 scipy supplies it
directly; where it is not, the design is a circulant of a known generator row with a final row of
$-1$.

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

Eleven factors are screened in twelve runs, and the check confirms that the columns are orthogonal.
A run count that is neither a power of 2 nor covered by a generator raises rather than falling back
on a different design.

### 3.3. General Fractional Designs

A general fractional design starts from a full factorial in a set of basic factors and defines the
remaining factors as products of them. The product that defines a factor is its generator, and the
generators fix both the design and its confounding.

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

Four factors in eight runs, half of the sixteen a full factorial would need.

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

The defining relation is generated by the words that the generators imply, and the resolution is
the shortest word in the subgroup those words generate.

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

The two eight-run designs cost the same and differ only in the generator, yet one confounds main
effects with two-way interactions and the other does not. The generator is the whole of the choice.

## 4. Response Surface Designs

### 4.1. Introduction

Once the factors that matter are known and the goal is optimisation, the model has to carry
curvature, because an optimum is a stationary point and a first-order model has none. The full
quadratic model in $k$ factors has $(k+1)(k+2)/2$ coefficients.

$$y = \beta_0 + \sum_{i=1}^{k} \beta_i x_i + \sum_{i \lt j} \beta_{ij} x_i x_j + \sum_{i=1}^{k} \beta_{ii} x_i^{2} + \varepsilon$$

A two-level design cannot fit it, since a square term takes the same value at both levels. The two
designs below add a third level in different ways.

### 4.2. Central Composite Designs

A central composite design puts a two-level factorial at the corners of a cube, star points on the
factor axes at a distance $\alpha$ from the centre, and one or more centre runs. Choosing
$\alpha = (2^k)^{1/4}$ makes the design rotatable, so the prediction variance depends on the
distance from the centre and not on the direction.

The three variants differ in where the star points fall. A circumscribed design places them outside
the cube, which needs factor settings beyond the two-level range. A faced design puts them on the
cube faces at $\alpha = 1$. An inscribed design keeps the circumscribed shape but rescales it so
that the star points, not the corners, sit at $\pm 1$.

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

The invalid `kind` raises instead of falling back on a default, so a misspelled variant cannot
silently produce a design the caller did not ask for.

### 4.3. Box-Behnken Designs

A Box-Behnken design also fits the full quadratic model but never sets two factors to an extreme at
the same time [[3](#ref-3)]. Its points sit at the midpoints of the edges of the design space and at
the centre, so the corners of the cube are avoided and no run combines the extremes of every factor.
There is no embedded factorial design.

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

The pairwise construction above reproduces the published Box-Behnken designs for three to five
factors. Beyond five the published designs use a balanced incomplete block design and are smaller
than every pair taken in turn, so this function stays valid but stops being minimal.

## 5. D-Optimal Designs

### 5.1. Introduction

The families above are built for regular situations: a cube of factor settings, a run count that is
a power of 2 or a multiple of 4, and every factor free to move. A D-optimal design drops all of
that. It takes the model and the run budget as given and searches for the set of runs that
minimises the covariance of the coefficients, by maximising the determinant of the information
matrix $\mathbf{X}^{\top}\mathbf{X}$.

D-efficiency normalises that determinant so that designs of different sizes can be compared. It is
1 for an orthogonal design and smaller otherwise, where $p$ is the number of model terms.

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

The two-level full factorial reaches the maximum for the linear model, which is the check that the
measure is scaled as intended: a D-optimal search cannot do better than an orthogonal design when
one exists.

### 5.2. Generating D-Optimal Designs

The search is iterative. The coordinate-exchange algorithm starts from a random design and repeats
one move until nothing improves: take one factor of one run, try every value on a grid, and keep
the best [[4](#ref-4)]. Since the result depends on where the search starts, the whole thing is
repeated from several random starts and the best design is kept.

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

Twelve runs for a quadratic model in three factors, three fewer than the smallest Box-Behnken design
of Table 3 and two above the ten coefficients the model has.

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

Asking for fewer runs than the model has terms raises from `d_efficiency`, since no design of that
size estimates the model and returning one would hide the mistake.

### 5.3. Augmenting D-Optimal Designs

An experiment often runs in stages, and the runs already performed are not available for
reconsideration. Augmentation holds those rows fixed and searches only for the new ones, so the
added runs are chosen for what the existing design is missing.

The search here exchanges whole rows against a candidate set rather than single coordinates, which
is the natural move when the rows are drawn from a fixed list.

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

The four corner runs cannot fit the six-term quadratic model at all. Two more runs make it
estimable, and the search puts both of them off the corners, where the existing design has nothing.

```text
[[ 0.  1.]
 [-1.  0.]]
0.42
```

### 5.4. Specifying Fixed Covariate Factors

Some factors are recorded rather than set: ambient temperature, the operator on shift, the age of a
batch. Their values are known in advance for each run but cannot be chosen. The design problem is
then to choose the controlled factors given the covariate column, so that the model separates the
controlled effects from the covariate effect.

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

The covariate drifts steadily across the eight runs. The search answers with a pattern in the two
controlled columns that is balanced against that drift, so neither controlled effect is confounded
with it.

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

A categorical factor has no numeric scale, so its levels cannot be pushed towards $\pm 1$. A factor
with $L$ levels enters the model as $L - 1$ coded columns, and the search runs on those columns.

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

Three categorical factors at three levels each give 27 candidate treatments. Asked for nine runs,
the search returns a design in which every level of every factor appears three times and every pair
of levels from two factors appears exactly once.

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

The row exchange of section 5.3 needs a list of allowed runs to choose from. Where the factor space
is a plain cube, that list is a grid over it, and generating it is mechanical.

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

The reason to pass a candidate set explicitly rather than let it be generated is that the factor
space is often not a cube. A mixture whose components must sum to one, a pair of settings that
cannot be high together, a machine that cannot run cold and fast: each of these is a constraint that
removes rows from the grid. Deleting those rows from the candidate set is enough, because the search
never proposes a run that is not on the list.

## References

<a id="ref-1"></a>
[1] MathWorks. [Design of Experiments — Statistics and Machine Learning Toolbox Documentation](https://kr.mathworks.com/help/stats/design-of-experiments.html).<br>
<a id="ref-2"></a>
[2] Plackett, R. L., & Burman, J. P. (1946). [The Design of Optimum Multifactorial Experiments](https://doi.org/10.1093/biomet/33.4.305).
*Biometrika*, 33(4), 305–325.<br>
<a id="ref-3"></a>
[3] Box, G. E. P., & Behnken, D. W. (1960). [Some New Three Level Designs for the Study of
Quantitative Variables](https://doi.org/10.1080/00401706.1960.10489912). *Technometrics*, 2(4), 455–475.<br>
<a id="ref-4"></a>
[4] Meyer, R. K., & Nachtsheim, C. J. (1995). [The Coordinate-Exchange Algorithm for Constructing
Exact Optimal Experimental Designs](https://doi.org/10.1080/00401706.1995.10485889). *Technometrics*, 37(1), 60–69.<br>
<a id="ref-5"></a>
[5] Montgomery, D. C. (2019). *Design and Analysis of Experiments* (10th ed.). Wiley.
ISBN 978-1-119-49249-8.

---

## Appendix A. Terminology

- **Basic factor**: a factor whose column comes from the full factorial a fractional design starts
  from, rather than from a generator.
- **Centre run**: a run with every continuous factor at the middle of its range.
- **Confounding**: two effects carried by the same design column, so that no analysis of the
  experiment can separate them.
- **Defining relation**: the set of products of factor columns that equal the column of ones in a
  fractional factorial design; its shortest word gives the resolution.
- **Factor**: an input whose value the experimenter sets or records.
- **Generator**: the product of basic factors that defines an added factor in a fractional
  factorial design.
- **Level**: one of the values a factor takes in a design.
- **Response**: the measured output of a run.
- **Rotatable**: a design whose prediction variance depends only on the distance from the centre of
  the factor space.
- **Run**: one execution of the experiment at one combination of factor levels; one row of a design.
- **Star point**: a run of a central composite design that moves one factor off the centre and
  holds the rest at it.
- **Treatment**: a combination of factor levels.
