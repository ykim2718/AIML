# Design of Experiments
Rev. 7 | Created: 2026-08-30 | Updated: 2026-10-09 19:38 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Placement](#31-placement)
- [4. Principle](#4-principle)
- [5. Full Factorial Designs](#5-full-factorial-designs)
  - [5.1 Multilevel Designs](#51-multilevel-designs)
  - [5.2 Two-Level Designs](#52-two-level-designs)
- [6. Fractional Factorial Designs](#6-fractional-factorial-designs)
  - [6.1 Confounding and Resolution](#61-confounding-and-resolution)
  - [6.2 Plackett-Burman Designs](#62-plackett-burman-designs)
  - [6.3 General Fractional Designs](#63-general-fractional-designs)
- [7. Response Surface Designs](#7-response-surface-designs)
  - [7.1 Quadratic Model](#71-quadratic-model)
  - [7.2 Central Composite Designs](#72-central-composite-designs)
  - [7.3 Box-Behnken Designs](#73-box-behnken-designs)
- [8. D-Optimal Designs](#8-d-optimal-designs)
  - [8.1 D-Efficiency](#81-d-efficiency)
  - [8.2 Generating D-Optimal Designs](#82-generating-d-optimal-designs)
  - [8.3 Augmenting D-Optimal Designs](#83-augmenting-d-optimal-designs)
  - [8.4 Specifying Fixed Covariate Factors](#84-specifying-fixed-covariate-factors)
  - [8.5 Specifying Categorical Factors](#85-specifying-categorical-factors)
  - [8.6 Specifying Candidate Sets](#86-specifying-candidate-sets)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Python Implementation](#appendix-b-python-implementation)
  - [B.1 Common Header](#b1-common-header)
  - [B.2 Full Factorial Designs](#b2-full-factorial-designs)
  - [B.3 Fractional Factorial Designs](#b3-fractional-factorial-designs)
  - [B.4 Response Surface Designs](#b4-response-surface-designs)
  - [B.5 D-Optimal Designs](#b5-d-optimal-designs)

## 1. Purpose

- **Problem Statement**: In passively collected data several factors explain the same change in the response, so a model fitted to that data cannot separate the effect of each factor.
- **Goal**: Build a DOE User's Guide. A reader who finishes it can pick the design that fits the target model, the run budget and the constraints on the factor space, and can state its assumptions and limits.
- **Non-Goal**: The analysis after the experiment (ANOVA, model fitting and testing) and the use of any particular software are out of scope.

## 2. Summary

Designing an experiment is choosing the rows of the design matrix so that the covariance of the coefficient estimates is small within a run budget.
The target model picks the design: a full factorial for every effect, a fractional factorial for screening many factors,
a response surface design for optimisation with curvature, and a D-optimal design for irregular budgets, constraints and covariates.

## 3. Taxonomy and its Hierarchy

Design families differ in what they assume about the model, and the stronger the assumption the fewer runs they need. This
document follows the four families of the MathWorks Statistics and Machine Learning Toolbox [[1](#ref-1)].

The five axes that separate designs are shown in [Fig 1](#fig-1).

```text
Axis               Values
-----------------  ----------------------------------------------------------------------
Target model       all effects | main effects + some interactions | main effects only
                   | full quadratic | any stated model
Levels per factor  2 | 3 | any
Run count          product of levels | power of 2 | multiple of 4 | set by geometry | any
Factor space       regular cube | constrained region
Factor type        continuous and set | categorical | recorded covariate
```

<a id="fig-1"></a>
Fig 1. Axes of the design taxonomy

The five axes are independent. With the same full quadratic target model, a cubic factor space calls for a response
surface design and a constrained region calls for a D-optimal design.

What each design assumes about the model, and what it pays for that, follows the hierarchy in [Fig 2](#fig-2).

```text
Family / design          Assumes                              Gives up
-----------------------  -----------------------------------  ---------------------------------------
Full factorial           nothing about the model              small run count: N1 x ... x Nk runs
+- General fractional    high-order interactions are small    separation of confounded effects
   +- Plackett-Burman    only main effects matter             every interaction (resolution III)
Response surface         response is at most quadratic        terms above second order
+- Central composite     settings at +/-alpha are reachable   fewest runs: embeds a full 2^k
+- Box-Behnken           three or more factors                corner runs and an embedded factorial
D-optimal                the stated model is correct          regular geometry and orthogonality
```

<a id="fig-2"></a>
Fig 2. Hierarchy of design families

Each step down from the full factorial adds an assumption about the model and removes runs. A general fractional design
saves runs by treating high-order interactions as small, and a Plackett-Burman design by treating every interaction as
small. The response surface family caps the model at second order and adds a third level to estimate curvature. A
D-optimal design chooses its runs for one stated model, so it needs neither a cubic factor space nor a fixed run count.

### 3.1 Placement

Table 1 gives the criteria for choosing a design.

Table 1. Placement of designs

| Design             | Model aimed at                  | Run count                        | Use when                                                                     |
| :----------------: | :-----------------------------: | :------------------------------: | :--------------------------------------------------------------------------: |
| Full factorial     | All effects and interactions    | $N_1 \times \cdots \times N_k$   | Few factors, every combination affordable                                    |
| General fractional | Main effects, some interactions | $2^{b}$                          | Screening many factors with a chosen resolution                              |
| Plackett-Burman    | Main effects                    | Smallest multiple of 4 above $k$ | Main-effect screening needing a run count between powers of 2                |
| Central composite  | Full quadratic model            | $2^k + 2k + n_c$                 | Optimisation near a known operating point                                    |
| Box-Behnken        | Full quadratic model            | $4\binom{k}{2} + n_c$            | Optimisation of a process that cannot set every factor to an extreme at once |
| D-optimal          | Any stated model                | Set to the budget                | Irregular budgets, constraints, covariates or categorical factors            |

$k$ is the number of factors, $b$ the number of basic factors, and $n_c$ the number of centre runs.

## 4. Principle

The design fixes the precision of the coefficient estimates before any response is measured. Write $\mathbf{y}$ for the
vector of $n$ measured responses, $\mathbf{X}$ for the model matrix built from the factor settings, and
$\boldsymbol{\beta}$ for the coefficients; the least squares estimate and its covariance are given by equation (1).

```math
\hat{\boldsymbol{\beta}} = \left( \mathbf{X}^{\top}\mathbf{X} \right)^{-1} \mathbf{X}^{\top}\mathbf{y}, \qquad \mathrm{Cov}\left[ \hat{\boldsymbol{\beta}} \right] = \sigma^{2} \left( \mathbf{X}^{\top}\mathbf{X} \right)^{-1} \hspace{19em} (1)
```

The covariance in equation (1) contains no $\mathbf{y}$, so once $\sigma^2$ is given it follows from the design alone. A
designed experiment sets the factor values deliberately, so the factors move independently of each other and the effect
of each one on the response is estimated apart. The design families differ in which model they aim at and in how they
spend the run budget (Table 1).

Two-level factors are coded as $-1$ and $+1$ throughout, and continuous factors are scaled so that their working range is
$[-1, 1]$. Terms used without definition are in [Appendix A](#appendix-a-terminology), and the Python implementation and
output of every design are in [Appendix B](#appendix-b-python-implementation).

## 5. Full Factorial Designs

A full factorial design measures the response at every combination of the factor levels and estimates every effect and
interaction the factors can produce. With $N_1, \ldots, N_k$ levels it needs $N_1 \times \cdots \times N_k$ runs, one per
treatment.

- **Assumption**: Few factors and levels, so a run at every combination is affordable.
- **Settings**: The level count $N_i$ of each factor.
- **Breaks when**: More factors multiply the run count by their level counts; even at two levels $2^k$ exceeds the budget.
- **Used for**: Experiments with few factors, and the starting point of the fractional and response surface families.

### 5.1 Multilevel Designs

Factors need not share a level count. The design is the cartesian product of the level sets, listed so that the first
column varies fastest. One two-level factor and one three-level factor give six runs
([Appendix B.2](#b2-full-factorial-designs)).

### 5.2 Two-Level Designs

When every factor has two levels the design has $2^k$ runs. The columns are orthogonal and balanced, so every effect is
estimated independently of the others and with the smallest variance the run count allows. The fractional and response
surface families start from this design.

## 6. Fractional Factorial Designs

A fractional factorial design measures a subset of the treatments, chosen so that the effects believed to matter stay
estimable. Most of the $2^k$ runs of a full factorial go to high-order interactions that are rarely large.

### 6.1 Confounding and Resolution

The price of fewer runs is confounding. Each retained column carries the sum of several effects, and no experiment run on
that design can tell them apart. The resolution names the degree of confounding and is the length of the shortest word in
the defining relation.

Table 2. Design resolution [[2](#ref-2)]

| Resolution | Main effects confounded with | Two-way interactions confounded with |
| :--------: | :--------------------------: | :----------------------------------: |
| III        | Two-way interactions         | Each other                           |
| IV         | Three-way interactions       | Each other                           |
| V          | Four-way interactions        | Three-way interactions               |

Under resolution III the main effects hold only if two-way interactions are small; resolution IV estimates the main
effects apart from two-way interactions; resolution V estimates both the main effects and the two-way interactions apart
from every other effect of second order or lower.

### 6.2 Plackett-Burman Designs

When only main effects are considered significant, a Plackett-Burman design gives a resolution III design whose run count
is a multiple of 4 rather than a power of 2, and so fills the gaps between the two-level factorials [[3](#ref-3)]. Eleven
factors are screened in 12 runs instead of 16.

The design is built from a Hadamard matrix. Where the run count is a power of 2 the columns of the Hadamard matrix are
used directly; otherwise a final row of $-1$ is placed under the circulant of a known generator row. The 11 columns of the
12-run design are mutually orthogonal ([Appendix B.3](#b3-fractional-factorial-designs)).

- **Assumption**: Interactions are negligible and only main effects are significant.
- **Settings**: The number of factors $k$. The run count is the smallest multiple of 4 above $k$, and needs a generator row when it is not a power of 2.
- **Breaks when**: A large two-way interaction is confounded with a main effect, so the main-effect estimate carries that interaction.
- **Used for**: Screening many factors.

### 6.3 General Fractional Designs

A general fractional design starts from a full factorial in a set of basic factors and defines the remaining factors as
products of them. The product that defines a factor is its generator, and the generators fix both the design and its
confounding. The generators `a b c abc` handle four factors in eight runs, half of the sixteen a full factorial needs.

The defining relation is generated by the words that the generators imply, and the resolution is the length of the
shortest word in the subgroup those words generate. Table 3 gives the resolution of four generator sets.

Table 3. Resolution of four generator sets

| Generators   | Runs  | Resolution |
| :----------: | :---: | :--------: |
| a b ab       | 4     | III        |
| a b c ab     | 8     | III        |
| a b c abc    | 8     | IV         |
| a b c d abcd | 16    | V          |

The two eight-run designs cost the same and differ only in the generator, yet `a b c ab` confounds main effects with
two-way interactions and `a b c abc` does not. At a fixed run count the generator sets the resolution.

- **Assumption**: The interactions that the resolution confounds are small (Table 2).
- **Settings**: The number of basic factors $b$ (run count $2^{b}$) and a generator for each added factor.
- **Breaks when**: A large confounded interaction makes the column estimate a sum of effects that cannot be told apart.
- **Used for**: Screening many factors when the run count stays a power of 2 and the resolution must be chosen.

## 7. Response Surface Designs

A response surface design adds a third level to each factor to fit a full quadratic model, which carries curvature. Once
the factors that matter are known and the goal is optimisation, the model needs curvature: an optimum is a stationary
point, and a first-order model has none.

### 7.1 Quadratic Model

The full quadratic model in $k$ factors is equation (2) and has $(k+1)(k+2)/2$ coefficients.

```math
y = \beta_0 + \sum_{i=1}^{k} \beta_i x_i + \sum_{i \lt j} \beta_{ij} x_i x_j + \sum_{i=1}^{k} \beta_{ii} x_i^{2} + \varepsilon \hspace{19em} (2)
```

A two-level design cannot fit equation (2). The square term $x_i^2$ takes the same value 1 at $-1$ and $+1$. The two
designs below add the third level in different ways.

### 7.2 Central Composite Designs

A central composite design consists of a two-level factorial at the corners of a cube, star points on the factor axes at
a distance $\alpha$ from the centre, and one or more centre runs. Choosing $\alpha = (2^k)^{1/4}$ makes the design
rotatable, so the prediction variance depends on the distance from the centre and not on the direction.

The three variants differ in where the star points sit.

- **Circumscribed**: Star points outside the cube at $\pm\alpha$, which needs factor settings beyond the two-level range.
- **Faced**: Star points on the faces of the cube at $\alpha = 1$.
- **Inscribed**: The circumscribed shape rescaled so that the star points, not the corners, sit at $\pm 1$.

A circumscribed design in two factors with one centre run has nine runs, with the star points at $\pm 1.4142$
([Appendix B.4](#b4-response-surface-designs)).

- **Assumption**: Near the operating point the response is approximated by a full quadratic model, and there are two or more factors.
- **Settings**: $\alpha$ ($(2^k)^{1/4}$ for rotatable, 1 for faced), the variant, and the centre run count $n_c \ge 1$. The run count is $2^k + 2k + n_c$.
- **Breaks when**: The process cannot reach the $\pm\alpha$ settings, which rules out the circumscribed design in favour of faced or inscribed.
- **Used for**: Optimisation near a known operating point.

### 7.3 Box-Behnken Designs

A Box-Behnken design fits the full quadratic model without ever setting two factors to an extreme at the same time
[[4](#ref-4)]. It puts a two-level factorial in each pair of factors with the rest held at the centre, so its points sit at
the midpoints of the edges of the design space and at the centre. The corners of the cube are left out, so no run
combines the extremes of every factor, and there is no embedded factorial design.

Table 4. Run counts of the two response surface designs, three centre runs each

| Factors | Quadratic terms | Box-Behnken | Central composite |
| :-----: | :-------------: | :---------: | :---------------: |
| 3       | 10              | 15          | 17                |
| 4       | 15              | 27          | 27                |
| 5       | 21              | 43          | 45                |

The pairwise construction reproduces the published Box-Behnken designs for three to five factors. Beyond five the
published designs use a balanced incomplete block design and are smaller than every pair taken in turn. The pairwise
construction stays valid there but is no longer minimal.

- **Assumption**: The response is approximated by a full quadratic model, and there are three or more factors.
- **Settings**: The centre run count $n_c$ (default 3 in the Appendix B implementation). The run count is $4\binom{k}{2} + n_c$.
- **Breaks when**: Beyond five factors the pairwise construction does not give the minimum run count.
- **Used for**: Optimising a process that cannot set every factor to an extreme at once.

## 8. D-Optimal Designs

A D-optimal design takes the model and the run budget as given and searches for the set of runs that minimises the
covariance of the coefficients by maximising the determinant of the information matrix $\mathbf{X}^{\top}\mathbf{X}$.
The cube of factor settings, the run count that is a power of 2 or a multiple of 4, and the factors free to move, all
presumed by the families above, are not required by a D-optimal design.

- **Assumption**: The stated model describes the response correctly. The design is optimal only for the coefficients of that model.
- **Settings**: The model (linear, interaction, quadratic), the run count, the grid level count of continuous factors, the number of random starts, and the candidate set.
- **Breaks when**: Fewer runs than model terms cannot estimate the model. With too few starts the search can stop at a design short of the optimum.
- **Used for**: Irregular run budgets, augmentation of staged experiments, recorded covariates, categorical factors, and constrained factor spaces.

### 8.1 D-Efficiency

D-efficiency normalises the determinant of the information matrix so that designs of different sizes can be compared. In
equation (3) $p$ is the number of model terms; D-efficiency is 1 for an orthogonal design and smaller otherwise.

```math
D = \frac{\left| \mathbf{X}^{\top}\mathbf{X} \right|^{1/p}}{n} \hspace{19em} (3)
```

The two-level full factorial reaches a D-efficiency of 1 for the linear model. When an orthogonal design exists, a
D-optimal search cannot find a better one.

### 8.2 Generating D-Optimal Designs

A D-optimal design comes from an iterative search. The coordinate-exchange algorithm starts from a random design and
repeats one move until nothing improves: take one factor of one run, try every value on a grid, and keep the best
[[5](#ref-5)]. The result depends on where the search starts, so the search is repeated from several random starts and the
best design is kept.

Twelve runs requested for a quadratic model in three factors give a design with D-efficiency 0.4498
([Appendix B.5](#b5-d-optimal-designs)). That is three runs fewer than the smallest Box-Behnken design of Table 4 and two
above the ten coefficients the model has.

### 8.3 Augmenting D-Optimal Designs

An experiment often runs in stages, and the runs already performed cannot be chosen again. Augmentation holds the
performed runs fixed and searches only for the new ones, so the added runs fill what the existing design is missing. The
search uses row exchange, which swaps whole runs against rows of a candidate set rather than single coordinates; when the
runs are drawn from a fixed list, exchanging rows is the fitting move.

The four corner runs alone cannot fit the six-term quadratic model. Two more runs make it estimable, and the search puts
both of them off the corners, where the existing design has nothing, at $(0, 1)$ and $(-1, 0)$.

### 8.4 Specifying Fixed Covariate Factors

A covariate is a factor that the experimenter records rather than sets, such as ambient temperature, the operator on
shift, or the age of a batch. Its values are known in advance for each run but cannot be chosen. The design problem is
then to choose the controlled factors given the covariate column, so that the model separates the controlled effects
from the covariate effect.

When the covariate drifts evenly from $-1$ to $+1$ across eight runs, the search places a pattern balanced against that
drift in the two controlled columns, so neither controlled effect is confounded with the covariate.

### 8.5 Specifying Categorical Factors

A categorical factor has no numeric scale, so its levels cannot be pushed towards $\pm 1$. A factor with $L$ levels enters
the model as $L - 1$ columns through effect coding, and the search runs on those columns.

Three categorical factors at three levels each give 27 candidate treatments. Asked for nine runs, the search returns a
design in which every level of every factor appears three times and every pair of levels from two factors appears exactly
once.

### 8.6 Specifying Candidate Sets

Row exchange takes a candidate set, the list of runs it may choose from. Where the factor space is a cube, the candidate
set is a grid over it.

Passing the candidate set explicitly handles factor spaces that are not cubes. A mixture whose components must sum to
one, a pair of settings that cannot be high together, and a machine that cannot run cold and fast are all constraints that
remove rows from the grid. The search never proposes a run that is not on the list, so deleting those rows from the
candidate set carries the constraint into the design.

## References

<a id="ref-1"></a>
[1] MathWorks. [Design of Experiments — Statistics and Machine Learning Toolbox Documentation](https://kr.mathworks.com/help/stats/design-of-experiments.html).<br>
<a id="ref-2"></a>
[2] Montgomery, D. C. (2019). *Design and Analysis of Experiments* (10th ed.). Wiley.
ISBN 978-1-119-49249-8.<br>
<a id="ref-3"></a>
[3] Plackett, R. L., & Burman, J. P. (1946). [The Design of Optimum Multifactorial Experiments](https://doi.org/10.1093/biomet/33.4.305).
*Biometrika*, 33(4), 305–325.<br>
<a id="ref-4"></a>
[4] Box, G. E. P., & Behnken, D. W. (1960). [Some New Three Level Designs for the Study of
Quantitative Variables](https://doi.org/10.1080/00401706.1960.10489912). *Technometrics*, 2(4), 455–475.<br>
<a id="ref-5"></a>
[5] Meyer, R. K., & Nachtsheim, C. J. (1995). [The Coordinate-Exchange Algorithm for Constructing
Exact Optimal Experimental Designs](https://doi.org/10.1080/00401706.1995.10485889). *Technometrics*, 37(1), 60–69.

---

## Appendix A. Terminology

- **Basic factor**: a factor whose column comes from the full factorial a fractional design starts
  from, rather than from a generator.
- **Candidate set**: the list of factor settings from which a D-optimal search may choose runs.
- **Centre run**: a run with every continuous factor at the middle of its range.
- **Confounding**: two effects carried by the same design column, so that no analysis of the
  experiment can separate them.
- **Covariate**: a factor that the experimenter records for each run rather than sets.
- **Defining relation**: the set of products of factor columns that equal the column of ones in a
  fractional factorial design; its shortest word gives the resolution.
- **Effect coding**: coding of a categorical factor with $L$ levels into $L - 1$ columns, with the last level set to $-1$ in every column.
- **Factor**: an input whose value the experimenter sets or records.
- **Generator**: the product of basic factors that defines an added factor in a fractional
  factorial design.
- **Hadamard matrix**: a square matrix of $\pm 1$ entries whose columns are mutually orthogonal.
- **Interaction**: the extent to which the effect of one factor depends on the level of another.
- **Level**: one of the values a factor takes in a design.
- **Main effect**: the change in the response when one factor changes level, averaged over the other factors.
- **Response**: the measured output of a run.
- **Rotatable**: a design whose prediction variance depends only on the distance from the centre of
  the factor space.
- **Run**: one execution of the experiment at one combination of factor levels; one row of a design.
- **Screening**: an experiment that picks out, from many factors, those with a significant effect on the response.
- **Star point**: a run of a central composite design that moves one factor off the centre and
  holds the rest at it.
- **Treatment**: a combination of factor levels.
- **Word**: a group of factor names whose product is the column of ones in the defining relation; its length is the number of factors in it.

## Appendix B. Python Implementation

The Python blocks of this appendix are meant to run in order. Each one uses the names defined before it, on top of the
header in B.1, and the `text` block right after it holds its output. Running them needs numpy and scipy.

### B.1 Common Header

```python
# Python
import itertools
from typing import Sequence

import numpy as np
from scipy.linalg import hadamard

np.set_printoptions(linewidth=100, suppress=True, precision=4)
```

### B.2 Full Factorial Designs

#### Multilevel Designs

The cartesian product construction of section 5.1, making six runs from a two-level and a three-level factor.

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

```text
[[0 0]
 [1 0]
 [0 1]
 [1 1]
 [0 2]
 [1 2]]
```

#### Two-Level Designs

The $2^k$ design of section 5.2, coded as $\pm 1$.

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

### B.3 Fractional Factorial Designs

#### Plackett-Burman Designs

The construction of section 6.2. The last line checks that the 11 columns of the 12-run design are mutually orthogonal. A
run count that is neither a power of 2 nor covered by a generator raises instead of falling back on a different design.

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

#### General Fractional Designs

The generator construction of section 6.3, making the eight-run design in four factors from the generators `a b c abc`.

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

#### Design Resolution

Computes the resolution of Table 3 as the shortest word in the subgroup generated by the words the generators imply.

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

### B.4 Response Surface Designs

#### Central Composite Designs

The three variants of section 7.2, chosen by `kind`. An invalid `kind` raises instead of falling back on a default, so a
misspelled variant cannot produce a design the caller did not ask for.

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

#### Box-Behnken Designs

The pairwise construction of section 7.3, making the fifteen runs of Table 4 from three factors and three centre runs.

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

### B.5 D-Optimal Designs

#### D-Efficiency

Computes equation (3) and checks the scale with the D-efficiency of 1 that the two-level full factorial reaches for the
linear model. Fewer runs than model terms raise an error.

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

#### Coordinate Exchange

The search of section 8.2, choosing twelve runs for a quadratic model in three factors. The first output line is the
D-efficiency.

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

#### Row Exchange and Augmentation

The search of section 8.3; rows passed in `fixed` are not exchanged. Two runs are added to the four corner runs.

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

```text
[[ 0.  1.]
 [-1.  0.]]
0.42
```

#### Fixed Covariates

The search of section 8.4; the last column is the evenly drifting covariate.

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

#### Categorical Factors

The effect coding and row exchange of section 8.5, with the chosen runs mapped back to their levels and sorted.

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

#### Candidate Sets

Builds the cubic grid of section 8.6 and chooses six runs for a quadratic model in two factors.

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
