# Scores in Partial Least Squares
Rev. 10 | Created: 2026-10-24 | Updated: 2026-10-26 05:46 UTC

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Comparison](#31-comparison)
- [4. Score Extraction](#4-score-extraction)
- [5. Dimension Reduction](#5-dimension-reduction)
- [6. Supervised Feature Extraction](#6-supervised-feature-extraction)
- [7. Multicollinearity and High Dimensionality](#7-multicollinearity-and-high-dimensionality)
- [8. Visualization and Anomaly Detection](#8-visualization-and-anomaly-detection)
  - [8.1 Score Plot](#81-score-plot)
  - [8.2 Hotelling's T-Squared on the Scores](#82-hotellings-t-squared-on-the-scores)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Comparison of PLS and OLS](#appendix-b-comparison-of-pls-and-ols)
  - [B.1 Ordinary Least Squares](#b1-ordinary-least-squares)
  - [B.2 Partial Least Squares](#b2-partial-least-squares)
  - [B.3 Summary](#b3-summary)
- [Appendix C. Derivation of Equations (1) to (11)](#appendix-c-derivation-of-equations-1-to-11)
  - [C.1 Equation (1)](#c1-equation-1)
  - [C.2 Equation (2)](#c2-equation-2)
  - [C.3 Equation (3)](#c3-equation-3)
  - [C.4 Equation (4)](#c4-equation-4)
  - [C.5 Equation (5)](#c5-equation-5)
  - [C.6 Equation (6)](#c6-equation-6)
  - [C.7 Equation (7)](#c7-equation-7)
  - [C.8 Equation (8)](#c8-equation-8)
  - [C.9 Equation (9)](#c9-equation-9)
  - [C.10 Equation (10)](#c10-equation-10)
  - [C.11 Equation (11)](#c11-equation-11)
- [Appendix D. Worked Example](#appendix-d-worked-example)

## 1. Purpose

- **Problem Statement**: 반도체 장비 센서나 스펙트럼 데이터는 입력 변수가 수백에서 수천 개이고, 그 변수들이 서로 강하게 상관되며, sample 수보다 많은 경우가 흔하다.
- **Goal**: PLS (Partial Least Squares, 부분최소제곱) 가 입력 $\mathbf{X}$ 와 타겟 $\mathbf{Y}$ 에서 score 를 추출하는 이유를 네 가지 쓸모로 나누어, 각 쓸모가 어떤 데이터 조건에서 필요한지를 독자가 가릴 수 있게 한다.
- **Non-Goal**: 성분 수 $K$ 를 교차 검증 (cross-validation) 으로 고르는 절차와 비선형 PLS 는 다루지 않는다.

## 2. Summary

PLS 가 score 를 추출하는 이유는 $\mathbf{X}$ 와 $\mathbf{Y}$ 사이의 공분산 (covariance) 을 최대로 하는 저차원 latent space 의 좌표값을 얻기 위해서다. 그 좌표값 하나로 차원 축소, $\mathbf{Y}$ 와 관련된 특징 추출, 다중공선성 (multicollinearity) 해소, 시각화와 이상 감지의 네 가지가 함께 이루어진다. 네 쓸모와 그것이 필요한 조건은 Table 1 에 있다.

## 3. Taxonomy and its Hierarchy

Latent variable 방법은 방향을 고를 때 $\mathbf{Y}$ 를 쓰는지와 무엇을 최대로 하는지의 두 축으로 갈리고, PLS 는 $\mathbf{Y}$ 를 쓰며 공분산을 최대로 하는 쪽에 놓인다. 두 축은 아래 Fig 1 과 같다.

```text
Axis 1. Supervision            Axis 2. Criterion maximized by a direction w
-------------------            ---------------------------------------------
unsupervised : X only          variance   : Var(Xw)
supervised   : X and Y         covariance : Cov(Xw, Yc)
```

<a id="fig-1"></a>
Fig 1. Two axes of latent variable methods

PCA 는 unsupervised 이고 variance 를 최대로 하며, PLS 는 supervised 이고 covariance 를 최대로 한다. 두 축의 값은 서로 독립이지만, 이 문서가 다루는 방법은 이 두 조합뿐이다.

OLS 에서 latent projection 으로 한 단계 내려가면 가정 하나를 버리는 대신 쓸 수 있는 데이터가 넓어진다. 그 계층은 아래 Fig 2 와 같다.

```text
OLS                       uses every column of X; needs X^T X invertible
 |
 +-- latent projection    regresses y on K scores instead of P columns
      |                   gives up: the full column space of X
      |                   gains   : works when P > N or columns are collinear
      |
      +-- PCR             directions by Var(Xw); y plays no part in the choice
      +-- PLS             directions by Cov(Xw, Yc); y steers every direction
```

<a id="fig-2"></a>
Fig 2. Hierarchy from OLS to latent projection

맨 위의 OLS 는 $\mathbf{X}$ 의 열을 모두 쓰므로 $\mathbf{X}^\top \mathbf{X}$ 의 역행렬이 있어야 한다. 한 단계 아래의 latent projection 은 열 $P$ 개 대신 score $K$ 개에 회귀하므로 역행렬이 필요 없지만, score 가 펼치는 부분공간 밖에 있는 $\mathbf{X}$ 의 정보는 버린다. 마지막 단계에서 PCR 과 PLS 는 방향을 고르는 기준만 다르며, PCR 은 $\mathbf{X}$ 의 variance 로, PLS 는 $\mathbf{Y}$ 와의 covariance 로 방향을 고른다. OLS 와 PLS 가 쓰이는 조건과 한계는 [Appendix B](#appendix-b-comparison-of-pls-and-ols) 에 있다.

### 3.1 Comparison

데이터 조건과 목적에 따라 쓰는 방법과 score 의 쓸모는 Table 1 과 같다.

Table 1. Method and use of the score by data condition

<table>
  <tr><th align="center">Method</th><th align="center">Condition</th><th align="center">Use of the score</th><th align="center">Section</th></tr>
  <tr><td align="center">OLS</td><td align="center">$N \gt P$, 열 사이의 상관이 낮음</td><td align="center">Score 불필요</td><td align="center">-</td></tr>
  <tr><td align="center" rowspan="4">PLS</td><td align="center">$P$ 가 수백 개 이상</td><td align="center">차원 축소</td><td align="center">section 5</td></tr>
  <tr><td align="center">$\mathbf{Y}$ 를 예측하는 것이 목적</td><td align="center">$\mathbf{Y}$ 와 관련된 특징 추출</td><td align="center">section 6</td></tr>
  <tr><td align="center">$P \gt N$ 이거나 열 사이의 상관이 높음</td><td align="center">다중공선성 해소</td><td align="center">section 7</td></tr>
  <tr><td align="center">Wafer·run 단위의 감시가 목적</td><td align="center">Score plot, Hotelling $T^2$</td><td align="center">section 8</td></tr>
  <tr><td align="center">PCA</td><td align="center">$\mathbf{Y}$ 가 없고 $\mathbf{X}$ 만 감시</td><td align="center">$\mathbf{X}$ 의 variance 요약</td><td align="center">section 6</td></tr>
</table>

$N$ 은 sample 수이고 $P$ 는 입력 변수의 수이다. PLS 의 네 행은 서로 배타적이지 않아서, 센서 데이터에서는 네 조건이 함께 성립하는 경우가 많다.

방법마다 가정, 설정값, 깨지는 조건, 쓰이는 곳은 Table 2 와 같다.

Table 2. Assumption, setting, failure condition and usage of each method

| Method | Assumption                                  | Setting                     | Breaks when                                               | Where used                                    |
| :---:  | :---:                                       | :---:                       | :---:                                                     | :---:                                         |
| PCA    | 큰 variance 방향에 쓸모 있는 정보가 있음    | 성분 수 $K$                 | $\mathbf{Y}$ 와 관련된 방향의 variance 가 작을 때         | $\mathbf{Y}$ 없는 장비 감시                   |
| PLS    | $\mathbf{X}$ 와 $\mathbf{Y}$ 의 관계가 선형 | 성분 수 $K$, 열의 centering | 관계가 비선형이거나 공정이 바뀌어 공분산 구조가 달라질 때 | 두께·수율 예측, 가상 계측 (virtual metrology) |

## 4. Score Extraction

PLS 의 score 는 $\mathbf{X}$ 와 $\mathbf{Y}$ 를 각각 한 방향으로 투영한 좌표이며, 두 좌표의 공분산이 최대가 되도록 방향을 고른다. $\mathbf{X} \in \mathbb{R}^{N \times P}$ 는 sample $N$ 개와 입력 변수 $P$ 개로 이루어진 입력 행렬이고, $\mathbf{Y} \in \mathbb{R}^{N \times M}$ 은 같은 sample 의 타겟 변수 $M$ 개로 이루어진 타겟 행렬이며, 두 행렬의 열은 평균을 빼서 centering 한다. 이 꼭지부터 꼭지 8 까지 나오는 식 (1) 에서 식 (11) 까지의 유도는 [Appendix C](#appendix-c-derivation-of-equations-1-to-11) 에 있다.

$\mathbf{w}$ 와 $\mathbf{c}$ 는 sample 을 투영할 방향을 나타내는 길이 1 인 벡터이고, $\mathbf{w}$ 는 $\mathbf{X}$ 쪽, $\mathbf{c}$ 는 $\mathbf{Y}$ 쪽 방향이다.

- $\mathbf{w} \in \mathbb{R}^P$ 는 입력 변수 $P$ 개에 하나씩 주는 가중치이다. $\mathbf{x}_i^\top$ 를 $\mathbf{X}$ 의 $i$ 번째 행, 곧 sample $i$ 의 입력값이라 하면 sample $i$ 의 X-score 는 $t_i = \mathbf{x}_i^\top \mathbf{w}$ 이고, 식 (1) 의 $\mathbf{t} = \mathbf{X}\mathbf{w}$ 는 이 값을 모든 sample 에 대해 모은 것이다.
- $\mathbf{c} \in \mathbb{R}^M$ 은 타겟 변수 $M$ 개에 하나씩 주는 가중치이고, 같은 방법으로 Y-score 는 $\mathbf{u} = \mathbf{Y}\mathbf{c}$ 이다.
- 두 벡터의 값은 식 (2) 에서 정해진다. 길이를 1 로 묶은 채 $\mathrm{Cov}(\mathbf{X}\mathbf{w}, \mathbf{Y}\mathbf{c})$ 를 최대로 하는 값이며, 타겟이 하나이고 그 열이 $\mathbf{y}$ 이면 $\mathbf{w} \propto \mathbf{X}^\top \mathbf{y}$ 이고 $\mathbf{c} = \pm 1$ 이므로, $\mathbf{w}$ 의 각 원소는 그 입력 변수와 $\mathbf{y}$ 의 공분산에 비례한다.

방향 $\mathbf{w} \in \mathbb{R}^{P}$ 와 $\mathbf{c} \in \mathbb{R}^{M}$ 으로 투영한 X-score $\mathbf{t}$ 와 Y-score $\mathbf{u}$ 는 아래와 같다.

```math
\mathbf{t} = \mathbf{X}\mathbf{w}, \qquad \mathbf{u} = \mathbf{Y}\mathbf{c} \hspace{19em} (1)
```

```math
(\mathbf{w}, \mathbf{c}) = \arg\max_{\lVert \mathbf{w} \rVert = \lVert \mathbf{c} \rVert = 1} \mathrm{Cov}(\mathbf{X}\mathbf{w}, \mathbf{Y}\mathbf{c}) \hspace{19em} (2)
```

첫 성분을 찾은 뒤에는 $\mathbf{t}$ 가 설명한 부분을 $\mathbf{X}$ 에서 빼고 (deflation), 남은 행렬에서 식 (2) 를 다시 풀어 다음 성분을 찾는다. $\mathbf{p}$ 는 $\mathbf{X}$ 의 열을 $\mathbf{t}$ 에 회귀한 계수인 loading 이다.

```math
\mathbf{p} = \frac{\mathbf{X}^\top \mathbf{t}}{\mathbf{t}^\top \mathbf{t}}, \qquad \mathbf{X} \leftarrow \mathbf{X} - \mathbf{t}\mathbf{p}^\top \hspace{19em} (3)
```

성분 $K$ 개를 찾으면 X-score 를 열로 모은 $\mathbf{T} \in \mathbb{R}^{N \times K}$ 와 Y-score 를 열로 모은 $\mathbf{U} \in \mathbb{R}^{N \times K}$ 를 얻고, $\mathbf{X}$ 와 $\mathbf{Y}$ 는 아래와 같이 분해된다. $\mathbf{P}$ 와 $\mathbf{Q}$ 는 각 행렬의 loading 을 열로 모은 행렬이고, $\mathbf{E}$ 와 $\mathbf{F}$ 는 성분 $K$ 개가 설명하지 못한 잔차이다.

```math
\mathbf{X} = \mathbf{T}\mathbf{P}^\top + \mathbf{E}, \qquad \mathbf{Y} = \mathbf{T}\mathbf{Q}^\top + \mathbf{F} \hspace{19em} (4)
```

새 sample 을 예측할 때는 $\mathbf{X}$ 만 있으므로, 식 (4) 는 $\mathbf{Y}$ 를 $\mathbf{U}$ 대신 $\mathbf{T}$ 로 분해한다. $\mathbf{U}$ 는 학습할 때 $\mathbf{T}$ 와의 공분산을 최대로 하는 짝으로만 쓰이며, 새 sample 의 예측값은 그 sample 의 $\mathbf{X}$ 에서 구한 score 에 $\mathbf{Q}^\top$ 를 곱한 값이다.

## 5. Dimension Reduction

PLS 는 입력 변수 $P$ 개를 score $K$ 개로 줄이며, $K$ 는 $P$ 보다 훨씬 작다. 반도체 장비 센서나 스펙트럼 데이터처럼 $P$ 가 수백에서 수천 개인 입력을 소수의 latent variable 좌표로 요약하는 것이다.

```math
\mathbf{X} \in \mathbb{R}^{N \times P} \;\longrightarrow\; \mathbf{T} \in \mathbb{R}^{N \times K}, \qquad K \ll P \hspace{19em} (5)
```

원래 입력 공간에서 sample 하나는 센서 $P$ 개의 값이고, score 공간에서는 핵심 요인 $K$ 개의 좌표값이다. 센서 수백 개는 gas 유량, RF power, chamber 온도 같은 소수의 공정 요인에 따라 함께 움직이는 경우가 많으므로, 그 요인의 수만큼의 $K$ 로 $\mathbf{X}$ 의 대부분을 식 (4) 의 $\mathbf{T}\mathbf{P}^\top$ 로 재구성할 수 있다. [Appendix D](#appendix-d-worked-example) 의 예제에서는 센서 300 개를 score 2 개로 줄여도 $\mathbf{y}$ 의 변동 95.6% 를 설명한다.

## 6. Supervised Feature Extraction

PCA 의 score 는 $\mathbf{X}$ 내부의 variance 만 최대로 하는 unsupervised 방식이고, PLS 의 score 는 $\mathbf{X}$ 와 $\mathbf{Y}$ 사이의 covariance 를 최대로 하는 supervised 방식이다. 두 방법이 방향 $\mathbf{w}$ 를 고르는 기준은 아래와 같다.

```math
\mathrm{PCA}: \max_{\lVert \mathbf{w} \rVert = 1} \mathrm{Var}(\mathbf{X}\mathbf{w}), \qquad \mathrm{PLS}: \max_{\lVert \mathbf{w} \rVert = 1} \mathrm{Cov}(\mathbf{X}\mathbf{w}, \mathbf{y}) \hspace{19em} (6)
```

PLS 의 기준인 공분산은 상관계수와 두 표준편차의 곱으로 나뉜다.

```math
\mathrm{Cov}(\mathbf{X}\mathbf{w}, \mathbf{y}) = \mathrm{Corr}(\mathbf{X}\mathbf{w}, \mathbf{y}) \cdot \sqrt{\mathrm{Var}(\mathbf{X}\mathbf{w})} \cdot \sqrt{\mathrm{Var}(\mathbf{y})} \hspace{19em} (7)
```

$\mathrm{Var}(\mathbf{y})$ 는 $\mathbf{w}$ 와 무관하므로, PLS 는 $\mathbf{y}$ 와의 상관과 $\mathbf{X}$ 의 variance 를 함께 키우는 방향을 고른다. PCA 는 식 (7) 의 상관 항이 없으므로 $\mathbf{y}$ 와 상관이 없더라도 variance 가 큰 방향을 먼저 고른다.

그 결과 PLS 는 $\mathbf{Y}$ (두께, 수율, 품질 등) 를 예측하는 데 직접 관련된 정보가 앞쪽 score 에 오도록 데이터의 좌표를 다시 배치한다. 두께에 관여하는 요인의 variance 가 가장 작은 [Appendix D](#appendix-d-worked-example) 의 예제에서, PCA 는 score 3 개를 모두 써야 $R^2 = 0.989$ 에 이르고 앞의 2 개로는 $R^2 = 0.080$ 이다. 같은 데이터에서 PLS 는 score 2 개로 $R^2 = 0.956$ 에 이른다.

## 7. Multicollinearity and High Dimensionality

PLS 는 서로 직교하는 score 에 회귀하므로, 입력 변수 사이의 상관이 높거나 $P \gt N$ 이어도 계수를 안정적으로 구한다. 센서 데이터에서는 변수 사이의 상관이 높고 feature 수 $P$ 가 sample 수 $N$ 보다 훨씬 많은 경우가 흔하다. 이때 OLS 의 $\mathbf{X}^\top \mathbf{X}$ 는 rank 가 $N$ 이하여서 $P \times P$ 행렬의 역행렬을 구할 수 없다.

식 (3) 의 deflation 을 거친 X-score 는 서로 정확히 직교한다.

```math
\mathbf{t}_j^\top \mathbf{t}_k = 0 \quad (j \neq k) \hspace{19em} (8)
```

$\mathbf{T}^\top \mathbf{T}$ 가 대각행렬이므로, $\mathbf{y}$ 를 $\mathbf{T}$ 에 회귀한 계수는 역행렬 없이 score 마다 따로 구해진다.

```math
q_k = \frac{\mathbf{t}_k^\top \mathbf{y}}{\mathbf{t}_k^\top \mathbf{t}_k}, \qquad k = 1, \ldots, K \hspace{19em} (9)
```

각 $q_k$ 는 다른 score 의 계수와 무관하게 정해지므로, 입력 변수가 서로 상관되어도 계수끼리 값을 주고받으며 흔들리지 않는다. [Appendix D](#appendix-d-worked-example) 의 예제에서 $\mathbf{X}^\top \mathbf{X}$ 의 rank 는 300 중 40 이고, PLS score 의 $\mathbf{T}^\top \mathbf{T}$ 에서 비대각 원소의 최댓값은 $2.10 \times 10^{-12}$ 로 대각 원소의 최솟값 $2.83 \times 10^{4}$ 에 비해 사실상 0 이다.

## 8. Visualization and Anomaly Detection

Score 는 각 sample (wafer 1 장, run 1 회) 이 저차원 latent space 의 어디에 있는지를 나타내는 좌표이므로, 그 좌표를 그리면 시각화가 되고 그 좌표의 거리를 재면 이상 감지가 된다.

### 8.1 Score Plot

Score plot 은 첫째 score $\mathbf{t}_1$ 을 가로축에, 둘째 score $\mathbf{t}_2$ 를 세로축에 두고 sample 을 점으로 찍은 그림이다. Wafer 나 장비 사이의 cluster 구조와 normal 운전에서 sample 이 지나는 궤적인 golden trace 를 한 장에서 볼 수 있다. PLS 의 score 는 $\mathbf{Y}$ 와 관련된 방향이므로, score plot 에서 떨어진 cluster 는 $\mathbf{Y}$ 가 다른 집단일 가능성이 높다.

### 8.2 Hotelling's T-Squared on the Scores

Score 공간에서 sample 이 normal 데이터의 중심에서 얼마나 떨어졌는지를 Hotelling $T^2$ 로 재고, 그 값이 관리 한계를 넘으면 공정이 관리 범위를 벗어난 것 (excursion) 으로 판정한다. Sample $i$ 의 $T^2$ 는 각 score 를 학습 데이터에서의 표준편차로 나누어 제곱한 값의 합이며, $s_k^2$ 는 학습 데이터에서 $k$ 번째 score 의 분산이다.

```math
T_i^2 = \sum_{k=1}^{K} \frac{t_{ik}^2}{s_k^2} \hspace{19em} (10)
```

학습에 쓰지 않은 새 sample 의 관리 한계는 유의수준 $\alpha$ 에서 $F$ 분포의 분위수로 정한다.

```math
T_{\mathrm{lim}}^2 = \frac{K (N^2 - 1)}{N (N - K)} \, F_{1-\alpha}(K, N - K) \hspace{19em} (11)
```

식 (8) 대로 score 가 서로 직교하므로 식 (10) 은 공분산 행렬의 역행렬 없이 score 마다 나누어 더하는 것으로 끝난다. $\alpha = 0.01$ 로 둔 [Appendix D](#appendix-d-worked-example) 의 예제에서 관리 한계는 10.96 이고, normal wafer 의 $T^2$ 는 3.82, 두께 요인을 벗어나게 한 wafer 의 $T^2$ 는 40.86 이다.

$T^2$ 는 score 가 펼치는 부분공간 안의 변화만 잰다. 식 (4) 의 잔차 $\mathbf{E}$ 에만 나타나는 변화, 곧 학습 데이터에 없던 방향으로 센서가 움직이는 경우는 $T^2$ 가 커지지 않으므로, 잔차의 크기를 재는 SPE 를 함께 감시한다.

## References

<a id="ref-1"></a>
[1] [Lagrange Multiplier Method](https://github.com/ykim2718/AIML/blob/main/Math/Optimization/Lagrange-Multiplier/lagrange-multiplier-ko.md). *AIML* repository, GitHub.

---

## Appendix A. Terminology

- **centering**: 각 열에서 그 열의 평균을 빼서 평균을 0 으로 만드는 처리.
- **cluster**: Score 공간에서 서로 가깝게 모인 sample 의 집단.
- **deflation**: 찾은 성분이 설명한 부분 $\mathbf{t}\mathbf{p}^\top$ 를 행렬에서 빼서, 다음 성분이 남은 정보에서 찾아지게 하는 단계.
- **eigenvector**: 행렬 $\mathbf{A}$ 에 곱해도 방향이 바뀌지 않는 벡터 $\mathbf{v}$ 이며, $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ 의 $\lambda$ 를 그 eigenvector 의 eigenvalue 라 한다.
- **excursion**: 공정 변수나 결과가 관리 범위를 벗어난 상태.
- **golden trace**: normal 운전에서 sample 이 score 공간이나 센서 값에서 지나는 기준 궤적.
- **Hotelling $T^2$**: 공분산으로 가중한 중심으로부터의 거리의 제곱이며, 다변량 데이터에서 sample 하나가 normal 운전의 분포에서 얼마나 드문지를 수 하나로 나타낸다.
- **Lagrange multiplier**: 제약조건 (constraint) 이 있는 최적화에서 제약조건마다 하나씩 두는 변수이며, 목적함수 (objective function) 에서 제약조건과의 곱을 뺀 식을 미분하여 0 으로 놓으면 제약조건 아래의 극값 조건을 얻는다 [[1](#ref-1)].
- **latent space**: Score 가 좌표가 되는 저차원 공간이며, 관측하지 않은 요인을 축으로 한다.
- **latent variable**: 직접 관측하지 않고 관측 변수의 선형결합으로 구한 변수이며, 이 문서에서는 score 하나가 latent variable 하나이다.
- **loading**: 원래 변수를 score 에 회귀한 계수이며, 각 변수가 그 성분에 얼마나 실려 있는지를 나타낸다.
- **OLS (Ordinary Least Squares)**: 잔차 제곱합을 최소로 하는 계수를 $(\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$ 로 구하는 회귀.
- **PCA (Principal Component Analysis)**: $\mathbf{X}$ 의 variance 를 최대로 하는 서로 직교하는 방향을 차례로 찾는 unsupervised 방법.
- **PCR (Principal Component Regression)**: PCA score 에 $\mathbf{y}$ 를 회귀하는 방법.
- **rank**: 행렬의 선형독립인 열의 수.
- **score**: Sample 을 한 방향으로 투영한 좌표값이며, PLS 에서는 X-score $\mathbf{t}$ 와 Y-score $\mathbf{u}$ 가 있다.
- **score plot**: 두 score 를 두 축으로 하여 sample 을 점으로 찍은 그림.
- **singular vector**: 행렬 $\mathbf{A}$ 에 대해 $\mathbf{A}\mathbf{c} = \sigma\mathbf{w}$ 와 $\mathbf{A}^\top\mathbf{w} = \sigma\mathbf{c}$ 를 함께 만족하는 길이 1 인 벡터 $\mathbf{w}$ 와 $\mathbf{c}$ 이며, 각각 왼쪽과 오른쪽 singular vector 라 하고 $\sigma \ge 0$ 을 singular value 라 한다.
- **SPE (Squared Prediction Error)**: 잔차 $\mathbf{E}$ 의 행 하나의 제곱합이며, score 부분공간 밖으로 벗어난 정도를 잰다.
- **supervised**: 방향을 고를 때 타겟 $\mathbf{Y}$ 를 쓰는 방식.
- **unsupervised**: 방향을 고를 때 입력 $\mathbf{X}$ 만 쓰는 방식.
- **Wishart distribution**: 정규분포를 따르는 벡터 여러 개로 구한 표본 공분산 행렬에 자유도를 곱한 행렬의 분포.
- **가상 계측 (virtual metrology)**: 계측 장비로 재지 않고 장비 센서 값에서 계측값을 예측하는 것.
- **과적합 (overfitting)**: Model 이 학습 데이터의 noise 까지 맞추어, 학습 데이터의 오차는 작지만 새 데이터의 예측 오차가 큰 상태.

## Appendix B. Comparison of PLS and OLS

PLS 와 OLS 는 회귀 계수를 구하는 대상이 다른 별개의 방법이다. OLS 는 입력 변수 $P$ 개에 $\mathbf{y}$ 를 직접 회귀하고, PLS 는 입력 변수를 score $K$ 개로 압축한 뒤 그 score 에 $\mathbf{y}$ 를 회귀한다.

### B.1 Ordinary Least Squares

OLS (Ordinary Least Squares, 일반최소제곱) 는 가장 전통적이고 기본적인 선형 회귀 방법이며, 잔차 (관측값과 예측값의 차이) 의 제곱합을 최소로 하는 회귀 계수를 구한다.

- 조건: 입력 변수의 수 $P$ 가 sample 수 $N$ 보다 적고, 변수 사이에 다중공선성이 없을 때 가장 잘 작동한다.
- 한계: 반도체 장비 센서 데이터처럼 입력 변수가 수백 개 이상이고 서로 강하게 상관되면, $\mathbf{X}^\top \mathbf{X}$ 의 역행렬을 구할 수 없거나 과적합 (overfitting) 이 일어난다. 앞의 경우는 계수를 하나로 정할 수 없고, 뒤의 경우는 학습 데이터에는 맞지만 새 데이터의 예측 오차가 크다.

### B.2 Partial Least Squares

PLS (Partial Least Squares, 부분최소제곱) 는 OLS 가 실패하는 고차원 데이터와 다중공선성이 심한 데이터를 위해 고안된, 차원 축소에 기반한 회귀 방법이다.

- 원리: 입력 변수를 그대로 쓰지 않고, $\mathbf{Y}$ 를 설명하는 정보가 가장 많은 소수의 latent variable (score) 로 압축한 뒤 회귀한다 (section 4, section 5).
- 장점: 변수 사이의 상관이나 noise 가 많아도 안정적으로 예측 model 을 만들며 (section 7), section 8 의 공정 모니터링과 식 (2) 의 $\mathbf{w}$ 로 각 입력 변수의 기여를 읽는 분석에도 쓰인다.
- OLS 와의 관계: 성분 수 $K$ 를 $\mathbf{X}$ 의 rank 까지 올리면 score 가 $\mathbf{X}$ 의 열 공간 전체를 펼치므로, $P \lt N$ 이고 $\mathbf{X}$ 의 열이 선형독립일 때 PLS 의 예측값은 OLS 의 예측값과 같다.

### B.3 Summary

두 방법의 차이는 Table 3 과 같다.

Table 3. OLS and PLS by data condition

| Method | Regresses y on           | Suits                                         | Fails when                                         |
| :---:  | :---:                    | :---:                                         | :---:                                              |
| OLS    | 입력 변수 $P$ 개         | 변수가 적고 서로의 상관이 낮은 데이터         | $P \gt N$ 이거나 변수 사이의 상관이 높을 때        |
| PLS    | Score $K$ 개 ($K \ll P$) | 변수가 매우 많고 서로 상관된 공정·센서 데이터 | $\mathbf{X}$ 와 $\mathbf{Y}$ 의 관계가 비선형일 때 |

## Appendix C. Derivation of Equations (1) to (11)

식 (1) 부터 식 (10) 까지는 centering 한 $\mathbf{X}$ 와 $\mathbf{Y}$ 에 내적, 최소제곱, Lagrange multiplier [[1](#ref-1)] 를 써서 유도하고, 식 (11) 은 정규분포와 Wishart distribution 의 성질로 유도한다. 아래에서 $\mathbf{x}_i^\top$ 는 $\mathbf{X}$ 의 $i$ 번째 행, 곧 sample $i$ 의 입력값이고, $\mathbf{X}_k$ 는 $k - 1$ 번의 deflation 을 거친 행렬이며 $\mathbf{X}_1 = \mathbf{X}$ 이다.

### C.1 Equation (1)

Score $t_i$ 는 sample $i$ 의 입력값 $\mathbf{x}_i$ 에서 방향 $\mathbf{w}$ 의 직선에 내린 수선의 발이 원점에서 떨어진 부호 있는 거리이며, 발이 $\mathbf{w}$ 와 같은 쪽에 있으면 양수, 반대쪽에 있으면 음수이다.

$\mathbf{x}_i$ 와 $\mathbf{w}$ 사이의 각을 $\theta$ 라 하면 원점에서 수선의 발까지의 부호 있는 거리는 $\lVert \mathbf{x}_i \rVert \cos\theta$ 이고, $\lVert \mathbf{w} \rVert = 1$ 이므로 이 값은 내적 $\mathbf{x}_i^\top \mathbf{w} = \lVert \mathbf{x}_i \rVert \lVert \mathbf{w} \rVert \cos\theta$ 와 같다. 따라서 $t_i = \mathbf{x}_i^\top \mathbf{w}$ 이고, 수선 $\mathbf{x}_i - t_i \mathbf{w}$ 는 $\mathbf{w}$ 와 직교한다.

$\mathbf{X}$ 의 $i$ 번째 행이 $\mathbf{x}_i^\top$ 이므로, $t_i$ 를 sample 마다 모은 열은 $\mathbf{X}\mathbf{w}$ 이다. $\mathbf{Y}$ 의 $i$ 번째 행을 길이 1 인 방향 $\mathbf{c}$ 위로 투영하면 같은 방법으로 $\mathbf{Y}\mathbf{c}$ 를 얻으므로, 두 score 는 아래와 같다.

```math
\mathbf{t} = \mathbf{X}\mathbf{w}, \qquad \mathbf{u} = \mathbf{Y}\mathbf{c} \hspace{19em} (1)
```

### C.2 Equation (2)

PLS 는 X-score 와 Y-score 가 함께 움직이는 정도, 곧 두 score 의 공분산이 가장 큰 방향을 고른다. $\mathbf{X}$ 와 $\mathbf{Y}$ 의 열을 centering 했으므로 $\mathbf{t}$ 와 $\mathbf{u}$ 의 평균도 0 이고, 식 (1) 을 넣으면 두 score 의 공분산은 아래와 같다.

```math
\mathrm{Cov}(\mathbf{t}, \mathbf{u}) = \frac{\mathbf{t}^\top \mathbf{u}}{N - 1} = \frac{\mathbf{w}^\top \mathbf{X}^\top \mathbf{Y} \mathbf{c}}{N - 1} \hspace{19em} (12)
```

식 (12) 는 $\mathbf{w}$ 나 $\mathbf{c}$ 의 길이를 $a$ 배 하면 값도 $a$ 배가 되므로, 길이를 묶지 않으면 최대값이 없다. 방향만 고르도록 두 벡터의 길이를 1 로 묶으면 PLS 가 푸는 문제는 아래와 같다.

```math
(\mathbf{w}, \mathbf{c}) = \arg\max_{\lVert \mathbf{w} \rVert = \lVert \mathbf{c} \rVert = 1} \mathrm{Cov}(\mathbf{X}\mathbf{w}, \mathbf{Y}\mathbf{c}) \hspace{19em} (2)
```

$N - 1$ 은 $\mathbf{w}$ 와 $\mathbf{c}$ 에 무관하므로, 식 (2) 는 아래 최적화 문제와 같다.

- 목적함수: $\mathbf{w}^\top \mathbf{X}^\top \mathbf{Y} \mathbf{c}$ 의 최대화
- 제약조건: $\mathbf{w}^\top \mathbf{w} = 1$ 과 $\mathbf{c}^\top \mathbf{c} = 1$ 의 두 개
- Lagrange multiplier: 두 제약조건에 하나씩 둔 $\lambda$ 와 $\mu$

목적함수에서 각 제약조건과 그 multiplier 의 곱을 뺀 $L = \mathbf{w}^\top \mathbf{X}^\top \mathbf{Y} \mathbf{c} - \lambda (\mathbf{w}^\top \mathbf{w} - 1) - \mu (\mathbf{c}^\top \mathbf{c} - 1)$ 을 $\mathbf{w}$ 와 $\mathbf{c}$ 로 미분하여 0 으로 놓는다.

```math
\mathbf{X}^\top \mathbf{Y} \mathbf{c} = 2\lambda \mathbf{w}, \qquad \mathbf{Y}^\top \mathbf{X} \mathbf{w} = 2\mu \mathbf{c} \hspace{19em} (13)
```

두 식의 왼쪽에 각각 $\mathbf{w}^\top$ 와 $\mathbf{c}^\top$ 를 곱하면 $2\lambda = 2\mu = \mathbf{w}^\top \mathbf{X}^\top \mathbf{Y} \mathbf{c}$ 이므로, 최대값 $\sigma$ 는 $\mathbf{X}^\top \mathbf{Y}$ 의 가장 큰 singular value 이고 $\mathbf{w}$ 와 $\mathbf{c}$ 는 그 singular value 의 왼쪽과 오른쪽 singular vector 이다.

타겟이 하나 ($M = 1$) 이면 $\mathbf{c}$ 는 길이 1 인 스칼라 $\pm 1$ 이고, 식 (13) 의 첫 식은 $\mathbf{X}^\top \mathbf{y} = \sigma \mathbf{w}$ 가 된다. 따라서 $\mathbf{w} = \mathbf{X}^\top \mathbf{y} / \lVert \mathbf{X}^\top \mathbf{y} \rVert$ 이며, 이것이 꼭지 4 의 $\mathbf{w} \propto \mathbf{X}^\top \mathbf{y}$ 이다.

### C.3 Equation (3)

첫 성분을 찾은 뒤에는 $\mathbf{t}$ 가 설명한 부분을 $\mathbf{X}$ 에서 빼야 다음 성분이 남은 정보에서 찾아진다. $\mathbf{t}$ 가 설명한 부분은 $\mathbf{X}$ 의 각 열을 $\mathbf{t}$ 하나에 회귀한 값 $\mathbf{t}\mathbf{p}^\top$ 이고, 계수 $\mathbf{p}$ 는 잔차 제곱합 $\lVert \mathbf{X} - \mathbf{t}\mathbf{p}^\top \rVert^2$ 를 최소로 하는 값이다. 잔차 제곱합을 $\mathbf{p}$ 로 미분하여 0 으로 놓으면 아래와 같다.

```math
-2\mathbf{X}^\top \mathbf{t} + 2\mathbf{p}\,\mathbf{t}^\top \mathbf{t} = \mathbf{0} \hspace{19em} (14)
```

식 (14) 를 $\mathbf{p}$ 에 대해 풀고, 구한 $\mathbf{t}\mathbf{p}^\top$ 를 $\mathbf{X}$ 에서 빼면 loading 과 deflation 은 아래와 같다.

```math
\mathbf{p} = \frac{\mathbf{X}^\top \mathbf{t}}{\mathbf{t}^\top \mathbf{t}}, \qquad \mathbf{X} \leftarrow \mathbf{X} - \mathbf{t}\mathbf{p}^\top \hspace{19em} (3)
```

Deflation 한 행렬 $\mathbf{X}_{k+1} = \mathbf{X}_k - \mathbf{t}_k \mathbf{p}_k^\top$ 는 $\mathbf{t}_k$ 와 직교한다.

```math
\mathbf{t}_k^\top \mathbf{X}_{k+1} = \mathbf{t}_k^\top \mathbf{X}_k - (\mathbf{t}_k^\top \mathbf{t}_k)\,\mathbf{p}_k^\top = \mathbf{t}_k^\top \mathbf{X}_k - \mathbf{t}_k^\top \mathbf{X}_k = \mathbf{0}^\top \hspace{19em} (15)
```

따라서 다음 성분은 $`\mathbf{t}_k`$ 가 설명한 부분이 빠진 $`\mathbf{X}_{k+1}`$ 에서 찾아진다.

### C.4 Equation (4)

식 (3) 의 deflation 을 $k = 1, \ldots, K$ 에 대해 차례로 더하면, 성분 $K$ 개가 설명한 부분과 마지막에 남은 행렬 $\mathbf{X}_{K+1}$ 로 $\mathbf{X}$ 가 나뉜다.

```math
\mathbf{X} = \mathbf{X}_1 = \sum_{k=1}^{K} \mathbf{t}_k \mathbf{p}_k^\top + \mathbf{X}_{K+1} = \mathbf{T}\mathbf{P}^\top + \mathbf{E}, \qquad \mathbf{E} = \mathbf{X}_{K+1} \hspace{19em} (16)
```

$\mathbf{Y}$ 도 같은 방법으로 각 $\mathbf{t}_k$ 에 회귀한 계수 $\mathbf{q}_k = \mathbf{Y}^\top \mathbf{t}_k / (\mathbf{t}_k^\top \mathbf{t}_k)$ 를 열로 모아 $\mathbf{Q}$ 로 두고, 남은 부분을 $\mathbf{F} = \mathbf{Y} - \mathbf{T}\mathbf{Q}^\top$ 로 둔다. C.8 에서 보이듯 $\mathbf{t}_k$ 가 서로 직교하므로, $\mathbf{Y}$ 를 성분마다 deflation 하든 원래 $\mathbf{Y}$ 를 그대로 쓰든 $\mathbf{q}_k$ 는 같다. 식 (16) 과 함께 쓰면 아래와 같다.

```math
\mathbf{X} = \mathbf{T}\mathbf{P}^\top + \mathbf{E}, \qquad \mathbf{Y} = \mathbf{T}\mathbf{Q}^\top + \mathbf{F} \hspace{19em} (4)
```

### C.5 Equation (5)

$\mathbf{t}_k = \mathbf{X}_k \mathbf{w}_k$ 이므로 deflation 은 $\mathbf{X}_k$ 의 오른쪽에 $P \times P$ 행렬을 곱하는 것과 같다.

```math
\mathbf{X}_{k+1} = \mathbf{X}_k - \mathbf{X}_k \mathbf{w}_k \mathbf{p}_k^\top = \mathbf{X}_k \left( \mathbf{I} - \mathbf{w}_k \mathbf{p}_k^\top \right) \hspace{19em} (17)
```

이것을 $`\mathbf{X}_1 = \mathbf{X}`$ 부터 되풀이하면 $`\mathbf{X}_k = \mathbf{X}\mathbf{G}_k`$ 이고, $`\mathbf{G}_k = (\mathbf{I} - \mathbf{w}_1 \mathbf{p}_1^\top) \cdots (\mathbf{I} - \mathbf{w}_{k-1} \mathbf{p}_{k-1}^\top)`$ 이며 $`\mathbf{G}_1 = \mathbf{I}`$ 이다. 따라서 $`\mathbf{t}_k = \mathbf{X}\mathbf{G}_k\mathbf{w}_k`$ 이고, $`\mathbf{r}_k = \mathbf{G}_k\mathbf{w}_k`$ 를 열로 모은 $`\mathbf{R} \in \mathbb{R}^{P \times K}`$ 로 아래를 얻는다.

```math
\mathbf{T} = \mathbf{X}\mathbf{R}, \qquad \mathbf{X} \in \mathbb{R}^{N \times P}, \;\; \mathbf{R} \in \mathbb{R}^{P \times K} \;\Rightarrow\; \mathbf{T} \in \mathbb{R}^{N \times K} \hspace{19em} (18)
```

모든 score 는 원래 $\mathbf{X}$ 의 선형결합이므로, 새 sample 의 score 도 그 sample 의 입력값에 $\mathbf{R}$ 을 곱해 구한다. $\mathbf{R}$ 의 열 수 $K$ 는 고른 성분 수이고 입력 변수의 수 $P$ 보다 훨씬 작게 고르므로, $\mathbf{X}$ 에서 $\mathbf{T}$ 로 가는 변환은 아래와 같다.

```math
\mathbf{X} \in \mathbb{R}^{N \times P} \;\longrightarrow\; \mathbf{T} \in \mathbb{R}^{N \times K}, \qquad K \ll P \hspace{19em} (5)
```

### C.6 Equation (6)

PCA 는 $\mathbf{X}\mathbf{w}$ 의 variance 를, PLS 는 $\mathbf{X}\mathbf{w}$ 와 $\mathbf{y}$ 의 공분산을 최대로 하는 길이 1 인 방향 $\mathbf{w}$ 를 고른다. $\mathbf{S} = \mathbf{X}^\top \mathbf{X} / (N - 1)$ 은 $\mathbf{X}$ 의 공분산 행렬이다.

- PCA
  - 목적함수: $\mathrm{Var}(\mathbf{X}\mathbf{w}) = \mathbf{w}^\top \mathbf{S} \mathbf{w}$ 의 최대화
  - 제약조건: $\mathbf{w}^\top \mathbf{w} = 1$
- PLS
  - 목적함수: $\mathrm{Cov}(\mathbf{X}\mathbf{w}, \mathbf{y}) = \mathbf{w}^\top \mathbf{X}^\top \mathbf{y} / (N - 1)$ 의 최대화
  - 제약조건: $\mathbf{w}^\top \mathbf{w} = 1$

두 목적함수와 제약조건을 한 줄씩 적으면 아래와 같다.

```math
\mathrm{PCA}: \max_{\lVert \mathbf{w} \rVert = 1} \mathrm{Var}(\mathbf{X}\mathbf{w}), \qquad \mathrm{PLS}: \max_{\lVert \mathbf{w} \rVert = 1} \mathrm{Cov}(\mathbf{X}\mathbf{w}, \mathbf{y}) \hspace{19em} (6)
```

PCA 는 제약조건에 Lagrange multiplier $\lambda$ 를 두고 $L = \mathbf{w}^\top \mathbf{S} \mathbf{w} - \lambda (\mathbf{w}^\top \mathbf{w} - 1)$ 을 미분하면 $\mathbf{S}\mathbf{w} = \lambda\mathbf{w}$ 이고, 이때 목적함수의 값은 $\mathbf{w}^\top \mathbf{S} \mathbf{w} = \lambda$ 이다. 따라서 PCA 의 첫 방향은 $\mathbf{S}$ 의 가장 큰 eigenvalue 의 eigenvector 이며, $\mathbf{y}$ 는 식에 들어 있지 않다.

PLS 는 Lagrange multiplier 없이 Cauchy–Schwarz 부등식으로 풀린다. 제약조건 $\mathbf{w}^\top \mathbf{w} = 1$ 아래에서 $\mathbf{w}^\top \mathbf{X}^\top \mathbf{y} \le \lVert \mathbf{w} \rVert \, \lVert \mathbf{X}^\top \mathbf{y} \rVert = \lVert \mathbf{X}^\top \mathbf{y} \rVert$ 이고, 등호는 $\mathbf{w}$ 가 $\mathbf{X}^\top \mathbf{y}$ 와 같은 방향일 때 성립하므로 C.2 의 $M = 1$ 결과와 같다.

### C.7 Equation (7)

$\mathbf{X}\mathbf{w}$ 와 $\mathbf{y}$ 의 상관계수는 공분산을 두 표준편차의 곱으로 나눈 값이다.

```math
\mathrm{Corr}(\mathbf{X}\mathbf{w}, \mathbf{y}) = \frac{\mathrm{Cov}(\mathbf{X}\mathbf{w}, \mathbf{y})}{\sqrt{\mathrm{Var}(\mathbf{X}\mathbf{w})} \, \sqrt{\mathrm{Var}(\mathbf{y})}} \hspace{19em} (19)
```

식 (19) 의 양변에 두 표준편차의 곱을 곱하면 공분산이 상관계수와 두 표준편차의 곱으로 나뉜다.

```math
\mathrm{Cov}(\mathbf{X}\mathbf{w}, \mathbf{y}) = \mathrm{Corr}(\mathbf{X}\mathbf{w}, \mathbf{y}) \cdot \sqrt{\mathrm{Var}(\mathbf{X}\mathbf{w})} \cdot \sqrt{\mathrm{Var}(\mathbf{y})} \hspace{19em} (7)
```

### C.8 Equation (8)

$`j \lt k`$ 일 때 $`\mathbf{t}_j^\top \mathbf{X}_k = \mathbf{0}^\top`$ 임을 $`k`$ 에 대한 귀납법으로 보인다. $`k = j + 1`$ 이면 식 (15) 의 결과이다. $`\mathbf{t}_j^\top \mathbf{X}_k = \mathbf{0}^\top`$ 이면 식 (17) 의 $`\mathbf{X}_{k+1} = \mathbf{X}_k (\mathbf{I} - \mathbf{w}_k \mathbf{p}_k^\top)`$ 에서 아래가 성립한다.

```math
\mathbf{t}_j^\top \mathbf{X}_{k+1} = \left( \mathbf{t}_j^\top \mathbf{X}_k \right) \left( \mathbf{I} - \mathbf{w}_k \mathbf{p}_k^\top \right) = \mathbf{0}^\top \hspace{19em} (20)
```

따라서 $j \lt k$ 이면 $\mathbf{t}_j^\top \mathbf{t}_k = \mathbf{t}_j^\top \mathbf{X}_k \mathbf{w}_k = 0$ 이고, $j \gt k$ 는 두 첨자를 바꾸어 같은 결과를 얻으므로 X-score 는 서로 직교한다.

```math
\mathbf{t}_j^\top \mathbf{t}_k = 0 \quad (j \neq k) \hspace{19em} (8)
```

### C.9 Equation (9)

$\mathbf{y}$ 를 $\mathbf{T}$ 에 회귀한 최소제곱 계수는 정규방정식 (normal equation) 의 해 $\mathbf{q} = (\mathbf{T}^\top \mathbf{T})^{-1} \mathbf{T}^\top \mathbf{y}$ 이다. 식 (8) 에서 $\mathbf{T}^\top \mathbf{T} = \mathrm{diag}(\mathbf{t}_1^\top \mathbf{t}_1, \ldots, \mathbf{t}_K^\top \mathbf{t}_K)$ 이므로, 역행렬은 각 대각 원소의 역수를 대각에 둔 행렬이다. $\mathbf{T}^\top \mathbf{y}$ 의 $k$ 번째 원소는 $\mathbf{t}_k^\top \mathbf{y}$ 이므로, $\mathbf{q}$ 의 $k$ 번째 원소는 아래와 같다.

```math
q_k = \frac{\mathbf{t}_k^\top \mathbf{y}}{\mathbf{t}_k^\top \mathbf{t}_k}, \qquad k = 1, \ldots, K \hspace{19em} (9)
```

### C.10 Equation (10)

Sample $`i`$ 의 score 를 모은 벡터 $`\mathbf{t}_{(i)} = (t_{i1}, \ldots, t_{iK})^\top`$ 의 Hotelling $`T^2`$ 는 score 의 공분산 행렬 $`\mathbf{S}_T`$ 로 가중한 거리의 제곱이다. $`\mathbf{X}`$ 를 centering 했으므로 score 의 평균은 0 이고, 아래와 같다.

```math
T_i^2 = \mathbf{t}_{(i)}^\top \mathbf{S}_T^{-1} \mathbf{t}_{(i)}, \qquad \mathbf{S}_T = \frac{\mathbf{T}^\top \mathbf{T}}{N - 1} \hspace{19em} (21)
```

식 (8) 에서 $\mathbf{S}_T = \mathrm{diag}(s_1^2, \ldots, s_K^2)$ 이고 $s_k^2 = \mathbf{t}_k^\top \mathbf{t}_k / (N - 1)$ 이다. 식 (21) 에 대각행렬의 역행렬을 넣어 이차형식을 풀면 score 마다 나누어 더한 합이 된다.

```math
T_i^2 = \sum_{k=1}^{K} \frac{t_{ik}^2}{s_k^2} \hspace{19em} (10)
```

### C.11 Equation (11)

학습 sample $N$ 개와 새 sample 의 score 가 서로 독립이고 같은 $K$ 변량 정규분포 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ 를 따른다고 둔다. 새 sample 의 score $\mathbf{t}_{\mathrm{new}}$ 와 학습 score 의 평균 $\bar{\mathbf{t}}$ 의 차는 아래 분포를 따르고, $(N - 1)\mathbf{S}_T$ 는 자유도 $N - 1$ 의 Wishart distribution 을 따르며 그 차와 독립이다.

```math
\mathbf{t}_{\mathrm{new}} - \bar{\mathbf{t}} \sim \mathcal{N}\!\left( \mathbf{0}, \; \left( 1 + \tfrac{1}{N} \right) \boldsymbol{\Sigma} \right) \hspace{19em} (22)
```

$\mathbf{d} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ 와 그와 독립인 자유도 $m$ 의 표본 공분산 행렬 $\mathbf{S}$ 에 대해 $\mathbf{d}^\top \mathbf{S}^{-1} \mathbf{d} \sim \frac{mK}{m - K + 1} F(K, m - K + 1)$ 이다. 식 (22) 의 차를 $\sqrt{1 + 1/N}$ 로 나누면 $\mathbf{d}$ 의 조건을 만족하므로, $m = N - 1$ 을 넣고 다시 $1 + 1/N = (N + 1)/N$ 을 곱하면 아래를 얻는다.

```math
T_{\mathrm{new}}^2 \sim \frac{N + 1}{N} \cdot \frac{(N - 1)K}{N - K} \, F(K, N - K) = \frac{K (N^2 - 1)}{N (N - K)} \, F(K, N - K) \hspace{19em} (23)
```

학습 score 의 평균 $\bar{\mathbf{t}}$ 는 0 이므로 $T_{\mathrm{new}}^2$ 는 새 sample 의 score 를 식 (10) 에 넣은 값과 같다. Score 의 방향 $\mathbf{R}$ 은 학습 데이터로 정한 값이므로, 식 (23) 의 분포는 $\mathbf{R}$ 을 고정된 값으로 볼 때 성립한다. 그 분포의 $1 - \alpha$ 분위수가 관리 한계이다.

```math
T_{\mathrm{lim}}^2 = \frac{K (N^2 - 1)}{N (N - K)} \, F_{1-\alpha}(K, N - K) \hspace{19em} (11)
```

## Appendix D. Worked Example

아래 code 는 숨은 요인 3 개가 센서 300 개를 움직이는 wafer 40 장의 데이터를 만들고, 그 가운데 variance 가 가장 작은 요인 하나가 두께를 정하게 한다. 이 데이터로 section 5 의 차원 축소, section 6 의 PCA 와 PLS 비교, section 7 의 rank 와 직교성, section 8.2 의 $T^2$ 판정을 차례로 보인다. 실행 환경은 NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1 이다.

```python
# Python
import numpy as np
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# 1. Sensor data: 300 sensors driven by three hidden factors, 40 wafers
rng = np.random.default_rng(0)
n_wafers, n_sensors = 40, 300

factors = rng.standard_normal((n_wafers, 3)) * [4.0, 2.0, 1.5]  # factor 1 has the largest spread
loadings = rng.standard_normal((3, n_sensors))
X = factors @ loadings + 0.5 * rng.standard_normal((n_wafers, n_sensors))

# Only factor 3, the one with the smallest spread, drives the thickness
y = 2.0 * factors[:, 2] + 0.3 * rng.standard_normal(n_wafers)

# 2. OLS cannot invert X^T X when sensors outnumber wafers
print(f"X: {X.shape}, rank of X^T X: {np.linalg.matrix_rank(X.T @ X)} of {n_sensors}")
print("-" * 60)

# 3. How much of y the first k scores explain, PCA against PLS
X_centered = X - X.mean(axis=0)
pca_scores = PCA(n_components=3).fit_transform(X_centered)
pls_scores = PLSRegression(n_components=3, scale=False).fit(X, y).x_scores_

for k in range(1, 4):
    r2_pca = LinearRegression().fit(pca_scores[:, :k], y).score(pca_scores[:, :k], y)
    r2_pls = LinearRegression().fit(pls_scores[:, :k], y).score(pls_scores[:, :k], y)
    print(f"first {k} score(s): R^2 of y  PCA {r2_pca:.3f}  PLS {r2_pls:.3f}")
print("-" * 60)

# 4. PLS scores are mutually orthogonal: T^T T is diagonal
gram = pls_scores.T @ pls_scores
off_diagonal = np.abs(gram - np.diag(np.diag(gram))).max()
print(f"largest off-diagonal of T^T T: {off_diagonal:.2e}, smallest diagonal: {np.diag(gram).min():.2e}")
print("-" * 60)

# 5. Hotelling T^2 on two PLS scores, with an excursion wafer added
n_components = 2
monitor = PLSRegression(n_components=n_components, scale=False).fit(X, y)
score_variance = monitor.x_scores_.var(axis=0, ddof=1)

normal_wafer = rng.standard_normal(3) * [4.0, 2.0, 1.5] @ loadings + 0.5 * rng.standard_normal(n_sensors)
excursion_wafer = normal_wafer + 6.0 * loadings[2]  # factor 3 pushed far outside its usual range
new_scores = monitor.transform(np.vstack([normal_wafer, excursion_wafer]))
t_squared = (new_scores ** 2 / score_variance).sum(axis=1)

alpha = 0.01
limit = (n_components * (n_wafers ** 2 - 1) / (n_wafers * (n_wafers - n_components))
         * stats.f.ppf(1 - alpha, n_components, n_wafers - n_components))
print(f"T^2 limit (alpha = {alpha}): {limit:.2f}")
for name, value in zip(["normal", "excursion"], t_squared):
    print(f"{name:>9} wafer: T^2 = {value:7.2f}  {'OUT' if value > limit else 'in'}")
```

실행 결과는 아래와 같다.

```text
X: (40, 300), rank of X^T X: 40 of 300
------------------------------------------------------------
first 1 score(s): R^2 of y  PCA 0.038  PLS 0.224
first 2 score(s): R^2 of y  PCA 0.080  PLS 0.956
first 3 score(s): R^2 of y  PCA 0.989  PLS 0.989
------------------------------------------------------------
largest off-diagonal of T^T T: 2.10e-12, smallest diagonal: 2.83e+04
------------------------------------------------------------
T^2 limit (alpha = 0.01): 10.96
   normal wafer: T^2 =    3.82  in
excursion wafer: T^2 =   40.86  OUT
```

- 첫 줄: 센서 300 개에 wafer 40 장이므로 $\mathbf{X}^\top \mathbf{X}$ 의 rank 가 40 에 그쳐 OLS 의 역행렬을 구할 수 없다.
- $R^2$ 세 줄: PCA 는 variance 가 큰 두 요인을 먼저 골라 앞의 score 2 개로 $R^2 = 0.080$ 이고, PLS 는 두께와의 공분산으로 방향을 골라 score 2 개로 $R^2 = 0.956$ 이다. Score 3 개를 모두 쓰면 두 방법이 같은 부분공간을 펼치므로 $R^2$ 가 같다.
- 직교성 줄: PLS score 의 비대각 원소가 대각 원소에 비해 부동소수점 오차 수준이어서 식 (8) 이 성립한다.
- $T^2$ 세 줄: 두께 요인을 평소 범위 밖으로 민 wafer 만 식 (11) 의 한계를 넘는다.
