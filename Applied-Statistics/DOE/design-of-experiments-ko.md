# Design of Experiments
Rev. 6 | Created: 2026-09-04 | Updated: 2026-10-09 19:38 CDT

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

- **Problem Statement**: 수동적으로 모은 자료에서는 여러 factor 가 반응의 같은 변화를 함께 설명하므로, 그 자료에 맞춘 model 이 factor 마다의 효과를 갈라내지 못한다.
- **Goal**: DOE User's Guide 를 만든다. 다 읽은 독자가 목표 model, run 예산, factor 공간의 제약에 맞는 design 을 고르고 그 design 의 가정과 한계를 말할 수 있어야 한다.
- **Non-Goal**: 실험을 마친 뒤의 분석 (ANOVA, model 적합과 검정) 과 특정 software 의 사용법은 다루지 않는다.

## 2. Summary

실험계획은 run 예산 안에서 계수 추정값의 공분산이 작아지도록 design matrix 의 행을 고르는 일이다.
Design 은 목표 model 로 고른다. 모든 효과는 full factorial, 많은 factor 의 screening 은 fractional factorial,
곡률이 있는 최적화는 response surface design, 불규칙한 예산·제약·covariate 는 D-optimal design 이 맡는다.

## 3. Taxonomy and its Hierarchy

Design 계열은 model 에 두는 가정으로 갈리며, 가정이 강할수록 run 이 적게 든다. 이 문서는 MathWorks Statistics and
Machine Learning Toolbox 가 나누는 네 계열을 따른다 [[1](#ref-1)].

Design 을 가르는 다섯 축은 [Fig 1](#fig-1) 과 같다.

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

다섯 축은 서로 독립이다. 목표 model 이 같은 완전 이차 model 이어도 factor 공간이 정육면체이면 response surface
design 을 쓰고, 제약이 있는 영역이면 D-optimal design 을 쓴다.

각 design 이 model 에 두는 가정과 그 대가는 [Fig 2](#fig-2) 의 계층을 따른다.

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

Full factorial 에서 한 단계 내려갈 때마다 model 에 두는 가정이 늘고 run 수가 준다. General fractional design 은
고차 교호작용을, Plackett-Burman design 은 모든 교호작용을 작다고 보는 대가로 run 을 줄인다. Response surface
계열은 model 의 차수를 이차로 묶고 세 번째 수준을 더해 곡률을 추정한다. D-optimal design 은 명시한 model 하나에
맞춰 run 을 고르므로 정육면체 모양의 factor 공간도, 정해진 run 수도 요구하지 않는다.

### 3.1 Placement

Design 을 고르는 기준은 Table 1 과 같다.

Table 1. Placement of designs

| Design             | Model aimed at         | Run count                       | Use when                                                        |
| :----------------: | :--------------------: | :-----------------------------: | :-------------------------------------------------------------: |
| Full factorial     | 모든 효과와 교호작용   | $N_1 \times \cdots \times N_k$  | factor 가 적고 모든 조합을 감당할 수 있다                       |
| General fractional | 주효과와 일부 교호작용 | $2^{b}$                         | 많은 factor 를 screening 하며 resolution 을 고른다              |
| Plackett-Burman    | 주효과                 | $k$ 보다 큰 가장 작은 4 의 배수 | 주효과만 보는 screening 에 2 의 거듭제곱 사이 run 수가 필요하다 |
| Central composite  | 완전 이차 model        | $2^k + 2k + n_c$                | 알려진 작업점 근처에서 최적화한다                               |
| Box-Behnken        | 완전 이차 model        | $4\binom{k}{2} + n_c$           | 모든 factor 를 동시에 극단에 둘 수 없는 공정을 최적화한다       |
| D-optimal          | 명시한 임의의 model    | 예산에 맞춰 지정                | 예산이 불규칙하거나 제약, covariate, categorical factor 가 있다 |

$k$ 는 factor 수, $b$ 는 basic factor 수, $n_c$ 는 centre run 수이다.

## 4. Principle

Design 은 반응을 재기 전에 계수 추정의 정밀도를 정한다. 측정한 $n$ 개의 반응 vector 를 $\mathbf{y}$, factor
설정으로 만든 model matrix 를 $\mathbf{X}$, 계수를 $\boldsymbol{\beta}$ 라 하면, 최소제곱추정량과 그 공분산은
식 (1) 과 같다.

```math
\hat{\boldsymbol{\beta}} = \left( \mathbf{X}^{\top}\mathbf{X} \right)^{-1} \mathbf{X}^{\top}\mathbf{y}, \qquad \mathrm{Cov}\left[ \hat{\boldsymbol{\beta}} \right] = \sigma^{2} \left( \mathbf{X}^{\top}\mathbf{X} \right)^{-1} \hspace{19em} (1)
```

식 (1) 의 공분산에는 $\mathbf{y}$ 가 없으므로, $\sigma^2$ 이 주어지면 design 만으로 정해진다. 계획된 실험은 factor
값을 실험자가 정하여 factor 들이 서로 독립으로 움직이게 하고, 그 결과 반응에 대한 각 factor 의 효과를 따로
추정한다. Design 계열들은 어떤 model 을 겨냥하는지와 run 예산을 어떻게 쓰는지에서 다르다 (Table 1).

이 문서에서 두 수준 factor 는 $-1$ 과 $+1$ 로 부호화하고, 연속 factor 는 작업 범위가 $[-1, 1]$ 이 되도록 척도를
맞춘다. 정의 없이 쓴 용어는 [Appendix A](#appendix-a-terminology) 에, 모든 design 의 Python 구현과 출력은
[Appendix B](#appendix-b-python-implementation) 에 있다.

## 5. Full Factorial Designs

Full factorial design 은 factor 수준의 모든 조합에서 반응을 재어, factor 들이 만들 수 있는 모든 효과와 교호작용을
추정한다. 수준이 $N_1, \ldots, N_k$ 이면 treatment 마다 하나씩 $N_1 \times \cdots \times N_k$ 개의 run 이 든다.

- **가정**: factor 와 수준이 적어 모든 조합의 run 을 감당할 수 있다.
- **설정값**: factor 마다의 수준 수 $N_i$.
- **깨지는 조건**: factor 가 늘면 run 수가 수준 수의 곱으로 늘고, 두 수준이어도 $2^k$ 이 예산을 넘는다.
- **쓰는 곳**: factor 가 적은 실험, 그리고 fractional 계열과 response surface 계열의 출발점.

### 5.1 Multilevel Designs

Factor 마다 수준 수가 달라도 된다. Design 은 수준 집합들의 곱집합이며, 첫 열이 가장 빨리 변하도록 나열한다. 두
수준 factor 하나와 세 수준 factor 하나는 run 6 개를 준다 ([Appendix B.2](#b2-full-factorial-designs)).

### 5.2 Two-Level Designs

모든 factor 가 두 수준이면 run 은 $2^k$ 개이다. 열들이 직교이고 균형을 이루므로, 모든 효과를 서로 독립으로, 그리고
그 run 수에서 가능한 가장 작은 분산으로 추정한다. Fractional 계열과 response surface 계열은 이 design 에서
출발한다.

## 6. Fractional Factorial Designs

Fractional factorial design 은 treatment 의 부분집합만 재되, 중요하다고 보는 효과는 추정할 수 있도록 부분집합을
고른다. Full factorial 의 $2^k$ 개 run 가운데 대부분은 좀처럼 크지 않은 고차 교호작용을 추정하는 데 쓰인다.

### 6.1 Confounding and Resolution

Run 을 줄인 대가는 confounding 이다. 남은 각 열은 여러 효과의 합을 지니며, 그 design 으로 한 실험은 그 효과들을
갈라내지 못한다. Resolution 은 confounding 의 정도를 나타내며, defining relation 에서 가장 짧은 word 의 길이이다.

Table 2. Design resolution [[2](#ref-2)]

| Resolution | Main effects confounded with | Two-way interactions confounded with |
| :--------: | :--------------------------: | :----------------------------------: |
| III        | 2차 교호작용                 | 서로                                 |
| IV         | 3차 교호작용                 | 서로                                 |
| V          | 4차 교호작용                 | 3차 교호작용                         |

Resolution III 은 2차 교호작용이 작아야 주효과를 믿을 수 있고, IV 는 주효과를 2차 교호작용과 갈라 추정하며, V 는
주효과와 2차 교호작용을 모두 다른 2차 이하 효과와 갈라 추정한다.

### 6.2 Plackett-Burman Designs

Plackett-Burman design 은 주효과만 유의하다고 볼 때 run 수가 2 의 거듭제곱이 아니라 4 의 배수인 resolution III
design 을 주어, 두 수준 factorial 사이의 빈틈을 메운다 [[3](#ref-3)]. Factor 11 개를 run 16 개가 아니라 12 개로
screening 한다.

Design 은 Hadamard matrix 로 만든다. Run 수가 2 의 거듭제곱이면 Hadamard matrix 의 열을 그대로 쓰고, 아니면
알려진 generator 행의 순환행렬 아래에 $-1$ 로 채운 행 하나를 붙인다. Run 12 개 design 의 열 11 개는 서로
직교한다 ([Appendix B.3](#b3-fractional-factorial-designs)).

- **가정**: 교호작용은 무시할 만하고 주효과만 유의하다.
- **설정값**: factor 수 $k$. Run 수는 $k$ 보다 큰 가장 작은 4 의 배수이며, 그 수가 2 의 거듭제곱이 아니면 generator 행이 있어야 한다.
- **깨지는 조건**: 2차 교호작용이 크면 주효과와 confound 되어, 주효과 추정값이 그 교호작용을 함께 지닌다.
- **쓰는 곳**: 많은 factor 의 screening.

### 6.3 General Fractional Designs

General fractional design 은 basic factor 들로 이루어진 full factorial 에서 출발해, 나머지 factor 를 basic factor
의 곱으로 정의한다. 한 factor 를 정의하는 곱이 그 factor 의 generator 이며, generator 가 design 과 그 confounding
을 함께 정한다. Generator `a b c abc` 는 factor 4 개를 full factorial 이 요구하는 16 개의 절반인 run 8 개로
다룬다.

Defining relation 은 generator 가 함의하는 word 들이 생성하며, resolution 은 그 word 들이 생성하는 부분군에서
가장 짧은 word 의 길이이다. 네 generator 집합의 resolution 은 Table 3 과 같다.

Table 3. Resolution of four generator sets

| Generators   | Runs  | Resolution |
| :----------: | :---: | :--------: |
| a b ab       | 4     | III        |
| a b c ab     | 8     | III        |
| a b c abc    | 8     | IV         |
| a b c d abcd | 16    | V          |

Run 8 개짜리 두 design 은 비용이 같고 generator 만 다르지만, `a b c ab` 는 주효과를 2차 교호작용과 confound
시키고 `a b c abc` 는 그러지 않는다. 같은 run 수에서 resolution 은 generator 가 정한다.

- **가정**: design 의 resolution 이 confound 시키는 교호작용은 작다 (Table 2).
- **설정값**: basic factor 수 $b$ (run 수 $2^{b}$) 와 더해지는 factor 마다의 generator.
- **깨지는 조건**: confound 된 교호작용이 크면 그 열의 추정값은 여러 효과의 합이 되어, 어느 효과인지 가릴 수 없다.
- **쓰는 곳**: 많은 factor 의 screening 에서 run 수를 2 의 거듭제곱으로 두고 resolution 을 골라야 할 때.

## 7. Response Surface Designs

Response surface design 은 각 factor 에 세 번째 수준을 더해, 곡률을 지닌 완전 이차 model 을 맞춘다. 중요한
factor 를 알고 목표가 최적화이면 model 에 곡률이 있어야 한다. 최적점은 정류점이고, 일차 model 에는 정류점이 없다.

### 7.1 Quadratic Model

Factor 가 $k$ 개인 완전 이차 model 은 식 (2) 이며, 계수는 $(k+1)(k+2)/2$ 개이다.

```math
y = \beta_0 + \sum_{i=1}^{k} \beta_i x_i + \sum_{i \lt j} \beta_{ij} x_i x_j + \sum_{i=1}^{k} \beta_{ii} x_i^{2} + \varepsilon \hspace{19em} (2)
```

두 수준 design 으로는 식 (2) 를 맞출 수 없다. 제곱항 $x_i^2$ 이 $-1$ 과 $+1$ 에서 같은 값 1 을 갖는다. 아래 두
design 은 세 번째 수준을 서로 다른 방식으로 더한다.

### 7.2 Central Composite Designs

Central composite design 은 정육면체 꼭짓점의 두 수준 factorial, factor 축 위에서 중심으로부터 거리 $\alpha$ 인
star point, 그리고 하나 이상의 centre run 으로 이루어진다. $\alpha = (2^k)^{1/4}$ 로 잡으면 design 이 rotatable
이 되어, 예측분산이 방향과 관계없이 중심으로부터의 거리에만 의존한다.

세 변형은 star point 를 두는 위치에서 다르다.

- **Circumscribed**: star point 를 정육면체 바깥 $\pm\alpha$ 에 두므로, 두 수준 범위를 넘는 factor 설정이 필요하다.
- **Faced**: $\alpha = 1$ 로 하여 star point 를 정육면체의 면 위에 둔다.
- **Inscribed**: circumscribed 의 모양을 유지하되, 꼭짓점이 아니라 star point 가 $\pm 1$ 에 오도록 척도를 줄인다.

Factor 2 개와 centre run 1 개의 circumscribed design 은 run 9 개이며, star point 는 $\pm 1.4142$ 에 놓인다
([Appendix B.4](#b4-response-surface-designs)).

- **가정**: 작업점 근처에서 반응이 완전 이차 model 로 근사되고, factor 가 2 개 이상이다.
- **설정값**: $\alpha$ (rotatable 이면 $(2^k)^{1/4}$, faced 이면 1), 변형, centre run 수 $n_c \ge 1$. Run 수는 $2^k + 2k + n_c$.
- **깨지는 조건**: 공정이 $\pm\alpha$ 설정을 낼 수 없으면 circumscribed design 을 쓸 수 없어, faced 나 inscribed 로 바꾼다.
- **쓰는 곳**: 알려진 작업점 근처에서의 최적화.

### 7.3 Box-Behnken Designs

Box-Behnken design 은 완전 이차 model 을 맞추면서 두 factor 를 동시에 극단에 두지 않는다 [[4](#ref-4)]. Factor
짝마다 두 수준 factorial 을 두고 나머지 factor 는 중심에 두므로, 점들이 design 공간의 모서리 중점과 중심에
놓인다. 정육면체의 꼭짓점이 빠져 모든 factor 의 극단을 묶은 run 이 없고, 안에 박힌 factorial design 도 없다.

Table 4. Run counts of the two response surface designs, three centre runs each

| Factors | Quadratic terms | Box-Behnken | Central composite |
| :-----: | :-------------: | :---------: | :---------------: |
| 3       | 10              | 15          | 17                |
| 4       | 15              | 27          | 27                |
| 5       | 21              | 43          | 45                |

짝 단위 구성은 factor 3 개에서 5 개까지 발표된 Box-Behnken design 을 그대로 재현한다. 5 개를 넘으면 발표된
design 은 balanced incomplete block design 을 써서 모든 짝을 차례로 도는 구성보다 작다. 짝 단위 구성은 그때도
타당하지만 최소가 아니다.

- **가정**: 반응이 완전 이차 model 로 근사되고, factor 가 3 개 이상이다.
- **설정값**: centre run 수 $n_c$ (Appendix B 구현의 기본값 3). Run 수는 $4\binom{k}{2} + n_c$.
- **깨지는 조건**: factor 가 5 개를 넘으면 짝 단위 구성은 최소 run 수를 주지 못한다.
- **쓰는 곳**: 모든 factor 를 동시에 극단에 둘 수 없는 공정의 최적화.

## 8. D-Optimal Designs

D-optimal design 은 model 과 run 예산을 주어진 것으로 받고, 정보행렬 $\mathbf{X}^{\top}\mathbf{X}$ 의 행렬식을
최대로 하여 계수의 공분산을 최소로 하는 run 집합을 찾는다. 앞의 계열들이 전제하는 정육면체 모양의 factor 설정,
2 의 거듭제곱이거나 4 의 배수인 run 수, 자유롭게 움직이는 factor 를 D-optimal design 은 요구하지 않는다.

- **가정**: 명시한 model 이 반응을 맞게 기술한다. Design 은 그 model 의 계수에 대해서만 최적이다.
- **설정값**: model (linear, interaction, quadratic), run 수, 연속 factor 의 grid 수준 수, 무작위 출발점 수, candidate set.
- **깨지는 조건**: run 수가 model 항 수보다 적으면 model 을 추정할 수 없다. 출발점이 적으면 탐색이 최적에 못 미치는 design 에 머물 수 있다.
- **쓰는 곳**: 불규칙한 run 예산, 단계 실험의 augmentation, 기록만 하는 covariate, categorical factor, 제약이 있는 factor 공간.

### 8.1 D-Efficiency

D-efficiency 는 정보행렬의 행렬식을 정규화하여 크기가 다른 design 끼리 견주게 한다. 식 (3) 에서 $p$ 는 model 항
수이며, D-efficiency 는 직교 design 에서 1 이고 그 밖에서는 1 보다 작다.

```math
D = \frac{\left| \mathbf{X}^{\top}\mathbf{X} \right|^{1/p}}{n} \hspace{19em} (3)
```

두 수준 full factorial 은 linear model 에서 D-efficiency 1 에 이른다. 직교 design 이 있으면 D-optimal 탐색은 그보다
나은 design 을 찾지 못한다.

### 8.2 Generating D-Optimal Designs

D-optimal design 은 반복 탐색으로 얻는다. Coordinate-exchange algorithm 은 무작위 design 에서 출발해, run 하나의
factor 하나를 골라 grid 위의 모든 값을 넣어 보고 가장 좋은 값을 남기는 이동을 더 나아지지 않을 때까지
되풀이한다 [[5](#ref-5)]. 결과가 출발점에 따라 다르므로, 무작위 출발점 여러 개에서 탐색을 되풀이하고 가장 좋은
design 을 남긴다.

Factor 3 개의 quadratic model 에 run 12 개를 요청하면 D-efficiency 0.4498 인 design 이 나온다
([Appendix B.5](#b5-d-optimal-designs)). Table 4 의 가장 작은 Box-Behnken design 보다 run 이 3 개 적고, model 의
계수 10 개보다 2 개 많다.

### 8.3 Augmenting D-Optimal Designs

실험은 흔히 단계를 나누어 진행하며, 이미 수행한 run 은 다시 고를 수 없다. Augmentation 은 수행한 run 을 고정한 채
새 run 만 탐색하므로, 기존 design 에 없는 정보를 채우는 run 을 고른다. 이 탐색은 좌표 하나가 아니라 run 전체를
candidate set 의 행과 맞바꾸는 row exchange 를 쓰며, run 을 고정된 목록에서 뽑을 때는 행 단위 교환이 알맞은
이동이다.

꼭짓점 run 4 개만으로는 항이 6 개인 quadratic model 을 맞출 수 없다. Run 2 개를 더하면 추정할 수 있게 되며,
탐색은 두 run 을 모두 기존 design 에 없는 꼭짓점 밖의 점 $(0, 1)$ 과 $(-1, 0)$ 에 둔다.

### 8.4 Specifying Fixed Covariate Factors

Covariate 는 실험자가 정하지 않고 기록만 하는 factor 이며, 주위 온도, 근무 중인 작업자, batch 의 경과 시간이 그
예이다. Covariate 값은 run 마다 미리 알려져 있지만 고를 수 없다. 이때 design 문제는 covariate 열이 주어진 상태에서
통제 가능한 factor 를 골라, model 이 통제 효과와 covariate 효과를 갈라내게 하는 일이 된다.

Run 8 개에 걸쳐 covariate 가 $-1$ 에서 $+1$ 로 고르게 변하면, 탐색은 통제 가능한 두 열에 그 변화와 균형을 이루는
무늬를 놓아 어느 통제 효과도 covariate 와 confound 되지 않게 한다.

### 8.5 Specifying Categorical Factors

Categorical factor 에는 numeric 척도가 없어 수준을 $\pm 1$ 쪽으로 옮길 수 없다. 수준이 $L$ 개인 factor 는 effect
coding 으로 $L - 1$ 개의 열이 되어 model 에 들어가고, 탐색은 그 열들 위에서 이루어진다.

수준이 3 개인 categorical factor 3 개는 후보 treatment 27 개를 준다. Run 9 개를 요청하면, 탐색은 모든 factor 의
모든 수준이 세 번씩 나타나고 두 factor 의 수준 짝이 모두 정확히 한 번씩 나타나는 design 을 돌려준다.

### 8.6 Specifying Candidate Sets

Row exchange 는 고를 수 있는 run 의 목록인 candidate set 을 받는다. Factor 공간이 정육면체이면 candidate set 은 그
위의 grid 이다.

Candidate set 을 명시적으로 건네면 정육면체가 아닌 factor 공간도 다룰 수 있다. 성분의 합이 1 이어야 하는 mixture,
함께 높을 수 없는 두 설정, 차가운 상태로 빠르게 돌 수 없는 기계는 모두 grid 에서 행을 덜어내는 제약이다. 탐색은
목록에 없는 run 을 제안하지 않으므로, candidate set 에서 그 행들을 지우면 제약이 design 에 반영된다.

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

- **Basic factor**: generator 가 아니라 fractional design 이 출발점으로 삼는 full factorial 에서 열이
  나오는 factor.
- **Candidate set**: D-optimal 탐색이 run 을 고를 수 있는 factor 설정의 목록.
- **Centre run**: 모든 연속 factor 가 자기 범위의 가운데에 있는 run.
- **Confounding**: 두 효과가 같은 design 열에 실려, 그 실험의 어떤 분석으로도 갈라낼 수 없는 상태.
- **Covariate**: 실험자가 정하지 않고 run 마다 기록하는 factor.
- **Defining relation**: fractional factorial design 에서 1 로 채워진 열과 같아지는 factor 열들의 곱의
  모임이며, 그중 가장 짧은 word 가 resolution 을 준다.
- **Effect coding**: 수준이 $L$ 개인 categorical factor 를 $L - 1$ 개의 열로 바꾸되, 마지막 수준을 모든 열에서 $-1$ 로 두는 부호화.
- **Factor**: 실험자가 값을 정하거나 기록하는 입력.
- **Generator**: fractional factorial design 에서 더해지는 factor 를 정의하는 basic factor 들의 곱.
- **Hadamard matrix**: 원소가 $\pm 1$ 이고 열들이 서로 직교하는 정사각행렬.
- **Interaction**: 한 factor 의 효과가 다른 factor 의 수준에 따라 달라지는 정도.
- **Level**: 한 design 에서 factor 가 갖는 값 가운데 하나.
- **Main effect**: 한 factor 의 수준을 바꿀 때 다른 factor 에 대해 평균한 반응의 변화.
- **Response**: run 의 측정된 출력.
- **Rotatable**: 예측분산이 factor 공간 중심으로부터의 거리에만 의존하는 design.
- **Run**: factor 수준의 한 조합에서 실험을 한 번 수행하는 것이며, design 의 한 행.
- **Screening**: 많은 factor 가운데 반응에 유의한 factor 를 가려내는 실험.
- **Star point**: central composite design 에서 factor 하나를 중심 밖으로 옮기고 나머지는 중심에 두는
  run.
- **Treatment**: factor 수준의 한 조합.
- **Word**: defining relation 에서 곱이 1 로 채워진 열이 되는 factor 이름들의 묶음이며, 그 길이는 묶인 factor 수이다.

## Appendix B. Python Implementation

이 Appendix 의 Python block 은 순서대로 실행하도록 되어 있다. 각 block 은 B.1 의 header 위에서 앞선 block 이 정의한
이름을 쓰며, 바로 뒤의 `text` block 은 그 출력이다. 실행에는 numpy 와 scipy 가 필요하다.

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

Section 5.1 의 곱집합 구성이며, 두 수준 factor 와 세 수준 factor 로 run 6 개를 만든다.

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

Section 5.2 의 $2^k$ design 을 $\pm 1$ 로 부호화한다.

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

Section 6.2 의 구성이다. 마지막 줄은 run 12 개 design 의 열 11 개가 서로 직교함을 확인한다. Run 수가 2 의
거듭제곱도 아니고 generator 도 없으면, 다른 design 으로 바꾸지 않고 오류를 낸다.

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

Section 6.3 의 generator 구성이며, generator `a b c abc` 로 factor 4 개의 run 8 개 design 을 만든다.

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

Generator 가 함의하는 word 들이 생성하는 부분군에서 가장 짧은 word 를 찾아 Table 3 의 resolution 을 계산한다.

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

Section 7.2 의 세 변형을 `kind` 로 고른다. 잘못된 `kind` 는 기본값으로 바꾸지 않고 오류를 내므로, 철자를 틀린
변형이 호출자가 요청하지 않은 design 을 만들지 않는다.

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

Section 7.3 의 짝 단위 구성이며, factor 3 개와 centre run 3 개로 Table 4 의 run 15 개를 만든다.

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

식 (3) 을 계산하며, 두 수준 full factorial 의 linear model D-efficiency 1 로 척도를 확인한다. Run 수가 model 항
수보다 적으면 오류를 낸다.

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

Section 8.2 의 탐색이며, factor 3 개의 quadratic model 에 run 12 개를 고른다. 출력의 첫 줄이 D-efficiency 이다.

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

Section 8.3 의 탐색이며, `fixed` 에 넣은 행은 바꾸지 않는다. 꼭짓점 run 4 개에 run 2 개를 더한다.

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

Section 8.4 의 탐색이며, 마지막 열이 고르게 변하는 covariate 이다.

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

Section 8.5 의 effect coding 과 row exchange 이며, 고른 run 을 원래 수준으로 되돌려 정렬해 보인다.

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

Section 8.6 의 정육면체 grid 를 만들고, factor 2 개의 quadratic model 에 run 6 개를 고른다.

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
