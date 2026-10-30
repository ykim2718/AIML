# Trace Feature Selection
Rev. 14 | Created: 2026-09-10 | Updated: 2026-10-30 01:17 UTC

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Comparison](#31-comparison)
- [4. Evaluation Criterion Axis](#4-evaluation-criterion-axis)
  - [4.1 Preprocessing](#41-preprocessing)
  - [4.2 Filter](#42-filter)
  - [4.3 Wrapper](#43-wrapper)
  - [4.4 Embedded](#44-embedded)
  - [4.5 Post-Hoc Ranking](#45-post-hoc-ranking)
  - [4.6 Error-Controlled Selection](#46-error-controlled-selection)
- [5. Selection Unit Axis](#5-selection-unit-axis)
- [6. Failure Modes](#6-failure-modes)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

- **Problem Statement**: Recipe 하나의 trace 는 sensor 수백 개에서 feature 수천 열을 만들지만 wafer 는 수백 장에 그쳐, 이 열을 모두 넣고 적합한 model 은 우연 상관을 학습하고 공정 담당자가 조정할 대상을 알려 주지 못한다.
- **Goal**: 선택 방법을 Fig 1 의 네 축, 곧 evaluation criterion, label availability, search strategy, selection unit 위에 놓아, 독자가 어떤 방법이 축마다 어느 값을 갖는지와 그 때문에 답이 다루지 못하는 것을 짚을 수 있게 한다.
- **Non-Goal**: Feature 하나에 점수를 매기는 일은 다루지 않는다. 이 주제는 [Feature Importance](../../Feature-Importance/feature-importance.md) 에 있다.

## 2. Summary

Trace feature 를 고르는 방법은 evaluation criterion, label availability, search strategy, selection unit 의 네 축 위에 놓아야 그 답이 다루지 못하는 것이 드러난다. Evaluation criterion 축은 답이 무엇의 성질인지와 비용을 정하고, selection unit 축은 고른 집합을 공정에서 실행할 수 있는지를 정한다. Trace data 는 wafer 수백 장에 열 수천 개여서 한 번의 적합으로 고른 집합이 wafer 가 바뀌면 달라지며, error-controlled 가지의 subsampling 구성원이 이 변화를 다룬다.

## 3. Taxonomy and its Hierarchy

Trace feature 를 고르는 방법은 네 축, 곧 evaluation criterion, label availability, search strategy, selection unit 에서 각각 값 하나를 갖는다. Evaluation criterion 축은 이름을 Liu & Yu [[1](#ref-1)] 의 evaluation criteria 에서 따왔으며, feature selection 문헌의 filter, wrapper, embedded 구분 [[2](#ref-2)], [[3](#ref-3)] 에 preprocessing, post-hoc ranking, error-controlled selection 을 더한 것이다. Selection unit 축은 trace data 의 모양에서 나온다. 열 하나가 sensor, recipe step, 요약통계를 함께 가리킨다.

네 축과 축마다의 값은 아래 [Fig 1](#fig-1) 과 같다.

```text
Trace feature selection
|
+-- Evaluation criterion ... dependence on the learning algorithm
|     Preprocessing | Filter | Wrapper | Embedded | Post-hoc ranking | Error-controlled
|
+-- Label availability ..... whether target labels are used
|     Supervised | Unsupervised | Semi-supervised
|
+-- Search strategy ........ how candidate subsets are generated
|     Exhaustive | Sequential | Randomized
|
+-- Selection unit ......... what one selected item stands for
      Column | Sensor | Step window | Statistic family
```

<a id="fig-1"></a>
Fig 1. Feature selection taxonomy for trace data

Evaluation criterion 과 search strategy 는 Liu & Yu [[1](#ref-1)] 의 차원 가운데 둘이고, label availability 는 Li 등 [[4](#ref-4)] 의 구분이며, selection unit 은 이 문서가 trace data 를 위해 더한 축이다. 네 축을 함께 쓴 출처는 없다. 한 방법은 축마다 값 하나를 갖는다. Sensor 로 묶은 group lasso 는 Embedded, Supervised, 부분집합 탐색 없음, Sensor 이고, recipe step 으로 묶으면 selection unit 값만 Step window 로 바뀐다. Evaluation criterion 축은 §4, selection unit 축은 §5 에서 다룬다.

Evaluation criterion 축은 선택이 학습 algorithm 을 어떻게 쓰는가로 다시 나뉜다. 계층은 학습 algorithm 도 target 도 쓰지 않는 가지에서 시작해 순위가 나온 뒤에야 쓰는 가지로 끝나며, 아래 [Fig 2](#fig-2) 와 같다. 오른쪽 열에는 줄마다 label availability, search strategy, selection unit 의 값을 두었다.

```text
Trace feature selection                                                                     Label availability | Search strategy | Selection unit
|
+-- 0. Preprocessing (the target is not consulted)
|   +-- Constancy ................... zero-variance drop, dead-sensor drop                  Unsupervised       | -               | Column, sensor
|   +-- Redundancy .................. correlation clustering, near-duplicate drop           Unsupervised       | -               | Column
|
+-- 1. Filter (scored before any model is fitted)
|   +-- 1.1 Univariate .............. Pearson r, Spearman rho, ANOVA F, mutual information  Supervised         | -               | Column
|   +-- 1.2 Multivariate ............ mRMR, correlation-filtered ranking                    Supervised         | Sequential      | Column
|   +-- 1.3 Neighbour contrast ...... Relief, ReliefF, RReliefF                             Supervised         | -               | Column
|
+-- 2. Wrapper (a model is refitted for each candidate subset)
|   +-- 2.1 Backward ................ RFE, backward elimination                             Supervised         | Sequential      | Column
|   +-- 2.2 Forward ................. forward selection, stepwise                           Supervised         | Sequential      | Column
|   +-- 2.3 Stochastic search ....... genetic search, simulated annealing                   Supervised         | Randomized      | Column
|
+-- 3. Embedded (selection is a term of the training objective)
|   +-- 3.1 Column penalty .......... lasso, elastic net                                    Supervised         | -               | Column
|   +-- 3.2 Group penalty ........... group lasso, sparse group lasso                       Supervised         | -               | Sensor, step window, statistic family
|   +-- 3.3 Split structure ......... MDI, split count, gain                                Supervised         | -               | Column
|
+-- 4. Post-hoc ranking (the fit is finished, the model is then probed)
|   +-- 4.1 Removal ................. permutation importance, drop-column                   Supervised         | -               | Column
|   +-- 4.2 Attribution ............. SHAP, LIME                                            Supervised         | -               | Column
|   +-- 4.3 Attention ............... attention weight, diagnostic only                     Supervised         | -               | Column
|
+-- 5. Error-controlled (a selected set with a stated error bound)
    +-- 5.1 Subsampling ............. stability selection                                   Supervised         | -               | Unit of the base selector
    +-- 5.2 Knockoffs ............... fixed-X knockoffs, model-X knockoffs                  Supervised         | -               | Column
```

<a id="fig-2"></a>
Fig 2. Hierarchy of the evaluation criterion axis by use of the learning algorithm

가지 0 은 가장 먼저 돌고 target 을 보지 않으므로, 빼야 할 열을 남길 수는 있어도 엉뚱한 열을 고르지는 않는다. 가지 1 부터 3 까지는 model 을 적합 전에 참조하는가, 적합을 감싸서 참조하는가, 적합 안에서 참조하는가로 갈린다. 가지 4 는 적합이 끝난 model 을 probe 한다. 가지 5 만 순위 대신 오차율로 답하며, 가지 1 부터 4 가운데 어느 것의 결과든 입력으로 받는다. 잘못 넘기 쉬운 가지 경계는 §4.1, §4.2, §4.4, §4.5 에서 정한다. 오른쪽 열의 - 는 그 줄의 구성원이 부분집합을 탐색하지 않는다는 뜻이다. Semi-supervised 나 exhaustive 에 해당하는 줄은 없고, 가지 5.1 의 selection unit 은 base selector 의 단위를 따른다.

### 3.1 Comparison

Trace feature list 에 묻는 질문마다 답하는 Fig 2 의 가지는 Table 1 과 같다.

Table 1. Question and the branch that answers it

| Question                                           | Branch                  | Note                                                         |
| :------------------------------------------------: | :---------------------: | :----------------------------------------------------------: |
| 어느 열이 죽었거나 상수이거나 중복인가             | 0                       | Target 을 보기 전에 실행                                     |
| 수천 열 가운데 무엇을 다음 단계로 넘길 것인가      | 1.1                     | 넉넉하게 자름, 조합에서만 의미를 갖는 feature 는 보이지 않음 |
| 어느 열이 이미 고른 열에 없는 정보를 갖는가        | 1.2                     | Relevance 와 함께 redundancy 를 기준으로 삼음                |
| 어느 열이 다른 열과 함께일 때만 의미를 갖는가      | 1.3                     | Neighbour contrast 로 marginal 점수가 놓치는 것을 봄         |
| 이 model 이 실제로 필요로 하는 부분집합은 무엇인가 | 2.1                     | 단계마다 재적합 한 번, filter 를 통과한 목록에서 실행        |
| 어느 sensor 를 수집에서 뺄 수 있는가               | 3.2, sensor 로 묶음     | 단위가 sensor 여야 실행할 수 있는 답                         |
| Target 이 어느 recipe step 에 반응하는가           | 3.2, step 으로 묶음     | 단위가 step 이어야 함                                        |
| 어느 sensor 를 남기고 통계량을 줄일 것인가         | 3.2, sparse group lasso | Group 사이 sparsity 와 group 안 sparsity 를 함께             |
| 적합된 black box model 이 어느 열에 의존하는가     | 4.1, held-out row 에서  | 그 적합의 의존도, 자료 안의 정보와는 별개                    |
| 이 wafer 하나의 예측값은 왜 그렇게 나왔는가        | 4.2                     | 행 하나의 설명, 선택과는 별개                                |
| 어느 feature 가 wafer 가 바뀌어도 다시 선택되는가  | 5.1                     | 부분집합 대신 선택 빈도를 보고                               |
| 무엇을 실재한다고 보고할 수 있는가                 | 5.2                     | 오차율을 보고하는 유일한 가지                                |

둘째 열의 번호는 Fig 2 의 가지 번호이고, 묶는 단위를 적은 행은 §5 의 selection unit 을 따른다.

## 4. Evaluation Criterion Axis

Evaluation criterion 축은 선택이 학습 algorithm 을 어떻게 쓰는가로 가지를 나누며, 이에 따라 답이 무엇의 성질인지와 비용이 정해진다. 가지마다의 값은 Table 2 와 같다.

Table 2. Branches of the evaluation criterion axis

| Branch              | Use of the learning algorithm   | The answer is a property of | Cost                                |
| :-----------------: | :-----------------------------: | :-------------------------: | :---------------------------------: |
| 0. Preprocessing    | 쓰지 않으며 target 도 보지 않음 | Feature 그 자체             | 한 번 훑기                          |
| 1. Filter           | 적합 전                         | 자료                        | Feature 마다 한 번 훑기             |
| 2. Wrapper          | 후보 부분집합마다 한 번         | 그 model 과 그 탐색         | 단계마다 재적합 한 번               |
| 3. Embedded         | 적합 도중                       | 적합된 model                | 적합 한 번                          |
| 4. Post-hoc ranking | 적합이 끝난 뒤                  | 그 적합 하나                | Feature 마다 probe 한 번            |
| 5. Error-controlled | 순위가 나온 뒤                  | 명시한 가정 아래의 분포     | 여러 번 실행하거나 knockoff 를 구성 |

고르는 일을 하는 가지에서는 구성원이 한 가지 성질로 갈리며, 각 절의 표에서 그 성질을 열 하나로 두었다. 가지 0 에는 표가 없다. 가지 0 의 구성원은 열 사이에서 고르지 않고 열을 덜어 낸다.

### 4.1 Preprocessing

Preprocessing 은 model 도 target 도 참조하지 않아 가지 0 이 된다. Pearson correlation, Spearman correlation, ANOVA F, mutual information 은 모두 feature 를 target 과 견주어 점수를 매기므로, 서로 비교할 수 있고 다른 지도 기준과도 비교할 수 있다. Variance thresholding 은 feature 를 그 자신의 분산으로만 재므로, 상관 검정과 한 묶음에 넣으면 성립하지 않는 비교를 하게 된다.

Constancy 묶음이 가장 먼저 실행된다. Trace 수집물에는 recipe 안에서 변하지 않는 setpoint 처럼 구조적으로 상수인 열과, 보고를 멈춘 sensor 의 열이 있다. 두 종류 모두 지도 기준을 돌리기 전에 빼야 한다. 분산이 0 인 열은 상관이 정의되지 않고, 거의 상수인 열은 상관이 양자화 오차에 좌우된다.

Redundancy 묶음은 같은 신호가 여러 열에 되풀이될 때 한 열만 남긴다. Trace table 에서는 한 recipe step 안에서 한 sensor 의 mean, median, integral 이 같은 물리량을 따라가므로, 세 열은 wafer 사이에서 함께 오르내린다. 열을 표준화한 뒤 모든 열 쌍의 Pearson correlation 절댓값 |r| 을 구하고, |r| 이 0.95 같은 threshold 를 넘는 열을 한 group 으로 묶는다. 묶을 때는 거리 1 − |r| 위의 hierarchical clustering 을 쓰거나 correlation matrix 를 차례로 훑는 greedy 제거를 쓰며, group 마다 한 열을 남긴다. 표준화한 열에서는 cosine similarity 가 Pearson r 과 같으므로, cosine 을 기준으로 삼아도 같은 방법이다. 남는 열은 group 의 평균이 아닌 원래 열 하나여서, 어느 sensor 의 어느 recipe step 의 어느 요약통계인지가 그대로 남는다 (§5).

Redundancy 묶음도 target 을 보지 않으므로, group 안에서 어느 열이 남을지는 target 과 상관없이 정해진다. 남길 열은 결측이 가장 적은 열이나 engineer 가 이미 읽는 요약통계처럼, target 을 보기 전에 정한 규칙으로 고른다. Target 과의 relevance 와 이미 고른 열과의 redundancy 를 함께 따지는 지도 기준은 가지 1.2 의 mRMR 이다 (§4.2).

### 4.2 Filter

Filter 가지는 trace feature list 전체에 돌릴 수 있을 만큼 계산이 싸며, 비싼 방법에 앞서 한 번 훑어 수천 열을 수백 열로 줄이는 데 쓴다.

Filter 의 구성원은 feature 를 혼자 재는 univariate 구성원과, 이미 고른 feature 와 견주어 재는 multivariate 구성원으로 나뉜다. Pearson correlation, Spearman correlation, ANOVA F, mutual information 은 한 번에 한 열씩 target 과 견주므로, 여러 요약통계가 한 물리량을 따라가는 sensor 에서는 거의 같은 점수가 여러 개 나온다. Univariate filter 는 그 sensor 의 열을 모두 남기므로, 열 수는 줄어도 redundancy 는 줄지 않는다. mRMR 은 target 과의 relevance 에서 이미 고른 집합과의 redundancy 를 빼서 점수를 매기므로, redundancy 를 다루는 구성원이다 [[5](#ref-5)].

세 번째 갈래는 neighbour contrast 다. Relief 와 그 변형은 각 행을 같은 class 와 다른 class 의 nearest neighbour 와 대비하여 feature 에 점수를 매기므로, 다른 feature 와 함께일 때만 의미를 갖는 feature 도 점수를 얻는다. Marginal 기준으로는 얻을 수 없는 결과다 [[6](#ref-6)]. 결과는 univariate 구성원처럼 feature 마다 숫자 하나지만 계산이 marginal 하지 않으므로, 가지 1.1 에 넣지 않고 별도 갈래 1.3 으로 둔다.

구성원끼리의 비교는 Table 3 과 같다.

Table 3. Filter methods

| Method               | Criterion                                  | Sees redundancy | Sees interaction |
| :------------------: | :----------------------------------------: | :-------------: | :--------------: |
| Pearson correlation  | 선형 연관, 부호 있음                       | 아니오          | 아니오           |
| Spearman correlation | 단조 연관, 부호 있음                       | 아니오          | 아니오           |
| ANOVA F              | 범주형 인자에 대한 집단 분리               | 아니오          | 아니오           |
| Mutual information   | 모든 의존성, 밀도 추정이 필요              | 아니오          | 아니오           |
| mRMR                 | 선택된 집합과의 redundancy 를 뺀 relevance | 예              | 아니오           |
| Relief, ReliefF      | Feature 공간에서의 neighbour contrast      | 부분적으로      | 예               |

자를 때는 넉넉하게 남긴다. Univariate 구성원은 어느 것도 다른 feature 와 함께일 때만 의미를 갖는 feature 를 보지 못하고, 여기서 버린 열은 뒤 단계에서 다시 들어오지 않는다.

### 4.3 Wrapper

Wrapper 가지는 부분집합마다 model 을 다시 적합하고 그 성능으로 점수를 매기므로, 답은 그 model 의 성질이고 비용은 단계마다 재적합 한 번이다 [[7](#ref-7)].

구성원마다의 탐색 방향과 비용은 Table 4 와 같다.

Table 4. Wrapper methods

| Method               | Search direction          | Refits per step           | Failure on a wide table                                          |
| :------------------: | :-----------------------: | :-----------------------: | :--------------------------------------------------------------: |
| RFE                  | 후진, 최하위 묶음을 제거  | 제거 단계마다 하나        | 첫 적합이 전체 열에서 이루어지며, 그때의 순위가 가장 믿기 어려움 |
| Backward elimination | 후진, 한 번에 하나        | 남은 feature 마다 하나    | 열 수가 행 수를 넘으면 정의되지 않음                             |
| Forward selection    | 전진, 한 번에 하나        | 단계마다 후보 하나에 하나 | 초반의 잘못된 선택을 다시 검토하지 않음                          |
| Stepwise             | 전진에 후진 단계를 섞음   | 위 두 가지 모두           | 최적해 보장 없이 탐색 비용만 듦                                  |
| Genetic search       | 부분집합 위의 확률적 탐색 | 세대마다 개체 하나에 하나 | 개체군에 따라 비용이 늘고, seed 없이는 결과가 재현되지 않음      |

RFE 는 열 수가 행 수보다 몇 자릿수 많은 gene expression 자료에서 나왔고, 단계마다 feature 하나 대신 한 묶음을 제거하여 그 조건을 감당한다 [[8](#ref-8)]. Trace 표도 같은 모양이어서 이 가지에서는 RFE 를 가장 흔히 쓴다. 그래도 단계마다 model 을 다시 적합하는 비용은 그대로이므로, wrapper 는 filter 에 들어가는 수천 열이 아니라 filter 를 통과한 수백 열에서 돌린다.

### 4.4 Embedded

Embedded 가지의 비용은 적합 한 번이다. 선택이 바깥 탐색 없이 학습 중 최소화하는 목적함수에서 바로 정해진다.

구성원마다의 penalty 와 selection unit 은 Table 5 와 같다.

Table 5. Embedded methods

| Method             | Penalty                             | Selection unit   | Effect of the unit                                          |
| :----------------: | :---------------------------------: | :--------------: | :---------------------------------------------------------: |
| Lasso              | 계수마다 L1                         | 열 하나          | 가장 작은 열 집합                                           |
| Elastic net        | L1 과 L2 의 혼합                    | 열 하나          | 상관된 열이 임의로 하나만 뽑히지 않고 함께 남거나 함께 빠짐 |
| Group lasso        | Group 안 L2, 이를 L1 으로 합산      | Group 하나       | Sensor 하나 또는 step 하나가 통째로 들어오거나 빠짐         |
| Sparse group lasso | Group penalty 와 열 penalty 의 혼합 | Group, 그다음 열 | Sensor 를 쓰되 그 통계량 가운데 일부만                      |
| MDI                | 없음, 분기에서 읽음                 | 열 하나          | 이미 끝낸 적합 외에는 없음                                  |

열끼리 강하게 상관된 trace 표의 조건에서 lasso 는 불안정하며, elastic net 이 이를 보완한다. 두 열이 강하게 상관되면 lasso 는 하나를 고르고 다른 하나를 0 으로 만드는데, 어느 쪽을 고를지는 자료가 안정적으로 정하지 못한다. Elastic net 은 L2 항을 더해 상관된 열이 함께 들어오고 함께 빠지게 한다 [[9](#ref-9)]. 이 보완은 §4.6 에서 다루는 wafer 사이의 불안정을 일부만 해결한다. 열을 상관으로 묶을 뿐, 담당자가 조정하는 물리적 대상으로 묶지는 않는다.

Tree importance 라는 이름 아래에는 서로 다른 두 방법이 있고, 그 가운데 MDI 만 embedded 다. MDI 는 어떤 feature 가 쓰인 분기들에서 누적한 불순도 감소량으로, 학습 절차가 만든 tree 구조에서 읽으므로 embedded 이며 가지 3.3 에 속한다. Permutation importance 는 완성된 model 의 입력에서 한 열을 섞어 손실의 변화를 보므로 post-hoc 이며 가지 4.1 에 속한다 (§4.5).

Trace 표에서는 두 방법을 구별하느냐에 따라 순위가 달라진다. MDI 는 서로 다른 값을 많이 갖는 feature 쪽으로 치우쳐, 연속형 열을 낮은 cardinality 의 열보다 높게 평가한다 [[10](#ref-10)]. Trace feature 표에는 평균이나 기울기 같은 연속형 요약통계와 cycle count 나 step index 같은 낮은 cardinality 의 계수값이 함께 들어 있어, MDI 는 target 과 무관한 이유로 연속형 열을 계수값보다 높은 순위에 둔다.

### 4.5 Post-Hoc Ranking

Post-hoc ranking 가지는 원자료 대신 적합된 model 에 적용한 filter 이며, filter 와 같은 결함이 있다. 순위만 주고 문턱은 주지 않는다.

구성원마다 흔드는 대상과 label 필요 여부는 Table 6 과 같다.

Table 6. Post-hoc ranking methods

| Method                 | What is disturbed              | Labels required | Scope of the result                      |
| :--------------------: | :----------------------------: | :-------------: | :--------------------------------------: |
| Permutation importance | 열 하나를 섞음                 | 예              | Held-out row 에서 잰 의존도              |
| Drop-column            | 열 하나를 빼고 model 을 재적합 | 예              | 열마다 재적합 한 번의 비용               |
| SHAP                   | Feature 부분집합을 marginalize | 아니오          | 예측마다의 attribution, 순위를 위해 집계 |
| LIME                   | 국소 이웃을 재표집             | 아니오          | 국소 surrogate model, 전역 결론과는 별개 |
| Attention weight       | 없음, 가중치를 읽을 뿐         | 아니오          | 진단 전용                                |

문턱이 없으므로 이 가지는 집합을 만드는 데 쓰지 않고, 짧은 후보 목록의 순서를 정하는 데 쓴다. Permutation 과 drop-column 은 다른 구성원에 없는 결정을 하나 더 요구한다. 학습 행을 쓸지 held-out row 를 쓸지다. Held-out row 를 쓰면 그 값은 표본 밖에서도 유지된 의존도이며, 서로 다른 값이 충분히 많은 잡음 열에서는 두 방식의 값이 그 열의 학습 자료 점수만큼 벌어진다.

SHAP, LIME, attention weight 는 post-hoc ranking 에 속한다. Embedded 는 선택이 학습 중 최소화하는 목적함수의 한 항이라는 뜻이다 [[2](#ref-2)]. Lasso 는 L1 penalty 가 손실 안에 있어 계수를 정확히 0 으로 만들면서 열을 고르는데 [[11](#ref-11)], SHAP [[12](#ref-12)] 과 LIME [[13](#ref-13)] 에는 그런 항이 없다. 둘 다 적합이 끝난 뒤에 계산하고, model 을 black box 로 다루며, 학습된 가중치를 바꾸지 않는다. 결과는 순위이고, 문턱은 여전히 방법 밖에서 정해야 한다.

Attention weight 에는 어느 가지에 속하는가와 별개로 반론이 하나 더 있다. 입력에 걸린 attention 은 예측이 어느 입력에 의존하는지를 믿을 만하게 가리키지 못한다. 같은 예측을 주는 다른 가중치 배정이 있고, 그 가중치는 gradient 기반 척도와 상관이 낮다 [[14](#ref-14)]. 이에 대한 반박은 이 결과를 없애지 않고 범위만 좁혀, model 의 나머지를 고정하면 attention 을 마음대로 바꿀 수 없음을 보였다 [[15](#ref-15)]. 따라서 attention weight 는 진단에는 쓸 수 있어도 선택 기준으로는 부족하며, Fig 2 의 가지 4.3 을 진단 전용으로 둔 근거가 이 반론이다.

### 4.6 Error-Controlled Selection

Error-controlled 가지는 비용이 더 드는 대신, 고른 feature 가운데 우연히 들어온 것이 몇 개인지에 한계를 준다. 다른 가지에는 이 한계가 없다.

구성원마다의 구성 방법과 한계의 대상은 Table 7 과 같다.

Table 7. Error-controlled methods

| Method              | Construction                              | What is bounded      | Price                               |
| :-----------------: | :---------------------------------------: | :------------------: | :---------------------------------: |
| Stability selection | Base selector 를 subsample 마다 다시 실행 | 거짓 양성의 기댓값   | Subsample 마다 base selector 한 번  |
| Fixed-X knockoffs   | 설계행렬로 만든 합성 열                   | False discovery rate | 열보다 행이 많아야 함               |
| Model-X knockoffs   | Feature 결합분포에서 뽑은 합성 열         | False discovery rate | 그 결합분포를 알거나 잘 추정해야 함 |

모든 가지 가운데 이 가지만 "무엇을 실재한다고 보고할 수 있는가" 에 답한다. Model-X knockoffs 는 각 feature 의 상관 구조를 유지하면서 나머지 feature 가 주어졌을 때 target 과 독립인 합성 사본을 만들고, 자기 사본을 정해진 차이 이상으로 이긴 feature 만 남겨 false discovery rate 를 정한 수준 아래로 유지한다 [[16](#ref-16)]. 대가는 feature 의 결합분포다. Trace 표에서는 구조를 가정해야만 결합분포를 추정할 수 있고, 보장은 그 가정보다 강해지지 않는다.

Stability selection 은 이 가지의 subsampling 구성원이다. Base selector 를 subsample 마다 다시 실행하고, feature 마다 선택 빈도를 세어 문턱을 넘는 feature 를 남기며, 명시한 조건 아래에서 잘못 선택된 feature 수의 기댓값에 유한표본 한계를 준다 [[17](#ref-17)]. 재표집은 가지 1 부터 4 까지의 어느 선택기든 바꾸지 않고 감싼다.

Trace data 에서는 한 번의 적합으로 고른 집합이 wafer 가 바뀌면 달라진다. Wafer 는 수백 장인데 feature 는 수천 개이고, 물리적으로 짝을 이루는 sensor — forward 와 reflected RF power, 인접한 thermocouple, 유량 setpoint 와 그 readback — 는 벌점 적합이 두 열 가운데 무엇을 고를지를 신호가 아닌 잡음이 정할 만큼 상관이 높다. 그래서 wafer 의 다른 부분집합으로 다시 적합하면 쌍마다 선택되는 열이 바뀐다.

선택 빈도의 분포 자체가 결과이며, 문턱으로 자른 집합만 보면 이 분포를 잃는다. 거의 모든 subsample 에서 선택된 feature 와 절반을 겨우 넘긴 feature 는 다른 결과인데, 한 번의 적합은 둘을 똑같이 보고한다. 선택 빈도가 뜻을 가지려면 두 조건이 성립해야 한다. Subsample 은 lot 단위로 나눠야 한다. 함께 처리된 wafer 는 독립 추출이 아니다. 또 처리 순서를 지켜야 한다. 앞뒤 wafer 를 섞어 나누면 선택이 견뎌야 할 drift 가 가려진다.

## 5. Selection Unit Axis

무엇을 단위로 고르는가가 답을 실행할 수 있는지를 정하며, trace 표에서 실행할 수 있는 단위는 열이 아니라 sensor 나 recipe step 이다. Trace feature 표는 sensor 마다, recipe step 마다 요약통계 한 벌을 계산해 만들므로, 열 하나가 어느 sensor, 어느 step, 어느 통계량인지를 함께 가리킨다. 이 셋 가운데 line 이 바꿀 수 있는 것은 sensor 와 step 뿐이다. 열 단위 선택은 forward RF power 의 최댓값은 남기고 평균은 버리면서 reflected RF power 의 최댓값을 남길 수 있다. 이 집합은 model 입력으로는 타당하지만 아무도 실행할 수 없는 지시다.

단위마다 선택된 항목 하나가 무엇이고 어떤 조치로 이어지는지는 Table 8 과 같다.

Table 8. Selection units

| Unit             | One selected item is                       | The action it maps to             | Method that selects at it             |
| :--------------: | :----------------------------------------: | :-------------------------------: | :-----------------------------------: |
| Column           | Sensor 하나의 step 하나에 대한 통계량 하나 | 그 자체로는 없음                  | Lasso, elastic net, univariate filter |
| Sensor           | Sensor 하나의 모든 통계량                  | 그 sensor 를 수집에서 남기거나 뺌 | Sensor 로 묶은 group lasso            |
| Step window      | Recipe step 하나에 대한 모든 통계량        | 그 step 을 바꾸거나 그대로 둠     | Step 으로 묶은 group lasso            |
| Statistic family | 모든 sensor 에 걸친 통계량 하나            | Trace 에서 무엇을 계산할지를 바꿈 | 통계량으로 묶은 group lasso           |

Group 은 적합 전에 선언하며 적합 뒤에는 되살릴 수 없으므로, selection unit 은 후처리의 선택이 아닌 축이다. Group lasso 는 group 마다 계수 vector 에 L2 norm 을 적용하고 그 norm 들을 L1 처럼 합하므로, group 전체가 함께 0 이 되거나 함께 남는다 [[18](#ref-18)]. Sparse group lasso 는 여기에 열 penalty 를 더해 "이 sensor 는 쓰되 통계량은 둘만" 같은 답을 준다. 수집 수준에서 실행할 수 있고 model 수준에서 경제적인 결정이다 [[19](#ref-19)].

열 단위는 model 이 어느 숫자를 필요로 하는가라는 다른 질문에 답하며, 수집 집합이 이미 고정되어 model 만 다듬을 때는 이 질문이 맞다. 잘못은 열 단위로 답해 놓고 그 답을 sensor 목록으로 보고하는 데 있다.

## 6. Failure Modes

아래 결함은 모두 구현이 아닌 방법에서 생기며, 단계마다 옳아 보이는 절차를 거쳐 나타난다.

- **Univariate filter over collinear sensors.** 한 물리 신호의 사본이 모두 남아, 열 수는 줄고 redundancy 는 그대로.
- **Wrapper score computed outside the resampling fold.** Selection bias 로 보고된 cross-validation 오차가 크게 낙관적 [[20](#ref-20)].
- **Random split over lot-grouped wafers.** 한 lot 의 wafer 가 분할 양쪽에 있어, held-out row 가 실제로는 학습에 쓰인 wafer 와 같은 lot.
- **A drift proxy selected.** Chamber 사용 시간이나 소모품 수명이 target 의 drift 를 따라가, 적합에서는 예측력이 있으나 조정해도 효과 없음.
- **MDI read as importance.** 연속형 요약통계가 cardinality 때문에 낮은 cardinality 의 계수값보다 높은 순위 [[10](#ref-10)].
- **A zero coefficient read as irrelevance.** 정확히 중복된 두 열이 모두 0 으로 채점됨.
- **Permutation over correlated columns.** 공정이 만들 수 없는 행에서 model 을 평가하고, 그 행에서의 거동이 점수에 들어감 [[21](#ref-21)].
- **A set selected on one chamber.** 공정 물리 대신 그 chamber 의 국소 상태를 반영해, 다음 chamber 에서 재현되지 않음.

## References

<a id="ref-1"></a>
[1] Liu, H. and Yu, L. (2005). [Toward Integrating Feature Selection Algorithms for Classification and Clustering](https://doi.org/10.1109/TKDE.2005.66). *IEEE Transactions on Knowledge and Data Engineering*, 17(4), 491–502.<br>
<a id="ref-2"></a>
[2] Guyon, I. and Elisseeff, A. (2003). [An Introduction to Variable and Feature Selection](https://www.jmlr.org/papers/v3/guyon03a.html). *Journal of Machine Learning Research*, 3, 1157–1182.<br>
<a id="ref-3"></a>
[3] Chandrashekar, G. and Sahin, F. (2014). [A survey on feature selection methods](https://doi.org/10.1016/j.compeleceng.2013.11.024). *Computers & Electrical Engineering*, 40(1), 16–28.<br>
<a id="ref-4"></a>
[4] Li, J., Cheng, K., Wang, S., Morstatter, F., Trevino, R. P., Tang, J. and Liu, H. (2017). [Feature Selection: A Data Perspective](https://doi.org/10.1145/3136625). *ACM Computing Surveys*, 50(6), 94.<br>
<a id="ref-5"></a>
[5] Peng, H., Long, F. and Ding, C. (2005). [Feature Selection Based on Mutual Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy](https://doi.org/10.1109/TPAMI.2005.159). *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 27(8), 1226–1238.<br>
<a id="ref-6"></a>
[6] Urbanowicz, R. J., Meeker, M., La Cava, W., Olson, R. S. and Moore, J. H. (2018). [Relief-based feature selection: Introduction and review](https://doi.org/10.1016/j.jbi.2018.07.014). *Journal of Biomedical Informatics*, 85, 189–203.<br>
<a id="ref-7"></a>
[7] Kohavi, R. and John, G. H. (1997). [Wrappers for feature subset selection](<https://doi.org/10.1016/S0004-3702(97)00043-X>). *Artificial Intelligence*, 97(1–2), 273–324.<br>
<a id="ref-8"></a>
[8] Guyon, I., Weston, J., Barnhill, S. and Vapnik, V. (2002). [Gene Selection for Cancer Classification using Support Vector Machines](https://doi.org/10.1023/A:1012487302797). *Machine Learning*, 46, 389–422.<br>
<a id="ref-9"></a>
[9] Zou, H. and Hastie, T. (2005). [Regularization and variable selection via the elastic net](https://doi.org/10.1111/j.1467-9868.2005.00503.x). *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.<br>
<a id="ref-10"></a>
[10] Strobl, C., Boulesteix, A.-L., Zeileis, A. and Hothorn, T. (2007). [Bias in random forest variable importance measures: illustrations, sources and a solution](https://doi.org/10.1186/1471-2105-8-25). *BMC Bioinformatics*, 8, 25.<br>
<a id="ref-11"></a>
[11] Tibshirani, R. (1996). [Regression Shrinkage and Selection via the Lasso](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x). *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.<br>
<a id="ref-12"></a>
[12] Lundberg, S. M. and Lee, S.-I. (2017). [A Unified Approach to Interpreting Model Predictions](https://papers.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html). *Advances in Neural Information Processing Systems*, 30, 4765–4774.<br>
<a id="ref-13"></a>
[13] Ribeiro, M. T., Singh, S. and Guestrin, C. (2016). ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://doi.org/10.1145/2939672.2939778). *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144.<br>
<a id="ref-14"></a>
[14] Jain, S. and Wallace, B. C. (2019). [Attention is not Explanation](https://doi.org/10.18653/v1/N19-1357). *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 3543–3556.<br>
<a id="ref-15"></a>
[15] Wiegreffe, S. and Pinter, Y. (2019). [Attention is not not Explanation](https://doi.org/10.18653/v1/D19-1002). *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing*, 11–20.<br>
<a id="ref-16"></a>
[16] Candès, E., Fan, Y., Janson, L. and Lv, J. (2018). [Panning for gold: 'model-X' knockoffs for high dimensional controlled variable selection](https://doi.org/10.1111/rssb.12265). *Journal of the Royal Statistical Society: Series B*, 80(3), 551–577.<br>
<a id="ref-17"></a>
[17] Meinshausen, N. and Bühlmann, P. (2010). [Stability selection](https://doi.org/10.1111/j.1467-9868.2010.00740.x). *Journal of the Royal Statistical Society: Series B*, 72(4), 417–473.<br>
<a id="ref-18"></a>
[18] Yuan, M. and Lin, Y. (2006). [Model selection and estimation in regression with grouped variables](https://doi.org/10.1111/j.1467-9868.2005.00532.x). *Journal of the Royal Statistical Society: Series B*, 68(1), 49–67.<br>
<a id="ref-19"></a>
[19] Simon, N., Friedman, J., Hastie, T. and Tibshirani, R. (2013). [A Sparse-Group Lasso](https://doi.org/10.1080/10618600.2012.681250). *Journal of Computational and Graphical Statistics*, 22(2), 231–245.<br>
<a id="ref-20"></a>
[20] Ambroise, C. and McLachlan, G. J. (2002). [Selection bias in gene extraction on the basis of microarray gene-expression data](https://doi.org/10.1073/pnas.102102699). *Proceedings of the National Academy of Sciences*, 99(10), 6562–6566.<br>
<a id="ref-21"></a>
[21] Hooker, G., Mentch, L. and Zhou, S. (2021). [Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance](https://doi.org/10.1007/s11222-021-10057-z). *Statistics and Computing*, 31, 82.

---

## Appendix A. Terminology

- **base selector**: Stability selection 이 subsample 마다 다시 실행하는 선택 방법.
- **embedded method**: 선택이 학습 중 최소화되는 목적함수의 한 항인 선택 방법.
- **filter method**: Model 을 적합하기 전에 feature 를 채점하는 선택 방법.
- **group lasso**: 선언된 그룹마다 L2 norm 을 씌우고 그 norm 들을 L1 으로 합산하여, 그룹이 통째로 남거나 통째로 빠지게 하는 penalty.
- **held-out row**: 적합에 쓰지 않고 적합된 model 을 평가하는 데만 쓰는 행.
- **knockoff**: 어떤 feature 의 상관 구조는 그대로 지니되 나머지 feature 가 주어졌을 때 target 과 독립인 합성 사본.
- **MDI**: Mean decrease in impurity, 어떤 feature 가 쓰인 분기들에 걸쳐 누적한 불순도 감소량.
- **mRMR**: Minimum redundancy maximum relevance, target 과의 relevance 에서 이미 선택된 집합과의 redundancy 를 빼서 채점하는 filter.
- **post-hoc ranking**: 학습이 끝난 뒤에 적합된 model 을 black box 로 다루어 계산한 순위.
- **recipe step**: 공정 recipe 에서 이름이 있는 한 단계이며, trace 를 그 단위로 따로 요약한다.
- **RFE**: Recursive feature elimination, 단계마다 model 을 재적합하고 최하위 feature 를 제거하는 wrapper.
- **selection unit**: 선택된 항목 하나가 대표하는 대상 — 열, sensor, recipe step, 통계량 계열.
- **stability selection**: Base selector 를 subsample 마다 다시 실행하고 선택 빈도가 문턱을 넘는 feature 를 남기는 절차.
- **subsample**: 행 가운데 비복원으로 뽑은 부분집합이며, 선택기를 그 위에서 다시 실행한다.
- **trace**: Sensor 가 wafer 한 장의 recipe 통과 동안 기록한 시계열.
- **wrapper method**: 후보 부분집합마다 model 을 재적합하고 그 model 의 성능으로 부분집합을 채점하는 선택 방법.
