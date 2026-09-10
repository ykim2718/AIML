# Trace Feature Selection (Korean)
Rev. 1 | Created: 2026-09-10 | Updated: 2026-09-10 21:13 UTC

> 반도체 장비 trace 가 만들어 내는 수천 개의 feature 가운데 무엇이 target 을 움직이는가, 그리고
> 그에 답하는 방법이 세 축 — model 을 언제 참조하는가, 무엇을 단위로 고르는가, wafer 가 바뀌어도
> 그 선택이 살아남는가 — 위 어디에 놓이는가.

## 1. Purpose

- **Problem Statement**: Recipe 하나의 trace 는 sensor 수백 개에서 feature 수천 열을 만들어 내는데 wafer 수는 수백에 머물러, 그 전부를 넣고 적합한 model 은 우연 상관을 학습하고 공정 담당자가 손댈 자리를 돌려주지 않는다.
- **Goal**: 선택 방법을 mechanism, selection unit, stability 의 세 축 위에 놓아, 어떤 방법이 각 축에서 어느 자리를 잡았고 따라서 그 답이 무엇을 덮지 못하는지 독자가 짚을 수 있게 한다.
- **Non-Goal**: Feature 하나에 숫자를 매기는 일은 여기서 다루지 않는다. 그것은 [Feature Importance](../Feature-Importance/feature-importance.md) 의 주제다.

## 2. Summary

Trace feature 를 고르는 방법의 자리를 정하려면 축이 셋 필요하고, mechanism 축은 그 가운데 첫째일 뿐이다. 이 축은 model 을 언제 참조하는가로 방법의 자리를 정하며, 그것이 답이 무엇의 성질인지와 비용이 얼마인지를 고정한다. §3 의 각 가지에 대한 그 자리를 Table 1 에 적는다.

Table 1. Branches of the mechanism axis

| Branch | When the model is consulted | The answer is a property of | Cost |
|--------|-----------------------------|-----------------------------|------|
| 0. Preprocessing | 참조하지 않으며 target 도 보지 않음 | Feature 그 자체 | 한 번 훑기 |
| 1. Filter | 적합 이전 | 자료 | Feature 마다 한 번 훑기 |
| 2. Wrapper | 후보 부분집합마다 한 번 | 그 model 과 그 탐색 | 단계마다 재적합 한 번 |
| 3. Embedded | 적합 도중 | 적합된 model | 적합 그 자체 |
| 4. Post-hoc ranking | 적합이 끝난 뒤 | 그 적합 하나 | Feature 마다 탐침 한 번 |
| 5. Error-controlled | 순위가 이미 나온 뒤 | 명시된 가정 아래의 분포 | 여러 번의 실행, 또는 knockoff 구성 |

나머지 두 축은 mechanism 축이 말하지 않는 것이며, trace data 에서는 그 둘이 답을 가른다. Selection unit (§5) 은 고른 집합을 실행에 옮길 수 있는지를 가른다. Trace 의 열 하나는 sensor 와 recipe step 과 요약 통계량을 한꺼번에 가리키는데, 그 가운데 line 이 바꿀 수 있는 것은 앞의 둘뿐이기 때문이다. Stability (§6) 는 그 집합이 실재하는지를 가른다. Wafer 는 수백인데 feature 는 수천이어서, 한 신호의 사본 둘 가운데 무엇을 고를지를 단일 적합에서는 잡음이 정하기 때문이다.

## 3. Hierarchy

계층은 model 을 언제 참조하는가로 나뉘며, model 도 target 도 전혀 참조하지 않는 가지에서 시작해 순위가 이미 나온 뒤에야 닿는 가지로 끝난다.

```
Trace feature selection
|
+-- 0. Preprocessing (the target is not consulted)
|   +-- Constancy ................... zero-variance drop, dead-sensor drop
|   +-- Redundancy .................. correlation clustering, near-duplicate drop
|
+-- 1. Filter (scored before any model is fitted)
|   +-- 1.1 Univariate .............. Pearson r, Spearman rho, ANOVA F, mutual information
|   +-- 1.2 Multivariate ............ mRMR, correlation-filtered ranking
|   +-- 1.3 Neighbour contrast ...... Relief, ReliefF, RReliefF
|
+-- 2. Wrapper (a model is refitted for each candidate subset)
|   +-- 2.1 Backward ................ RFE, backward elimination
|   +-- 2.2 Forward ................. forward selection, stepwise
|   +-- 2.3 Stochastic search ....... genetic search, simulated annealing
|
+-- 3. Embedded (selection is a term of the training objective)
|   +-- 3.1 Column penalty .......... lasso, elastic net
|   +-- 3.2 Group penalty ........... group lasso, sparse group lasso
|   +-- 3.3 Split structure ......... MDI, split count, gain
|
+-- 4. Post-hoc ranking (the fit is finished, the model is then probed)
|   +-- 4.1 Removal ................. permutation importance, drop-column
|   +-- 4.2 Attribution ............. SHAP, LIME
|   +-- 4.3 Attention ............... attention weight, diagnostic only
|
+-- 5. Error-controlled (a selected set with a stated error bound)
    +-- 5.1 Subsampling ............. stability selection
    +-- 5.2 Knockoffs ............... fixed-X knockoffs, model-X knockoffs
```

Fig 1. Hierarchy of trace feature selection methods by when the model is consulted

가지 0 은 가장 먼저 돌고 target 을 참조하지 않으므로 잘못 고를 수는 없고 덜 고를 수만 있다. 가지 1 부터 3 까지는 model 을 적합 이전에 참조하는가, 적합 둘레에서 참조하는가, 적합 안에서 참조하는가로 갈린다. 가지 4 는 이미 끝난 model 을 탐침한다. 가지 5 는 순위가 아니라 오차율로 답하는 유일한 가지이고, 1 부터 4 가운데 어디에서든 이어진다. 아래 네 절이 잘못 넘기 쉬운 경계 넷을 정한다.

### 3.1 The Boundary Between Embedded And Post-Hoc

SHAP 과 LIME 과 attention weight 는 embedded 방법이 아니다. Embedded 는 선택이 학습 중에 최소화되는 목적함수의 한 항이라는 뜻이기 때문이다 [[1](#ref-1)]. Lasso 가 고르는 것은 L1 penalty 가 손실 안에 있어 계수를 정확히 0 으로 몰기 때문인데 [[7](#ref-7)], SHAP [[12](#ref-12)] 이나 LIME [[13](#ref-13)] 에는 그에 해당하는 일이 없다. 둘 다 적합이 끝난 뒤에 계산되고, model 을 black box 로 다루며, 학습된 가중치를 건드리지 않는다. 그 산출물은 순위이고 문턱은 여전히 방법 밖에서 정해야 한다. 그것은 filter 의 거동이며, 다만 원자료가 아니라 적합된 model 에 적용되었을 뿐이다.

Attention weight 에는 자리 문제 말고 또 하나의 반론이 붙는다. 입력에 걸린 attention 은 예측이 어느 입력에 의존하는지를 믿을 만하게 짚어 내지 못한다. 같은 예측을 내는 다른 가중치 배정이 존재하고, 그 가중치는 gradient 기반 척도와 상관이 낮다 [[14](#ref-14)]. 이에 대한 반박은 그 결과를 없애는 것이 아니라 좁히는 것으로, model 의 나머지를 고정하면 attention 을 마음대로 바꿔 놓을 수 없음을 보인다 [[15](#ref-15)]. 남는 것은 진단으로는 충분하고 선택 기준으로는 모자라며, 가지 4.3 이 적는 것이 그것이다.

### 3.2 The Two Methods Under Tree Importance

Tree importance 는 한 이름 아래 놓인 서로 다른 두 방법이고, 그 가운데 하나만 embedded 다. MDI 는 어떤 feature 가 쓰인 분기들에 걸쳐 누적한 불순도 감소량으로, 학습 절차가 세운 구조에서 읽어 내므로 embedded 이며 가지 3.3 에 놓인다. Permutation importance 는 완성된 model 의 입력에서 한 열을 섞고 손실을 지켜보므로 post-hoc 이며 가지 4.1 에 속한다.

Trace 표에서 이 구별은 겉치레가 아니다. MDI 는 서로 다른 값을 많이 가지는 feature 쪽으로 치우쳐, 연속형 열을 낮은 cardinality 의 열보다 부풀린다 [[11](#ref-11)]. Trace feature 표가 바로 그 혼합이다. 평균이나 기울기 같은 연속형 요약 통계량 옆에 cycle count 나 step index 같은 낮은 cardinality 의 계수값이 나란히 놓이고, MDI 는 target 과 무관한 이유로 앞의 무리를 뒤의 무리 위에 세운다.

### 3.3 The Place Of Variance Thresholding

Variance thresholding 은 상관 검정 옆이 아니라 mechanism 축 밖에 놓인다. Target 을 전혀 보지 않기 때문이다. Pearson correlation, Spearman correlation, ANOVA F, mutual information 은 모두 feature 를 target 에 대고 점수를 매기므로 서로 비교할 수 있고 다른 지도 기준과도 비교할 수 있다. Variance thresholding 은 feature 를 그 자신에 대고 재므로, 같은 상자에 넣으면 성립하지 않는 비교를 암시하게 된다.

그 일 자체는 실제로 필요하고 가장 먼저 이루어진다. Trace 수집물에는 recipe 안에서 움직이지 않는 setpoint 처럼 구조적으로 상수인 열과, 보고를 멈춘 sensor 의 열이 들어 있다. 둘 다 어떤 지도 기준보다 먼저 빠져야 한다. 분산이 0 인 열은 상관이 정의되지 않고, 거의 상수인 열은 상관이 양자화에 지배되기 때문이다. 그것이 전처리이고, Fig 1 의 가지 0 이며, 거의 중복인 열의 제거도 여기에 함께 놓인다.

### 3.4 The Division Inside The Filter Branch

Filter 가지는 feature 를 혼자 재는가 이미 고른 feature 에 대어 재는가로 나뉘며, 그것이 trace 표에서 filter 의 거동을 가르는 단 하나의 성질이다. Pearson correlation, Spearman correlation, ANOVA F, mutual information 은 한 번에 한 열씩 target 에 대어 점수를 매기므로, 여러 요약 통계량이 모두 한 물리량을 따라가는 sensor 는 거의 같은 점수를 여러 개 내놓게 된다. Filter 는 그것을 모두 남기고, 열 수는 줄지만 중복은 함께 줄지 않는다. mRMR 은 target 에 대한 관련성을 이미 선택된 집합과의 중복에 대어 점수를 매기며, 중복 질문에 답하는 구성원이다 [[5](#ref-5)].

그 둘 옆에 세 번째 갈래가 있다. Relief 와 그 후손은 각 행을 같은 class 와 다른 class 의 최근접 이웃에 대비하여 feature 에 점수를 매기므로, 다른 feature 와 함께일 때만 의미를 가지는 feature 도 점수를 얻을 수 있다. 주변부 기준으로는 만들어 낼 수 없는 결과다 [[6](#ref-6)]. 산출물은 univariate 구성원과 마찬가지로 feature 당 숫자 하나이지만 계산이 주변부가 아니며, 그래서 1.1 의 구성원이 아니라 별도의 갈래다.

## 4. Mechanism Axis

한 가지 안에서 구성원은 한 가지 성질로 갈리며, 아래의 각 표가 적는 것이 그 성질이다. 다섯 절은 Fig 1 에서 고르는 일을 하는 다섯 가지를 다룬다. 가지 0 에는 표가 없다. 그 구성원은 열 사이에서 고르는 것이 아니라 열을 덜어 내기 때문이다.

### 4.1 Filter

이 가지는 trace feature list 전량에 돌릴 만큼 싸고, 그것이 이 가지의 쓰임이다. 비싼 것을 시도하기 전에 한 번 훑어 수천 열을 수백 열로 자른다.

Table 2. Filter methods

| Method | Criterion | Sees redundancy | Sees interaction |
|--------|-----------|-----------------|------------------|
| Pearson correlation | 선형 연관, 부호 있음 | 아니오 | 아니오 |
| Spearman correlation | 단조 연관, 부호 있음 | 아니오 | 아니오 |
| ANOVA F | 범주형 인자에 대한 집단 분리 | 아니오 | 아니오 |
| Mutual information | 모든 의존성, 밀도 추정을 대가로 | 아니오 | 아니오 |
| mRMR | 선택된 집합과의 중복을 뺀 관련성 | 예 | 아니오 |
| Relief, ReliefF | Feature 공간에서의 이웃 대비 | 부분적으로 | 예 |

자름은 넉넉하게 한다. Univariate 구성원 가운데 어느 것도 다른 feature 와 함께일 때만 의미를 가지는 feature 를 볼 수 없고, 여기서 버린 열은 뒤에서 되찾지 못한다.

### 4.2 Wrapper

이 가지는 부분집합을 그 위에 다시 적합한 model 의 성능으로 채점하므로, 그 답은 그 model 의 성질이고 비용은 단계마다 재적합 한 번이다 [[2](#ref-2)].

Table 3. Wrapper methods

| Method | Search direction | Refits per step | Failure on a wide table |
|--------|------------------|-----------------|-------------------------|
| RFE | 후진, 최하위 묶음을 떨굼 | 제거 단계마다 하나 | 첫 적합이 전체 열 위에서 이루어지며, 그때의 순위가 가장 못 믿을 것 |
| Backward elimination | 후진, 한 번에 하나 | 남은 feature 마다 하나 | 열 수가 행 수를 넘으면 정의되지 않음 |
| Forward selection | 전진, 한 번에 하나 | 단계마다 후보 하나에 하나 | 초반의 잘못된 선택이 다시 검토되지 않음 |
| Stepwise | 전진에 후진 단계를 섞음 | 위 두 가지 모두 | 최적해 보장 없이 탐색 비용만 |
| Genetic search | 부분집합 위의 확률적 탐색 | 세대마다 개체 하나에 하나 | 개체군에 따라 비용이 늘고, seed 없이는 결과가 재현되지 않음 |

RFE 는 열 수가 행 수를 자릿수 단위로 넘는 유전자 발현 자료에서 나왔고, 그 경우를 단계마다 feature 하나가 아니라 묶음을 제거하는 것으로 감당한다 [[4](#ref-4)]. Trace 표가 바로 같은 모양이며, 그래서 이 가지에서 흔히 집어 드는 구성원이 RFE 다. 그래도 이 가지를 정의하는 비용은 그대로다. Model 을 단계마다 다시 적합하므로, 이 가지는 filter 에 들어간 수천 열이 아니라 filter 를 통과해 남은 수백 열 위에서 돌린다.

### 4.3 Embedded

이 가지의 비용은 적합 한 번이다. 선택이 둘레에 두른 탐색에서가 아니라 최소화하던 목적함수에서 떨어져 나오기 때문이다.

Table 4. Embedded methods

| Method | Penalty | Selection unit | What the unit buys |
|--------|---------|----------------|--------------------|
| Lasso | 계수마다 L1 | 열 하나 | 가장 작은 열 집합 |
| Elastic net | L1 과 L2 의 혼합 | 열 하나 | 상관된 열이 임의로 하나만 뽑히지 않고 함께 남거나 함께 빠짐 |
| Group lasso | 그룹 안 L2, 그것들을 L1 으로 합산 | 그룹 하나 | Sensor 하나 또는 step 하나가 통째로 들어오거나 빠짐 |
| Sparse group lasso | 그룹 penalty 와 열 penalty 의 혼합 | 그룹, 그다음 열 | Sensor 를 쓰되 그 통계량 가운데 일부만 |
| MDI | 없음, 분기에서 읽음 | 열 하나 | 이미 끝낸 적합 말고는 아무것도 |

Elastic net 이 있는 까닭은 lasso 가 바로 trace 표가 내미는 조건에서 불안정하기 때문이다. 두 열이 강하게 상관되면 lasso 는 그중 하나를 고르고 다른 하나를 0 으로 만드는데, 어느 쪽을 고를지는 자료가 안정적으로 정해 주지 않는다. Elastic net 이 더한 L2 항은 상관된 열이 함께 들어오고 함께 나가게 만든다 [[8](#ref-8)]. 그것은 §6 의 안정성 문제에 대한 부분적인 답이지 완전한 답이 아니다. 담당자가 손대는 물리적 대상이 아니라 상관으로 묶기 때문이다.

### 4.4 Post-Hoc Ranking

이 가지는 원자료가 아니라 적합된 model 에 적용된 filter 이며, filter 의 결함을 그대로 물려받는다. 문턱이 들어 있지 않은 순위라는 결함이다.

Table 5. Post-hoc ranking methods

| Method | What is disturbed | Labels required | Standing |
|--------|-------------------|-----------------|----------|
| Permutation importance | 열 하나를 섞음 | 예 | 보류 행 위에서 잰 의존도 |
| Drop-column | 열 하나를 빼고 model 을 재적합 | 예 | 열마다 재적합 한 번의 비용 |
| SHAP | Feature 부분집합을 주변화 | 아니오 | 예측마다의 귀속, 순위를 위해 집계 |
| LIME | 국소 이웃을 재표집 | 아니오 | 국소 대리 model 이며 전역 진술이 아님 |
| Attention weight | 없음, 가중치를 읽을 뿐 | 아니오 | 진단 전용 (§3.1) |

그 결함 때문에 이 가지는 집합을 만드는 데가 아니라 짧은 후보 목록의 차례를 정하는 데 쓰인다. Permutation 과 drop-column 은 나머지가 강요하지 않는 결정을 하나 더 요구한다. 쓰는 행이 학습 행인가 보류 행인가이다. 보류 행 위에서면 그 숫자는 표본 밖에서 살아남은 의존도를 보고하며, 서로 다른 값을 충분히 많이 가지는 잡음 열은 그 학습 자료 점수의 폭만큼 두 읽기를 갈라 놓는다.

### 4.5 Error-Controlled Selection

이 가지는 비용을 치르고 다른 어느 가지도 하지 않는 진술을 얻는다. 고른 feature 가운데 몇 개가 우연히 거기 있는지에 대한 한계다.

Table 6. Error-controlled methods

| Method | Construction | What is bounded | Price |
|--------|--------------|-----------------|-------|
| Stability selection | 기저 선택기를 부분표본 위에서 다시 돌림 | 거짓 양성의 기댓값 | 부분표본마다 기저 선택기 한 번 |
| Fixed-X knockoffs | 설계행렬에서 만든 합성 열 | False discovery rate | 열보다 행이 많아야 함 |
| Model-X knockoffs | Feature 의 결합분포에서 뽑은 합성 열 | False discovery rate | 그 결합분포를 알거나 잘 추정해야 함 |

이 가지는 "무엇이 가장 높은 점수를 받았는가" 가 아니라 "무엇을 실재한다고 보고할 수 있는가" 에 답하는 유일한 가지다. Model-X knockoffs 는 각 feature 의 상관 구조는 그대로 지니되 나머지가 주어졌을 때 target 과 독립인 합성 사본을 만들고, 자기 사본을 정해진 폭만큼 이기는 feature 를 남겨 false discovery rate 를 고른 수준 아래로 묶는다 [[17](#ref-17)]. 대가는 feature 의 결합분포이며, trace 표에서 그것은 구조를 가정해야만 추정할 수 있고, 보장은 그 가정보다 강해지지 않는다.

## 5. Selection Unit Axis

무엇을 단위로 고르는가가 그 답을 실행에 옮길 수 있는지를 정하며, trace 표에서 그 단위는 열이 아니라 sensor 이거나 recipe step 이다. Trace feature 표는 sensor 마다 recipe step 마다 요약 통계량 한 벌을 적용해 만들므로, 열 하나가 어느 sensor 인지, 어느 step 인지, 어느 통계량인지를 한꺼번에 가리키고, 그 가운데 line 이 바꿀 수 있는 것에 대응하는 것은 앞의 둘뿐이다. 열 단위 선택은 forward RF power 의 최댓값을 남기고 그 평균을 버리면서 reflected RF power 의 최댓값을 남길 수 있다. 그 집합은 타당한 model 입력이면서 아무도 수행할 수 없는 지시다.

Table 7. Selection units

| Unit | One selected item is | The action it maps to | Method that selects at it |
|------|----------------------|-----------------------|---------------------------|
| Column | Sensor 하나의 step 하나에 대한 통계량 하나 | 그 자체로는 없음 | Lasso, elastic net, univariate filter |
| Sensor | Sensor 하나의 모든 통계량 | 그 sensor 를 수집에서 남기거나 뺌 | Sensor 로 묶은 group lasso |
| Step window | Recipe step 하나에 대한 모든 통계량 | 그 step 을 바꾸거나 그대로 둠 | Step 으로 묶은 group lasso |
| Statistic family | 모든 sensor 에 걸친 통계량 하나 | Trace 에서 무엇을 뽑을지를 바꿈 | 통계량으로 묶은 group lasso |

그룹은 적합 전에 선언되고 적합 뒤에는 되찾을 수 없으며, 그래서 이것이 후처리 선택이 아니라 축이다. Group lasso 는 그룹마다의 계수 vector 에 L2 norm 을 씌우고 그 norm 들을 L1 처럼 합산하므로, 그룹 전체가 함께 0 이 되거나 함께 남는다 [[9](#ref-9)]. Sparse group lasso 는 그 위에 열 penalty 를 더하며, "이 sensor 는 쓰되 그 통계량 가운데 둘만" 에 답하는 형태다. 수집 수준에서 실행할 수 있고 model 수준에서 경제적인 결정이다 [[10](#ref-10)].

열 단위가 틀린 것은 아니다. 그것은 다른 질문, 곧 model 이 어느 숫자를 필요로 하는가에 답하며, 수집 집합이 이미 고정되어 있고 model 만 다듬는 상황에서는 그 질문이 옳은 질문이다. 잘못은 그것에 답해 놓고 그 답을 sensor 목록이라고 보고하는 데 있다.

## 6. Stability Axis

단일 적합에서 보고한 선택은 trace data 에서 재현되지 않으므로, 이 축에서 중요한 자리는 재표집 자리다. Wafer 는 수백인데 feature 는 수천이고, 물리적으로 짝을 이루는 sensor — forward 와 reflected RF power, 인접한 열전대, 유량 setpoint 와 그 readback — 의 상관이 벌점 적합에서 두 열 가운데 무엇을 고를지가 신호가 아니라 잡음으로 정해지는 수준에 가깝다. Wafer 의 다른 부분집합 위에서 다시 적합하면 각 쌍에서 선택되는 구성원이 바뀐다.

Table 8. Stability positions

| Position | Procedure | What is reported | Cost |
|----------|-----------|------------------|------|
| Single fit | 선택기를 전체 행 위에서 한 번 | 폭이 붙지 않은 부분집합 하나 | 한 번 |
| Resampled | 선택기를 B 개 부분표본마다 한 번 | Feature 마다의 선택 빈도 | B 번 |
| Error-bounded | 재표집에 한계에서 정한 문턱을 씌움 | 거짓 양성 기댓값에 한계가 붙은 부분집합 | B 번, 더하여 한계의 성립 조건 |

Stability selection 은 재표집 자리를 방법으로 만든 것이다. 기저 선택기를 부분표본 위에서 다시 돌리고, feature 마다 선택 빈도를 세고, 문턱을 넘는 것을 남기며, 명시된 조건 아래에서 잘못 선택된 feature 수의 기댓값에 유한표본 한계를 준다 [[16](#ref-16)]. 그 한계 때문에 Fig 1 에서 가지 5.1 로 나타나고, 동시에 축의 한 자리이기도 하다. 재표집이 가지 1 부터 4 까지의 어떤 선택기든 바꾸지 않고 감싸기 때문이다.

빈도 곡선 자체가 결과이고, 문턱으로 자른 집합만 읽으면 그것을 버리는 셈이다. 거의 모든 부분표본에서 선택된 feature 와 절반을 갓 넘긴 feature 는 서로 다른 결과인데, 단일 적합은 둘을 똑같이 보고한다. 그 빈도가 뜻을 가지려면 두 조건이 서야 한다. 부분표본은 lot 을 지켜야 한다. 함께 처리된 wafer 는 독립 추출이 아니기 때문이다. 그리고 처리 순서를 지켜야 한다. 앞뒤 wafer 를 섞는 분할은 선택이 견뎌 내야 할 drift 를 가리기 때문이다.

## 7. Selection Guide

Table 9 는 trace feature list 에 던지는 질문에서 Fig 1 의 답하는 가지로 간다.

Table 9. Question and the branch that answers it

| Question | Branch | Note |
|----------|--------|------|
| 어느 열이 죽었거나 상수이거나 중복인가 | 0 | Target 을 보기 전에 돈다 |
| 수천 열 가운데 무엇을 더 끌고 갈 만한가 | 1.1 | 넉넉하게 자를 것, 조합에서만 사는 feature 는 여기서 안 보인다 |
| 어느 열이 이미 고른 것들에 없는 정보를 지니는가 | 1.2 | 기준은 관련성만이 아니라 중복이다 |
| 어느 열이 조합에서만 의미를 가지는가 | 1.3 | 이웃 대비는 주변부 점수가 못 보는 것을 본다 |
| 바로 이 model 이 실제로 필요로 하는 부분집합은 | 2.1 | 단계마다 재적합 한 번이므로 filter 를 통과한 목록 위에서 |
| 어느 sensor 를 수집에서 뺄 수 있는가 | 3.2, sensor 로 묶음 | 단위가 sensor 여야 답이 실행 가능하다 |
| Target 이 어느 recipe step 에 반응하는가 | 3.2, step 으로 묶음 | 단위가 step 이어야 한다 |
| 어느 sensor 를 남기되 통계량을 줄일 것인가 | 3.2, sparse group lasso | 그룹 간과 그룹 안 희소성을 동시에 |
| 적합된 black box model 이 어느 열에 기대는가 | 4.1, 보류 행 위에서 | 그 적합의 의존도이지 자료 안의 정보가 아니다 |
| 이 wafer 하나의 예측이 왜 거기 나왔는가 | 4.2 | 행 하나에 대한 설명이지 선택이 아니다 |
| 어느 feature 가 wafer 가 바뀌어도 살아남는가 | 5.1 | 부분집합이 아니라 선택 빈도를 보고할 것 |
| 무엇을 실재한다고 보고할 수 있는가 | 5.2 | 오차율을 진술하는 유일한 가지 |

## 8. Failure Modes

아래 결함은 어떤 구현의 성질이 아니라 방법의 성질이며, 하나같이 각 단계에서는 옳아 보이는 절차를 거쳐 도달한다.

- **Univariate filter over collinear sensors.** 한 물리 신호의 사본이 모두 남아, 열 수는 줄고 중복은 줄지 않음.
- **Wrapper score computed outside the resampling fold.** 선택 편향, 보고된 교차검증 오차가 큰 폭으로 낙관적 [[20](#ref-20)].
- **Random split over lot-grouped wafers.** 한 lot 의 wafer 가 분할 양쪽에 놓여, 보류 행이 보류되지 않음.
- **A drift proxy selected.** Chamber 사용 시간이나 소모품 수명이 target 자신의 drift 를 따라가, 적합에서는 예측력이 있고 개입에는 무력함.
- **MDI read as importance.** 연속형 요약 통계량이 cardinality 때문에 낮은 cardinality 의 계수값 위에 놓임 [[11](#ref-11)].
- **A zero coefficient read as irrelevance.** 정확히 중복된 쌍의 두 구성원이 모두 0 으로 채점됨.
- **Permutation over correlated columns.** 공정이 만들어 낼 수 없는 행 위에서 model 이 평가되고, 거기서의 거동이 점수에 들어감 [[18](#ref-18)].
- **A set selected on one chamber.** 공정 물리가 아니라 그 chamber 의 국소 상태여서 다음 chamber 로 옮겨 가지 않음.

## 9. Further Work

- **Selection at the trace-segment unit.** Recipe step 보다 잘고 열보다 굵은 단위로, step 안에서 반응을 담고 있는 시간 구간을 고르는 일이다. 지금 가능한 까닭은 virtual metrology 의 feature 기반 틀이 trace 전체 요약에서 물리적 뜻이 명시된 구간 feature 로 옮겨 왔기 때문이다 [[19](#ref-19)]. 필요한 것은 step 안에서 공통 시간축에 정렬된 trace 이며, step 길이가 wafer 마다 달라지면 step 경계만으로는 그것이 얻어지지 않는다.
- **Group knockoffs for the sensor unit.** Error-controlled 가지를 §5 의 선택 단위에 적용하여, 보고하는 sensor 목록이 순위가 아니라 false discovery rate 를 지니게 하는 일이다. 지금 가능한 까닭은 model-X knockoffs 가 요구 조건을 열보다 많은 행에서 feature 의 알려진 결합분포로 옮겨 놓았기 때문이며 [[17](#ref-17)], 넓은 trace 표는 원리상 그것을 충족할 수 있고 긴 표는 그럴 수 없다. 필요한 것은 sensor 집합에 대해 검증된 공분산 model 이다. 보장은 합성 사본에 가정한 분포보다 강해지지 않기 때문이다.

## References

<a id="ref-1"></a>
[1] Guyon, I. and Elisseeff, A. (2003). [An Introduction to Variable and Feature Selection](https://www.jmlr.org/papers/v3/guyon03a.html). *Journal of Machine Learning Research*, 3, 1157–1182.<br>
<a id="ref-2"></a>
[2] Kohavi, R. and John, G. H. (1997). [Wrappers for feature subset selection](<https://doi.org/10.1016/S0004-3702(97)00043-X>). *Artificial Intelligence*, 97(1–2), 273–324.<br>
<a id="ref-3"></a>
[3] Chandrashekar, G. and Sahin, F. (2014). [A survey on feature selection methods](https://doi.org/10.1016/j.compeleceng.2013.11.024). *Computers & Electrical Engineering*, 40(1), 16–28.<br>
<a id="ref-4"></a>
[4] Guyon, I., Weston, J., Barnhill, S. and Vapnik, V. (2002). [Gene Selection for Cancer Classification using Support Vector Machines](https://doi.org/10.1023/A:1012487302797). *Machine Learning*, 46, 389–422.<br>
<a id="ref-5"></a>
[5] Peng, H., Long, F. and Ding, C. (2005). [Feature Selection Based on Mutual Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy](https://doi.org/10.1109/TPAMI.2005.159). *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 27(8), 1226–1238.<br>
<a id="ref-6"></a>
[6] Urbanowicz, R. J., Meeker, M., La Cava, W., Olson, R. S. and Moore, J. H. (2018). [Relief-based feature selection: Introduction and review](https://doi.org/10.1016/j.jbi.2018.07.014). *Journal of Biomedical Informatics*, 85, 189–203.<br>
<a id="ref-7"></a>
[7] Tibshirani, R. (1996). [Regression Shrinkage and Selection via the Lasso](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x). *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.<br>
<a id="ref-8"></a>
[8] Zou, H. and Hastie, T. (2005). [Regularization and variable selection via the elastic net](https://doi.org/10.1111/j.1467-9868.2005.00503.x). *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.<br>
<a id="ref-9"></a>
[9] Yuan, M. and Lin, Y. (2006). [Model selection and estimation in regression with grouped variables](https://doi.org/10.1111/j.1467-9868.2005.00532.x). *Journal of the Royal Statistical Society: Series B*, 68(1), 49–67.<br>
<a id="ref-10"></a>
[10] Simon, N., Friedman, J., Hastie, T. and Tibshirani, R. (2013). [A Sparse-Group Lasso](https://doi.org/10.1080/10618600.2012.681250). *Journal of Computational and Graphical Statistics*, 22(2), 231–245.<br>
<a id="ref-11"></a>
[11] Strobl, C., Boulesteix, A.-L., Zeileis, A. and Hothorn, T. (2007). [Bias in random forest variable importance measures: illustrations, sources and a solution](https://doi.org/10.1186/1471-2105-8-25). *BMC Bioinformatics*, 8, 25.<br>
<a id="ref-12"></a>
[12] Lundberg, S. M. and Lee, S.-I. (2017). [A Unified Approach to Interpreting Model Predictions](https://papers.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html). *Advances in Neural Information Processing Systems*, 30, 4765–4774.<br>
<a id="ref-13"></a>
[13] Ribeiro, M. T., Singh, S. and Guestrin, C. (2016). ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://doi.org/10.1145/2939672.2939778). *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144.<br>
<a id="ref-14"></a>
[14] Jain, S. and Wallace, B. C. (2019). [Attention is not Explanation](https://doi.org/10.18653/v1/N19-1357). *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 3543–3556.<br>
<a id="ref-15"></a>
[15] Wiegreffe, S. and Pinter, Y. (2019). [Attention is not not Explanation](https://doi.org/10.18653/v1/D19-1002). *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing*, 11–20.<br>
<a id="ref-16"></a>
[16] Meinshausen, N. and Bühlmann, P. (2010). [Stability selection](https://doi.org/10.1111/j.1467-9868.2010.00740.x). *Journal of the Royal Statistical Society: Series B*, 72(4), 417–473.<br>
<a id="ref-17"></a>
[17] Candès, E., Fan, Y., Janson, L. and Lv, J. (2018). [Panning for gold: 'model-X' knockoffs for high dimensional controlled variable selection](https://doi.org/10.1111/rssb.12265). *Journal of the Royal Statistical Society: Series B*, 80(3), 551–577.<br>
<a id="ref-18"></a>
[18] Hooker, G., Mentch, L. and Zhou, S. (2021). [Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance](https://doi.org/10.1007/s11222-021-10057-z). *Statistics and Computing*, 31, 82.<br>
<a id="ref-19"></a>
[19] Suthar, K., Shah, D., Wang, J. and He, Q. P. (2019). [Next-generation virtual metrology for semiconductor manufacturing: A feature-based framework](https://doi.org/10.1016/j.compchemeng.2019.05.016). *Computers & Chemical Engineering*, 127, 140–149.<br>
<a id="ref-20"></a>
[20] Ambroise, C. and McLachlan, G. J. (2002). [Selection bias in gene extraction on the basis of microarray gene-expression data](https://doi.org/10.1073/pnas.102102699). *Proceedings of the National Academy of Sciences*, 99(10), 6562–6566.

---

## Appendix A. Terminology

- **embedded method**: 선택이 학습 중 최소화되는 목적함수의 한 항인 선택 방법.
- **filter method**: Model 을 적합하기 전에 feature 를 채점하는 선택 방법.
- **group lasso**: 선언된 그룹마다 L2 norm 을 씌우고 그 norm 들을 L1 으로 합산하여, 그룹이 통째로 남거나 통째로 빠지게 하는 penalty.
- **knockoff**: 어떤 feature 의 상관 구조는 그대로 지니되 나머지 feature 가 주어졌을 때 target 과 독립인 합성 사본.
- **MDI**: Mean decrease in impurity, 어떤 feature 가 쓰인 분기들에 걸쳐 누적한 불순도 감소량.
- **mRMR**: Minimum redundancy maximum relevance, target 에 대한 관련성을 이미 선택된 집합과의 중복에 대어 채점하는 filter.
- **post-hoc ranking**: 학습이 끝난 뒤에 적합된 model 을 black box 로 다루어 계산한 순위.
- **recipe step**: 공정 recipe 의 이름 붙은 한 국면이며, trace 를 그 단위로 따로 요약한다.
- **RFE**: Recursive feature elimination, 단계마다 model 을 재적합하고 최하위 feature 를 떨구는 wrapper.
- **selection unit**: 선택된 항목 하나가 대표하는 대상 — 열, sensor, recipe step, 통계량 계열.
- **stability selection**: 기저 선택기를 부분표본 위에서 다시 돌리고 선택 빈도가 문턱을 넘는 feature 를 남기는 절차.
- **trace**: Sensor 가 wafer 한 장의 recipe 통과 동안 기록한 시계열.
- **virtual metrology**: Wafer 를 재는 대신 장비 자료에서 계측값을 예측하는 것.
- **wrapper method**: 후보 부분집합마다 model 을 재적합하고 그 model 의 성능으로 부분집합을 채점하는 선택 방법.
