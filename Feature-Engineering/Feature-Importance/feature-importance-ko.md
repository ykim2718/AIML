# Feature Importance (Korean)
Rev. 0 | Created: 2026-08-31 | Updated: 2026-08-31 18:02 CDT

> Feature 에 숫자를 매기는 방법들을 그 숫자가 어디서 나오는가로 나누어 정리한 글이며, 이미
> import 해 둔 library 가 아니라 묻고 있는 물음에서 방법이 골라지도록 하려는 것이다.

## 1. Scope

Feature importance 는 하나의 양이 아니다. 이 문서의 모든 방법은 feature 마다 숫자 하나를
돌려주지만, 그 숫자를 어느 방법이 냈는지 모르는 독자는 그것이 무엇을 뜻하는지 말할 수 없다.
같은 feature 에 대해 두 방법이 낸 두 숫자가 자릿수만큼 달라도 둘 다 옳을 수 있으며, 서로 다른
물음에 답했기 때문이다.

첫 갈림은 같은 낱말이 덮고 있는 두 물음 사이에 있다.

- **적합된 model 의 의존도.** 학습된 이 model 이 그 feature 를 얼마나 쓰는가. 같은 데이터를 다른 seed 로 다시 적합하면 달라진다.
- **데이터 안의 정보.** 어떤 model 을 쓰든 그 feature 가 target 에 대해 지니는 양. 어느 한 적합이 아니라 결합분포의 성질이다.

발표된 방법 대부분은 앞의 것에 답하면서 뒤의 것으로 읽히며, 그로부터 따라오는 것을 section 12
가 모은다. Section 2 는 방법이 자리를 잡는 축을 세우고, section 3 은 계열을 하나의 위계에
놓으며, section 4 부터 10 까지가 일곱 계열을 차례로 다루고, section 11 이 선택을 한자리에
모은다.

정착된 기법이 아직 움직이는 기법과 섞이지 않도록, 모든 방법에 status 를 붙인다.

- **Standard.** 주요 library 에 들어 있고, 교과서에 실려 있으며, 실패 양상이 기록되어 있다.
- **Recent.** 아직 연구가 진행 중인 갈래이며, 구현은 대개 연구용 package 이고, 실무 관행이 정해지지 않았다.

Status 는 그 방법이 얼마나 정착했는지를 적은 것이지 얼마나 좋은지를 적은 것이 아니다. Standard
인 방법이 옳은 선택인 경우가 매우 흔하며, recent 인 방법은 새로워서가 아니라 물음이 그 방법이
더하는 것을 필요로 할 때 고른다.

## 2. Axes Of The Question

아래 일곱 소절은 일곱 개의 축이지 일곱 개의 범주가 아니다. 방법은 그 모두에 대해 동시에 자리를
잡으며, 두 방법은 그 자리가 같은 축에서만 견줄 수 있다.

### 2.1. Object

Importance 가 무엇의 성질인가.

- **적합된 model.** 학습된 model 하나와, 그 적합이 우연히 갖게 된 의존도.
- **Model class.** 거의 같은 정도로 잘 맞는 모든 model 이며, 숫자 하나를 구간으로 바꾼다 [[4](#ref-4)].
- **분포.** 모집단 수준의 양이며, model 은 nuisance 추정에만 쓰인다 [[18](#ref-18)].

한 번의 적합에서 나온 숫자를 문제의 성질로 보고하는 것이 이 분야에서 가장 흔한 과장이며,
model class 라는 object 는 그 간극에 이름을 붙이려고 있다.

### 2.2. Scope

숫자가 성립하는 범위.

- **Global.** 데이터 전체에 대해 feature 마다 숫자 하나.
- **Cohort.** 부분집단 안에서 feature 마다 숫자 하나이며, model 이 영역마다 다르게 움직일 때 쓴다.
- **Local.** 예측마다 feature 마다 숫자 하나.

Global 숫자를 local 숫자의 집계로 만들 수 있으나, 부호 있는 local 값은 평균을 내면 상쇄되므로
절댓값의 평균을 대신 쓴다. 그 집계량은 곧바로 계산한 global 측도와는 다른 양이며, 둘의 순위가
일치해야 할 이유는 없다.

### 2.3. Effect Measured

Feature 를 흔들었을 때 무엇을 보는가.

- **Loss.** 예측이 얼마나 나빠지는가. Label 이 필요하다.
- **Output.** 예측이 어느 쪽으로든 얼마나 움직이는가. Label 이 필요 없다.
- **Structure.** Model 이 어떻게 지어졌는가를 그 parameter 나 split 에서 읽는다. 아무것도 흔들지 않는다.

앞의 둘은 중복 아래에서 갈라선다. 상관된 사본이 loss 를 메울 수 있으면 loss 기반 측도는 model 이
크게 기대는 feature 에 0 에 가까운 값을 주고, output 기반 측도는 온전한 무게를 준다. Section 4
의 data-only 계열은 이 셋 밖에 있으며, target 과의 association 을 곧바로 재고 흔들 model 이
없기 때문이다.

### 2.4. Conditioning

Feature 를 흔들 때 무엇으로 바꾸는가.

- **Marginal.** 나머지 행과 무관하게 그 feature 자신의 주변분포에서 뽑는다. 데이터에 없는 입력 조합이 만들어지고, model 은 적합된 적 없는 자리에서 평가된다 [[5](#ref-5)].
- **Conditional.** 다른 feature 가 주어졌을 때의 조건분포에서 뽑는다. 행은 그럴듯한 채로 남지만, 그 조건분포를 추정해야 한다.

Marginal 은 model 이 입력을 어떻게 다루는지 묻는다. Conditional 은 나머지가 주어졌을 때 그
feature 가 무엇을 더하는지 묻는다. 상관된 feature 가 둘이 갈라지는 자리이며, 그 갈라짐이 겉으로
비슷해 보이는 방법들 사이 불일치의 가장 큰 원천이다.

### 2.5. Data

숫자를 어느 행에서 계산하는가.

- **학습 데이터.** Split 통계와 계수이며, model 이 적합한 것을 그대로 보상하고 과적합한 것까지 보상한다.
- **Held-out 데이터.** Model 이 보지 않은 행에서의 permutation 과 재학습 측도이며, 표본 밖에서도 남는 의존도를 보고한다.

값이 충분히 다양한 순수한 잡음 feature 는 학습 데이터에서 높은 점수를 얻고 held-out 에서 0 에
가까운 점수를 얻으며, 둘의 차이 자체가 진단이 된다.

### 2.6. Unit

숫자 하나가 무엇에 붙는가.

- **열 하나.** 어디서나 기본값.
- **열의 묶음.** 범주형 변수 하나의 one-hot block, 또는 sensor 하나의 channel 들을 함께 채점한다.
- **쌍 또는 집합.** 주효과가 아니라 interaction 의 세기.

One-hot 열을 하나씩 채점하면 그것이 나온 범주형 변수를 과소평가하며, 변수의 기여가 level 들로
쪼개지기 때문이다. 채점 전에 묶는 것이 그 해법이고, structure 기반 계열을 뺀 모든 계열에서 쓸
수 있다.

### 2.7. Guarantee

숫자에 대해 무엇을 주장하는가.

- **순위만.** 오류에 대한 진술이 없으며, 대부분의 방법이 여기에 있다.
- **신뢰구간.** 추정량의 표본변동에 대한 진술.
- **통제된 오류율.** 거짓 발견 비율이 정해진 수준 아래로 눌린 선택 집합.

Section 10 의 계열만이 세 번째를 주며, 그래서 "어느 feature 가 높은 점수를 받았는가" 가 아니라
"어느 feature 를 사실로 보고해도 되는가" 에 답하는 유일한 계열이다.

Table 1. Axes and the choice each one forces

| Axis | Positions | What is lost by not choosing |
|------|-----------|------------------------------|
| Object | 적합된 model, model class, 분포 | 한 적합의 우연을 문제의 성질로 읽음 |
| Scope | Global, cohort, local | 영역마다 다르게 움직이는 model 을 평평하게 평균냄 |
| Effect | Loss, output, structure | 중복된 feature 가 0 으로, 잡음이 높게 채점됨 |
| Conditioning | Marginal, conditional | Model 이 적합된 적 없는 자리에서 나온 숫자 |
| Data | 학습, held-out | 과적합이 importance 로 보고됨 |
| Unit | 열, 묶음, 집합 | 범주형 변수가 자신의 level 들로 쪼개짐 |
| Guarantee | 순위, 구간, 오류율 | 뒤에 오류 진술이 없는 cut-off |

## 3. Hierarchy

계열은 숫자가 어디서 오는가로 갈라지며, 그것만이 안정된 구분이다. Model 종류로 가르는 구분은
안정되지 않는데 같은 계열이 model 마다 되풀이해 나타나기 때문이고, global 과 local 로 가르는
구분도 안정되지 않는데 대부분의 계열이 둘 다 내놓기 때문이다.

```
Feature importance
|
+-- A. Data-only association (no model is fitted)
|   +-- A1. Linear and monotone ............. Pearson r, Spearman rho, ANOVA F, chi-square
|   +-- A2. General dependence .............. mutual information, distance correlation, HSIC
|   +-- A3. Redundancy-aware ................ mRMR, correlation-filtered ranking
|
+-- B. Model-internal structure (read off the fitted model)
|   +-- B1. Coefficients .................... standardized beta, lasso path, elastic net
|   +-- B2. Split statistics ................ MDI, split count, cover
|   +-- B3. Component and attention weights . PLS VIP, PCA loading, attention weight
|
+-- C. Perturbation and removal (change an input, watch the model)
|   +-- C1. Marginal permutation ............ MDA, drop-and-shuffle variants
|   +-- C2. Conditional permutation ......... conditional MDA, grid-conditioned permutation
|   +-- C3. Retraining ...................... drop-column, LOCO, model class reliance
|   +-- C4. Curve summaries ................. PDP spread, ALE spread, H-statistic
|
+-- D. Gradient attribution (differentiate the model)
|   +-- D1. Plain gradient .................. saliency, gradient times input
|   +-- D2. Path integral ................... integrated gradients, DeepLIFT
|   +-- D3. Layer propagation ............... LRP, Grad-CAM
|
+-- E. Game-theoretic attribution (share the payoff among features)
|   +-- E1. Local additive .................. LIME, KernelSHAP, TreeSHAP, DeepSHAP
|   +-- E2. Global additive ................. SAGE, Shapley effects
|   +-- E3. Amortized ....................... FastSHAP and learned explainers
|
+-- F. Variance-based sensitivity (decompose output variance)
|   +-- F1. Screening ....................... Morris elementary effects
|   +-- F2. Variance decomposition .......... Sobol indices, FAST, functional ANOVA
|
+-- G. Error-controlled selection (attach a guarantee)
    +-- G1. Knockoffs ....................... linear-model knockoffs, model-X knockoffs
    +-- G2. Nonparametric inference ......... LOCO inference, VIM, decorrelated LOCO
```

Fig 1. Hierarchy of feature importance methods by the source of the number

계열의 차례는 그것이 필요해진 차례이다. A 는 가장 오래되었고 model 이 필요 없다. B 는 model 을
적합하고 나면 비용이 들지 않는다. C 는 model-agnostic 이면서 loss 기반일 수 있는 첫 계열이다.
D 는 입력이 백만 개인 network 에 C 가 너무 느려서 있다. E 는 C 와 D 가 어긋나므로 그 어긋남을
공리로 정리하려는 데서 나왔다. F 는 computer experiment 라는 다른 문헌에서 와서 같은 자리에
닿는다. G 만이 오류율과 함께 답한다.

## 4. Data-Only Association

이 계열은 사이에 model 없이 feature 를 target 에 견주어 채점한다. 아무것도 적합하지 않으므로
아무것도 과적합될 수 없고, 숫자는 데이터만의 성질이다. 모든 구성원이 태생적으로 marginal 인데,
feature 를 하나씩 target 에 견주고 나머지 feature 는 계산에 들어오지 않기 때문이다.

Table 2. Data-only association measures

| Method | What it scores | Status |
|--------|----------------|--------|
| Pearson correlation | 선형 종속, 부호 있음 | Standard |
| Spearman, Kendall | 단조 종속, 부호 있음, 순위 기반 | Standard |
| ANOVA F, chi-square | 범주형 target 또는 feature 에 대한 집단 분리 | Standard |
| Mutual information | 임의의 종속, 밀도 추정을 대가로 함 | Standard |
| Distance correlation | 임의의 종속, 독립일 때만 0 | Standard |
| HSIC | Kernel feature space 에서의 임의의 종속 | Standard |
| mRMR [[21](#ref-21)] | Target 과의 관련성에서 이미 고른 것과의 중복을 뺀 값 | Standard |

한계는 모두에게 같다. 혼자서는 아무것도 지니지 않고 조합에서 전부를 지니는 feature 는 0 점을
받으며, exclusive-or 가 가장 작은 예이다. 각 입력은 target 과 독립이고 둘이 함께면 target 이
결정된다. 그래서 순수한 filter 는 정작 중요했던 두 feature 만 떨어뜨릴 수 있으며, 이 계열이
model 을 설명하는 데가 아니라 아주 넓은 표를 modeling 전에 줄이는 데 쓰이는 까닭이 그것이다.

mRMR 은 feature 하나 너머를 보는 유일한 구성원으로, 후보에게 이미 고른 것과의 중복만큼 벌점을
준다. 그 때문에 출력이 순위가 아니라 집합이 되고, 그 집합은 만들어진 순서에 달려 있다.

## 5. Model-Internal Structure

여기서는 숫자를 적합된 model 자체에서 읽는다. 아무것도 흔들지 않고 아무것도 다시 평가하지
않으므로 학습이 끝난 뒤의 비용이 0 이며, 이 계열이 널리 쓰이는 까닭이 그것이다. 약점의 까닭도
같다. 숫자는 model 이 어떻게, 어느 데이터에서 지어졌는지를 적은 것이므로 section 2.5 가 온전히
적용된다.

Table 3. Model-internal structure measures

| Method | What it scores | Status |
|--------|----------------|--------|
| Standardized coefficient | Feature 의 표준편차당 반응의 변화, 부호 있음 | Standard |
| Lasso path [[22](#ref-22)] | 정규화 경로를 따른 진입 순서와 계수 크기 | Standard |
| MDI | 그 feature 로 낸 split 전체의 불순도 감소 합 | Standard |
| Split count, cover | 그 feature 로 split 한 횟수와 그 split 이 닿은 행의 수 | Standard |
| PLS VIP | 반응을 예측하는 잠재성분에 대한 feature 의 기여 | Standard |
| PCA loading | 재구성 분산에 대한 기여이며 target 을 참조하지 않음 | Standard |
| Attention weight | Model 이 입력 위치에 두는 가중치 | Standard |

특정 구성원에게만 붙는 주의가 둘 있다. 계수는 feature 를 한 척도에 올린 뒤에야 서로 견줄 수
있으며, 표준화하지 않은 계수는 importance 가 아니라 단위에 대한 진술이다. MDI 는 학습 데이터에서
계산되고 값이 다양한 feature 쪽으로 치우치는데, random forest 에 대해 그 치우침과 그것을 낳는
기제가 함께 기록되어 있다 [[2](#ref-2)]. 연속형 잡음 열이 실제 이진 예측변수를 이 까닭만으로
앞지를 수 있다.

PCA loading 은 나머지에서 덜어내려고 표에 넣었다. 그것은 입력을 재구성하는 몫으로 feature 를
채점하며, 어떤 feature 는 target 에 대해 아무것도 지니지 않은 채 주성분을 지배할 수 있다.

## 6. Perturbation And Removal

이 계열은 입력을 흔들고 model 에 무슨 일이 생기는지 잰다. Model 을 새 행에 대해 부를 수만
있으면 되므로 model-agnostic 인 첫 계열이고, 흔든 행을 held-out label 에 견주어 채점할 수
있으므로 loss 기반일 수 있는 첫 계열이다.

Table 4. Perturbation and removal measures

| Method | What it scores | Status |
|--------|----------------|--------|
| Permutation importance, MDA [[1](#ref-1)] | 한 열을 섞어 target 과의 연결을 끊었을 때의 loss 증가 | Standard |
| Conditional permutation [[3](#ref-3)] | 같은 것이되, 상관된 feature 의 칸 안으로 섞기를 제한 | Standard |
| Drop-column, LOCO | 그 feature 없이 model 을 다시 적합했을 때의 loss 증가 | Standard |
| Model class reliance [[4](#ref-4)] | 거의 같은 정도로 잘 맞는 모든 model 에 걸친 permutation importance 의 구간 | Recent |
| PDP spread | 그 feature 의 partial dependence 곡선의 표준편차 | Standard |
| ALE spread [[15](#ref-15)] | 같은 것이되, 주변 평균 대신 국소 차분을 쓰는 accumulated local effects 에서 | Recent |
| H-statistic [[14](#ref-14)] | 가법 부분으로 설명되지 않는 예측 분산의 몫, 한 쌍씩 | Standard |

Permutation importance 는 이 계열의 기본값이고 결함이 가장 잘 기술된 방법이다. 한 열을 나머지와
무관하게 섞는 것은 section 2.4 의 marginal 자리이므로 상관된 feature 는 데이터에 없는 조합에서
평가되고, 나온 숫자는 model 의 의존도와 데이터 manifold 밖에서의 거동을 섞는다 [[5](#ref-5)].
같은 비판이 partial dependence 에 닿으며, accumulated local effects 는 model 을 주변분포 전체에
대해 평균내는 대신 이웃 안에서 국소 차분을 평균내어 그것을 피하려고 만들어졌다 [[15](#ref-15)].

재학습은 permutation 과 다른 물음에 답하며, 그 차이는 눈에 두고 있을 값어치가 있다. Permutation
은 이 model 이 그 feature 를 얼마나 필요로 하는지 묻는다. 재학습은 그 feature 없이 적합한 model
이 얼마나 잃는지 묻는데, 상관된 두 feature 는 따로따로, 그리고 참되게 "아무것도" 라고 답할 수
있다. Model class reliance 는 그 간극을 반대쪽에서 닫는데, 마침 학습된 하나의 의존도가 아니라
잘 맞는 model 전체에 걸친 의존도의 구간을 보고한다 [[4](#ref-4)].

## 7. Gradient Attribution

Model 을 입력에 대해 미분하는 방식은 model 이 미분 가능한 데서만 쓸 수 있고, 기울기를 한 점에서
잡으므로 태생적으로 local 이다. 이 계열이 있는 까닭은 비용이다. 입력이 십만 개인 network 는 열
하나씩 섞을 수 없고, 역전파 한 번이 모든 입력에 숫자를 한꺼번에 준다.

Table 5. Gradient attribution measures

| Method | What it scores | Status |
|--------|----------------|--------|
| Saliency | 입력에서의 출력 기울기의 크기 | Standard |
| Gradient times input | 같은 것에 입력값을 곱한 것이며, 합을 출력과 견줄 수 있게 함 | Standard |
| Integrated gradients [[10](#ref-10)] | Baseline 에서 입력까지의 경로를 따라 적분한 기울기 | Standard |
| DeepLIFT | 기준점과의 차이를 layer 별 규칙으로 거슬러 전파한 것 | Standard |
| LRP [[11](#ref-11)] | 출력의 relevance 를 layer 마다 입력으로 되나눈 것 | Standard |
| Grad-CAM [[12](#ref-12)] | 마지막 convolution layer 의 기울기 가중 활성 지도이며, 성긴 공간 mask | Standard |

Integrated gradients 는 정해진 성질들을 만족하는 attribution 으로서 공리적으로 도입되었고,
baseline 의 선택이 답을 정하는 자유 parameter 이다 [[10](#ref-10)]. 검은 image, 흐린 image,
평균 image 는 같은 예측에 대한 세 개의 다른 물음이다.

이 계열이 축 위에서 어디에 있는지는 분명히 적어 둘 값어치가 있다. Output 기반이고 결코 loss
기반이 아니므로, 예측이 어디서 왔는지는 말하고 그 예측이 좋았는지는 결코 말하지 않는다. 확신에
차서 틀린 예측도 깔끔한 attribution 지도를 낸다.

## 8. Game-Theoretic Attribution

이 계열은 feature 를 참가자로, 예측을 그들에게 나눌 배당으로 다룬다. Shapley value 는 작은
공정성 조건 묶음을 만족하는 유일한 분배이며, 그것이 이 계열을 추천하는 근거이자 비용을 feature
수에 지수적으로 만드는 까닭인데 그 값이 모든 부분집합에 걸친 평균이기 때문이다.

Table 6. Game-theoretic attribution measures

| Method | What it scores | Status |
|--------|----------------|--------|
| LIME [[6](#ref-6)] | 한 행의 이웃에서 black box 에 맞춘 희소 선형 model 의 계수 | Standard |
| KernelSHAP [[7](#ref-7)] | 예측의 Shapley value 이며, 표집한 부분집합에 대한 가중 최소제곱으로 추정 | Standard |
| TreeSHAP [[8](#ref-8)] | Tree ensemble 에 대해 같은 것을 정확히, 다항 시간에 | Standard |
| DeepSHAP | Network 에 대해 같은 것을 layer 별 attribution 을 합성하여 | Standard |
| SAGE [[9](#ref-9)] | 예측이 아니라 loss 의 Shapley value 이며, feature 마다 global 숫자 하나 | Recent |
| Shapley effects | 출력 분산의 Shapley value 이며, 이 계열을 section 9 에 잇는다 | Recent |
| FastSHAP [[20](#ref-20)] | 그 목적으로 학습된 explainer 가 순전파 한 번에 내놓는 Shapley value | Recent |

통합이 이 계열의 기여이다. LIME, DeepLIFT, layer-wise relevance propagation 이 하나의 가법
attribution 방법 부류의 구성원임이 보여졌고, Shapley value 가 그 부류의 특별한 원소이다
[[7](#ref-7)]. SAGE 가 속한 가법 importance 틀은 또 다른 global 측도 묶음을 같은 방식으로 덮는다
[[9](#ref-9)].

쓰임을 정하는 실무적 논점이 셋 있다. 첫째, 부분집합 평균에는 일부 feature 가 빠진 채로 낸 예측의
값이 있어야 하고, 그것을 공급하는 두 길인 marginal 과 conditional 이 section 2.4 의 갈림을 그대로
되풀이하며 상관된 feature 에 다른 attribution 을 준다. 둘째, 추정 algorithm 이 여럿이고 정확도가
서로 다르며, 그것이 한자리에서 조사되고 비교되어 있다 [[24](#ref-24)]. 셋째, 공리는 배당을 어떻게
나눌지를 묶을 뿐 그 나눔이 사람의 물음에 답하는지에 대해서는 아무 말도 하지 않으며, 그것이 이
계열에 대한 상시 비판의 알맹이다 [[19](#ref-19)].

## 9. Variance-Based Sensitivity

Variance-based sensitivity 는 통계학이 아니라 computer model 연구에서 왔고, 출력의 분산을 입력
각각과 interaction 각각의 기여로 분해한다. 분해되는 것이 model 의 출력이므로 label 이 끼어들지
않고, 입력은 관측된 것이 아니라 대개 설계에서 표집된다.

Table 7. Variance-based sensitivity measures

| Method | What it scores | Status |
|--------|----------------|--------|
| Morris elementary effects | 성긴 격자 위 한 번에 하나씩 바꾼 변화의 평균과 퍼짐이며, screening 단계 | Standard |
| Sobol first-order index [[13](#ref-13)] | 그 feature 혼자로 설명되는 출력 분산의 몫 | Standard |
| Sobol total-effect index [[13](#ref-13)] | 그 feature 와 그것이 끼는 모든 interaction 으로 설명되는 몫 | Standard |
| FAST | 같은 지수를 표집이 아니라 주파수 해석으로 추정 | Standard |
| Functional ANOVA | 주효과와 차수가 오르는 interaction 항으로의 완전 분해 | Standard |

Sobol 지수 한 쌍이 이 계열이 내놓은 가장 쓸모 있는 것이다. First-order 지수와 total-effect 지수가
feature 를 아래와 위에서 묶고, 둘의 간격이 바로 그 영향 가운데 interaction 에 사는 몫이다.
First-order 지수가 0 에 가깝고 total-effect 지수가 큰 feature 는 조합에서만 중요하며, 바로 그것이
section 4 의 data-only 계열이 전혀 볼 수 없는 경우이다.

확인해야 할 가정은 입력의 독립이다. 분산 분해는 독립인 입력에 대해 유도되었고 그것은 설계된
실험이 공급하고 관측 데이터는 공급하지 않으므로, 상관된 공정 데이터에서 이 지수들은 marginal
permutation 과 같은 manifold 밖 문제를 안는다.

## 10. Error-Controlled Selection

뒤따르는 것은 이 문서의 나머지와 다른 물음에 답한다. 이 방법들은 feature 의 순위를 매기지 않고
집합을 돌려주며, 그 집합에서 헛된 것의 비율에 상한을 붙인다. 그것이 section 2.7 의 세 번째
guarantee 이고, 귀무 feature 가 어떻게 생겼을지에 대한 명시적 model 을 대가로 산다.

Table 8. Error-controlled selection methods

| Method | What it scores | Status |
|--------|----------------|--------|
| Knockoffs [[16](#ref-16)] | 합성된 음성 대조 사본과의 대비로 얻은, 거짓 발견 비율이 통제된 선택 집합 | Recent |
| LOCO inference [[17](#ref-17)] | 그 feature 를 뺐을 때의 예측오차 증가에 대한 신뢰구간 | Recent |
| VIM [[18](#ref-18)] | 적합된 model 을 nuisance 로 두고 얻은 모집단 importance parameter 의 신뢰구간 | Recent |
| Decorrelated LOCO [[23](#ref-23)] | LOCO 와 같되, parameter 에서 공변량 사이 상관의 영향을 걷어낸 것 | Recent |

Knockoff 구성은 feature 마다 상관 구조는 지키면서 target 에 대한 정보는 지니지 않는 합성 사본을
만든다. 자기 사본을 이기지 못하는 feature 는 귀무에 대한 증거이고, 그것을 세면 반응의 분포에
아무 가정 없이 오류율이 나온다. 이 절차는 선형 model 에서 시작했고 model-X 구성으로 임의의
model 까지 넓혀졌으며, 그 구성은 가정을 feature 의 분포 쪽으로 옮긴다 [[16](#ref-16)].

Inference 쪽 구성원은 importance 를 모집단 대비 — 모든 feature 로 얻을 수 있는 예측력과 문제의
feature 를 뺀 채로 얻을 수 있는 예측력의 차이 — 로 정의하고, 그 대비를 어떤 algorithm 으로든
추정한다 [[18](#ref-18)]. Leave-one-out parameter 를 읽기 어렵게 만드는 상관 의존성이 바로
decorrelated 판이 고치는 것이다 [[23](#ref-23)].

## 11. Selection Guide

Table 9 는 일곱 계열을 section 2 의 축 위에 놓고, Table 10 은 반대 방향으로 물음에서 그것에
답하는 계열로 간다.

Table 9. Families on the axes

| Family | Object | Scope | Effect | Conditioning | Labels | Cost |
|--------|--------|-------|--------|--------------|--------|------|
| A. Data-only association | 분포 | Global | Association | Marginal | 필요 | 가장 낮음 |
| B. Model-internal structure | 적합된 model | Global | Structure | 해당 없음 | 쓰지 않음 | 학습 외에 없음 |
| C. Perturbation and removal | 적합된 model 또는 model class | Global | Loss 또는 output | 둘 다 가능 | Loss 에 필요 | Feature 마다 한 번의 통과, 또는 한 번의 재적합 |
| D. Gradient attribution | 적합된 model | Local | Output | Marginal | 쓰지 않음 | 역전파 한 번 |
| E. Game-theoretic attribution | 적합된 model | Local, SAGE 는 global | Output, SAGE 는 loss | 둘 다 가능 | SAGE 에 필요 | 원리상 지수적, 실무에서는 표집 |
| F. Variance-based sensitivity | 적합된 model | Global | Output variance | Marginal, 독립 가정 | 쓰지 않음 | 큰 설계 표본 |
| G. Error-controlled selection | 분포 | Global | Loss | Conditional | 필요 | 가장 높음 |

Table 10. Question and the family that answers it

| Question | Family | Note |
|----------|--------|------|
| 아주 넓은 표에서 modeling 전에 남길 열은 어느 것인가 | A | Interaction 이 보이지 않으므로 자르는 선을 넉넉히 |
| 이 tree ensemble 은 어느 feature 로 split 했는가 | B | 학습 데이터뿐이며 값이 다양한 열 쪽으로 치우침 |
| 배포된 model 이 실제로 필요로 하는 feature 는 무엇인가 | C1 또는 C3, held-out 행에서 | 의존도에는 permutation, 필요성에는 재학습 |
| 상관을 감안하고도 중요한 feature 는 무엇인가 | C2 또는 G2 | 조건분포가 필요한 conditional 물음 |
| 이 예측 하나는 왜 나왔는가 | D 또는 E1 | 입력이 많은 network 에는 gradient, 표에는 Shapley |
| Interaction 을 넣어 데이터 전체에서 중요한 feature 는 무엇인가 | E2 또는 F2 | 관측 데이터에는 SAGE, 설계 표본에는 Sobol |
| 오류율과 함께 사실로 보고할 수 있는 feature 는 무엇인가 | G1 | 이 물음에 답하는 유일한 계열 |
| 답이 어느 model 을 적합했는가에 얼마나 달려 있는가 | C3, model class reliance | 숫자 하나 대신 구간 |

## 12. Failure Modes

아래 결함은 구현이 아니라 방법의 성질이며, 계열을 가로질러 되풀이된다.

- **상관이 공을 쪼갠다.** 같은 정보를 지닌 두 feature 는 어느 방법에서는 절반씩 받고 다른 방법에서는 각각 온전히 받으며, 어느 답도 틀리지 않았다. 그 선택이 section 2.4 의 marginal 대 conditional 선택이고, 어느 쪽을 택했는지 밝히지 않은 채 숫자를 보고하면 독자가 그것을 읽을 수 없다.
- **Structure 기반 점수는 과적합을 보상한다.** MDI 와 벌점 없는 계수는 model 이 적합된 그 데이터에서 계산되므로, 값이 다양한 잡음 열이 실제 예측변수를 앞지를 수 있다 [[2](#ref-2)].
- **Permutation 은 model 을 적합된 적 없는 자리에서 평가한다.** 상관된 열을 독립으로 섞으면 데이터 manifold 밖의 행이 만들어지고, 그 자리에서의 model 거동이 점수에 들어온다 [[5](#ref-5)].
- **Importance 가 0 이라고 무관한 것은 아니다.** Drop-one 재학습에서는 표에 정확한 사본이 있는 feature 가 0 점을 받고, 그 사본도 0 점을 받는다.
- **Importance 는 인과가 아니다.** 이 문서의 모든 방법은 association 이나 model 의 의존도를 보고한다. 어떤 feature 는 model 의 가장 강한 입력이면서 개입했을 때 target 에 아무 영향이 없을 수 있고, 숫자를 어떻게 다시 가중해도 그것은 달라지지 않는다.
- **순위는 불안정하다.** Seed 를 바꿔 다시 적합하거나 행을 재표집하면 순위의 가운데가 뒤바뀐다. 퍼짐 없이 내놓은 단일 순위는 데이터가 뒷받침하는 것보다 과장하며, section 10 의 구간 방법이 공급하는 것이 그 퍼짐이다.
- **Local 값을 집계하면 양이 바뀐다.** 부호 있는 local attribution 은 평균에서 상쇄되므로 절댓값의 평균을 쓰는데, 그것은 global 방법의 출력을 요약한 것이 아니라 다른 측도이다.
- **One-hot 열은 자기 변수를 과소평가한다.** Level 을 따로 채점하면 한 변수의 기여가 그 level 들로 쪼개지며, 묶기를 쓸 수 없는 structure 기반 계열을 뺀 모든 계열에서 그렇다.
- **공리는 물음까지 옮겨 가지 않는다.** Shapley value 의 유일성은 배당을 어떻게 나누는가에 대한 진술이고, 상시 비판은 그것이 그 나눔을 사람이 알고자 한 것에 대한 답으로 만들어 주지는 않는다는 것이다 [[19](#ref-19)]. 보통의 network 만큼 풍부한 model 부류에서는 complete 하고 linear 한 attribution 방법 — integrated gradients 와 Shapley 계열이 그 안에 든다 — 이 여러 자연스러운 최종 과제에서 무작위 추측보다 나을 것이 없음이 보여졌다 [[25](#ref-25)].

## References

<a id="ref-1"></a>
[1] Breiman, L. (2001). [Random Forests](https://doi.org/10.1023/A:1010933404324). *Machine Learning*, 45(1), 5–32.

<a id="ref-2"></a>
[2] Strobl, C., Boulesteix, A.-L., Zeileis, A. and Hothorn, T. (2007). [Bias in random forest variable importance measures: illustrations, sources and a solution](https://doi.org/10.1186/1471-2105-8-25). *BMC Bioinformatics*, 8, 25.

<a id="ref-3"></a>
[3] Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T. and Zeileis, A. (2008). [Conditional variable importance for random forests](https://doi.org/10.1186/1471-2105-9-307). *BMC Bioinformatics*, 9, 307.

<a id="ref-4"></a>
[4] Fisher, A., Rudin, C. and Dominici, F. (2019). [All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously](https://jmlr.org/papers/v20/18-760.html). *Journal of Machine Learning Research*, 20(177), 1–81.

<a id="ref-5"></a>
[5] Hooker, G., Mentch, L. and Zhou, S. (2021). [Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance](https://doi.org/10.1007/s11222-021-10057-z). *Statistics and Computing*, 31, 82.

<a id="ref-6"></a>
[6] Ribeiro, M. T., Singh, S. and Guestrin, C. (2016). ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://doi.org/10.1145/2939672.2939778). *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144.

<a id="ref-7"></a>
[7] Lundberg, S. M. and Lee, S.-I. (2017). [A Unified Approach to Interpreting Model Predictions](https://papers.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html). *Advances in Neural Information Processing Systems*, 30, 4765–4774.

<a id="ref-8"></a>
[8] Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N. and Lee, S.-I. (2020). [From local explanations to global understanding with explainable AI for trees](https://doi.org/10.1038/s42256-019-0138-9). *Nature Machine Intelligence*, 2, 56–67.

<a id="ref-9"></a>
[9] Covert, I., Lundberg, S. M. and Lee, S.-I. (2020). [Understanding Global Feature Contributions With Additive Importance Measures](https://papers.neurips.cc/paper/2020/hash/c7bf0b7c1a86d5eb3be2c722cf2cf746-Abstract.html). *Advances in Neural Information Processing Systems*, 33.

<a id="ref-10"></a>
[10] Sundararajan, M., Taly, A. and Yan, Q. (2017). [Axiomatic Attribution for Deep Networks](https://proceedings.mlr.press/v70/sundararajan17a.html). *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 3319–3328.

<a id="ref-11"></a>
[11] Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R. and Samek, W. (2015). [On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation](https://doi.org/10.1371/journal.pone.0130140). *PLoS ONE*, 10(7), e0130140.

<a id="ref-12"></a>
[12] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D. (2020). [Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization](https://doi.org/10.1007/s11263-019-01228-7). *International Journal of Computer Vision*, 128, 336–359.

<a id="ref-13"></a>
[13] Sobol', I. M. (2001). [Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates](https://doi.org/10.1016/S0378-4754(00)00270-6). *Mathematics and Computers in Simulation*, 55(1–3), 271–280.

<a id="ref-14"></a>
[14] Friedman, J. H. and Popescu, B. E. (2008). [Predictive learning via rule ensembles](https://doi.org/10.1214/07-AOAS148). *The Annals of Applied Statistics*, 2(3), 916–954.

<a id="ref-15"></a>
[15] Apley, D. W. and Zhu, J. (2020). [Visualizing the effects of predictor variables in black box supervised learning models](https://doi.org/10.1111/rssb.12377). *Journal of the Royal Statistical Society: Series B*, 82(4), 1059–1086.

<a id="ref-16"></a>
[16] Candès, E., Fan, Y., Janson, L. and Lv, J. (2018). [Panning for gold: 'model-X' knockoffs for high dimensional controlled variable selection](https://doi.org/10.1111/rssb.12265). *Journal of the Royal Statistical Society: Series B*, 80(3), 551–577.

<a id="ref-17"></a>
[17] Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. and Wasserman, L. (2018). [Distribution-Free Predictive Inference for Regression](https://doi.org/10.1080/01621459.2017.1307116). *Journal of the American Statistical Association*, 113(523), 1094–1111.

<a id="ref-18"></a>
[18] Williamson, B. D., Gilbert, P. B., Simon, N. R. and Carone, M. (2023). [A General Framework for Inference on Algorithm-Agnostic Variable Importance](https://doi.org/10.1080/01621459.2021.2003200). *Journal of the American Statistical Association*, 118(543), 1645–1658.

<a id="ref-19"></a>
[19] Kumar, I. E., Venkatasubramanian, S., Scheidegger, C. and Friedler, S. (2020). [Problems with Shapley-value-based explanations as feature importance measures](https://proceedings.mlr.press/v119/kumar20e.html). *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, 5491–5500.

<a id="ref-20"></a>
[20] Jethani, N., Sudarshan, M., Covert, I., Lee, S.-I. and Ranganath, R. (2022). [FastSHAP: Real-Time Shapley Value Estimation](https://arxiv.org/abs/2107.07436). *International Conference on Learning Representations*.

<a id="ref-21"></a>
[21] Peng, H., Long, F. and Ding, C. (2005). [Feature Selection Based on Mutual Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy](https://doi.org/10.1109/TPAMI.2005.159). *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 27(8), 1226–1238.

<a id="ref-22"></a>
[22] Tibshirani, R. (1996). [Regression Shrinkage and Selection via the Lasso](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x). *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.

<a id="ref-23"></a>
[23] Verdinelli, I. and Wasserman, L. (2024). [Decorrelated Variable Importance](https://www.jmlr.org/papers/volume25/22-0801/22-0801.pdf). *Journal of Machine Learning Research*, 25(7), 1–27.

<a id="ref-24"></a>
[24] Chen, H., Covert, I. C., Lundberg, S. M. and Lee, S.-I. (2023). [Algorithms to estimate Shapley value feature attributions](https://doi.org/10.1038/s42256-023-00657-x). *Nature Machine Intelligence*, 5, 590–601.

<a id="ref-25"></a>
[25] Bilodeau, B., Jaques, N., Koh, P. W. and Kim, B. (2024). [Impossibility theorems for feature attribution](https://doi.org/10.1073/pnas.2304406120). *Proceedings of the National Academy of Sciences*, 121(2), e2304406120.

---

## Appendix A. Terminology

- **ALE**: Accumulated local effects 이며, model 을 주변분포에 대해 평균낸 것이 아니라 이웃 안의 국소 차분으로 지은 곡선.
- **Family**: Fig 1 의 최상위 묶음 일곱 가운데 하나이며, 숫자가 같은 출처에서 계산되는 방법들을 담는다. 문헌이 이 층위가 아니라 축에 이름을 붙이고 이 층위에 정해진 낱말이 없어 여기서 쓴다.
- **FAST**: Fourier amplitude sensitivity test 이며, model 출력의 주파수 해석에 기반한 Sobol 지수의 추정량.
- **FDR**: False discovery rate 이며, 선택된 집합에서 헛된 것이 차지하는 비율의 기댓값.
- **HSIC**: Hilbert-Schmidt independence criterion 이며, kernel feature space 에서 계산하는 종속 측도.
- **LIME**: Local interpretable model-agnostic explanations 이며, 한 행의 이웃에서 black box 에 맞춘 희소 선형 model.
- **LOCO**: Leave out covariates 이며, 그 feature 없이 model 을 다시 적합했을 때의 예측오차 증가.
- **LRP**: Layer-wise relevance propagation 이며, 출력의 relevance 를 layer 별 규칙으로 입력에 되나누는 것.
- **MDA**: Mean decrease in accuracy 이며, 한 열을 무작위로 섞었을 때의 loss 증가.
- **MDI**: Mean decrease in impurity 이며, tree ensemble 이 그 feature 로 낸 split 전체의 불순도 감소 합.
- **mRMR**: Minimum redundancy maximum relevance 이며, target 과의 관련성에서 이미 고른 feature 와의 중복을 빼는 greedy 선택 기준.
- **PDP**: Partial dependence plot 이며, 나머지 feature 의 주변분포에 대해 평균낸 model 출력.
- **SAGE**: Shapley additive global importance 이며, 예측 하나가 아니라 model loss 의 Shapley value.
- **SHAP**: Shapley additive explanations 이며, 예측의 Shapley value 를 추정하는 local attribution 계열.
- **Shapley value**: Efficiency, symmetry, null player 성질, linearity 를 만족하는 참가자 사이 배당의 유일한 분배.
- **Total-effect index**: 한 feature 와 그것이 끼는 모든 interaction 에 돌릴 수 있는 출력 분산의 몫.
- **VIM**: Variable importance measure 이며, 여기서는 model 을 nuisance 로 두고 추정하는 예측력의 모집단 대비를 가리킨다.
- **VIP**: Variable importance in projection 이며, partial least squares model 의 잠재성분에 대한 feature 의 기여.
