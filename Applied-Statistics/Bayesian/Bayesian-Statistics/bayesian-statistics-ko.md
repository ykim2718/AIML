# Bayesian Statistics
Rev. 7 | Created: 2026-09-20 | Updated: 2026-09-23 11:29 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Placement](#31-placement)
- [4. Conditional Probability and Bayes' Rule](#4-conditional-probability-and-bayes-rule)
- [5. The Pool Table Problem](#5-the-pool-table-problem)
  - [5.1 Setup](#51-setup)
  - [5.2 The Prior](#52-the-prior)
  - [5.3 The Classical Answer](#53-the-classical-answer)
  - [5.4 The Bayesian Answer](#54-the-bayesian-answer)
  - [5.5 Comparison](#55-comparison)
- [6. The Coin Example](#6-the-coin-example)
  - [6.1 Setup](#61-setup)
  - [6.2 The Maximum Likelihood Estimate](#62-the-maximum-likelihood-estimate)
  - [6.3 The Prior as a Choice](#63-the-prior-as-a-choice)
  - [6.4 The Posterior](#64-the-posterior)
  - [6.5 What Enough Data Does](#65-what-enough-data-does)
- [7. Limits](#7-limits)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Derivations](#appendix-b-derivations)
  - [B.1 Bob's Win Probability at a Given p](#b1-bobs-win-probability-at-a-given-p)
  - [B.2 Evaluation of the Expectation](#b2-evaluation-of-the-expectation)

## 1. Purpose

- **Problem Statement**: 같은 자료로 같은 물음에 답해도 고전 통계학과 Bayes 통계학이 서로 다른 값을 내며, 어느 계열로 푼 값인지 밝히지 않으면 그 값이 무엇을 뜻하는지 말할 수 없다.
- **Goal**: 두 계열이 무엇을 확률변수로 보고 조건부 확률의 어느 쪽을 구하는지 가르고, 한 문제에서 두 답이 갈리는 폭을 수치로 보여, 손에 있는 문제에 어느 계열을 쓸지 고르게 한다.
- **Non-Goal**: Posterior 를 실제로 계산하는 수치 기법은 다루지 않는다. 여기서 푸는 예제는 적분이 닫힌 형태로 풀리는 경우 하나뿐이다.

## 2. Summary

고전 통계학과 Bayes 통계학은 조건부 확률의 어느 쪽을 구하는가로 갈린다. 고전 통계학은 model 을 고정해 두고 $P(\mathrm{data} \mid \mathrm{model})$ 을 구해 그 model 이 타당한지 검정하고, Bayes 통계학은 자료를 고정해 두고 $P(\mathrm{model} \mid \mathrm{data})$ 를 구해 model 의 가능성을 계산한다 [[1](#ref-1)].

5 꼭지의 당구대 문제에서 고전 통계학은 어느 점추정값을 고르는가에 따라 약 18 : 1 이나 7 : 1 을 내고, Bayes 통계학은 10 : 1 하나를 낸다. 같은 자료에서 밥이 이길 확률이 $27/512$, $1/8$, $1/11$ 로 갈린다.

6 꼭지의 동전 예제는 prior 가 선택일 때를 다룬다. 100 번의 던지기에서는 서로 다른 네 prior 가 거의 같은 posterior 를 내며, prior 의 선택이 답을 정하는 것은 자료가 적을 때이다.

Bayes 통계학이 언제나 낫다고 말할 수는 없다. 적분이 닫힌 형태로 풀리지 않는 경우가 많고, prior 를 무엇으로 둘지의 가정이 타당하지 않으면 결과를 믿을 수 없으며, model 을 확률변수로 다루는 용법 자체에 이견이 있다. 7 꼭지가 이 셋을 적는다.

## 3. Taxonomy and its Hierarchy

Model 과 자료를 두고 세울 수 있는 조건부 확률이 둘이며, 어느 쪽을 구하는가가 고전 통계학과 Bayes 통계학을 가른다. 아래 <a href="#fig-1">Fig 1</a> 이 그 갈림과 각 계열이 딸려 요구하는 것을 한 장에 담는다.

```text
A question about a model and the data it might produce
|
+-- classical statistics --- P(data | model)
|                            the model parameter is an unknown fixed value
|                            route: assume one value, test it, predict with it
|                            input: the data
|
+-- Bayesian statistics ---- P(model | data)
                             the model parameter is a random variable
                             route: weight every value by the posterior, then average
                             input: the data and a prior over the parameter
```

<a id="fig-1"></a>
Fig 1. The two schools and what each takes as given

갈림은 무엇을 고정으로 두는가에서 시작한다. 고전 통계학은 model 을 고정으로 두므로 parameter 하나를 골라야 하고, Bayes 통계학은 자료를 고정으로 두므로 parameter 전체에 분포를 얹어야 한다. 아래로 내려갈수록 답이 쓰는 정보가 늘어나고, 그만큼 입력으로 요구하는 것도 늘어난다.

### 3.1 Placement

Table 1. The two schools side by side

| #   | Aspect             | Classical statistics                   | Bayesian statistics                    |
| :-: | :----------------: | :------------------------------------: | :------------------------------------: |
| 1   | 구하는 확률        | $P(\mathrm{data} \mid \mathrm{model})$ | $P(\mathrm{model} \mid \mathrm{data})$ |
| 2   | Model 의 parameter | 알지 못하는 고정값                     | 확률변수                               |
| 3   | 답을 얻는 길       | 값 하나를 가정하고 검정                | 모든 값에 걸쳐 평균                    |
| 4   | 요구하는 입력      | 자료                                   | 자료와 prior                           |
| 5   | 5 꼭지의 답        | 약 18 : 1 또는 7 : 1                   | 10 : 1                                 |

2 행이 나머지를 결정한다. Parameter 를 고정값으로 보면 그 값을 추정해 검정하는 길밖에 없고, 확률변수로 보면 그 분포를 자료로 갱신한 뒤 평균내는 길이 열린다. 4 행이 Bayes 쪽의 대가다. Prior 는 자료에서 나오지 않으므로 분석자가 넣어야 하며, 그 선택이 5 행의 답을 움직인다.

## 4. Conditional Probability and Bayes' Rule

Bayes 통계학의 계산은 조건부 확률과 Bayes 정리 둘로 이루어진다. 조건부 확률은 두 사건이 함께 일어날 확률을 조건이 되는 사건의 확률로 나눈 값이다.

```math
P(A \mid B) = \frac{P(A \cap B)}{P(B)} \hspace{19em} (1)
```

사건이 아니라 확률변수로 적으면 아래와 같고, $B$ 가 값 $b$ 로 관측되었을 때 $A$ 가 값 $a$ 를 가질 확률을 뜻한다.

```math
P(A = a \mid B = b) = \frac{P(A = a,\, B = b)}{P(B = b)} \hspace{19em} (2)
```

식 (2) 는 두 확률을 과거의 자료에서 셀 수 있으면 그대로 쓴다. 셀 수 없을 때 Bayes 정리가 방향을 뒤집는다. 전확률공식을 분모에 넣은 꼴이 아래이다 [[2](#ref-2)].

```math
P(A = a \mid B = b) = \frac{P(B = b \mid A = a)\, P(A = a)}{\sum_{a' \in A} P(B = b \mid A = a')\, P(A = a')} \hspace{19em} (3)
```

방향을 뒤집는 것이 이득이 되는 자리가 있다. 어떤 체질을 가진 사람이 특정 암에 걸릴 확률은 직접 구하기 어렵지만, 그 암에 걸린 사람 가운데 그 체질을 가진 사람의 비율은 상대적으로 구하기 쉽다. 식 (3) 은 구하기 쉬운 쪽을 재료로 삼아 구하기 어려운 쪽을 내놓는다.

Table 2. Terms of Bayes' rule

| #   | Term       | Position in equation (3)     | Reading                             |
| :-: | :--------: | :--------------------------: | :---------------------------------: |
| 1   | Prior      | 분자의 $P(A = a)$            | 자료를 보기 전 $A$ 에 두는 분포     |
| 2   | Likelihood | 분자의 $P(B = b \mid A = a)$ | $A$ 를 고정했을 때 관측이 나올 확률 |
| 3   | Evidence   | 분모 전체                    | 모든 $a'$ 에 걸쳐 분자를 합한 값    |
| 4   | Posterior  | 좌변                         | 관측을 반영해 갱신한 $A$ 의 분포    |

3 행은 좌변을 확률분포로 만드는 분모다. $a$ 가 이산값이면 합이고 연속값이면 적분이며, 5 꼭지가 쓰는 것은 적분 쪽이다.

## 5. The Pool Table Problem

두 계열의 답이 갈리는 것을 한 문제에서 볼 수 있다. Eddy 가 든 당구대 문제를 그대로 쓰며
[[1](#ref-1)], Bayesian billiards problem 이라는 이름으로도 불린다.

### 5.1 Setup

두 사람이 보지 못하는 상태에서 제 3 자가 당구대에 공을 하나 굴려, 멈춘 자리에 기준선을 긋는다. 이후
공을 한 개씩 굴려 공이 기준선의 앨리스 쪽에 서면 앨리스가 1 점을, 밥 쪽에 서면 밥이 1 점을 가져간다.
두 사람은 기준선이 어디에 그어졌는지 끝까지 알지 못한다.

득점 확률이 영역의 넓이에 비례한다고 두면, 앨리스가 한 판에서 점수를 얻을 확률이 $p$ 일 때 밥의
확률은 $1 - p$ 이다. 승리 조건은 6 점 선취이고, 현재 앨리스가 5 점, 밥이 3 점이다. 이때 공평한 베팅
비율을 묻는다.

앨리스는 1 점만 더 얻으면 이기고, 밥은 남은 세 판을 연속으로 가져가야 이긴다. 따라서 $p$ 가 주어졌을
때 밥이 이길 확률은 아래와 같다.

```math
P(\mathrm{Bob\ wins} \mid p) = (1 - p)^3 \hspace{19em} (4)
```

식 (4) 의 유도는 [Appendix B](#appendix-b-derivations) 에 있다. 식 (4) 에는 관측되지 않은 $p$ 가
남아 있고, 고전 통계학과 Bayes 통계학이 갈리는 자리가 여기이며, 5.3 과 5.4 가 이 $p$ 를 서로 다르게
처리한다.

### 5.2 The Prior

기준선이 그어지는 방식이 $p$ 의 prior 를 정한다. 공이 서는 자리가 무작위이므로 기준선도 당구대 위에
고르게 퍼져 있고, 따라서 $p$ 는 0 과 1 사이에서 uniform distribution 을 따른다. Prior 를 분석자가 임의로
고른 것이 아니라 게임의 규칙에서 읽어낸 것이며, 5.4 가 $P(p)$ 를 상수로 두는 근거가 이것이다.

### 5.3 The Classical Answer

고전 통계학은 $p$ 를 point estimate 하나로 고정한 뒤 식 (4) 에 넣는다. 관측한 득점 비율을 쓰면
$\hat{p} = 5/8$ 이고, 밥이 이길 확률은 아래와 같다.

```math
P(\mathrm{Bob\ wins}) = (1 - \hat{p})^3 = \left(\frac{3}{8}\right)^3 = \frac{27}{512} \approx 0.053 \hspace{19em} (5)
```

Prior 의 기댓값 $0.5$ 를 쓰면 같은 자리에서 $1/8 = 0.125$ 가 나온다. 두 값이 두 배 넘게 갈리는 것은 어느
점추정값을 고르는가가 답을 정하기 때문이고, 값 하나로 고정하는 순간 그 선택이 얼마나 불확실한지는 답에
들어가지 않는다. 앨리스가 5 : 3 으로 앞서 있다는 사실이 $p$ 에 대해 말해 주는 것도 폭이 아니라 점 하나로
줄어든다.

### 5.4 The Bayesian Answer

Bayes 통계학은 $p$ 를 하나로 정하지 않고, 자료가 가리키는 $p$ 의 분포에 걸쳐 식 (4) 를 평균낸다.

```math
E = \int_{0}^{1} (1 - p)^3\, P(p \mid A = 5,\, B = 3)\, dp \hspace{19em} (6)
```

$p$ 가 0 과 1 사이의 실수이므로 합이 아니라 적분이다. 피적분 함수의 posterior 는 식 (3) 을 연속값에
적용해 얻는다.

```math
P(p \mid A = 5,\, B = 3) = \frac{P(A = 5,\, B = 3 \mid p)\, P(p)}{\int_{0}^{1} P(A = 5,\, B = 3 \mid p)\, P(p)\, dp} \hspace{19em} (7)
```

분자의 likelihood $P(A = 5, B = 3 \mid p)$ 는 한 판의 승률이 $p$ 일 때 여덟 판에서 앨리스가 5 점, 밥이
3 점을 얻을 확률이며 binomial distribution 을 따른다. 여기에 5.2 의 uniform prior 를 곱하면 상수 배를
빼고 likelihood 의 모양이 그대로 남는다.

```math
P(p \mid A = 5,\, B = 3) \propto p^5 (1 - p)^3 \hspace{19em} (8)
```

이 모양은 아래 밀도에서 $\alpha = 6$, $\beta = 4$ 인 경우와 같다. 즉 posterior 는 $\mathrm{Beta}(6, 4)$
이며, 일반적으로 uniform prior 아래 앨리스가 $a$ 판, 밥이 $b$ 판을 가져가면 $\mathrm{Beta}(a+1,\, b+1)$
이 된다.

```math
f(p) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\, \Gamma(\beta)}\, p^{\alpha - 1} (1 - p)^{\beta - 1} \hspace{19em} (9)
```

식 (6) 의 적분을 [Appendix B](#appendix-b-derivations) 가 풀어 아래를 낸다.

```math
E = \frac{1}{11} \approx 0.091 \hspace{19em} (10)
```

문헌에 따라 $p$ 를 밥의 득점 확률로 두기도 한다. 그때 likelihood 는 $p^3 (1 - p)^5$, posterior 는
$\mathrm{Beta}(4, 6)$, 밥의 승리 확률은 $p^3$ 이 된다. 두 규약은 $p$ 와 $1 - p$ 를 맞바꾼 것이어서 답은
같다.

### 5.5 Comparison

Table 3. Three routes to the same question

| #   | Route                        | Value used for $p$    | Bob's win probability  | Fair odds |
| :-: | :--------------------------: | :-------------------: | :--------------------: | :-------: |
| 1   | 고전 통계학. 관측 비율       | $5/8$                 | $27/512 \approx 0.053$ | 약 18 : 1 |
| 2   | 고전 통계학. Prior 기댓값    | $0.5$                 | $1/8 = 0.125$          | 7 : 1     |
| 3   | Bayes 통계학. Posterior 평균 | $\mathrm{Beta}(6, 4)$ | $1/11 \approx 0.091$   | 10 : 1    |

1 행과 2 행은 같은 자료에서 서로 다른 값을 고른 결과이고, 그 선택을 바꾸면 답이 세 배 가까이 움직인다.
3 행만이 $p$ 를 하나로 고르지 않고 자료가 남긴 분포를 그대로 들고 가므로, 고를 것이 없어 답이 하나로
정해진다.

## 6. The Coin Example

5 꼭지에서 prior 는 게임의 규칙이 정해 주었다. Prior 를 분석자가 골라야 할 때 그 선택이 답을 얼마나
움직이는지는 동전 던지기 예제가 보여 준다.

### 6.1 Setup

한쪽에 0, 다른 쪽에 1 이 적힌 동전을 100 번 던져 1 이 57 번, 0 이 43 번 나왔다. 1 이 나올 확률
$\theta$ 는 얼마이며, 이 동전은 공정한가를 묻는다.

가장 단순한 답은 관측 비율 $57/100 = 0.57$ 이다. 이 값이 $0.5$ 에서 얼마나 떨어져 있어야 공정하지
않다고 할 수 있는지는 이 값 하나만으로 말할 수 없다.

### 6.2 The Maximum Likelihood Estimate

한 번의 던지기에서 1 이 나올 확률이 $\theta$ 이고 0 이 나올 확률이 $1 - \theta$ 이다.

```math
p(1 \mid \theta) = \theta, \qquad p(0 \mid \theta) = 1 - \theta \hspace{19em} (11)
```

던지기가 서로 독립이므로 100 번의 결합확률은 각 던지기의 곱이고, 순서와 무관하게 1 과 0 의 개수만
남는다.

```math
p(\mathrm{data} \mid \theta) = \theta^{57} (1 - \theta)^{43} \hspace{19em} (12)
```

식 (12) 를 최대로 만드는 $\theta$ 가 maximum likelihood 추정값이며, 그 값은 $0.57$ 로 관측 비율과
같다. 두 값이 같아지는 것은 이 model 에서 그럴 뿐이고 언제나 그런 것은 아니다.

### 6.3 The Prior as a Choice

Bayes 쪽으로 가려면 $\theta$ 에 prior 를 얹어야 한다. $\theta$ 가 확률이므로 prior 는 구간 $[0, 1]$
위의 분포여야 하고, 그 조건만 지키면 어떤 모양이든 둘 수 있다. Beta distribution 이 이 모양들을 한
족으로 묶는다.

```math
f(\theta \mid a, b) = \frac{1}{\mathrm{B}(a, b)}\, \theta^{a-1} (1 - \theta)^{b-1}, \qquad \mathrm{B}(a, b) = \int_{0}^{1} \theta^{a-1} (1 - \theta)^{b-1}\, d\theta \hspace{19em} (13)
```

$\mathrm{B}(a, b)$ 는 밀도의 적분을 1 로 만드는 정규화 상수이며, 식 (17) 의 beta integral 과 같은
적분이다. 아래 네 쌍이 서로 다른 믿음을 같은 족 안에서 나타낸다.

Table 4. Prior beliefs and the beta parameters that express them

| #   | Prior belief             | Parameters $(a, b)$ | Shape                     |
| :-: | :----------------------: | :-----------------: | :-----------------------: |
| 1   | 공정에 가깝다            | $(10, 10)$          | $0.5$ 에서 뾰족한 봉우리  |
| 2   | 0 쪽으로 치우쳤다        | $(1, 10)$           | 0 에서 급히 내려가는 모양 |
| 3   | 0 이나 1 쪽으로 치우쳤다 | $(0.5, 0.5)$        | 양 끝이 솟은 U 자         |
| 4   | 아무것도 모른다          | $(1, 1)$            | 평평한 uniform            |

속이려는 상대를 가정하지 않는다면 1 행과 4 행 사이의 완만한 모양이면 충분하다. 여기서는
$\mathrm{Beta}(2, 2)$ 를 골랐고, 이는 $0.5$ 에 완만한 봉우리를 둔 prior 이다.

### 6.4 The Posterior

식 (12) 의 likelihood 에 $\mathrm{Beta}(2, 2)$ prior 를 곱하면 지수가 하나씩 올라간 같은 꼴이 된다.

```math
p(\theta \mid \mathrm{data}) \propto \theta^{57} (1 - \theta)^{43} \cdot \theta (1 - \theta) = \theta^{58} (1 - \theta)^{44} \hspace{19em} (14)
```

이것은 $\mathrm{Beta}(59, 45)$ 의 밀도이며, prior 와 posterior 가 같은 족에 남는 이 성질이 conjugate
이다. Conjugate 이면 식 (13) 의 $\mathrm{B}(a, b)$ 가 정규화 상수를 맡으므로, Bayes 정리의 분모를 따로
적분하지 않아도 posterior 가 닫힌 형태로 나온다.

Posterior 의 최빈값과 평균은 모두 $0.57$ 부근으로 maximum likelihood 추정값과 겹친다. 점 하나가 아니라
분포이므로 구간도 함께 나오며, 양쪽 꼬리를 같게 잡은 99 % credible interval 은 $[0.441, 0.688]$ 이다.
공정한가라는 물음에는 posterior 의 $8.4\,\%$ 가 $0.5$ 아래에 있다고 답한다. 공정한 동전을 배제할 만큼
작지 않으므로, 가르려면 더 던져야 한다.

### 6.5 What Enough Data Does

Table 4 의 네 prior 를 모두 넣어도 posterior 는 넷 다 $0.57$ 부근에 몰린 비슷한 모양이 된다. 관측이
100 개이면 식 (12) 의 likelihood 가 prior 의 차이를 덮기 때문이며, 실제로 이 자료를 만든 값은 $0.55$
였다. Prior 의 선택이 답을 정하는 것은 자료가 적을 때이고, 자료가 쌓이면 posterior 는 prior 가 무엇이든
같은 곳으로 모인다.

이 예제가 5 꼭지에 더하는 것이 셋이다. 첫째, prior 를 고를 수 있으면 아는 것을 계산에 넣을 수 있고
아는 것이 없으면 평평한 prior 를 둔다. 둘째, 답이 값 하나가 아니라 분포여서 상한과 하한이 함께 나온다.
셋째, 자료가 충분하면 그 답이 maximum likelihood 추정값으로 다가간다.

## 7. Limits

Bayes 통계학을 고전 통계학보다 낫다고 단정할 수 없게 만드는 것이 셋이다.

- **계산**: 적분이 닫힌 형태로 풀리는 경우가 드물다. 이 문서의 예제는 beta integral 로 떨어지는 드문 경우이며, 일반적으로는 수치 기법이 필요하고 정밀도를 확보하기 어렵다.
- **가정**: Prior 가 결과를 움직인다. 5.2 가 $P(p)$ 를 uniform 으로 둔 것과 6.3 이 $\mathrm{Beta}(2, 2)$ 를 고른 것이 그 예이며, 자료가 적을수록 그 선택이 답에 더 많이 남는다.
- **해석**: Model 을 확률변수로 다루는 용법 자체에 이견이 있다. Parameter 에 분포를 얹는 것이 무엇을 뜻하는지가 명확하게 정해지지 않아, 이 용법을 받아들이지 않는 쪽이 있다 [[3](#ref-3)].

## References

<a id="ref-1"></a>
[1] Eddy, S. R. (2004). [What is Bayesian statistics?](https://doi.org/10.1038/nbt0904-1177) *Nature Biotechnology*, 22(9), 1177–1178.<br>
<a id="ref-2"></a>
[2] Puga, J. L., Krzywinski, M., & Altman, N. (2015). [Bayes' theorem](https://doi.org/10.1038/nmeth.3335). *Nature Methods*, 12(4), 277–278.<br>
<a id="ref-3"></a>
[3] Puga, J. L., Krzywinski, M., & Altman, N. (2015). [Bayesian statistics](https://doi.org/10.1038/nmeth.3368). *Nature Methods*, 12(5), 377–378.<br>
<a id="ref-4"></a>
[4] National Institute of Standards and Technology. [DLMF §5.12 Beta Function](https://dlmf.nist.gov/5.12). *NIST Digital Library of Mathematical Functions*.

---

## Appendix A. Terminology

- **Bernoulli trial**: 결과가 둘뿐이고 성공 확률이 매번 같은 시행.
- **beta distribution**: 식 (9) 의 밀도를 갖는 0 과 1 사이의 분포. Uniform prior 와 binomial likelihood 의 posterior 가 이 분포이다.
- **beta integral**: 식 (17) 의 적분. Gamma function 의 비로 닫힌 형태로 풀린다.
- **binomial distribution**: 성공 확률이 같은 시행을 여러 번 되풀이했을 때 성공 횟수가 따르는 분포.
- **conjugate**: Prior 와 posterior 가 같은 분포족에 남게 만드는 prior 와 likelihood 의 짝.
- **credible interval**: Posterior 의 확률이 정해진 만큼 담기는 구간.
- **evidence**: Bayes 정리의 분모. Posterior 를 확률분포로 만드는 정규화 상수이다.
- **Gamma function**: 계승을 실수로 확장한 함수. 자연수에서 $\Gamma(n+1) = n!$ 이다.
- **likelihood**: Parameter 를 고정했을 때 관측이 나올 확률.
- **maximum likelihood**: Likelihood 를 최대로 만드는 parameter 값을 고르는 추정 방법.
- **point estimate**: Parameter 를 분포가 아니라 값 하나로 나타낸 추정값.
- **posterior**: 관측을 반영해 갱신한 parameter 의 분포.
- **prior**: 관측을 보기 전 parameter 에 두는 분포.
- **uniform distribution**: 주어진 구간 안에서 밀도가 일정한 분포.
## Appendix B. Derivations

### B.1 Bob's Win Probability at a Given p

5.1 의 식 (4) 를 세운다. 앨리스가 5 점, 밥이 3 점인 상태에서 앨리스는 한 판만 더 가져가면 6 점에 닿고,
밥은 세 판을 더 가져가야 6 점에 닿는다. 따라서 밥이 최종 승리하는 경우는 남은 세 판을 모두 가져가는
하나뿐이다. 셋 중 한 판이라도 앨리스가 가져가면 그 자리에서 앨리스가 6 점에 닿아 게임이 끝나기
때문이다.

$p$ 가 주어지면 각 판은 밥이 이길 확률이 $1 - p$ 인 서로 독립인 Bernoulli trial 이므로, 세 판의
결합확률은 곱으로 분해된다.

```math
P(\mathrm{Bob\ wins} \mid p) = P(R_1 = R_2 = R_3 = \mathrm{Bob} \mid p) = (1-p)(1-p)(1-p) = (1-p)^3 \hspace{19em} (15)
```

독립이 성립하는 근거는 기준선이 게임 내내 움직이지 않는다는 데 있다. 기준선이 정해 준 $p$ 를 조건으로
걸고 나면 판과 판 사이에 남는 연결이 없으므로, 앞 판의 결과가 뒤 판의 확률을 바꾸지 않는다.

### B.2 Evaluation of the Expectation

식 (6) 에 식 (7) 을 넣으면 분자와 분모가 모두 $p$ 에 대한 적분이 된다. Likelihood 는 여덟 판 가운데 다섯 판을 앨리스가 가져간 확률이므로 $P(A = 5, B = 3 \mid p) = \binom{8}{5} p^5 (1-p)^3$ 이고, 이항계수와 상수 prior 는 분자와 분모에 함께 있어 약분된다. 남는 것은 아래와 같다.

```math
E = \frac{\int_{0}^{1} p^5 (1-p)^3 (1-p)^3\, dp}{\int_{0}^{1} p^5 (1-p)^3\, dp} = \frac{\int_{0}^{1} p^5 (1-p)^6\, dp}{\int_{0}^{1} p^5 (1-p)^3\, dp} \hspace{19em} (16)
```

두 적분은 Euler 의 beta integral 이며, Gamma function 으로 닫힌 형태로 풀린다 [[4](#ref-4)].

```math
\int_{0}^{1} p^{m-1} (1-p)^{n-1}\, dp = \frac{\Gamma(m)\, \Gamma(n)}{\Gamma(m+n)} \hspace{19em} (17)
```

분자는 $m = 6$, $n = 7$ 이고 분모는 $m = 6$, $n = 4$ 이다. $\Gamma(n+1) = n!$ 을 써서 정리하면 값이 나온다.

```math
E = \frac{\Gamma(6)\Gamma(7) / \Gamma(13)}{\Gamma(6)\Gamma(4) / \Gamma(10)} = \frac{5!\, 6! / 12!}{5!\, 3! / 9!} = \frac{6!}{3!} \cdot \frac{9!}{12!} = \frac{120}{1320} = \frac{1}{11} \hspace{19em} (18)
```
