# Bayesian Statistics
Rev. 1 | Created: 2026-09-20 | Updated: 2026-09-20 14:15 CDT

## 1. Purpose

- **Problem Statement**: 같은 자료로 같은 물음에 답해도 고전 통계학과 Bayes 통계학이 서로 다른 값을 내며, 어느 계열로 푼 값인지 밝히지 않으면 그 값이 무엇을 뜻하는지 말할 수 없다.
- **Goal**: 두 계열이 무엇을 확률변수로 보고 조건부 확률의 어느 쪽을 구하는지 가르고, 한 문제에서 두 답이 갈리는 폭을 수치로 보여, 손에 있는 문제에 어느 계열을 쓸지 고르게 한다.
- **Non-Goal**: Posterior 를 실제로 계산하는 수치 기법은 다루지 않는다. 여기서 푸는 예제는 적분이 닫힌 형태로 풀리는 경우 하나뿐이다.

## 2. Summary

두 계열은 조건부 확률의 어느 쪽을 구하는가로 갈린다. 고전 통계학은 model 을 고정해 두고 $P(\mathrm{data} \mid \mathrm{model})$ 을 구해 그 model 이 타당한지 검정하고, Bayes 통계학은 자료를 고정해 두고 $P(\mathrm{model} \mid \mathrm{data})$ 를 구해 model 의 가능성을 계산한다 [[1](#ref-1)].

이 차이는 말의 차이로 그치지 않는다. 5 꼭지의 당구대 문제에서 고전 통계학은 베팅 비율 7 : 1 을, Bayes 통계학은 10 : 1 을 낸다. 같은 자료에서 한쪽은 밥이 이길 확률을 $1/8$ 로, 다른 쪽은 $1/11$ 로 본다.

Bayes 통계학이 언제나 낫다고 말할 수는 없다. 적분이 닫힌 형태로 풀리지 않는 경우가 많고, prior 를 무엇으로 둘지의 가정이 타당하지 않으면 결과를 믿을 수 없으며, model 을 확률변수로 다루는 용법 자체에 이견이 있다. 6 꼭지가 이 셋을 적는다.

## 3. Taxonomy and its Hierarchy

Model 과 자료를 두고 세울 수 있는 조건부 확률이 둘이며, 어느 쪽을 구하는가가 두 계열을 가른다. 아래 <a href="#fig-1">Fig 1</a> 이 그 갈림과 각 계열이 딸려 요구하는 것을 한 장에 담는다.

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
| 5   | 5 꼭지의 답        | 7 : 1                                  | 10 : 1                                 |

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

두 계열의 답이 갈리는 것을 한 문제에서 볼 수 있다. Eddy 가 든 당구대 문제를 그대로 쓴다 [[1](#ref-1)].

### 5.1 Setup

앨리스와 밥은 당구대 위에 공을 하나 먼저 굴려 그 자리를 기준으로 당구대를 둘로 나눈다. 이후 공을 한 개씩 굴려, 공이 앨리스 쪽에 서면 앨리스가 1 점을, 밥 쪽에 서면 밥이 1 점을 가져간다. 두 사람은 기준 공이 어디에 놓였는지 알지 못한다.

득점 확률이 영역의 넓이에 비례한다고 두면, 앨리스가 한 판에서 점수를 얻을 확률이 $p$ 일 때 밥의 확률은 $1 - p$ 이다. 승리 조건은 6 점 선취이고, 현재 앨리스가 5 점, 밥이 3 점이다. 이때 공평한 베팅 비율을 묻는다.

앨리스는 1 점만 더 얻으면 이기고, 밥은 남은 세 판을 연속으로 가져가야 이긴다. 따라서 밥이 이길 확률은 아래와 같다.

```math
P(\mathrm{Bob\ wins} \mid p) = (1 - p)^3 \hspace{19em} (4)
```

### 5.2 The Classical Answer

고전 통계학은 $p$ 에 값 하나를 놓는다. 문제가 $p$ 를 주지 않으므로 기댓값 $0.5$ 를 취하면 밥이 이길 확률은 $(1 - 0.5)^3 = 1/8$ 이고, 앨리스가 이길 확률은 $7/8$ 이다. 베팅 비율은 7 : 1 이 된다.

이 답에는 물을 것이 남는다. $0.5$ 는 $p$ 의 기댓값일 뿐이며, 지금 진행 중인 게임의 $p$ 가 $0.5$ 라는 보장이 없다. 앨리스가 5 : 3 으로 앞서 있다는 사실 자체가 $p$ 가 $0.5$ 보다 클 가능성을 가리키는데, 7 : 1 은 그 사실을 쓰지 않는다.

### 5.3 The Bayesian Answer

Bayes 통계학은 $p$ 를 하나로 정하지 않고, 자료가 가리키는 $p$ 의 분포에 걸쳐 식 (4) 를 평균낸다.

```math
E = \int_{0}^{1} (1 - p)^3\, P(p \mid A = 5,\, B = 3)\, dp \hspace{19em} (5)
```

$p$ 가 0 과 1 사이의 실수이므로 합이 아니라 적분이다. 피적분 함수의 posterior 는 식 (3) 을 연속값에 적용해 얻는다.

```math
P(p \mid A = 5,\, B = 3) = \frac{P(A = 5,\, B = 3 \mid p)\, P(p)}{\int_{0}^{1} P(A = 5,\, B = 3 \mid p)\, P(p)\, dp} \hspace{19em} (6)
```

분자의 likelihood $P(A = 5, B = 3 \mid p)$ 는 한 판의 승률이 $p$ 일 때 여덟 판에서 앨리스가 5 점, 밥이 3 점을 얻을 확률이다. Prior $P(p)$ 는 $p$ 가 어느 값일 확률이며, 여기서는 상수로 둔다. 식 (5) 와 식 (6) 을 합쳐 적분하면 [Appendix B](#appendix-b-evaluation-of-the-expectation) 의 계산으로 아래를 얻는다.

```math
E = \frac{1}{11} \hspace{19em} (7)
```

Table 3. Two answers to the betting problem

| #   | School                                 | Bob's win probability | Fair odds |
| :-: | :------------------------------------: | :-------------------: | :-------: |
| 1   | 고전 통계학. $p$ 를 $0.5$ 로 고정      | $1/8$                 | 7 : 1     |
| 2   | Bayes 통계학. $p$ 를 posterior 로 평균 | $1/11$                | 10 : 1    |

두 값의 차이는 5 : 3 이라는 자료를 쓰는가 쓰지 않는가에서 온다. 1 행은 $p$ 를 자료와 무관하게 $0.5$ 로 놓아 앨리스가 앞서 있다는 사실을 버리고, 2 행은 그 사실로 $p$ 의 분포를 앨리스 쪽으로 옮긴 뒤 평균낸다. 그래서 2 행이 밥의 승률을 더 낮게 본다. Prior 가 uniform 인 근거, 점추정값을 달리 골랐을 때의 값, posterior 의 이름과 parameter 는 [Appendix C](#appendix-c-the-pool-table-problem-in-detail) 에 있다.

## 6. Limits

Bayes 통계학을 고전 통계학보다 낫다고 단정할 수 없게 만드는 것이 셋이다.

- **계산**: 적분이 닫힌 형태로 풀리는 경우가 드물다. 이 문서의 예제는 beta integral 로 떨어지는 드문 경우이며, 일반적으로는 수치 기법이 필요하고 정밀도를 확보하기 어렵다.
- **가정**: Prior 가 결과를 움직인다. 5.3 이 $P(p)$ 를 상수로 둔 것이 그 예이며, 그 가정이 타당하지 않으면 뒤따르는 값도 믿을 수 없다.
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

- **beta distribution**: 식 (13) 의 밀도를 갖는 0 과 1 사이의 분포. Uniform prior 와 binomial likelihood 의 posterior 가 이 분포이다.
- **beta integral**: 식 (9) 의 적분. Gamma function 의 비로 닫힌 형태로 풀린다.
- **evidence**: Bayes 정리의 분모. Posterior 를 확률분포로 만드는 정규화 상수이다.
- **Gamma function**: 계승을 실수로 확장한 함수. 자연수에서 $\Gamma(n+1) = n!$ 이다.
- **likelihood**: Parameter 를 고정했을 때 관측이 나올 확률.
- **point estimate**: Parameter 를 분포가 아니라 값 하나로 나타낸 추정값.
- **posterior**: 관측을 반영해 갱신한 parameter 의 분포.
- **prior**: 관측을 보기 전 parameter 에 두는 분포.
- **total probability**: 한 사건의 확률을 서로 배타적인 조건들에 걸쳐 나누어 합하는 공식.
- **uniform distribution**: 주어진 구간 안에서 밀도가 일정한 분포.

## Appendix B. Evaluation of the Expectation

식 (5) 에 식 (6) 을 넣으면 분자와 분모가 모두 $p$ 에 대한 적분이 된다. Likelihood 는 여덟 판 가운데 다섯 판을 앨리스가 가져간 확률이므로 $P(A = 5, B = 3 \mid p) = \binom{8}{5} p^5 (1-p)^3$ 이고, 이항계수와 상수 prior 는 분자와 분모에 함께 있어 약분된다. 남는 것은 아래와 같다.

```math
E = \frac{\int_{0}^{1} p^5 (1-p)^3 (1-p)^3\, dp}{\int_{0}^{1} p^5 (1-p)^3\, dp} = \frac{\int_{0}^{1} p^5 (1-p)^6\, dp}{\int_{0}^{1} p^5 (1-p)^3\, dp} \hspace{19em} (8)
```

두 적분은 Euler 의 beta integral 이며, Gamma function 으로 닫힌 형태로 풀린다 [[4](#ref-4)].

```math
\int_{0}^{1} p^{m-1} (1-p)^{n-1}\, dp = \frac{\Gamma(m)\, \Gamma(n)}{\Gamma(m+n)} \hspace{19em} (9)
```

분자는 $m = 6$, $n = 7$ 이고 분모는 $m = 6$, $n = 4$ 이다. $\Gamma(n+1) = n!$ 을 써서 정리하면 값이 나온다.

```math
E = \frac{\Gamma(6)\Gamma(7) / \Gamma(13)}{\Gamma(6)\Gamma(4) / \Gamma(10)} = \frac{5!\, 6! / 12!}{5!\, 3! / 9!} = \frac{6!}{3!} \cdot \frac{9!}{12!} = \frac{120}{1320} = \frac{1}{11} \hspace{19em} (10)
```

## Appendix C. The Pool Table Problem in Detail

5 꼭지가 두 답을 견주는 데 필요한 만큼만 적은 자리를, 여기서 단계별로 푼다. 이 문제는 Bayesian
billiards problem 이라는 이름으로도 불린다.

### C.1 The Prior

두 사람이 보지 못하는 상태에서 제 3 자가 첫 공을 굴려 멈춘 자리에 기준선을 긋는다. 공이 서는 자리가
무작위이므로 기준선도 당구대 위에 고르게 퍼져 있고, 따라서 앨리스 쪽 영역이 차지하는 비율 $p$ 는 0 과
1 사이에서 uniform distribution 을 따른다. 5.3 이 $P(p)$ 를 상수로 둔 근거가 이것이며, 상수라는 가정이
임의로 고른 것이 아니라 공을 굴리는 방식에서 나온 것임을 여기서 확인한다.

### C.2 The Point Estimate Route

빈도주의는 $p$ 를 자료에서 추정해 하나의 값으로 고정한다. 가장 단순한 추정값은 관측한 득점 비율
$\hat{p} = 5/8$ 이고, 이때 밥이 이길 확률은 아래와 같다.

```math
P(\mathrm{Bob\ wins}) = (1 - \hat{p})^3 = \left(\frac{3}{8}\right)^3 = \frac{27}{512} \approx 0.053 \hspace{19em} (11)
```

5.2 가 쓴 $0.5$ 는 같은 계열 안의 다른 선택이며 $1/8$ 을 낸다. 두 값이 두 배 넘게 갈리는 것은 어느
점추정값을 고르는가가 답을 정하기 때문이고, 값 하나로 고정하는 순간 그 선택이 얼마나 불확실한지는
답에 들어가지 않는다.

### C.3 The Posterior

Uniform prior 에 binomial likelihood 를 곱하면 posterior 는 beta distribution 이 된다. 상수 배를 빼면
posterior 는 likelihood 의 모양을 그대로 따른다.

```math
P(p \mid A = 5,\, B = 3) \propto p^5 (1 - p)^3 \hspace{19em} (12)
```

이 모양은 아래 밀도에서 $\alpha = 6$, $\beta = 4$ 인 경우와 같다. 즉 posterior 는 $\mathrm{Beta}(6, 4)$
이며, 일반적으로 uniform prior 아래 앨리스가 $a$ 판, 밥이 $b$ 판을 가져가면 $\mathrm{Beta}(a+1,\, b+1)$
이 된다.

```math
f(p) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\, \Gamma(\beta)}\, p^{\alpha - 1} (1 - p)^{\beta - 1} \hspace{19em} (13)
```

문헌에 따라 $p$ 를 밥의 득점 확률로 두기도 한다. 그때 likelihood 는 $p^3 (1 - p)^5$, posterior 는
$\mathrm{Beta}(4, 6)$, 밥의 승리 확률은 $p^3$ 이 된다. 두 규약은 $p$ 와 $1 - p$ 를 맞바꾼 것이어서 답은
같다.

### C.4 The Answer

밥이 역전하려면 남은 세 판을 모두 가져가야 하므로 그 확률은 $(1 - p)^3$ 이고, 이것을 $\mathrm{Beta}(6, 4)$
에 걸쳐 평균낸 것이 식 (5) 다. [Appendix B](#appendix-b-evaluation-of-the-expectation) 가 그 적분을 풀어
$1/11$ 을 낸다.

Table 4. Three routes to the same question

| #   | Route                | Value used for $p$    | Bob's win probability  | Fair odds |
| :-: | :------------------: | :-------------------: | :--------------------: | :-------: |
| 1   | 점추정, 관측 비율    | $5/8$                 | $27/512 \approx 0.053$ | 약 18 : 1 |
| 2   | 점추정, prior 기댓값 | $0.5$                 | $1/8 = 0.125$          | 7 : 1     |
| 3   | Posterior 평균       | $\mathrm{Beta}(6, 4)$ | $1/11 \approx 0.091$   | 10 : 1    |

1 행과 2 행은 같은 자료에서 서로 다른 값을 고른 결과이고, 그 선택을 바꾸면 답이 세 배 가까이 움직인다.
3 행만이 $p$ 를 하나로 고르지 않고 자료가 남긴 분포를 그대로 들고 가므로, 고를 것이 없어 답이 하나로
정해진다.
