# Marginal Likelihood and its Laplace Approximation
Rev. 4 | Created: 2026-09-20 | Updated: 2026-09-20 13:50 CDT

## 1. Purpose

- **Problem Statement**: Model 을 고르려면 그 model 이 자료를 낼 확률 $p(D)$ 가 필요한데, 정의가 적분이고 대부분의 model 에서 그 적분의 닫힌 해가 없다.
- **Goal**: $p(D)$ 가 무엇을 재는 양인지와 적분이 풀리지 않는 이유를 가리고, Laplace approximation 이 그 적분을 무엇으로 바꾸며 어떤 조건에서 그 값을 믿을 수 있는지를 적어, 손에 있는 model 에 대해 evidence 를 어느 방법으로 구할지 고르게 한다.
- **Non-Goal**: Bridge sampling 과 nested sampling 의 실행 절차는 다루지 않는다. 이름과 계층에서의 자리만 적는다.

## 2. Summary

Posterior 가 단봉이고 표본이 충분하면 Laplace approximation 이 $p(D)$ 를 얻는 가장 싼 방법이며, 값은 최적화 한 번과 Hessian 한 번으로 나온다. 정확한 값은 prior 와 likelihood 가 conjugate 인 경우에만 손으로 적을 수 있고, 그 밖에서는 근사나 표본이 유일한 길이다. 수치 적분은 차원이 커지면 격자 점의 수가 감당되지 않아 대안이 되지 못한다.

Laplace approximation 은 posterior 의 봉우리에서 2차까지 전개해 적분 대상을 Gaussian 으로 바꾼다. 그 결과가 식 (3) 이고, 여기서 표본 수에 따라 커지지 않는 항을 버리면 BIC 가 된다. 따라서 BIC 를 쓰는 것은 이미 Laplace approximation 을 쓰되 그 항들을 버린 것과 같다.

믿을 수 없게 되는 조건은 셋이다. Posterior 에 봉우리가 여럿이면 하나만 세고, 표본이 적으면 2차 전개의 잔차가 남으며, 봉우리가 정의역의 경계에 있으면 Gaussian 적분의 구간이 맞지 않는다.

## 3. Taxonomy and its Hierarchy

$p(D)$ 를 구하는 방법은 적분을 다루는 방식으로 갈리고, 그 방식은 posterior 의 모양에 무엇을 가정하는가로 다시 갈린다. 아래 <a href="#fig-1">Fig 1</a> 이 두 축과 계층을 한 장에 담는다.

```text
p(D) = INTEGRAL of p(D|theta) p(theta) d(theta)
|
+-- analytic ------ conjugate pair      exact value, closed form exists
|
+-- approximate --- Laplace             unimodal posterior, Gaussian at the mode
|                   +-- BIC             Laplace with the terms that do not grow with n dropped
|                   variational         posterior replaced by a chosen family, lower bound
|
+-- sampling ------ bridge sampling     no assumption on the shape, many draws
                    nested sampling     no assumption on the shape, many draws
```

<a id="fig-1"></a>
Fig 1. Method families for the marginal likelihood and the assumption each makes

세로축은 적분을 다루는 방식이고, 가로로 붙은 글은 그 방법이 posterior 의 모양에 두는 가정이다. 위에서 아래로 내려갈수록 가정이 약해져 쓸 수 있는 model 이 넓어지고, 대신 값 하나를 얻는 비용이 커진다. BIC 가 Laplace 아래에 붙은 것은 계열이 달라서가 아니라 같은 전개에서 항을 더 버린 것이기 때문이다.

### 3.1 Placement

Placement 는 <a href="#fig-1">Fig 1</a> 이 세운 축과 계층 위에서 각 방법이 어느 자리에 놓이는지를 뜻한다. Table 1 이 그 자리를 posterior 에 두는 가정, 값 하나를 얻는 비용, 내놓는 산출의 세 가지로 적으며, 손에 있는 model 에 어느 방법을 쓸지는 이 표에서 고른다.

Table 1. Methods for the marginal likelihood

| #   | Method                | Assumption on the posterior                          | Cost                      | Output            |
| :-: | :-------------------: | :--------------------------------------------------: | :-----------------------: | :---------------: |
| 1   | Conjugate closed form | Prior 와 likelihood 가 conjugate                     | 없음                      | 정확한 값         |
| 2   | Laplace approximation | 단봉. 표본이 충분                                    | 최적화 1 회, Hessian 1 회 | 근사값            |
| 3   | BIC                   | 단봉. 표본이 충분                                    | 최적화 1 회               | 대략값            |
| 4   | Variational inference | 고른 분포족 (family of distributions) 으로 근사 가능 | 반복 최적화               | 하한값            |
| 5   | Bridge sampling       | 모양에 가정 없음                                     | 표본 다수                 | 근사값            |
| 6   | Nested sampling       | 모양에 가정 없음                                     | 표본 다수                 | 근사값, 증거 구간 |

1 행은 Gaussian likelihood 와 Gaussian prior 처럼 지수의 어깨를 완전제곱으로 묶을 수 있는 경우에만 성립한다. 2 행과 3 행은 같은 전개에서 나오며 3 행이 2 행에서 표본 수에 따라 커지지 않는 항을 버린 것이다. 4 행이 내는 값은 $p(D)$ 자체가 아니라 그 하한이므로, 서로 다른 model 의 하한을 견주는 일은 하한의 느슨한 정도가 model 마다 다를 때 뒤집힌다. 5 행과 6 행은 posterior 의 모양을 묻지 않는 대신 표본 수가 비용을 정하고, 수렴을 따로 진단해야 한다 [[1](#ref-1)].

## 4. Marginal Likelihood

Marginal likelihood 는 model 이 자료를 낼 확률을 prior 의 가중치로 모든 $\theta$ 에 걸쳐 평균낸 값이며, Bayes 정리의 분모가 그것이다.

```math
p(\theta \mid D) = \frac{p(D \mid \theta)\, p(\theta)}{p(D)} \hspace{19em} (1)
```

식 (1) 에서 $D$ 는 관측한 자료 전체이고, $\theta$ 는 그 자료를 설명하는 model 의 parameter 를 모은 vector 이다. $\theta$ 의 성분 수를 $d$, $D$ 가 담은 관측의 수를 $n$ 으로 두며, 두 값은 식 (3) 과 식 (4) 에서 다시 쓰인다. Model 을 하나 정한다는 것은 $\theta$ 가 어떤 성분으로 이루어지는지와 $p(D \mid \theta)$ 가 어떤 함수인지를 정한다는 뜻이다.

Table 2. Terms of Bayes' theorem

| #   | Term                | Symbol             | Reading                                  |
| :-: | :-----------------: | :----------------: | :--------------------------------------: |
| 1   | Prior               | $p(\theta)$        | 자료를 보기 전 $\theta$ 에 두는 분포     |
| 2   | Likelihood          | $p(D \mid \theta)$ | $\theta$ 를 고정했을 때 자료가 나올 확률 |
| 3   | Posterior           | $p(\theta \mid D)$ | 자료를 본 뒤 $\theta$ 에 두는 분포       |
| 4   | Marginal likelihood | $p(D)$             | $\theta$ 를 적분해 없앤 자료의 확률      |

1 행과 2 행이 입력이고 3 행이 출력이며, 4 행은 출력을 확률분포 (probability distribution) 로 만드는 분모다. 4 행에 붙은 marginal 은 결합분포 (joint distribution) 에서 $\theta$ 를 적분해 없앴다는 뜻이고, likelihood 는 남은 값이 model 하나를 놓고 잰 자료의 확률이라는 뜻이다. Model 을 비교할 때 이 값이 likelihood 의 자리에 들어간다.

2 행의 이름에는 조건이 하나 붙는다. $p(D \mid \theta)$ 라는 식은 하나이지만, 무엇을 변수로 두는가에 따라 이름이 갈린다. $\theta$ 를 고정하고 $D$ 를 변수로 보면 자료에 대한 확률분포이고, $D$ 에 대해 적분하면 1 이 된다. $D$ 를 관측된 값으로 고정하고 $\theta$ 를 변수로 보면 그것이 likelihood 이며, 어느 $\theta$ 가 그 자료를 더 잘 내놓는지를 재는 $\theta$ 의 함수다. Bayes 정리 안에서는 $D$ 가 이미 관측된 값이므로 언제나 뒤쪽 읽기이고, 이때 $\theta$ 에 대한 적분은 1 이 되지 않는다. Likelihood 를 분포가 아니라 함수라 부르는 이유가 이것이다.

### 4.1 Derivation

유도의 출발점은 결합분포 $p(D, \theta) = p(D \mid \theta)\, p(\theta)$ 하나다. 여기서 $\theta$ 를 적분해 없애면 $D$ 만의 분포가 남는다.

```math
p(D) = \int p(D, \theta)\, d\theta = \int p(D \mid \theta)\, p(\theta)\, d\theta \hspace{19em} (2)
```

같은 값이 다른 길로도 나온다. 식 (1) 의 좌변이 확률분포이려면 $\theta$ 에 대한 적분이 1 이어야 하므로, 분모는 분자를 $\theta$ 에 대해 적분한 값일 수밖에 없다. 두 길이 같은 식에 닿는 것은 두 조건이 같은 것을 요구하기 때문이며, 어느 쪽으로 세워도 $p(D)$ 는 정의상 적분이다.

식 (2) 가 재는 것은 한 점 $\theta$ 에서의 적합도가 아니라 prior 가 퍼뜨린 만큼의 평균이다. 그래서 parameter 가 많아 prior 가 넓게 퍼진 model 은 자료를 잘 맞히는 $\theta$ 를 가지고 있어도 평균이 깎이며, 이 깎임을 Occam factor 라 부른다 [[2](#ref-2)]. Model 선택에서 $p(D)$ 를 쓰는 이유가 여기에 있다. 적합도와 복잡도를 따로 재어 더하는 대신 한 적분이 둘을 함께 센다.

### 4.2 Why the Integral Resists

식 (2) 의 닫힌 해가 없다는 것은 적분 구간의 문제가 아니라 피적분 함수의 원시함수가 표준 함수의 유한 조합으로 적히지 않는다는 뜻이다. 구간을 $(-\infty, \infty)$ 로 두든 유한 구간으로 두든 같다.

- **되는 경우**: Gaussian likelihood 와 Gaussian prior. Conjugate 라 지수의 어깨를 완전제곱으로 묶으면 Gaussian 적분 공식이 그대로 적용된다.
- **안 되는 경우**: Logistic regression 처럼 $\sigma(\theta^{\top} x)$ 가 든 likelihood 와 Gaussian prior. 곱이 어떤 표준 분포 꼴도 아니어서 적을 답이 없다.

수치 적분이 이 벽을 넘지 못하는 이유는 따로 있다. 차원마다 격자를 $m$ 개 두면 점이 $m^d$ 개로 늘어, $d$ 가 열만 넘어도 계산이 불가능해진다. 그래서 근사나 표본으로 우회한다.

## 5. Laplace Approximation

Laplace approximation 은 log posterior 를 MAP 점 $\hat\theta$ 둘레에서 2차까지 전개해 적분 대상을 Gaussian 으로 바꾸고, 그 Gaussian 적분을 닫힌 형태로 푼다. 결과는 아래와 같으며, $A$ 는 $\hat\theta$ 에서의 음의 Hessian 이다. 유도는 [Appendix B](#appendix-b-derivation-of-equations-3-and-4) 에 있다.

```math
\log p(D) \approx \log p(D \mid \hat\theta) + \log p(\hat\theta) + \frac{d}{2}\log 2\pi - \frac{1}{2}\log |A| \hspace{19em} (3)
```

우변의 네 항이 각각 하는 일이 다르다. 첫 항은 봉우리에서의 적합도, 둘째 항은 그 자리에 prior 가 주는 가중치, 셋째와 넷째 항은 봉우리 둘레에서 posterior 가 차지하는 부피다. 부피 항이 Occam factor 를 담는다. 봉우리가 뾰족하면 $|A|$ 가 커져 값이 깎이고, 넓으면 덜 깎인다.

관측 수 $n$ 이 커지면 $A$ 가 $n$ 에 비례해 $\log|A| \approx d \log n$ 이 되고, $n$ 에 따라 커지지 않는 항을 버리면 식 (3) 이 BIC 로 줄어든다 [[3](#ref-3)].

```math
\mathrm{BIC} = -2 \log p(D \mid \hat\theta) + d \log n \hspace{19em} (4)
```

### 5.1 Conditions

- **가정**: Posterior 가 단봉이고, 봉우리가 정의역의 내부에 있으며, 표본이 2차 전개를 지탱할 만큼 많다.
- **설정값**: $\hat\theta$ 는 MAP 추정값이고 $A$ 는 그 자리의 음의 Hessian 이다. Prior 를 상수로 두면 $\hat\theta$ 는 최대가능도 추정값이 된다.
- **깨지는 조건**: 봉우리가 여럿이면 하나만 세어 값이 낮게 나온다. 표본이 적으면 상대오차가 $O(1/n)$ 의 규모로 남는다 [[4](#ref-4)]. 봉우리가 경계에 있으면 Gaussian 적분의 구간이 맞지 않는다. Parameter 가 identifiable 하지 않아 $A$ 가 특이하면 $\log|A|$ 가 발산한다.
- **만나는 자리**: Model 간 Bayes factor 계산 [[1](#ref-1)], Gaussian process classification 의 evidence 계산 [[5](#ref-5)], 그리고 BIC 로 model 을 고르는 모든 자리. Gaussian process regression 은 conjugate 라 이 근사가 필요하지 않다.

## References

<a id="ref-1"></a>
[1] Kass, R. E., & Raftery, A. E. (1995). [Bayes Factors](https://doi.org/10.1080/01621459.1995.10476572). *Journal of the American Statistical Association*, 90(430), 773–795.<br>
<a id="ref-2"></a>
[2] MacKay, D. J. C. (2003). [Information Theory, Inference, and Learning Algorithms](http://www.inference.org.uk/mackay/itila/book.html). Cambridge University Press. ISBN 978-0521642989.<br>
<a id="ref-3"></a>
[3] Schwarz, G. (1978). [Estimating the Dimension of a Model](https://doi.org/10.1214/aos/1176344136). *The Annals of Statistics*, 6(2), 461–464.<br>
<a id="ref-4"></a>
[4] Tierney, L., & Kadane, J. B. (1986). [Accurate Approximations for Posterior Moments and Marginal Densities](https://doi.org/10.1080/01621459.1986.10478240). *Journal of the American Statistical Association*, 81(393), 82–86.<br>
<a id="ref-5"></a>
[5] Rasmussen, C. E., & Williams, C. K. I. (2006). [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/). MIT Press. ISBN 978-0262182539.

---

## Appendix A. Terminology

- **Bayes factor**: 두 model 의 marginal likelihood 의 비. Model 비교의 척도이다.
- **BIC**: 식 (4) 의 값. Laplace approximation 에서 표본 수에 따라 커지지 않는 항을 버린 model 선택 기준이다.
- **bridge sampling**: 보조 분포를 하나 두고 두 분포의 표본으로 marginal likelihood 의 비를 추정하는 방법.
- **conjugate**: Prior 와 posterior 가 같은 분포족에 속하게 만드는 prior 와 likelihood 의 짝.
- **evidence**: Marginal likelihood 의 다른 이름. Model 선택의 맥락에서 이 이름을 쓴다.
- **Gaussian process**: 임의의 유한 개 입력점에서의 함수값이 결합 정규분포 (joint normal distribution) 를 이루는 확률과정.
- **Hessian**: 다변수 함수의 2차 편도함수를 모은 행렬.
- **identifiable**: 서로 다른 parameter 값이 서로 다른 분포를 내는 성질. 깨지면 봉우리가 한 점이 아니라 능선이 된다.
- **MAP**: Posterior 를 최대로 만드는 parameter 값.
- **nested sampling**: Likelihood 의 등고선을 따라 prior 를 적분하여 marginal likelihood 를 추정하는 방법.
- **Occam factor**: Prior 가 넓게 퍼진 model 에서 marginal likelihood 가 깎이는 몫.
- **positive definite**: 0 이 아닌 모든 vector 에 대해 이차형식이 양수인 행렬의 성질.
- **posterior**: 자료를 본 뒤의 parameter 분포.
- **prior**: 자료를 보기 전의 parameter 분포.
- **variational inference**: Posterior 를 고른 분포족 안에서 가장 가까운 분포로 바꾸어 푸는 근사.

## Appendix B. Derivation of Equations (3) and (4)

식 (2) 의 피적분 함수를 지수 하나로 묶는 것에서 시작한다. $h(\theta)$ 를 정규화하지 않은 log posterior 로 둔다.

```math
h(\theta) = \log p(D \mid \theta) + \log p(\theta), \qquad p(D) = \int e^{h(\theta)}\, d\theta \hspace{19em} (5)
```

$\hat\theta$ 를 $h$ 의 최대점, 곧 MAP 로 두고 그 자리에서 2차까지 Taylor 전개한다.

```math
h(\theta) \approx h(\hat\theta) + \nabla h(\hat\theta)^{\top}(\theta - \hat\theta) - \frac{1}{2}(\theta - \hat\theta)^{\top} A\, (\theta - \hat\theta) \hspace{19em} (6)
```

여기서 $A = -\nabla^2 h(\hat\theta)$ 이다. $\hat\theta$ 가 내부의 최대점이므로 $\nabla h(\hat\theta) = 0$ 이고, 1차 항이 사라진다. 남은 것을 식 (5) 에 넣으면 상수가 적분 밖으로 나온다.

```math
p(D) \approx e^{h(\hat\theta)} \int \exp\left(-\frac{1}{2}(\theta - \hat\theta)^{\top} A\, (\theta - \hat\theta)\right) d\theta \hspace{19em} (7)
```

적분은 Gaussian 적분이므로 닫힌 형태로 풀린다. $A$ 가 positive definite 이면 값은 아래와 같다.

```math
\int \exp\left(-\frac{1}{2} x^{\top} A\, x\right) dx = (2\pi)^{d/2} |A|^{-1/2} \hspace{19em} (8)
```

식 (7) 과 식 (8) 을 합치고 $h(\hat\theta) = \log p(D \mid \hat\theta) + \log p(\hat\theta)$ 를 되돌리면 식 (9) 가 되며, 양변에 로그를 취한 것이 식 (3) 이다.

```math
p(D) \approx p(D \mid \hat\theta)\, p(\hat\theta)\, (2\pi)^{d/2} |A|^{-1/2} \hspace{19em} (9)
```

BIC 는 여기서 한 걸음 더 간다. 관측 하나당 정보량을 $A_1$ 으로 두면 관측이 독립일 때 $A \approx n A_1$ 이므로 $|A| \approx n^d |A_1|$ 이고 $\log|A| \approx d \log n + \log|A_1|$ 이다. 식 (3) 에서 $n$ 에 따라 커지지 않는 항, 곧 $\log p(\hat\theta)$ 와 $\frac{d}{2}\log 2\pi$ 와 $\log|A_1|$ 을 버리면 $\log p(D) \approx \log p(D \mid \hat\theta) - \frac{d}{2}\log n$ 이 남고, 양변에 $-2$ 를 곱한 것이 식 (4) 다 [[3](#ref-3)].
