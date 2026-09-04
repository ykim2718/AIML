# Bayesian Optimization for Semiconductor Process Tuning
Rev. 1 | Created: 2026-08-27 | Updated: 2026-09-04 20:10 UTC

공정 조건 하나를 시험하려면 장비를 세우고 wafer 를 걸어야 하므로 시도할 수 있는 횟수가 수십 회에 그친다. Grid 로 훑거나 무작위로 뿌리는 방법은 그 예산 안에서 답에 닿지 못한다. Bayesian optimization 은 지금까지의 결과로 아직 해 보지 않은 조건의 결과를 확률로 그려 두고, 그 그림을 근거로 다음 한 점을 고른다. 이 문서는 그 방법이 무엇을 가정하고 어떻게 도는지, 그리고 fab 이 그 위에 얹는 제약이 무엇인지 정리한다.

## 1. Problem Setting

공정 parameter 를 모은 벡터를 $x$ 라 하고 그 조건에서 나오는 결과를 $f(x)$ 라 하면, 하려는 일은 $f$ 를 가장 크게 하는 $x$ 를 찾는 것이다. 이 문제에는 네 가지 성질이 있다.

- 한 번 재는 값이 비싸다. 시도 횟수가 예산으로 정해진다.
- $f$ 의 식이 없다. 조건을 넣고 결과를 받는 것 말고는 안을 볼 수 없다.
- Gradient 가 없다. 어느 쪽으로 가야 나아지는지 계산으로 알 수 없다.
- 잰 값에 잡음이 섞인다. 같은 조건을 두 번 걸어도 결과가 다르다.

첫째 성질 때문에 전수 탐색이 막히고, 나머지 셋 때문에 gradient 를 따라가는 최적화가 막힌다. 남는 길은 지금까지 본 것으로 아직 보지 않은 곳을 추측하고, 그 추측이 가장 값진 곳을 다음에 시험하는 것이다.

## 2. Method

Bayesian optimization 은 부품이 둘이다. 비싼 $f$ 를 대신하는 surrogate model 과, 그 surrogate 를 읽어 다음 점을 고르는 acquisition function 이다.

Surrogate 로는 보통 Gaussian process 를 쓴다 [2](#ref-2). Gaussian process 는 각 $x$ 에서 예측값 하나가 아니라 평균 $\mu(x)$ 와 표준편차 $\sigma(x)$ 를 함께 낸다. 이 $\sigma(x)$ 가 방법 전체를 떠받친다. 관측한 점 가까이에서는 작고 멀리 떨어진 곳에서는 커지므로, model 이 어디를 모르는지 스스로 말해 주기 때문이다.

Loop 는 이렇게 돈다.

1. 공간을 고르게 덮는 몇 점으로 시작해 초기 관측을 얻는다.
2. 관측에 Gaussian process 를 맞춘다.
3. Acquisition function 을 최대로 만드는 $x$ 를 찾아 다음 시험 조건으로 삼는다.
4. 그 조건을 실제로 걸어 $f$ 를 재고 관측에 더한다.
5. 예산이 다할 때까지 2 단계로 돌아간다.

3 단계가 이 방법의 요령이다. 비싼 $f$ 를 최적화하는 문제가 surrogate 위에서 계산만으로 푸는 값싼 최적화 문제로 바뀐다. 장비를 걸어야 하는 것은 4 단계의 한 점뿐이다.

## 3. Acquisition Function

Acquisition function 은 $\mu(x)$ 와 $\sigma(x)$ 를 한 수로 합쳐 후보마다 점수를 매긴다. 두 값을 어떻게 섞느냐가 곧 exploration 과 exploitation 사이의 저울이다. $\sigma$ 가 큰 곳을 고르면 모르는 곳을 넓게 살피는 쪽이고, $\mu$ 가 큰 곳을 고르면 좋아 보이는 곳을 파고드는 쪽이다.

Table 1. Common acquisition functions

| Name | Rule | Behavior |
|------|------|----------|
| Probability of improvement | 지금까지의 최고를 넘을 확률 | 저울이 exploitation 쪽으로 치우친다 |
| Expected improvement | 넘는 정도까지 셈한 기댓값 | 얼마나 넘는지를 함께 보아 저울이 덜 치우친다 |
| Upper confidence bound | $\mu(x) + \kappa \sigma(x)$ | $\kappa$ 로 저울을 직접 돌린다 |

세 이름 모두 $f$ 를 크게 하는 문제를 기준으로 적었다. 작게 하는 문제에서는 부호를 뒤집어 그대로 쓰며, 이름까지 바뀌는 것은 셋째뿐이라 lower confidence bound 가 된다.

Expected improvement 가 기본값처럼 쓰이는 이유는 넘을 확률만이 아니라 넘는 양까지 함께 보기 때문이다 [1](#ref-1). Probability of improvement 는 조금이라도 넘기만 하면 점수를 주므로 이미 좋은 곳 주변을 맴돌기 쉽다. Upper confidence bound 는 저울을 $\kappa$ 하나로 드러내 두어 조절이 쉬운 대신, 그 값을 직접 정해야 한다.

## 4. Comparison with Classical DOE

Table 2. Classical DOE and Bayesian optimization

| Item | Classical DOE | Bayesian optimization |
|------|---------------|-----------------------|
| When the points are fixed | 실험 전에 전부 정한다 | 한 점씩 결과를 보고 정한다 |
| Model | 저차 다항식 | Gaussian process 같은 비모수 model |
| Goal | 요인이 결과에 미치는 효과를 추정한다 | 최적점을 찾는다 |
| Parallelism | 한 번에 모두 돌린다 | 순차가 기본이다 |
| Number of runs | 설계가 정한다 | 예산이 다할 때까지 이어진다 |

두 방법은 서로를 밀어내지 않는다. 고전 DOE 는 어느 요인이 결과를 움직이는지 먼저 가려 주므로, 그렇게 좁힌 parameter 로 Bayesian optimization 을 돌리면 탐색할 차원이 줄어 같은 예산으로 더 멀리 간다. 표에서 가장 크게 갈리는 줄은 Parallelism 이며, 그것이 다음 절의 제약으로 이어진다.

## 5. Practice in Semiconductor

적용하는 자리는 세 갈래이다. Recipe 를 최적화해 etch rate 와 uniformity 를 맞추는 일, chamber matching, 그리고 virtual metrology model 자체의 hyperparameter 를 고르는 일이다. 앞의 둘은 실험 한 번에 wafer 가 들어가고, 마지막 하나는 계산만 들어간다.

Fab 은 이 loop 위에 다섯 가지 제약을 얹는다.

Table 3. Constraints a fab puts on the loop

| Constraint | Why it appears | What it forces |
|------------|----------------|----------------|
| Specification | 목표 하나만 올려서는 안 되고 uniformity 와 defect 상한을 함께 지켜야 한다 | 제약을 건 최적화로 푼다 |
| Trade-off | Etch rate 와 uniformity 처럼 두 목표가 서로 밀어낸다 | 한 점이 아니라 Pareto front 를 찾는다 |
| Batch | Wafer 를 lot 단위로 한 번에 돌린다 | 다음 한 점이 아니라 다음 여러 점을 함께 고른다 |
| Safety | Acquisition 이 process cliff 밖의 점을 고를 수 있다 | 탐색 범위에 상한을 두어 장비와 wafer 를 지킨다 |
| Drift | 장비가 시간에 따라 흐른다 | 오래된 관측의 무게를 줄이거나 시간을 입력에 넣는다 |

Batch 와 Safety 두 줄이 교과서와 현장을 가장 크게 갈라 놓는다. Batch 는 순차로 도는 loop 를 깨뜨린다. 한 점을 걸고 결과를 본 뒤 다음 점을 정하는 것이 원래 방식인데, lot 하나가 한 번에 돌아가므로 아직 결과를 모르는 채로 여러 점을 함께 골라야 한다. 그러려면 고른 점들이 서로 붙지 않도록 따로 손을 써야 한다.

Safety 는 목적 함수에 없는 제약이다. Acquisition function 은 모르는 곳을 값지게 보므로 아무도 가 보지 않은 극단을 기꺼이 고르는데, 그 극단이 장비를 상하게 하거나 lot 을 버리게 만드는 자리일 수 있다. 그러므로 탐색 범위는 최적화가 정하도록 두지 말고 공정을 아는 사람이 미리 잘라 두어야 한다.

## 6. Relation to Active Learning

Bayesian optimization 과 active learning 은 surrogate model 과 acquisition function 이라는 같은 장치를 쓴다. 다른 것은 그 장치로 무엇을 얻으려 하는지이다. Active learning 은 model 의 오차를 가장 빨리 줄이는 점을 고르고, Bayesian optimization 은 최적점에 가장 빨리 닿는 점을 고른다.

그러므로 둘은 상하 관계가 아니라 목적이 다른 두 방법이다. 어느 쪽을 쓸지는 다음 실험으로 model 을 고치려는지 공정을 고치려는지가 정한다. 공정 전체를 설명하는 model 이 필요하면 active learning 이고, 조건 하나만 좋으면 되고 그 바깥은 몰라도 되면 Bayesian optimization 이다.

## References

<a id="ref-1"></a>[1] Jones, D. R., Schonlau, M., Welch, W. J. [Efficient Global Optimization of Expensive Black-Box Functions](https://doi.org/10.1023/A:1008306431147). Journal of Global Optimization, 13(4), 455-492, 1998.<br>
<a id="ref-2"></a>[2] Rasmussen, C. E., Williams, C. K. I. [Gaussian Processes for Machine Learning](http://gaussianprocess.org/gpml/). MIT Press, 2006. ISBN 978-0-262-18253-9.<br>
<a id="ref-3"></a>[3] Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., de Freitas, N. [Taking the Human Out of the Loop: A Review of Bayesian Optimization](https://doi.org/10.1109/JPROC.2015.2494218). Proceedings of the IEEE, 104(1), 148-175, 2016.

---

## Appendix A. Terminology

- **Acquisition function** 은 surrogate model 이 낸 평균과 표준편차를 한 수로 합쳐 후보마다 매기는 점수이며, 다음에 시험할 점은 이 점수가 가장 큰 곳이다.
- **Black-box function** 은 입력을 넣고 출력을 받는 것 말고는 안을 들여다볼 수 없는 함수이다.
- **Chamber matching** 은 같은 기종의 장비 여럿이 같은 조건에서 같은 결과를 내도록 맞추는 일이다.
- **Exploitation** 은 지금까지 좋아 보이는 곳을 더 파고드는 쪽으로 다음 점을 고르는 것이다.
- **Exploration** 은 아직 모르는 곳을 살피는 쪽으로 다음 점을 고르는 것이다.
- **Gaussian process** 는 함수 자체에 확률 분포를 두는 model 이며, 관측을 주면 각 입력에서 평균과 표준편차를 낸다.
- **Pareto front** 는 서로 밀어내는 목표들 사이에서 어느 하나를 나쁘게 하지 않고는 다른 하나를 좋게 할 수 없는 해들의 모음이다.
- **Process cliff** 는 공정 parameter 가 조금 더 벗어나면 수율이 급격히 무너지는 구간이다.
- **Recipe** 는 한 공정 단계를 돌리는 데 필요한 조건과 순서를 적어 둔 것이다.
- **Surrogate model** 은 비싼 실제 함수를 대신해 값을 내주는 값싼 model 이다.
- **Virtual metrology** 는 실제로 재지 않은 계측값을 장비의 sensor 기록으로부터 model 이 대신 내주는 것이다.
