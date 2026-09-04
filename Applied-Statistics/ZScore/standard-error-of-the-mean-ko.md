# Standard Deviation of a Population and of Its Sample Mean (Korean)
Rev. 3 | Created: 2026-08-30 | Updated: 2026-09-04 20:10 UTC

> 본래 분포의 표준편차와 거기에서 뽑은 표본의 평균이 갖는 표준편차 사이의 관계, 표본 크기가
> 그 관계에 미치는 영향, 그리고 모두 sigma 로 읽히는 여러 기호의 구분에 대한 기록.

## 1. Scope

표준편차라 불리고 sigma 로 적는 양이 둘 있으나, 두 양이 가리키는 것은 다르다. 하나는
모집단의 개별 값이 흩어진 정도이다. 다른 하나는 같은 크기의 표본을 되풀이해 뽑았을 때 그
표본평균이 모평균 둘레에 흩어진 정도이다. 두 양을 잇는 것은 표본의 크기뿐이다.

이 문서는 그 관계를 적고, 두 양이 각각 무엇을 뜻하는지 밝히고, sigma 라는 이름을 함께 쓰는
기호를 갈라 놓는다. 유도는 [Appendix B](#appendix-b-derivation) 에 둔다.

## 2. Relation

### 2.1. Statement

표준편차가 $\sigma$ 인 모집단에서 크기 $n$ 의 표본을 독립으로 뽑고, 그 표본의 평균을
$\bar{X}$ 라 하자. $\bar{X}$ 의 표준편차는 $\sigma_{\bar{X}}$ 로 적고 standard error 라
부른다.

$$\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}$$

여기서 $n$ 은 뽑아낸 개별 관측값의 개수이므로, 이 관계에 들어 있는 것은 두 표준편차와 그
개수뿐이다.

### 2.2. What the Two Describe

Table 1. The two standard deviations compared.

| Aspect | Population standard deviation | Standard error of the mean |
|---|---|---|
| Symbol | $\sigma$ | $\sigma_{\bar{X}}$ |
| Object measured | 모집단의 개별 값 | 되풀이해 뽑은 표본의 평균 |
| Sample size | 관여하지 않음 | $1/\sqrt{n}$ 이라는 인자로 관여 |
| Relative size | 더 큼 | $n \gt 1$ 이면 더 작음 |

두 번째 것이 더 작은 까닭은 평균이 상쇄하기 때문이다. 한 번 뽑은 값은 어느 쪽 꼬리로든 멀리
떨어질 수 있고 그것을 상쇄할 것이 없다. 평균이 그만큼 움직이려면 극단값끼리 방향이 맞아야
하는데, 그렇게 맞는 일은 극단값 하나가 나오는 일보다 드물다.

## 3. Effect of the Sample Size

Standard error 는 표본 크기 자체가 아니라 그 제곱근에 반비례해 줄어든다. 줄어드는 비율은
Table 2 와 같다.

Table 2. Standard error as a fraction of the population standard deviation.

| Sample size | Square root | Standard error |
|---:|---:|---:|
| 1 | 1.000 | 1.000 |
| 2 | 1.414 | 0.707 |
| 4 | 2.000 | 0.500 |
| 9 | 3.000 | 0.333 |
| 16 | 4.000 | 0.250 |
| 25 | 5.000 | 0.200 |
| 100 | 10.000 | 0.100 |

두 행이 이 표의 전부를 말한다. $n = 1$ 에서는 평균이 곧 그 하나의 관측값이므로 standard error
가 모집단의 표준편차와 같아지고 두 양이 겹친다. $n = 100$ 에서는 그 10분의 1 이 된다.

정밀도를 비싸게 만드는 것이 이 제곱근이다. Standard error 를 반으로 줄이려면 표본이 네 배로
들고, 10분의 1 로 줄이려면 100 배로 든다. 그 대신 이 관계는 표본평균이 어떤 개별 관측값보다
모평균에 대해 더 날카로운 진술이라는 것과, 그 날카로움이 자료를 보기 전에 $n$ 과 $\sigma$
만으로 이미 정해져 있다는 것을 함께 말해 준다.

## 4. Symbols Read as Sigma

아래 기호 가운데 셋은 sigma 로 읽고, 나머지 하나는 sigma 를 쓰지 않을 자리에 쓰는 기호이다.
서로 바꾸어 쓸 수 없다.

Table 3. Symbols read as sigma.

| Symbol | Name | Meaning |
|---|---|---|
| $\sum$ | Capital sigma | 항을 모두 더하라는 합계 연산 기호 |
| $\sigma$ | Lower-case sigma | 모집단의 표준편차 |
| $s$ | Latin s | 표본 하나에서 계산한 표준편차 |
| $\sigma_{\bar{X}}$ | Sigma with a subscript | 표본평균의 표준편차 |

가장 자주 놓치는 것이 $\sigma$ 와 $s$ 의 구분이다. 둘 다 개별 값의 흩어진 정도를 재지만,
$\sigma$ 는 모집단의 성질이어서 실제로는 알 수 없고, $s$ 는 손에 있는 관측값에서 계산하므로
표본이 바뀌면 값도 바뀐다. $\sigma$ 를 모를 때는 그 자리에 $s$ 를 넣어 $s/\sqrt{n}$ 으로
standard error 를 추정한다. 이것은 추정값이라 그 자체의 불확실성을 지니며, $\sigma/\sqrt{n}$
에는 그런 불확실성이 없다.

## References

<a id="ref-1"></a>
[1] Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury.
ISBN 978-0-534-24312-8.<br>
<a id="ref-2"></a>
[2] Cochran, W. G. (1977). [*Sampling Techniques*](https://www.wiley.com/en-us/Sampling+Techniques,+3rd+Edition-p-9780471162407) (3rd ed.). Wiley. ISBN 978-0-471-16240-7.

---

## Appendix A. Terminology

- **Population**: 진술의 대상이 되는 값의 전체 집합.
- **Sample**: 실제로 관측한 population 의 부분집합.
- **Sample mean**: 한 sample 안의 관측값을 산술평균한 값이며 $\bar{X}$ 로 적는다.
- **Standard error**: sample 에서 계산한 통계량의 표준편차이며, 여기서는 sample mean 의
  표준편차.
- **Variance**: 표준편차의 제곱.

## Appendix B. Derivation

평균이 $\mu$ 이고 분산이 $\sigma^2$ 인 모집단에서 $X_1, \ldots, X_n$ 을 독립으로 뽑는다고
하자. 각 추출은 같은 분포를 따르며, 어느 추출도 다른 추출에 대한 정보를 지니지 않는다.

$$E[X_i] = \mu, \qquad \mathrm{Var}[X_i] = \sigma^{2}, \qquad i = 1, \ldots, n$$

표본평균은 그 합을 개수로 나눈 것이다.

$$\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i$$

분산의 성질 두 가지가 필요하다. 확률변수에 상수를 곱하면 분산은 그 상수의 제곱만큼 커지고,
독립인 확률변수를 더한 것의 분산은 각 분산의 합이다.

$$\mathrm{Var}[aY] = a^{2} \mathrm{Var}[Y], \qquad \mathrm{Var}\left[ \sum_{i=1}^{n} X_i \right] = \sum_{i=1}^{n} \mathrm{Var}[X_i]$$

$a = 1/n$ 으로 앞의 성질을 쓰고 이어서 뒤의 성질을 쓴다.

$$\mathrm{Var}\left[ \bar{X} \right] = \frac{1}{n^{2}} \mathrm{Var}\left[ \sum_{i=1}^{n} X_i \right] = \frac{1}{n^{2}} \sum_{i=1}^{n} \sigma^{2} = \frac{n\sigma^{2}}{n^{2}} = \frac{\sigma^{2}}{n}$$

표준편차는 분산의 양의 제곱근이므로 section 2.1 의 관계가 나온다.

$$\sigma_{\bar{X}} = \sqrt{\mathrm{Var}\left[ \bar{X} \right]} = \frac{\sigma}{\sqrt{n}}$$

같은 합에 기댓값을 취하면 표본평균이 모평균에 놓여 있음이 드러난다. Standard error 가 치우침이
아니라 정확도에 대한 진술이 되는 것은 이 때문이다 [[1](#ref-1)].

$$E\left[ \bar{X} \right] = \frac{1}{n} \sum_{i=1}^{n} E[X_i] = \frac{n\mu}{n} = \mu$$

이 유도에서 독립성은 분산의 두 번째 성질에서만 쓰인다. 그것이 깨지는 경우가 둘 있다. 추출끼리
상관이 있으면 분산의 합이 빠뜨린 공분산 항이 더해져 결과가 성립하지 않는다. 크기 $N$ 의 유한
모집단에서 비복원으로 뽑으면 추출이 조금씩 종속되고, 분산에 finite population correction 인자가
붙는다 [[2](#ref-2)].

$$\mathrm{Var}\left[ \bar{X} \right] = \frac{\sigma^{2}}{n} \cdot \frac{N-n}{N-1}$$

$n$ 을 고정한 채 $N$ 이 커지면 이 인자는 1 로 간다. 앞의 관계는 $n$ 개를 덜어내도 달라지지
않을 만큼 모집단이 큰 극한인 셈이다.
