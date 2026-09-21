# Standard Deviation of a Population and of Its Sample Mean
Rev. 5 | Created: 2026-08-30 | Updated: 2026-09-21 16:52 CDT

> 본래 분포의 표준편차와 거기에서 뽑은 표본의 평균이 갖는 표준편차 사이의 관계, 표본 크기가
> 그 관계에 미치는 영향, 그리고 모두 sigma 로 읽히는 여러 기호의 구분에 대한 기록.

## 1. Scope

표준편차라 불리고 sigma 로 적는 양이 둘 있으나, 두 양이 가리키는 것은 다르다. 하나는
모집단의 개별 값이 흩어진 정도이다. 다른 하나는 같은 크기의 표본을 되풀이해 뽑았을 때 그
표본평균이 모평균 둘레에 흩어진 정도이다. 두 양을 잇는 것은 표본의 크기뿐이다.

이 문서는 그 관계를 적고, 두 양이 각각 무엇을 뜻하는지 밝히고, sigma 라는 이름을 함께 쓰는
기호를 갈라 놓는다. 유도는 [Appendix B](#appendix-b-derivation) 에 두고, 용어의 뜻은
[Appendix A](#appendix-a-terminology) 에 둔다.

## 2. Relation

### 2.1. Statement

표준편차가 $\sigma$ 인 모집단에서 크기 $n$ 의 표본을 독립으로 뽑고, 그 표본의 평균을
$\bar{X}$ 라 하자. $\bar{X}$ 의 표준편차는 $\sigma_{\bar{X}}$ 로 적고 standard error 라
부른다.

```math
\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}} \hspace{19em} (1)
```

여기서 $n$ 은 뽑아낸 개별 관측값의 개수이므로, 이 관계에 들어 있는 것은 두 표준편차와 그
개수뿐이다. 이 관계는 하나의 모집단에서 독립으로 뽑는 것을 전제로 하며, 그 전제가 깨지는 두
경우는 [Appendix B](#appendix-b-derivation) 에 둔다.

### 2.2. What the Two Describe

Table 1. The two standard deviations compared.

| Aspect          | Population standard deviation | Standard error of the mean      |
| :-------------: | :---------------------------: | :-----------------------------: |
| Symbol          | $\sigma$                      | $\sigma_{\bar{X}}$              |
| Object measured | 모집단의 개별 값              | 되풀이해 뽑은 표본의 평균       |
| Sample size     | 관여하지 않음                 | $1/\sqrt{n}$ 이라는 인자로 관여 |
| Relative size   | 더 큼                         | $n \gt 1$ 이면 더 작음          |

평균은 극단값을 상쇄한다. 한 번 뽑은 값은 상쇄할 다른 값 없이 어느 쪽 꼬리로든 멀리 떨어질
수 있으나, 평균이 그만큼 움직이려면 크기 $n$ 의 표본을 이루는 $X_1, \ldots, X_n$ 가운데
여럿이 같은 쪽으로 치우쳐야 하고, 그렇게 치우치는 일은 극단값 하나가 나오는 일보다 드물다.

## 3. Effect of the Sample Size

Standard error 는 표본 크기 자체가 아니라 그 제곱근에 반비례해 줄어든다. 줄어드는 비율은
Table 2 와 같다.

Table 2. Standard error as a fraction of the population standard deviation.

| Sample size | Square root | Standard error |
| :---------: | :---------: | :------------: |
| 1           | 1.000       | 1.000          |
| 2           | 1.414       | 0.707          |
| 4           | 2.000       | 0.500          |
| 9           | 3.000       | 0.333          |
| 16          | 4.000       | 0.250          |
| 25          | 5.000       | 0.200          |
| 100         | 10.000      | 0.100          |

두 행이 이 관계의 전부를 말한다. $n = 1$ 에서는 평균이 곧 그 하나의 관측값이므로 standard
error 가 모집단의 표준편차와 같아지고 두 양이 겹친다. $n = 100$ 에서는 모집단 표준편차의 10분의 1 이 된다.

제곱근이 정밀도의 값을 정한다. Standard error 를 반으로 줄이려면 표본이 네 배로 들고, 10분의
1 로 줄이려면 100 배로 든다. 그 대신 이 관계는 표본평균이 어떤 개별 관측값보다 모평균에 대해
더 날카로운 진술이라는 것과, 그 날카로움이 자료를 보기 전에 $n$ 과 $\sigma$ 만으로 이미
정해져 있다는 것을 함께 말해 준다.

## 4. Symbols Read as Sigma

Table 3 은 기호 넷을 싣는다. 그 가운데 셋은 sigma 로 읽고, 나머지 하나인 Latin $s$ 는 표본
하나에서 계산한 개별 값의 흩어진 정도를 담는다. 넷이 각각 다른 양을 가리키므로 서로 바꾸어
쓸 수 없다.

Table 3. Symbols read as sigma.

| Symbol             | Name                   | Meaning                           |
| :----------------: | :--------------------: | :-------------------------------: |
| $\sum$             | Capital sigma          | 항을 모두 더하라는 합계 연산 기호 |
| $\sigma$           | Lower-case sigma       | 모집단의 표준편차                 |
| $s$                | Latin s                | 표본 하나에서 계산한 표준편차     |
| $\sigma_{\bar{X}}$ | Sigma with a subscript | 표본평균의 표준편차               |

가장 자주 놓치는 것이 $\sigma$ 와 $s$ 의 구분이다. 둘 다 개별 값의 흩어진 정도를 재지만,
$\sigma$ 는 모집단의 성질이어서 실제로는 알 수 없고, $s$ 는 손에 있는 관측값에서 계산하므로
표본이 바뀌면 값도 바뀐다. $\sigma$ 를 모를 때는 그 자리에 $s$ 를 넣어 $s/\sqrt{n}$ 으로
standard error 를 추정한다.

```math
\hat{\sigma}_{\bar{X}} = \frac{s}{\sqrt{n}} \hspace{19em} (2)
```

식 (2) 는 추정값이라 그 자체의 불확실성을 지닌다. 그래서 식 (2) 로 구간을 잡을 때는 식 (1)
이 허용하는 정규분포의 분위수 대신 자유도 $n - 1$ 의 $t$ 분포를 쓴다 [[1](#ref-1)].

## References

<a id="ref-1"></a>
[1] Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury.
ISBN 978-0-534-24312-8.<br>
<a id="ref-2"></a>
[2] Cochran, W. G. (1977). [*Sampling Techniques*](https://www.wiley.com/en-us/Sampling+Techniques,+3rd+Edition-p-9780471162407) (3rd ed.). Wiley. ISBN 978-0-471-16240-7.

---

## Appendix A. Terminology

- **Draw**: 모집단에서 관측값 하나를 뽑는 행위, 또는 그렇게 뽑힌 값 하나.
- **Population**: 진술의 대상이 되는 값의 전체 집합.
- **Sample**: 실제로 관측한 population 의 부분집합.
- **Sample mean**: 한 sample 안의 관측값을 산술평균한 값이며 $\bar{X}$ 로 적는다.
- **Standard error**: sample 에서 계산한 통계량의 표준편차이며, 여기서는 sample mean 의
  표준편차.
- **Variance**: 표준편차의 제곱.

## Appendix B. Derivation

평균이 $\mu$ 이고 variance 가 $\sigma^2$ 인 모집단에서 $X_1, \ldots, X_n$ 을 독립으로 뽑는다고
하자. 각 draw 는 같은 분포를 따르며, 어느 draw 도 다른 draw 에 대한 정보를 지니지 않는다.

```math
E[X_i] = \mu, \qquad \mathrm{Var}[X_i] = \sigma^{2}, \qquad i = 1, \ldots, n \hspace{19em} (3)
```

표본평균은 그 합을 개수로 나눈 것이다.

```math
\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i \hspace{19em} (4)
```

Variance 의 성질 두 가지가 필요하다. 확률변수에 상수를 곱하면 variance 는 그 상수의 제곱만큼
커지고, 독립인 확률변수를 더한 것의 variance 는 각 variance 의 합이다.

```math
\mathrm{Var}[aY] = a^{2} \mathrm{Var}[Y], \qquad \mathrm{Var}\left[ \sum_{i=1}^{n} X_i \right] = \sum_{i=1}^{n} \mathrm{Var}[X_i] \hspace{19em} (5)
```

식 (5) 의 앞의 성질을 $a = 1/n$ 으로 쓰고 이어서 뒤의 성질을 쓴다.

```math
\mathrm{Var}\left[ \bar{X} \right] = \frac{1}{n^{2}} \mathrm{Var}\left[ \sum_{i=1}^{n} X_i \right] = \frac{1}{n^{2}} \sum_{i=1}^{n} \sigma^{2} = \frac{n\sigma^{2}}{n^{2}} = \frac{\sigma^{2}}{n} \hspace{19em} (6)
```

표준편차는 variance 의 양의 제곱근이므로 식 (1) 이 나온다.

```math
\sigma_{\bar{X}} = \sqrt{\mathrm{Var}\left[ \bar{X} \right]} = \frac{\sigma}{\sqrt{n}} \hspace{19em} (7)
```

식 (4) 에 기댓값을 취하면 표본평균이 모평균에 놓여 있음이 드러난다. 식 (7) 이 재는 것은
흩어진 정도뿐이다 [[1](#ref-1)].

```math
E\left[ \bar{X} \right] = \frac{1}{n} \sum_{i=1}^{n} E[X_i] = \frac{n\mu}{n} = \mu \hspace{19em} (8)
```

이 유도에서 독립성은 식 (5) 의 두 번째 성질에서만 쓰인다. 독립성이 깨지는 경우가 둘 있다.
Draw 끼리 상관이 있으면 variance 의 합이 빠뜨린 covariance 항이 더해져 식 (6) 이 성립하지
않는다. 크기 $N$ 의 유한 모집단에서 비복원으로 뽑으면 draw 가 조금씩 종속되고, variance 에
finite population correction 인자가 붙는다 [[2](#ref-2)].

```math
\mathrm{Var}\left[ \bar{X} \right] = \frac{\sigma^{2}}{n} \cdot \frac{N-n}{N-1} \hspace{19em} (9)
```

$n$ 을 고정한 채 $N$ 이 커지면 이 인자는 1 로 간다. 식 (1) 은 $n$ 개를 덜어내도 달라지지
않을 만큼 모집단이 큰 극한인 셈이다. $N = 1000$ 이고 $n = 100$ 이면 인자가 0.901 이므로
standard error 는 식 (1) 이 주는 값의 0.949 배가 된다.
