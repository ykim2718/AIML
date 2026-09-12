# Polynomial Feature Expansion (Korean)
Rev. 53 | Created: 2026-09-07 | Updated: 2026-09-12 00:55 CDT

Polynomial feature expansion 은 한 변수의 거듭제곱과 서로 다른 변수의 곱을 함께 만드는 연산이다. 이 문서는 그 두 가지 열로 numeric tabular data 의 non-linear behavior 를 model 에 담는 방법을 다룬다.

## 1. Purpose

- **Problem Statement**: Numeric tabular data 에서 선형 model 로는 표현하기 힘든 경우가 있다.
- **Goal**: Original variable 의 power term ($x^2$, $x^3$) 와 interaction term ($x_1 \ast x_2$) 을 추가하여 비선형 model 을 만든다.
- **Non-Goal**: Derived variable 은 다루지 않고, expanded data set 의 학습을 다루지 않는다.

## 2. Summary

Expansion 은 표에 이미 있는 열로 곱과 제곱을 계산해 새 열로 붙이는 연산이며, 행은 그대로 두고 열만 늘린다. 열이 $x_1$, $x_2$ 인 표는 열이 $x_1$, $x_2$, $x_1^2$, $x_1 x_2$, $x_2^2$ 인 표가 되고, 새로 생긴 그 세 열이 선형 model 에 곡선과 변수 사이의 상호작용을 준다.

대가는 열의 개수 증가이다. 열 수가 행 수에 근접하면 계수, 곧 각 열에 곱해지는 $\beta$ 값을 하나로 정할 수 없다 (5.1 절).

이 차원의 저주를 감소시키기 위해서, 아래 세 가지를 기본으로 둔다. 기본이란 자료에서 달리 할 근거가 나오기 전까지 그대로 쓰는 설정이라는 뜻이며, 셋을 함께 두면 열 수가 행 수보다 충분히 적게 남고 표본을 다시 뽑아도 그 $\beta$ 가 크게 흔들리지 않는다.

- **Degree 2** — 만들 항을 한 변수의 제곱과 두 변수의 곱까지로 제한한다 (5.1 절).
- **Standardization** — Expansion 앞뒤로 각 열을 평균 0, 표준편차 1 로 맞춘다. 평균을 빼면 열 사이의 상관이 낮아지고 계수를 읽을 수 있게 되며, 표준편차로 나누면 열마다 다른 크기가 없어져 penalty 가 공평하게 걸린다 (4.2, 4.3, 5.2 절).
- **Penalty** — Expansion 이 만든 열에 ridge 나 lasso 를 건다. Ridge 는 모든 $\beta$ 를 같은 비율로 줄일 뿐 0 으로 만들지 않고, lasso 는 작은 $\beta$ 를 정확히 0 으로 만든다 (5.2 절).

세 가지 가운데 자주 빠지는 것은 standardization 과 penalty 다. Centering 하지 않은 물리 단위에서 $x$ 와 $x^2$ 의 상관은 1 에 가깝고 (4.2 절), expansion 이 만든 열은 원 변수가 서로 직교하더라도 서로 직교하지 않는다. 그래서 expansion 의 실패는 model 이 자료를 못 맞추는 모습이 아니라, 표본을 다시 뽑을 때마다 계수의 부호가 뒤집히는 모습으로 나타난다. Centering 은 그 상관을 낮출 뿐 0 으로 만들지 못하므로 (4.2 절), 부호가 뒤집히는 일이 centering 만으로 사라지지는 않는다. 남는 몫은 penalty 가 맡아, 닮은 열들이 서로 상쇄하는 큰 계수를 갖지 못하게 한다 (5.2 절).

Expansion 을 쓰지 않아야 하는 자리도 분명하다. 변수가 수십 개를 넘으면 열 수가 표본 수를 넘고, 한 변수 안에서 여러 번 꺾이는 모양이 필요하면 degree 를 올리는 대신 spline 으로 가야 하며, 훈련 구간 밖을 예측해야 하면 그 구간 밖에서 다항식이 폭주하는 성질, 곧 extrapolation 이 그대로 위험이 된다.

## 3. Objective

Expansion 이 노리는 것은 두 가지다. 한 변수 안의 비선형 관계와 변수 사이의 상호작용이며, 선형 model 은 둘 다 표현하지 못한다. 앞의 것은 power term 이, 뒤의 것은 interaction term 이 맡으며, 둘 다 열로 만들어지므로 model 자체는 선형으로 남는다.

### 3.1 Power Term

한 변수 안의 비선형 관계는 그 변수의 거듭제곱, 곧 power term 이 맡는다. 변수 $x$ 에 $x^2$ 와 $x^3$ 을 더하면 model 이 학습하는 식은 (1) 이며, model 은 계수에 대해 선형인 채로 곡선을 그린다.

$$\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 \hspace{19em} (1)$$

여기서 선형이라는 말은 $x$ 가 아니라 계수 $\beta$ 에 대한 것이며, 그래서 잔차 제곱합을 가장 작게 하는 계수를 푸는 최소제곱 (least squares) 이 그대로 쓰인다. 계수 $\beta_2$ 는 한 번 꺾이는 곡률, 곧 정점이나 포화를 담고, $\beta_3$ 은 한 번 더 꺾이는 모양을 담는다. 비선형은 model 이 아니라 열에 들어 있으므로, 쓰던 선형 model 과 그 위에 쌓인 추론·penalty 를 그대로 둔 채 비선형을 얻는다.

### 3.2 Interaction Term

변수 사이의 상호작용은 서로 다른 두 변수의 곱, 곧 interaction term 이 맡는다. 변수 $x_1$, $x_2$ 에 곱 $x_1 x_2$ 를 더하면 식 (2) 가 된다.

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1 x_2 \hspace{19em} (2)$$

곱항은 한 변수의 기울기를 다른 변수가 바꾸도록 허용한다. 식 (2) 를 $x_1$ 로 미분한 식 (3) 이 그 뜻이다.

$$\frac{\partial \hat{y}}{\partial x_1} = \beta_1 + \beta_3 x_2 \hspace{19em} (3)$$

$\beta_3$ 이 0 이 아니면 $x_1$ 의 효과는 $x_2$ 의 수준마다 다르다. 공정으로 옮기면 압력의 효과가 온도에 따라 달라진다는 뜻이며, $x_1$ 과 $x_2$ 의 1차 항 두 개만으로는 적을 수 없다. 두 변수가 함께 높을 때만 나타나는 효과는 이 곱항에만 담긴다. 응답면 (response surface), 곧 공정 조건 위에 응답이 그리는 곡면을 2차 다항식으로 적는 오랜 관행이 곡률과 상호작용의 조합이며, 최적 조건을 그 곡면에서 기울기가 0 이 되는 점 (stationary point) 으로 읽는 방법이 거기서 나왔다 [[1](#ref-1)].

## 4. Mechanism

### 4.1 Expansion

Expansion 이 만드는 열은 원 변수의 거듭제곱을 곱한 항, 곧 monomial 이다. 변수가 $n$ 개이고 최고 차수를 $d$ 로 두면 새 열은 식 (4) 의 집합이다.

$$\Phi_d(\mathbf{x}) = \left\lbrace \prod_{i=1}^{n} x_i^{a_i} \ \middle|\ a_i \in \mathbb{Z}_{\ge 0}, \ 1 \le \sum_{i=1}^{n} a_i \le d \right\rbrace \hspace{19em} (4)$$

변수가 $[X_1, X_2]$ 이고 $d = 2$ 이면 모든 행에서 값이 1 인 상수 열, 곧 절편 (intercept) 을 포함한 열은 $[1, X_1, X_2, X_1^2, X_1 X_2, X_2^2]$ 이며, 3.1 절의 제곱항과 3.2 절의 곱항이 그 안에 함께 들어 있다. 변수가 $n$ 개인 2차 model 의 일반형은 식 (5) 이다.

$$y = \beta_0 + \sum_{i=1}^{n} \beta_i x_i + \sum_{1 \le i \le j \le n} \beta_{ij} x_i x_j + \varepsilon \hspace{19em} (5)$$

### 4.2 Standardization

Expansion 전에 각 열을 평균 0, 표준편차 1 로 맞춘다. 이것이 standardization 이며, 평균을 빼는 부분만 따로 centering 이라 한다. 두 부분이 하는 일은 다르다. 평균을 빼면 열 사이의 상관이 낮아지고 계수를 읽을 수 있게 되며, 이것이 아래 두 문단이다. 표준편차로 나누면 열마다 다른 크기가 없어지고, 그 몫은 4.3 절과 5.2 절에 있다.

Centering 의 첫 번째 이유는 상관의 감소다. 상관은 두 열의 공분산을 두 열의 표준편차로 나눈 값이므로, 상관이 어디서 오는지는 분자인 공분산에서 읽는다. 표본을 $x_1, \dots, x_N$, 그 평균을 $\bar{x}$, 평균을 뺀 값을 $u_i = x_i - \bar{x}$ 로 두면 $u$ 의 평균은 0 이다. 윗줄은 표본 평균을 뜻하며, 공분산의 정의가 식 (6) 이다.

$$\mathrm{cov}(x, x^2) = \frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})(x_i^2 - \overline{x^2}) \hspace{19em} (6)$$

첫 인자 $x - \bar{x}$ 가 $u$ 이고, 괄호를 풀면 평균이 두 항으로 갈라져 식 (7) 이 된다.

$$\mathrm{cov}(x, x^2) = \frac{1}{N} \sum_{i=1}^{N} u_i x_i^2 - \overline{x^2} \cdot \overline{u} \hspace{19em} (7)$$

$\overline{u} = 0$ 이므로 뒤의 항이 사라지고 식 (8) 이 남는다.

$$\mathrm{cov}(x, x^2) = \frac{1}{N} \sum_{i=1}^{N} u_i x_i^2 \hspace{19em} (8)$$

$x^2 = u^2 + 2\bar{x}u + \bar{x}^2$ 을 넣고 항마다 평균을 취하면 식 (9) 가 된다.

$$\mathrm{cov}(x, x^2) = \overline{u^3} + 2 \bar{x} \overline{u^2} + \bar{x}^2 \overline{u} \hspace{19em} (9)$$

여기서도 $\overline{u} = 0$ 이 마지막 항을 지우므로, 공분산은 식 (10) 으로 닫힌다.

$$\mathrm{cov}(x, x^2) = 2 \bar{x} \overline{u^2} + \overline{u^3} \hspace{19em} (10)$$

식 (10) 의 두 항은 출처가 다르다. 앞의 항 $2 \bar{x} \overline{u^2}$ 는 평균이 0 에서 얼마나 떨어져 있는지에서만 오고, 뒤의 항 $\overline{u^3}$ 는 분포가 한쪽으로 기운 정도, 곧 3차 중심적률에서만 온다. 평균을 빼는 일은 앞의 항을 0 으로 만들고 뒤의 항은 그대로 둔다.

두 항의 몫을 견주려면 공분산을 표준편차로 나누어 상관으로 적어야 하며, 그 나눗셈이 필요한 이유와 아래 두 식의 유도는 [Appendix B](#appendix-b-covariance-and-correlation) 에 있다. $t = \bar{x} / \sqrt{\overline{u^2}}$, $s = \overline{u^3} / (\overline{u^2})^{3/2}$, $k = \overline{u^4} / (\overline{u^2})^2$ 로 두면 식 (10) 은 $(\overline{u^2})^{3/2} (2t + s)$ 이고, $x$ 의 분산은 $\overline{u^2}$, $x^2$ 의 분산은 $(\overline{u^2})^2 (k - 1 + 4t^2 + 4ts)$ 이므로 상관은 식 (11) 이다.

$$r(x, x^2) = \frac{2t + s}{\sqrt{k - 1 + 4t^2 + 4ts}} \hspace{19em} (11)$$

Centering 은 $\bar{x}$ 를 0 으로 만들어 $t = 0$ 을 강제하므로, 식 (11) 은 식 (12) 로 줄어든다.

$$r(u, u^2) = \frac{s}{\sqrt{k - 1}} \hspace{19em} (12)$$

두 식의 차이가 centering 이 상관을 낮추는 근거다. 식 (11) 에서 $\lvert t \rvert$ 를 키우면 분자는 $2t$, 분모는 $2 \lvert t \rvert$ 에 가까워지므로 $\lvert r \rvert$ 는 1 로 간다. $t$ 는 평균이 흩어짐의 몇 배만큼 0 에서 떨어져 있는지를 재는 값이어서 물리 단위의 자료에서는 크고, 그래서 그 자료의 $x$ 와 $x^2$ 의 상관은 1 에 닿는다. 식 (12) 에는 $t$ 가 없다. Centering 뒤의 상관은 평균의 위치와 무관하게 분포의 모양 $s$ 와 $k$ 로만 정해지며, 대칭 분포에서는 $s = 0$ 이므로 정확히 0 이다.

Centering 의 두 번째 이유는 해석이다. Centering 한 자료에서 $\beta_1$ 은 다른 변수가 평균일 때의 기울기여서 읽을 수 있는 값이 된다. Centering 하지 않으면 그것은 다른 변수가 0 일 때의 기울기이고, 그 0 은 자료에 없는 점인 경우가 많다 [[2](#ref-2)].

다만 centering 은 상관을 낮출 뿐 없애지 못한다. Expansion 이 만든 collinearity 는 자료의 성질이 아니라 expansion 자체의 성질이므로, penalty 가 함께 필요하다 (5.2 절).

### 4.3 Conditioning

Conditioning 은 design matrix 를 푸는 일이 입력의 작은 오차에 얼마나 민감한지를 말하며, 그것을 재는 값이 조건수 (condition number) 다. Design matrix 는 행이 관측이고 열이 model 이 쓰는 항인 행렬로, 계수는 이 행렬을 풀어 얻는다. 조건수는 그 오차가 푼 결과에서 몇 배로 커지는지를 나타낸다.

조건수를 올리는 것은 degree 와 열 사이의 collinearity 이고, 내리는 것은 standardization 이다. 평균을 빼면 열 사이의 겹침이 줄고 표준편차로 나누면 열마다 다른 크기가 없어지므로, 두 부분을 함께 해야 조건수가 가장 낮아진다.

### 4.4 Heredity

곱항을 남기면 그 곱을 이루는 두 변수의 1차 항, 곧 main effect 도 함께 남긴다. 이 규칙을 heredity 라 하며, 근거는 통계가 아니라 좌표계에 있다.

$y = \beta_{12} x_1 x_2$ 처럼 곱항만 있는 model 에 원점 이동 $x_1 = z_1 + a$, $x_2 = z_2 + b$ 를 넣으면 식 (13) 이 된다.

$$\beta_{12} (z_1 + a)(z_2 + b) = \beta_{12} z_1 z_2 + \beta_{12} b z_1 + \beta_{12} a z_2 + \beta_{12} ab \hspace{19em} (13)$$

Main effect 가 저절로 생긴다. 곧 main effect 없는 곱항 model 은 원점을 어디에 두었느냐에 따라 달라져, 온도를 섭씨로 재느냐 절대온도로 재느냐가 model 을 바꾼다. Main effect 를 함께 두면 그 이동이 계수의 재배열로 흡수된다. 곱을 이루는 변수 가운데 하나만 있어도 된다는 약한 형태 (weak heredity) 를 근거로 main effect 를 지우는 관행이 있으나, 그것이 정당화되는 조건은 실무에서 거의 성립하지 않는다 [[4](#ref-4)]. 변수 선택을 자동화할 때도 heredity 를 Bayes 의 사전 분포 (prior) 나 최적화의 제약으로 걸어 두는 편이 낫다 [[5](#ref-5)] [[6](#ref-6)].

## 5. Caution

Expansion 의 대가는 두 가지다. 하나는 열 수가 빠르게 늘어 overfitting, 곧 훈련 자료에는 맞지만 새 자료에서는 어긋나는 상태를 부르고 계산 비용을 올리는 것이고, 다른 하나는 expansion 이 만든 열이 서로 닮아 계수가 흔들리는 것이다. 열 수는 degree 로, 계수의 흔들림은 penalty 로 잡는다.

### 5.1 Dimensionality And Overfitting

열의 수는 변수의 수에 대해 $d$ 차로 늘어난다. 열이 늘수록 그 열들이 이루는 공간을 같은 밀도로 채우는 데 필요한 관측 수는 지수로 늘어나며, 이것을 curse of dimensionality 라 한다. Expansion 은 행 수를 그대로 둔 채 열만 늘리므로 그 현상을 자초한다. 절편을 뺀 전체 expansion 의 열 수는 식 (14), 서로 다른 변수의 곱만 남기는 `interaction_only` 의 열 수는 식 (15) 이다.

$$p_{\mathrm{full}} = \binom{n+d}{d} - 1 \hspace{19em} (14)$$

$$p_{\mathrm{inter}} = \sum_{j=1}^{\min(d,\ n)} \binom{n}{j} \hspace{19em} (15)$$

두 식은 모두 식 (4) 의 집합에서 나오며, 그 유도는 [Appendix C](#appendix-c-term-count-derivation) 에 있다.

Table 1. Column count after expansion, bias column excluded

| Variables | Degree 2, full | Degree 2, interaction only | Degree 3, full | Degree 3, interaction only |
| --- | --- | --- | --- | --- |
| 5 | 20 | 15 | 55 | 25 |
| 10 | 65 | 55 | 285 | 175 |
| 20 | 230 | 210 | 1,770 | 1,350 |
| 50 | 1,325 | 1,275 | 23,425 | 20,875 |
| 100 | 5,150 | 5,050 | 176,850 | 166,750 |

Table 1 에서 읽을 것은 `interaction_only` 가 줄여 주는 몫이 작다는 사실이다. $d = 2$ 에서 그 차이는 제곱항 $n$ 개뿐이어서 $n = 100$ 의 5,150 이 5,050 이 될 뿐이다. 곧 이 option 은 열 수를 줄이려고 켜는 것이 아니라, 한 변수 안의 곡률을 model 에 넣지 않겠다는 판단을 적어 두는 것이다.

열 수를 실제로 정하는 것은 degree 다. $d$ 를 2 에서 3 으로 올리면 $n = 20$ 에서 열은 230 에서 1,770 으로 늘어난다. 열 수가 행 수에 가까워지면 최소제곱의 해는 불안정해지고 넘어서면 유일하지 않으므로, expansion 의 상한을 정하는 것은 degree 가 아니라 표본 수이다.

그래서 degree 는 이론이 아니라 적합에 쓰지 않고 남겨 둔 자료의 오차, 곧 held-out 오차로 고르며, 후보는 좁다. 실무의 거의 모든 경우에 2 이고, 3 이 필요한 자료는 드물며, 4 이상이 이기는 것처럼 보이면 expansion 이 아니라 다른 방법을 써야 한다는 신호다.

<img src="polynomial-feature-expansion_fig/fig1.png" width="1100" style="max-width: 100%;" alt="Fig 1">

Fig 1. Degree and extrapolation, conditioning, and the cost of expansion

Fig 1(a) 는 첫 번째 이유다. 60 개 표본에 degree 2, 5, 9 를 맞춘 것으로, 훈련 구간 (회색) 안에서는 degree 5 와 9 가 모두 그럴듯하지만 구간을 벗어나면 차수가 높은 곡선이 먼저 폭주한다. 다항식의 바깥 거동은 최고차항이 지배하므로, extrapolation 이 필요한 곳에서 degree 를 올리면 훈련 구간 안의 적합은 좋아져도 구간 밖 예측의 오차는 커진다.

Fig 1(b) 는 4.3 절의 조건수를 차수별로 그린 것이고, Fig 1(c) 는 항 수와 행 수의 관계다. 변수 5 개, 행 60 개, 참 model 이 곱항 하나인 자료에서 held-out RMSE, 곧 그 오차를 제곱 평균의 제곱근으로 잰 값은 degree 1 의 1.34 에서 degree 2 의 0.34 로 내려갔다가 degree 3 에서 1.08 로 되돌아간다. degree 3 의 열 수는 55 로 행 수 60 에 거의 닿는다. 같은 자리에서 ridge 는 0.75 여서 그 악화의 절반 가까이를 막는다.

### 5.2 Regularization

Expansion 이 만든 열에는 penalty 를 반드시 함께 건다. Penalty 는 계수의 크기에 값을 매겨 적합 기준에 더하는 항이다. Expansion 은 열 수를 늘리는 동시에 서로 닮은 열을 만드는데, penalty 없는 최소제곱은 그 닮음을 서로 상쇄하는 두 개의 큰 계수로 흡수하며, 그래서 자료가 조금만 흔들려도 적합이 크게 움직인다. Ridge 는 계수 제곱합에 비례하는 penalty 를 걸어 그 상쇄를 막는다 [[3](#ref-3)].

둘 중 기본은 ridge 다. Ridge 는 닮은 열들에 계수를 나누어 주어 예측을 안정시키고, lasso 는 그 가운데 하나만 남기고 나머지를 지운다. Expansion 이 만든 열에서 lasso 는 곱항을 남기고 그 main effect 를 지워 4.4 절의 heredity 를 깨뜨릴 수 있으므로, 홀로 쓰기보다 계층 제약과 함께 쓴다 [[6](#ref-6)].

Ridge 가 계수를 0 으로 만들지 않는다는 것은 ridge 로는 열을 지울 수 없다는 뜻이다. 그래도 기본으로 두는 이유는 expansion 에서 penalty 가 버는 것이 열의 개수가 아니라 예측의 안정이기 때문이며, 그 크기는 5.1 절의 degree 3 에서 held-out RMSE 가 1.08 에서 0.75 로 내려가는 차이다. 열의 개수를 실제로 줄여야 하면 그것은 lasso 나 elastic net 의 몫이다.

Penalty 는 열의 크기에 걸리므로 expansion 이 만든 열을 standardization 한 뒤에 적용하며, [Appendix E](#appendix-e-implementation) 의 pipeline 에 두 번째 standardization 이 들어가는 이유가 그것이다. 세 penalty 의 목적 함수와 각각이 계수를 얼마나 움직이는지는 [Appendix D](#appendix-d-ridge-and-lasso-on-expanded-columns) 에 있다.

### 5.3 Failure Modes

Expansion 이 실패하는 모습은 여섯 가지로 정리된다. 대부분은 model 이 못 맞추는 모습이 아니라 계수나 예측이 불안정해지는 모습으로 온다.

Table 2. Failure modes of a polynomial expansion

| Symptom | Cause | Countermeasure |
| --- | --- | --- |
| Held-out error worse at degree 2 than at degree 1 | Term count close to the row count | Ridge or lasso, `interaction_only`, selective expansion |
| Coefficient signs flipping across resamples | Collinearity manufactured by the expansion | Centering, a penalty, reading predictions instead of coefficients |
| Prediction diverging just outside the training range | Extrapolation behaviour of a polynomial | Spline, a range guard on the input, no extrapolation |
| A handful of rows dominating the fit | Squares amplifying leverage | Outlier handling before expansion, robust loss |
| Duplicate or all-zero columns | Dummy columns squared and crossed | `interaction_only=True`, expansion restricted to continuous columns |
| Imputed values amplified | Imputation error squared inside a product | Imputation before expansion, an indicator column for what was imputed |

Table 2 의 다섯째 줄은 expansion 이 스스로 걸러 주지 않으므로 따로 적는다. 범주형 변수는 범주 하나에 열 하나를 두고 그 범주면 1, 아니면 0 을 적어 수치로 바꾸며, 그 열을 dummy 라 한다. Dummy 는 제곱이 자기 자신이어서 완전히 중복된 열이 되고, 한 행이 두 범주에 함께 속할 수 없으므로 같은 범주형 변수에서 나온 두 dummy 의 곱은 언제나 0 이다. Expansion 은 그것을 알지 못하므로, 범주형에서 나온 열은 expansion 대상에서 빼거나 `interaction_only` 로 다루어야 한다.

### 5.4 Diagnostics

Expansion 이 도움이 되었는지는 네 가지로 확인한다.

- Degree 를 1 부터 올리며 그린 held-out 오차 곡선. 최저점이 2 를 넘지 않는지 본다.
- Expansion 뒤 design matrix 의 조건수와 열별 VIF (Variance Inflation Factor). Centering 뒤에도 큰 값이면 penalty 가 필요하다.
- 자료에서 복원추출로 다시 뽑은 표본 (bootstrap) 에서 계수 부호가 유지되는 비율. 곱항의 부호가 뒤집히면 그 항은 해석하지 않는다.
- 잔차를 곱항에 대해 그린 산점도. Expansion 전에 남아 있던 구조가 사라졌는지 확인한다.

## 6. Further Work

- **Sparse polynomial chaos expansion** — 서로 직교하는 다항식들의 모음 위에서 항을 희소하게 골라 고차 expansion 의 항 수를 줄이는 방법이다 [[10](#ref-10)]. 최소각 회귀 (least angle regression) 로 항을 고르는 절차가 자리 잡아 수백 개 후보에서 수십 개만 남기는 일이 계산으로 가능해졌다. 착수에는 입력 변수의 분포 가정 (기저가 그 분포에 따라 정해진다) 과 설계된 표본이 필요하다.
- **Hierarchical interaction selection at scale** — heredity 를 convex 제약으로 걸어 곱항을 고르는 lasso 계열이다 [[6](#ref-6)]. 제약이 convex 여서 찾은 최적해가 유일하고 수백 변수까지 풀리므로, 4.4 절의 규칙을 사람이 지키는 대신 최적화가 지키게 할 수 있다. 착수에는 곱항 후보의 범위를 미리 좁히는 규칙과 계산 예산이 필요하다.
- **Learned basis** — 고정된 monomial 기저 대신 1차원 함수를 학습해 쌓는 model 이다 [[11](#ref-11)]. 2024 년에 spline 기반 구현이 공개되어 같은 자료에서 expansion + ridge 와 직접 견줄 수 있게 되었다. 착수에는 held-out 비교 절차와, 학습되는 기저가 표본 수에 비해 과하지 않은지 판단할 기준이 필요하다.

## References

<a id="ref-1"></a>
[1] Box, G. E. P. and Wilson, K. B. (1951). [On the Experimental Attainment of Optimum Conditions](https://doi.org/10.1111/j.2517-6161.1951.tb00067.x). *Journal of the Royal Statistical Society: Series B*, 13(1), 1–38.<br>
<a id="ref-2"></a>
[2] Marquardt, D. W. (1980). [Comment: You Should Standardize the Predictor Variables in Your Regression Models](https://doi.org/10.1080/01621459.1980.10477430). *Journal of the American Statistical Association*, 75(369), 87–91.<br>
<a id="ref-3"></a>
[3] Hoerl, A. E. and Kennard, R. W. (1970). [Ridge Regression: Biased Estimation for Nonorthogonal Problems](https://doi.org/10.1080/00401706.1970.10488634). *Technometrics*, 12(1), 55–67.<br>
<a id="ref-4"></a>
[4] Nelder, J. A. (1998). [The Selection of Terms in Response-Surface Models—How Strong is the Weak-Heredity Principle?](https://doi.org/10.1080/00031305.1998.10480588) *The American Statistician*, 52(4), 315–318.<br>
<a id="ref-5"></a>
[5] Chipman, H. (1996). [Bayesian variable selection with related predictors](https://doi.org/10.2307/3315687). *The Canadian Journal of Statistics*, 24(1), 17–36.<br>
<a id="ref-6"></a>
[6] Bien, J., Taylor, J. and Tibshirani, R. (2013). [A lasso for hierarchical interactions](https://doi.org/10.1214/13-AOS1096). *The Annals of Statistics*, 41(3), 1111–1141.<br>
<a id="ref-7"></a>
[7] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, É. (2011). [Scikit-learn: Machine Learning in Python](https://www.jmlr.org/papers/v12/pedregosa11a.html). *Journal of Machine Learning Research*, 12, 2825–2830.<br>
<a id="ref-8"></a>
[8] Rahimi, A. and Recht, B. (2007). [Random Features for Large-Scale Kernel Machines](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html). *Advances in Neural Information Processing Systems*, 20.<br>
<a id="ref-9"></a>
[9] Friedman, J. H. and Popescu, B. E. (2008). [Predictive learning via rule ensembles](https://doi.org/10.1214/07-AOAS148). *The Annals of Applied Statistics*, 2(3), 916–954.<br>
<a id="ref-10"></a>
[10] Blatman, G. and Sudret, B. (2011). [Adaptive sparse polynomial chaos expansion based on least angle regression](https://doi.org/10.1016/j.jcp.2010.12.021). *Journal of Computational Physics*, 230(6), 2345–2367.<br>
<a id="ref-11"></a>
[11] Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y. and Tegmark, M. (2024). [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756). *arXiv:2404.19756*.<br>
<a id="ref-12"></a>
[12] Tibshirani, R. (1996). [Regression Shrinkage and Selection via the Lasso](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x). *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.<br>
<a id="ref-13"></a>
[13] Zou, H. and Hastie, T. (2005). [Regularization and variable selection via the elastic net](https://doi.org/10.1111/j.1467-9868.2005.00503.x). *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.

---

## Appendix A. Terminology

- **collinearity**: 두 개 이상의 열이 거의 같은 방향을 가리켜 계수를 따로 추정할 수 없는 상태.
- **condition number**: 행렬의 최대 특이값과 최소 특이값의 비. 입력의 작은 오차가 해에서 얼마나 커지는지를 나타낸다.
- **curse of dimensionality**: 열이 늘수록 그 공간을 같은 밀도로 채우는 데 필요한 관측 수가 지수로 늘어나는 현상.
- **degree**: expansion 이 허용하는 monomial 의 최고 차수. $X_1^2 X_2$ 의 차수는 3 이다.
- **derived variable**: 도메인 지식으로 두 개 이상의 열을 묶어 새로 만든 변수. 비나 율이 그 예다.
- **design matrix**: 행이 관측이고 열이 model 이 쓰는 항인 행렬. 계수는 이 행렬을 풀어 얻는다.
- **dummy**: 범주형 변수의 한 범주에 대응하여 그 범주면 1, 아니면 0 을 담는 열.
- **extrapolation**: 훈련 자료가 덮지 않는 입력 범위에 대한 예측.
- **held-out**: 적합에 쓰지 않고 적합한 model 의 오차를 재는 데만 쓰는 자료.
- **heredity**: 곱항을 model 에 넣으면 그것을 이루는 낮은 차수 항도 함께 넣는 규칙.
- **leverage**: 한 관측이 자신의 예측값을 끌어당기는 정도. 입력이 중심에서 멀수록 커진다.
- **main effect**: 변수 하나의 1차 항 $\beta_i x_i$.
- **monomial**: 변수들의 거듭제곱을 곱한 항. $X_1^2 X_2$ 가 그 예다.
- **overfitting**: 훈련 자료에는 맞지만 새 자료에서는 어긋나는 상태.
- **penalty**: 계수의 크기에 값을 매겨 적합 기준에 더하는 항. ridge 와 lasso 가 그것이다.
- **RMSE**: 제곱 오차의 평균에 제곱근을 취한 값 (Root Mean Squared Error).
- **standardization**: 각 열에서 그 열의 평균을 빼고 표준편차로 나누어 평균 0, 표준편차 1 로 맞추는 연산.
- **VIF**: 한 열을 나머지 열로 회귀했을 때의 $R^2$ 로 계산하는 분산 팽창 계수. $1/(1-R^2)$ 이다.

## Appendix B. Covariance And Correlation

공분산은 두 열이 함께 움직이는 정도를 재는 값이다. 모집단에 대한 정의가 식 (16) 이며, $E[\cdot]$ 는 기댓값 (expected value), $\mu_X$ 와 $\mu_Y$ 는 두 변수의 기댓값이다.

$$\mathrm{Cov}(X, Y) = \sigma_{XY} = E[(X - \mu_X)(Y - \mu_Y)] \hspace{19em} (16)$$

기댓값의 성질로 괄호를 풀면 계산하기 쉬운 식 (17) 이 된다. 두 변수의 곱의 기댓값에서 각 변수의 기댓값의 곱을 뺀 것이다.

$$\mathrm{Cov}(X, Y) = E[XY] - E[X]E[Y] \hspace{19em} (17)$$

표본 $n$ 개로 재는 공분산은 식 (18) 이다. $x_i$ 와 $y_i$ 는 $i$ 번째 관측이고 $\bar{x}$ 와 $\bar{y}$ 는 표본 평균이며, $n - 1$ 로 나누는 것은 모집단의 값을 치우침 없이 추정 (unbiased estimator) 하도록 자유도를 하나 줄인 것이다.

$$s_{XY} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) \hspace{19em} (18)$$

부호가 뜻하는 것은 셋이다. $\mathrm{Cov}(X, Y) \gt 0$ 이면 $X$ 가 커질 때 $Y$ 도 커지고, $\mathrm{Cov}(X, Y) \lt 0$ 이면 $X$ 가 커질 때 $Y$ 는 작아지며, $\mathrm{Cov}(X, Y) = 0$ 이면 두 변수 사이에 선형 관계가 없다. 자기 자신과의 공분산 $\mathrm{Cov}(X, X)$ 는 분산 $\mathrm{Var}(X)$ 다.

4.2 절과 아래의 유도는 $1/N$ 로 나눈 평균을 쓴다. 상관은 공분산을 두 표준편차로 나눈 값이고 분자와 분모가 같은 약수를 가지므로, $1/N$ 을 쓰든 $1/(n-1)$ 을 쓰든 상관의 값은 같다.

4.2 절은 공분산을 두 표준편차로 나누어 상관으로 적고, 그 상관이 식 (11) 과 식 (12) 라고 했다. 나누어야 하는 이유와 두 식의 유도가 아래다.

나누어야 하는 이유는 단위다. 열 $x$ 를 $c \gt 0$ 배 하면 $\mathrm{cov}(cx, (cx)^2) = c^3 \mathrm{cov}(x, x^2)$ 이므로, 공분산의 크기는 자료의 단위를 바꾸기만 해도 달라져 두 항의 몫을 재는 데 쓸 수 없다. 표준편차로 나누면 $\mathrm{sd}(cx) = c \cdot \mathrm{sd}(x)$ 와 $\mathrm{sd}((cx)^2) = c^2 \cdot \mathrm{sd}(x^2)$ 이 그 $c^3$ 을 약분하므로 식 (19) 의 왼쪽이 성립하고, Cauchy–Schwarz 부등식이 그 값을 $[-1, 1]$ 안에 묶어 오른쪽이 성립한다.

$$r(cx, (cx)^2) = r(x, x^2), \qquad \lvert r(x, x^2) \rvert \le 1 \hspace{12em} (19)$$

유도는 분모를 구하는 일이다. 분자는 식 (10) 이고, $\mathrm{var}(x) = \overline{u^2}$ 는 정의 그대로다. $x^2 = u^2 + 2\bar{x}u + \bar{x}^2$ 에서 상수 $\bar{x}^2$ 은 분산을 바꾸지 않으므로, 남은 두 항의 분산을 펼치면 식 (20) 이 된다.

$$\mathrm{var}(x^2) = \mathrm{var}(u^2 + 2\bar{x}u) = \overline{u^4} - (\overline{u^2})^2 + 4 \bar{x}^2 \overline{u^2} + 4 \bar{x} \overline{u^3} \hspace{6em} (20)$$

여기에 $t = \bar{x} / \sqrt{\overline{u^2}}$, $s = \overline{u^3} / (\overline{u^2})^{3/2}$, $k = \overline{u^4} / (\overline{u^2})^2$ 를 넣으면 분자와 두 분모가 식 (21) 로 적힌다.

$$\mathrm{cov}(x, x^2) = (\overline{u^2})^{3/2} (2t + s), \quad \mathrm{sd}(x) = (\overline{u^2})^{1/2}, \quad \mathrm{sd}(x^2) = \overline{u^2} \sqrt{k - 1 + 4t^2 + 4ts} \hspace{2em} (21)$$

상관은 $\mathrm{cov}(x, x^2) / (\mathrm{sd}(x) \cdot \mathrm{sd}(x^2))$ 이므로 $(\overline{u^2})^{3/2}$ 이 약분되어 4.2 절의 식 (11) 이 남는다. $s$ 와 $k$ 는 평균을 뺀 값 $u$ 로만 적혀 있어 centering 이 바꾸지 않고, centering 은 $\bar{x} = 0$ 곧 $t = 0$ 만 만들므로 식 (11) 에 $t = 0$ 을 넣은 것이 식 (12) 이다. $\lvert t \rvert$ 를 키우면 분모는 $2 \lvert t \rvert \sqrt{1 + s / t + (k - 1) / (4t^2)}$ 여서 $2 \lvert t \rvert$ 에 가까워지고 분자는 $2t$ 에 가까워지므로 $\lvert r \rvert$ 는 1 로 간다.

## Appendix C. Term Count Derivation

집합 표기를 읽는 법이 먼저다. 집합은 원소를 늘어놓아 $\lbrace 2, 4, 6 \rbrace$ 처럼 적거나, 조건으로 $\lbrace \cdot \mid \cdot \rbrace$ 꼴로 적는다. 뒤의 꼴에서는 세로줄이 중괄호 안을 둘로 나누어, 왼쪽에 원소가 취하는 모양을, 오른쪽에 그 모양이 만족해야 할 조건을 적는다. 그래서 $\lbrace n^2 \mid n \in \mathbb{Z}, \ 1 \le n \le 3 \rbrace$ 은 $n$ 이 1 부터 3 까지의 정수일 때의 $n^2$ 을 모두 모은 것, 곧 $\lbrace 1, 4, 9 \rbrace$ 이다. 세로줄 자리에는 콜론도 그만큼 자주 쓰이며, 이 문서는 둘을 함께 쓴다.

4.1 절의 식 (4) 를 여기에 다시 적는다.

$$\Phi_d(\mathbf{x}) = \left\lbrace \prod_{i=1}^{n} x_i^{a_i} \ \middle|\ a_i \in \mathbb{Z}_{\ge 0}, \ 1 \le \sum_{i=1}^{n} a_i \le d \right\rbrace \hspace{19em} (4)$$

식 (4) 는 기호가 빽빽하지만 읽는 법은 간단하다. 왼쪽의 $\Phi_d(\mathbf{x})$ 는 변수 값 한 벌 $\mathbf{x} = (x_1, \dots, x_n)$ 에서 만들어지는 새 열들의 모음이다. 세로줄 왼쪽의 $\prod_{i=1}^{n} x_i^{a_i}$ 는 변수 $x_i$ 를 각각 $a_i$ 제곱하여 모두 곱한 것, 곧 monomial 하나다. 지수 $a_i$ 는 0 이상의 정수이며 ($a_i \in \mathbb{Z}_{\ge 0}$), 0 이면 그 변수는 곱에서 빠진다. 지수의 합 $\sum_i a_i$ 가 그 항의 차수이므로, 조건 $1 \le \sum_i a_i \le d$ 는 합이 0 인 상수항을 빼고 차수를 $d$ 까지만 허용한다는 뜻이다.

변수가 두 개이고 $d = 2$ 이면 그 조건을 만족하는 지수 짝은 다섯이다. Table 3 이 그 다섯이다.

Table 3. Exponent pairs admitted by equation (4) at two variables and degree 2

| Exponent of $x_1$ | Exponent of $x_2$ | Degree | Term |
| --- | --- | --- | --- |
| 1 | 0 | 1 | $x_1$ |
| 0 | 1 | 1 | $x_2$ |
| 2 | 0 | 2 | $x_1^2$ |
| 1 | 1 | 2 | $x_1 x_2$ |
| 0 | 2 | 2 | $x_2^2$ |

빠진 짝은 $(0, 0)$ 하나이며, 그것이 상수항이다.

식 (4) 는 만들 열의 집합을 정의할 뿐 그 크기를 말하지 않는다. 그 크기가 식 (14) 와 식 (15) 이며, 아래가 그 유도다.

차수가 정확히 $k$ 인 monomial 하나는 합이 $k$ 인 음이 아닌 정수 지수 $(a_1, \dots, a_n)$ 하나에 대응하므로, 그 차수의 monomial 을 세는 일은 그런 지수 벌을 세는 일이다. 그 수가 식 (22) 이며, 왼쪽의 세로줄 둘 $\lvert \cdot \rvert$ 은 그 안에 든 집합의 원소 개수를 뜻한다.

$$\left| \lbrace (a_1, \dots, a_n) : a_i \in \mathbb{Z}_{\ge 0}, \ \sum_{i=1}^{n} a_i = k \rbrace \right| = \binom{k+n-1}{n-1} \hspace{19em} (22)$$

세는 방법은 별과 막대 (stars and bars) 다. 차수 $k$ 를 같은 별 $k$ 개로 놓고, 변수 $n$ 개를 막대 $n-1$ 개로 나눈 칸 $n$ 개로 놓으면, 한 칸에 든 별의 수가 그 변수의 지수 $a_i$ 가 된다. 그러면 지수 벌을 세는 일은 별 $k$ 개와 막대 $n-1$ 개, 모두 $k+n-1$ 개를 한 줄로 늘어놓고 그중 어느 $n-1$ 자리를 막대로 삼을지 고르는 일과 같아져 $\binom{k+n-1}{n-1}$ 이 된다.

$n = 2$, $k = 2$ 로 확인하면 $\binom{3}{1} = 3$ 이고, 배열 $\ast\ast\mid$, $\ast\mid\ast$, $\mid\ast\ast$ 가 각각 지수 $(2, 0)$, $(1, 1)$, $(0, 2)$, 곧 Table 3 의 차수 2 항 $x_1^2$, $x_1 x_2$, $x_2^2$ 셋과 같다.

차수를 0 부터 $d$ 까지 더하면 식 (23) 이 된다. 남는 몫을 담을 지수 $a_0 \ge 0$ 을 하나 더 두어 $a_0 + \sum_i a_i = d$ 로 적으면, 이 합은 물건 $d$ 개를 $n+1$ 개의 칸에 담는 경우의 수 하나로 묶인다.

$$\sum_{k=0}^{d} \binom{k+n-1}{n-1} = \binom{n+d}{d} \hspace{19em} (23)$$

식 (4) 의 집합은 $k = 0$ 인 상수항을 뺀 것이므로 그 크기는 $\binom{n+d}{d} - 1$ 이고, 이것이 식 (14) 이다.

`interaction_only` 에서는 같은 변수를 두 번 쓰지 않으므로, 남는 항 하나는 변수 $n$ 개에서 고른 크기 $j$ 의 부분집합 하나에 대응한다. $j$ 는 1 부터 $\min(d, n)$ 까지이고, 그 수를 더한 것이 식 (15) 이다. $d \ge n$ 이면 모든 부분집합이 허용되어 그 합은 식 (24) 로 닫힌다.

$$\sum_{j=1}^{n} \binom{n}{j} = 2^n - 1 \hspace{19em} (24)$$

## Appendix D. Ridge And Lasso On Expanded Columns

Expansion 이 만든 열에 거는 penalty 는 셋 가운데 하나다. 목적 함수로 적으면 ridge 는 식 (25), lasso 는 식 (26) 이며 [[12](#ref-12)], $\alpha$ 가 penalty 를 누르는 세기다.

$$\hat{\boldsymbol{\beta}}_{\mathrm{ridge}} = \arg\min_{\boldsymbol{\beta}} \lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2^2 + \alpha \lVert \boldsymbol{\beta} \rVert_2^2 \hspace{15em} (25)$$

$$\hat{\boldsymbol{\beta}}_{\mathrm{lasso}} = \arg\min_{\boldsymbol{\beta}} \lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2^2 + \alpha \lVert \boldsymbol{\beta} \rVert_1 \hspace{15em} (26)$$

차이는 penalty 의 모양에서 온다. 열이 standardization 되어 있고 서로 직교하면 두 해는 식 (27) 로 닫힌 꼴이 된다. Ridge 는 모든 계수를 같은 비율로 나누어 줄이고 0 에는 닿지 않으며, lasso 는 크기가 $\alpha / 2$ 에 못 미치는 계수를 정확히 0 으로 만들고 나머지는 그만큼 0 쪽으로 당긴다.

$$\hat{\beta}_j^{\mathrm{ridge}} = \frac{\hat{\beta}_j^{\mathrm{ols}}}{1 + \alpha}, \qquad \hat{\beta}_j^{\mathrm{lasso}} = \mathrm{sign}(\hat{\beta}_j^{\mathrm{ols}}) \max \left( \lvert \hat{\beta}_j^{\mathrm{ols}} \rvert - \frac{\alpha}{2}, \ 0 \right) \hspace{9em} (27)$$

Expansion 이 만든 열은 직교와 거리가 멀고 (4.2 절), 서로 닮은 열이 무리를 이룬다. Ridge 는 그 무리에 계수를 나누어 주고, lasso 는 하나만 남기고 나머지를 0 으로 만든다. 어느 것이 남을지는 표본이 조금만 달라져도 바뀌므로, lasso 가 돌려주는 항의 목록은 그 자체로 불안정하다. 둘을 $\rho$ 로 섞은 elastic net 이 식 (28) 이며 [[13](#ref-13)], $\rho$ 가 1 이면 lasso, 0 이면 ridge 다. 제곱 항이 닮은 무리를 함께 남기거나 함께 지우므로, 항을 고르면서도 목록이 덜 흔들린다.

$$\hat{\boldsymbol{\beta}}_{\mathrm{enet}} = \arg\min_{\boldsymbol{\beta}} \lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2^2 + \alpha \left( \rho \lVert \boldsymbol{\beta} \rVert_1 + \frac{1 - \rho}{2} \lVert \boldsymbol{\beta} \rVert_2^2 \right) \hspace{9em} (28)$$

Table 4. Penalties on expanded columns

| Penalty | Term added | A group of columns that resemble one another | Where it fits |
| --- | --- | --- | --- |
| Ridge | Sum of the squared coefficients | Coefficient shared across the group | The default on expanded columns |
| Lasso | Sum of the absolute coefficients | One kept, the rest at zero | A short term list, under a heredity constraint |
| Elastic net | Both, mixed by $\rho$ | Kept or dropped together | Selection wanted with a list that holds |

$\alpha$ 는 held-out 오차로 고르며, 후보는 10 의 거듭제곱 간격으로 잡는다. standardization 한 열 위에서만 뜻이 있고 (5.2 절), 그 탐색을 `RidgeCV`, `LassoCV`, `ElasticNetCV` 가 대신한다. 절편은 penalty 에서 뺀다. 절편에 penalty 를 걸면 적합된 수준이 0 쪽으로 끌려가 model 이 자료의 중심에서 벗어난다.

Penalty 를 건다고 degree 를 4 로 올릴 수 있는 것은 아니다. Penalty 가 버는 것은 열 수가 행 수에 가까울 때 적합이 무너지느냐 버티느냐의 차이이며, 그 차이의 크기는 5.1 절에 있다.

## Appendix E. Implementation

### E.1 Options

Expansion 자체는 `sklearn.preprocessing.PolynomialFeatures` 한 줄이며, 정할 것은 네 인자뿐이다 [[7](#ref-7)].

Table 5. PolynomialFeatures arguments

| Argument | Effect | Note |
| --- | --- | --- |
| `degree` | Highest degree of the monomials | A `(min, max)` tuple for the lowest degree as well, so `(2, 2)` for second-order terms only |
| `interaction_only` | Products of distinct variables only | First-order terms kept, powers of a single variable dropped |
| `include_bias` | A constant column of ones | False where the estimator carries its own intercept |
| `order` | Memory layout of the output array | 'C' or 'F', a choice of layout rather than of content |

```python
# Python
from sklearn.preprocessing import PolynomialFeatures

# interaction_only=True keeps X1*X2 and drops X1^2, X2^2
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_expanded = poly.fit_transform(X)
term_name = poly.get_feature_names_out()
```

`get_feature_names_out()` 이 돌려주는 이름은 계수를 다시 열에 되짚는 유일한 통로다. Expansion 뒤에 이름을 잃으면 계수는 남아도 그것이 어느 곱의 계수인지 말할 수 없다.

### E.2 Pipeline

Expansion 은 홀로 쓰지 않고 standardization 과 penalty 사이에 둔다. 순서는 원 변수의 standardization, expansion, expansion 이 만든 열의 두 번째 standardization, 그리고 penalty 를 건 적합이다.

앞의 standardization 은 4.3 절의 조건수 문제를 없애고, 뒤의 standardization 은 penalty 가 열마다 공평하게 걸리게 한다. 곱항의 분산은 원 변수 분산의 곱에 가까워 열마다 크게 벌어지므로, 두 번째 standardization 없이 ridge 를 걸면 penalty 가 사실상 분산이 큰 열에만 걸린다.

```python
# Python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

pipeline = Pipeline([
    ('raw_scale', StandardScaler()),
    ('expand', PolynomialFeatures(include_bias=False)),
    ('term_scale', StandardScaler()),
    ('fit', Ridge()),
])
grid = {'expand__degree': [1, 2, 3], 'fit__alpha': [0.1, 1.0, 10.0, 100.0]}
search = GridSearchCV(pipeline, grid, scoring='neg_root_mean_squared_error', cv=5)
search.fit(X, y)
```

Expansion 을 pipeline 안에 두는 이유는 편의가 아니다. Expansion 자체는 행마다 독립이라 누수 (leakage) 를 만들지 않지만, 앞뒤의 standardization 은 cross-validation 이 자료를 나눈 조각 (fold) 의 훈련 부분에서만 평균과 분산을 얻어야 한다. degree 와 penalty 를 함께 고르는 일도 pipeline 안에서만 한 번의 탐색으로 끝난다.

### E.3 Cost

Expansion 의 비용은 열 수에 선형이고, 그 열 수는 식 (14) 로 늘어난다. 행 100,000, 변수 100, $d = 2$ 이면 열은 5,150 개이고, 값을 하나도 빠뜨리지 않고 담는 dense 행렬로 두면 64-bit 실수 기준 4.1 GB 다. Expansion 결과를 memory 에 두지 않는 길이 둘 있다.

첫째는 kernel 이다. 다항 kernel 식 (29) 는 expansion 한 공간의 내적을 expansion 없이 계산한다.

$$K(\mathbf{x}, \mathbf{z}) = (\gamma \mathbf{x}^{\top} \mathbf{z} + c)^{d} \hspace{19em} (29)$$

`KernelRidge(kernel='poly')` 가 그 형태이며, 비용이 열이 아니라 행에 걸리므로 변수가 많고 행이 적은 자료에 맞는다. 대신 계수가 개별 monomial 에 붙지 않아 어느 곱이 기여했는지 읽을 수 없다.

둘째는 근사다. `PolynomialCountSketch` 는 다항 kernel 이 쓰는 항들을 정해진 수의 열로 줄여 담고 (sketch), `Nystroem` 은 표본의 부분집합으로 kernel 행렬을 근사한다. 둘 다 kernel 을 유한한 수의 열로 근사해 선형 model 의 속도를 지키는 계열이며 [[8](#ref-8)], 열 수를 사용자가 정한 값으로 묶는다.

희소 입력은 그대로 받는다. 0 이 아닌 값만 저장하는 CSR 형식의 행렬을 넣으면 expansion 결과도 같은 형식으로 나오므로, dummy 열이 많은 자료가 dense 로 부풀지 않는다.

### E.4 Selective Expansion

모든 짝을 만들 필요는 없다. 곱할 열을 골라 넘기면 열 수는 Table 1 이 아니라 고른 개수로 끝난다.

```python
# Python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures

expand_column = ['temperature', 'pressure']
transformer = ColumnTransformer(
    [('expand', PolynomialFeatures(degree=2, include_bias=False), expand_column)],
    remainder='passthrough',
)
```

고를 근거는 셋이다. 공정이 이미 아는 상호작용, 잔차가 두 변수의 조합에서 구조를 보이는 경우, 그리고 tree 여러 개를 합친 model (tree ensemble) 을 먼저 돌려 상호작용의 세기를 재고 상위 짝만 남기는 방법이다 [[9](#ref-9)]. 셋 다 없으면 전체 expansion 에 penalty 를 거는 편이 낫다. 근거 없이 짝을 고르면 어느 상호작용이 model 에 들어갈지를 자료가 아니라 분석자가 정하게 된다.
