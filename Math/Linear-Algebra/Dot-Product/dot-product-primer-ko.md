# Dot Product Primer
Rev. 3 | Created: 2026-10-25 | Updated: 2026-10-25 16:10 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Notation](#3-notation)
- [4. Interpretation](#4-interpretation)
  - [4.1 Algebraic Form](#41-algebraic-form)
  - [4.2 Geometric Form](#42-geometric-form)
  - [4.3 Projection](#43-projection)
- [5. Worked Example](#5-worked-example)
- [6. Application](#6-application)
  - [6.1 Linear Regression and Perceptron](#61-linear-regression-and-perceptron)
  - [6.2 PLS and PCA Scores](#62-pls-and-pca-scores)
  - [6.3 Breaking Conditions](#63-breaking-conditions)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Derivation](#appendix-b-derivation)

## 1. Purpose

- **Problem Statement**: 내적 $\mathbf{x}_i^\top \mathbf{w}$ 는 원소를 곱해 더하는 계산으로 쓰이지만, 그 값이 두 vector 의 길이와 각도로 무엇을 뜻하는지는 계산식에 드러나지 않는다.
- **Goal**: 독자가 내적 값 하나를 보고 vector 의 길이와 방향 일치도 가운데 무엇이 그 값을 정했는지 가릴 수 있게 한다.
- **Non-Goal**: 복소 vector 의 내적과 일반 inner product space 의 공리는 다루지 않는다.

## 2. Summary

내적의 대수적 정의인 원소별 곱의 합은 두 vector 의 길이의 곱에 사잇각의 cosine 을 곱한 값과 같다.
$\lVert \mathbf{w} \rVert = 1$ 이면 내적은 $\mathbf{x}_i$ 를 $\mathbf{w}$ 축에 정사영 (projection) 한 부호 있는 길이이다.
선형 회귀의 예측값과 PLS·PCA 의 score 가 모두 이 정사영 길이로 계산된다.

## 3. Notation

이 문서가 쓰는 기호는 아래 Table 1 과 같다.

Table 1. Symbols

| Symbol                     | Meaning                                         | Shape           | Assumption                                                           |
| :------------------------: | :---------------------------------------------: | :-------------: | :------------------------------------------------------------------: |
| $\mathbf{x}_i$             | $i$ 번째 sample 의 입력, 센서값이나 feature     | $p$ 차원 vector | 실수 원소                                                            |
| $\mathbf{w}$               | model 의 weight vector                          | $p$ 차원 vector | 실수 원소                                                            |
| $p$                        | vector 의 차원                                  | scalar          | 양의 정수                                                            |
| $x_{ij}$, $w_j$            | $\mathbf{x}_i$ 와 $\mathbf{w}$ 의 $j$ 번째 원소 | scalar          | $j = 1, \dots, p$                                                    |
| $\lVert \mathbf{v} \rVert$ | vector $\mathbf{v}$ 의 Euclidean norm           | scalar          | 0 이상                                                               |
| $\theta$                   | $\mathbf{x}_i$ 와 $\mathbf{w}$ 의 사잇각        | scalar          | $0^\circ \le \theta \le 180^\circ$, 두 vector 모두 영 vector 가 아님 |
| $b$                        | 선형 model 의 절편                              | scalar          | 실수                                                                 |
| $\hat{y}_i$                | sample $i$ 의 예측값                            | scalar          | 실수                                                                 |
| $t_i$                      | sample $i$ 의 score                             | scalar          | $\lVert \mathbf{w} \rVert = 1$                                       |

특수 기호는 아래와 같이 읽는다.

- $\mathbf{x}_i^\top$: 영어로 "x i transpose", 한국어로 "x i 의 전치" 또는 "엑스 아이 트랜스포즈" 라 읽는다.
- $\sum$: 영어로 "sum" 또는 "sigma", 한국어로 "합" 또는 "시그마" 라 읽는다. 식 (1) 처럼 아래위에 $j = 1$ 과 $p$ 가 붙으면 영어로 "the sum over j from 1 to p", 한국어로 "j 가 1 부터 p 까지의 합" 이라 읽는다.
- $\lVert \mathbf{v} \rVert$: 영어로 "the norm of v" 또는 "norm v", 한국어로 "v 의 노름" 이라 읽는다.
- $\hat{y}_i$: 영어로 "y hat i" 또는 "y hat sub i", 한국어로 "와이 햇 아이" 라 읽는다. 기호 위의 hat 은 그 값이 model 의 추정값임을 나타낸다.

## 4. Interpretation

내적은 대수적 정의와 기하학적 정의 두 가지로 적을 수 있고, 두 정의는 같은 값을 준다.

```math
\mathbf{x}_i^\top \mathbf{w} = \sum_{j=1}^{p} x_{ij} w_j = \lVert \mathbf{x}_i \rVert \lVert \mathbf{w} \rVert \cos\theta \hspace{19em} (1)
```

두 정의가 같다는 것은 코사인 법칙에서 나오며, 유도는 [Appendix B](#appendix-b-derivation) 에 있다.

### 4.1 Algebraic Form

식 (1) 의 가운데 변은 두 vector 에서 같은 위치의 원소를 곱한 뒤 모두 더한 값이다. 컴퓨터는 내적을 이 식으로 계산하며, 이 변은 영 vector 를 포함한 모든 실수 vector 에서 성립한다.

Vector 의 길이 (norm) 는 자기 자신과의 내적의 제곱근이다.

```math
\lVert \mathbf{x}_i \rVert = \sqrt{\mathbf{x}_i^\top \mathbf{x}_i} = \sqrt{\sum_{j=1}^{p} x_{ij}^2} \hspace{19em} (2)
```

### 4.2 Geometric Form

식 (1) 의 오른쪽 변에서 내적은 두 vector 의 길이와 사잇각의 cosine 을 곱한 값이다. 내적을 두 길이의 곱으로 나눈 $\cos\theta$ 가 두 vector 의 방향 일치도이다.

```math
\cos\theta = \frac{\mathbf{x}_i^\top \mathbf{w}}{\lVert \mathbf{x}_i \rVert \lVert \mathbf{w} \rVert} \hspace{19em} (3)
```

식 (3) 은 분모가 0 이 아닐 때, 곧 두 vector 가 모두 영 vector 가 아닐 때만 성립한다. 영 vector 에는 방향이 없어 $\theta$ 가 정의되지 않는다.

사잇각에 따른 내적의 값은 아래 Table 2 와 같다.

Table 2. Dot product by angle

| Angle       | $\cos\theta$ | Dot product            | Meaning                           |
| :---------: | :----------: | :--------------------: | :-------------------------------: |
| $0^\circ$   | $1$          | 최대, 길이의 곱        | 같은 방향                         |
| $90^\circ$  | $0$          | $0$                    | 직교, $\mathbf{w}$ 방향 성분 없음 |
| $180^\circ$ | $-1$         | 최소, 길이의 곱의 음수 | 반대 방향                         |

$\theta$ 가 $0^\circ$ 에서 $180^\circ$ 로 커지는 동안 $\cos\theta$ 는 1 에서 -1 로 줄어들므로, 길이가 같으면 내적은 방향이 어긋날수록 작아진다.

### 4.3 Projection

$\lVert \mathbf{w} \rVert = 1$ 인 단위 vector 이면 식 (1) 은 아래와 같이 줄어든다.

```math
\mathbf{x}_i^\top \mathbf{w} = \lVert \mathbf{x}_i \rVert \cos\theta \hspace{19em} (4)
```

식 (4) 의 값은 $\mathbf{x}_i$ 의 끝점에서 $\mathbf{w}$ 축에 수선을 내렸을 때 원점에서 수선의 발까지의 부호 있는 거리이다. 수선의 발은 $(\mathbf{x}_i^\top \mathbf{w})\,\mathbf{w}$ 이고, 수선 $\mathbf{x}_i - (\mathbf{x}_i^\top \mathbf{w})\,\mathbf{w}$ 는 $\mathbf{w}$ 와 직교한다. $\lVert \mathbf{w} \rVert \ne 1$ 이면 내적은 정사영 길이에 $\lVert \mathbf{w} \rVert$ 를 곱한 값이 된다.

## 5. Worked Example

반도체 증착 또는 식각 chamber 에서 wafer 한 장을 처리하며 얻은 정규화 센서값을 입력 vector 로, 두께 예측에서 각 센서의 중요도를 weight vector 로 둔다.

```math
\mathbf{x}_i = \begin{bmatrix} \mathrm{RF\ Power} \\ \mathrm{Chamber\ Pressure} \\ \mathrm{Gas\ Flow} \end{bmatrix} = \begin{bmatrix} 10 \\ 5 \\ 2 \end{bmatrix}, \quad \mathbf{w} = \begin{bmatrix} 0.6 \\ 0.8 \\ 0.0 \end{bmatrix} \hspace{19em} (5)
```

$\lVert \mathbf{w} \rVert = \sqrt{0.6^2 + 0.8^2 + 0.0^2} = 1.0$ 이므로 $\mathbf{w}$ 는 단위 vector 이다.

대수적 정의는 식 (1) 의 가운데 변을 그대로 따른다.

```math
\mathbf{x}_i^\top \mathbf{w} = (10 \times 0.6) + (5 \times 0.8) + (2 \times 0.0) = 6 + 4 + 0 = 10 \hspace{19em} (6)
```

기하학적 정의는 내적을 쓰지 않고 세 변의 길이만으로 구한다. $\mathbf{x}_i - \mathbf{w} = [9.4, 4.2, 2.0]^\top$ 이므로 $\lVert \mathbf{x}_i - \mathbf{w} \rVert^2 = 110$ 이고, 코사인 법칙 ([Appendix B](#appendix-b-derivation) 의 첫 단계) 에 넣으면 아래와 같다.

```math
\lVert \mathbf{x}_i \rVert \lVert \mathbf{w} \rVert \cos\theta = \frac{\lVert \mathbf{x}_i \rVert^2 + \lVert \mathbf{w} \rVert^2 - \lVert \mathbf{x}_i - \mathbf{w} \rVert^2}{2} = \frac{129 + 1 - 110}{2} = 10 \hspace{19em} (7)
```

식 (6) 과 식 (7) 이 모두 10 이므로 식 (1) 의 두 변이 같다. 길이와 각도를 따로 보면 $\lVert \mathbf{x}_i \rVert = \sqrt{129} \approx 11.36$, $\cos\theta = 10 / 11.36 \approx 0.88$, $\theta \approx 28^\circ$ 이다.

정사영으로 보면 수선의 발은 $10\,\mathbf{w} = [6, 8, 0]^\top$ 이고 수선은 $\mathbf{x}_i - 10\,\mathbf{w} = [4, -3, 2]^\top$ 이다. 수선과 $\mathbf{w}$ 의 내적은 $4 \times 0.6 + (-3) \times 0.8 + 2 \times 0.0 = 0$ 이므로 식 (4) 의 해석대로 둘은 직교한다.

RF Power 와 압력의 비율이 달라진 wafer 의 입력이 $[2, 10, 8]^\top$ 이면 값은 아래 Table 3 과 같이 바뀐다.

Table 3. Normal and shifted wafer

| Wafer   | $\mathbf{x}_i$ | $\lVert \mathbf{x}_i \rVert$ | $\mathbf{x}_i^\top \mathbf{w}$ | $\cos\theta$ | $\theta$   |
| :-----: | :------------: | :--------------------------: | :----------------------------: | :----------: | :--------: |
| Normal  | $[10, 5, 2]$   | $11.36$                      | $10.0$                         | $0.88$       | $28^\circ$ |
| Shifted | $[2, 10, 8]$   | $12.96$                      | $9.2$                          | $0.71$       | $45^\circ$ |

Shifted wafer 는 길이가 11.36 에서 12.96 으로 늘었으나 사잇각이 $28^\circ$ 에서 $45^\circ$ 로 커져, 내적은 10.0 에서 9.2 로 0.8 만 줄었다. 같은 변화에서 $\cos\theta$ 는 0.88 에서 0.71 로 줄었으므로, 이상 감지에서는 내적 하나보다 길이와 식 (3) 의 $\cos\theta$ 를 나누어 본다.

## 6. Application

선형 model 은 입력 vector 를 weight vector 축에 정사영한 길이로 출력을 계산한다.

### 6.1 Linear Regression and Perceptron

선형 회귀와 perceptron 의 출력은 내적에 절편 $b$ 를 더한 값이다.

```math
\hat{y}_i = \mathbf{x}_i^\top \mathbf{w} + b \hspace{19em} (8)
```

식 (8) 의 $\mathbf{x}_i^\top \mathbf{w}$ 는 입력이 $\mathbf{w}$ 방향으로 가진 성분이 예측값에 더하는 몫이다. $\mathbf{w}$ 와 직교하는 입력 성분은 예측값을 바꾸지 않는다.

### 6.2 PLS and PCA Scores

PLS 와 PCA 는 단위 길이의 weight vector $\mathbf{w}$ 를 구한 뒤, centering 한 sample $\mathbf{x}_i$ 의 score 를 식 (4) 의 정사영 길이로 계산한다.

```math
t_i = \mathbf{x}_i^\top \mathbf{w}, \quad \lVert \mathbf{w} \rVert = 1 \hspace{19em} (9)
```

수백 개의 센서로 이루어진 $p$ 차원 입력이 축 하나에서 숫자 하나 $t_i$ 로 줄어든다. 축을 여러 개 쓰면 score 도 그 개수만큼 생겨 저차원 latent space 의 좌표가 된다.

### 6.3 Breaking Conditions

내적을 방향 일치도나 정사영 길이로 읽을 수 없는 조건은 아래 셋이다.

- **영 vector**: $\mathbf{x}_i$ 나 $\mathbf{w}$ 가 영 vector 이면 식 (3) 의 분모가 0 이 되어 $\cos\theta$ 와 $\theta$ 가 정의되지 않는다. 내적 자체는 0 이다.
- **단위 vector 가 아닌 weight**: $\lVert \mathbf{w} \rVert \ne 1$ 이면 내적은 정사영 길이에 $\lVert \mathbf{w} \rVert$ 를 곱한 값이어서, 식 (9) 의 score 로 쓰기 전에 $\mathbf{w}$ 를 그 길이로 나누어야 한다.
- **centering 하지 않은 입력**: PLS 와 PCA 의 축은 centering 한 자료에서 구하므로, centering 하지 않은 $\mathbf{x}_i$ 를 식 (9) 에 넣으면 평균 vector 의 정사영 길이만큼 score 가 밀린다.

---

## Appendix A. Terminology

- **centering**: 각 열에서 그 열의 평균을 빼서 평균을 0 으로 만드는 전처리.
- **latent space**: 원래 입력을 몇 개의 축으로 정사영하여 얻은 저차원 좌표 공간.
- **norm**: Vector 의 길이. 이 문서에서는 식 (2) 의 Euclidean norm 이다.
- **PCA**: Principal Component Analysis. 입력의 분산이 가장 큰 방향을 차례로 축으로 고르는 차원 축소 방법.
- **perceptron**: 입력과 weight 의 내적에 절편을 더한 값의 부호로 class 를 정하는 선형 분류기.
- **PLS**: Partial Least Squares. 입력의 score 와 출력의 공분산이 가장 큰 방향을 차례로 축으로 고르는 회귀 방법.
- **score**: Sample 을 한 축에 정사영한 부호 있는 길이.
- **weight vector**: Model 이 입력의 각 원소에 곱하는 계수를 모은 vector.

## Appendix B. Derivation

식 (1) 의 두 변이 같다는 것은 $\mathbf{x}_i$, $\mathbf{w}$, $\mathbf{x}_i - \mathbf{w}$ 를 세 변으로 하는 삼각형에서 나온다. 두 vector 는 모두 영 vector 가 아니라고 둔다. 단계마다 쓴 성질은 아래 Table 4 와 같다.

Table 4. Derivation of equation (1)

| Step  | Expression                                                                                                                                                                 | Property                                                                |
| :---: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------: |
| 1     | $\lVert \mathbf{x}_i - \mathbf{w} \rVert^2 = \lVert \mathbf{x}_i \rVert^2 + \lVert \mathbf{w} \rVert^2 - 2 \lVert \mathbf{x}_i \rVert \lVert \mathbf{w} \rVert \cos\theta$ | 코사인 법칙                                                             |
| 2     | $\lVert \mathbf{x}_i - \mathbf{w} \rVert^2 = (\mathbf{x}_i - \mathbf{w})^\top (\mathbf{x}_i - \mathbf{w})$                                                                 | 식 (2) 의 norm 정의                                                     |
| 3     | $(\mathbf{x}_i - \mathbf{w})^\top (\mathbf{x}_i - \mathbf{w}) = \mathbf{x}_i^\top \mathbf{x}_i - 2\,\mathbf{x}_i^\top \mathbf{w} + \mathbf{w}^\top \mathbf{w}$             | 분배법칙, $\mathbf{w}^\top \mathbf{x}_i = \mathbf{x}_i^\top \mathbf{w}$ |
| 4     | $\lVert \mathbf{x}_i - \mathbf{w} \rVert^2 = \lVert \mathbf{x}_i \rVert^2 + \lVert \mathbf{w} \rVert^2 - 2\,\mathbf{x}_i^\top \mathbf{w}$                                  | 식 (2) 를 단계 3 에 대입                                                |
| 5     | $\mathbf{x}_i^\top \mathbf{w} = \lVert \mathbf{x}_i \rVert \lVert \mathbf{w} \rVert \cos\theta$                                                                            | 단계 1 과 단계 4 의 우변 비교                                           |

단계 5 의 왼쪽 $\mathbf{x}_i^\top \mathbf{w}$ 를 원소로 풀면 식 (1) 의 가운데 변인 원소별 곱의 합이 되므로, 식 (1) 의 세 변이 모두 같다.
