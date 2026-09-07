# Overfitting
Rev. 1 | Created: 2026-09-07 | Updated: 2026-09-07 16:33 CDT

이 폴더는 overfitting 을 자료의 모양에 따라 나누어 다룬다. 열이 행보다 많은 자료와 행이 열보다 많은 자료는 같은 이름의 실패를 서로 다른 이유로 겪으므로, 방어도 검증도 다르게 설계해야 한다.

## 1. Documents

Table 1. Documents in this folder

| # | Document | Subject |
|---|----------|---------|
| 1 | [overfitting-in-wide-data.md](overfitting-in-wide-data.md) | 열의 수가 행의 수보다 크거나 비슷한 자료. 세 가지 원인과 다섯 갈래의 방어, 그리고 그 효과를 정직하게 재는 절차 |
| 2 | [overfitting-in-long-data.md](overfitting-in-long-data.md) | 행의 수가 열의 수보다 훨씬 큰 자료. Model 용량, 서로 닮은 행, 절차의 누수라는 세 경로와 그 각각의 장치, 그리고 어느 경로인지 가려내는 검사 |

## 2. Which One To Read

$p$ 를 열의 수, $n$ 을 행의 수라 할 때 갈림길은 하나이다.

- $p$ 가 $n$ 에 견주어 크거나 더 크면 wide data 이다. 자유도가 열에서 오므로 열을 줄이거나 계수를 묶는 것이 먼저이다.
- $n$ 이 $p$ 보다 훨씬 크면 long data 이다. 자유도가 model 의 유연성에서 오므로 용량을 묶고 분할을 고치는 것이 먼저이다.
- 둘 다 크면 두 문서를 함께 읽는다. 방어는 겹쳐 쓸 수 있고, 분할 규칙은 어느 경우에나 자료가 묶인 방식을 따른다.

두 문서는 각각 독립으로 읽히며, 어느 쪽도 다른 쪽을 읽었다고 가정하지 않는다.
