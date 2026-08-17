# Referenced R² — Specifying the Dispersion in the R² Denominator
Rev. 1 | Created: 2026-08-15 | Updated: 2026-08-16 20:30 CDT

> 표준 R² 의 분모를 데이터에서 계산하지 않고 지정한 기준으로 바꿀 수 있는지,
> 바꾸면 그 값이 물리적으로 무엇을 뜻하게 되는지 정리한 문서.

## 1. Question

표준 R² 는 아래와 같이 정의된다.

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

분모 `SS_tot` 는 데이터 자체의 평균 `y_bar` 를 기준으로 한 총제곱합이며, 이것이
"데이터의 전체 산포" 로 불리는 양이다. 이 분모를 데이터에서 계산하지 않고 밖에서
지정한 값으로 바꿀 수 있는가 하는 것이 이 문서의 질문이다.

답은 가능하다는 것이다. 원리상 자연스러울 뿐 아니라 이미 확립된 형태가 여럿 있다.
다만 분모를 바꾸는 순간 지표의 의미가 함께 바뀌므로, 무엇으로 바꿨는지를 반드시
같이 보고해야 한다.

이렇게 분모를 지정해서 만든 지표를 이 문서에서는 **Referenced R²** 라 부르고, 본문에서는
`Ref_R2`, 수식에서는 Ref R² 로 표기한다. 분모를 무엇으로 지정했는지를 늘 함께 적어야
하므로, 이름 자체에 "기준이 있다" 는 사실을 박아 둔 것이다.

## 2. Structure

분모를 왜 바꿔도 되는지는 R² 를 다시 읽으면 드러난다. `SS_tot` 는 "항상 `y_bar` 를
예측하는 모델" 의 오차제곱합과 정확히 같다. 즉 R² 는 처음부터 아래 형태였다.

```text
                error left by the model
R2 = 1 -  ───────────────────────────────────
            error left by a chosen baseline
```

표준 R² 는 baseline 을 "평균만 내놓는 모델" 로 고정한 특수한 경우이고, 이 구조는
skill score 라는 이름으로 널리 쓰인다. **분모를 지정한다는 것은 baseline 을 바꾸는
것이며, 그 이상도 이하도 아니다.** 이것이 이 문서의 나머지 내용을 지배하는 사실이다.

따라서 `Ref_R2` 는 새로운 계산이 아니라 baseline 을 명시적으로 고른 R² 이고, 표준 R² 는
그 baseline 을 데이터의 평균으로 암묵적으로 정해 둔 `Ref_R2` 의 한 사례다.

분자와 분모가 모두 y 의 제곱 단위이므로 비율은 무차원이고, `Ref_R2` 는 "baseline 이
감수해야 했던 오차 중 몇 %를 모델이 없앴는가" 로 읽힌다. Baseline 을 무엇으로 잡든 이
해석은 유지된다.

## 3. Variants

**Table 1. Ways to fix the denominator**

| Variant | Denominator | Baseline it encodes | Typical use |
|---|---|---|---|
| Standard | `Σ(y_i − y_bar)²` | 이 데이터셋의 평균 | 단일 데이터셋 안에서의 적합도 |
| Fixed reference point | `Σ(y_i − y_train_bar)²` | 학습 때 알던 평균 | Test set 평가 |
| Fixed reference dispersion | `N · sigma_ref²` | 외부에서 정한 산포 | Lot, batch, 기간 간 비교 |
| Baseline model | `Σ(y_i − y_base_i)²` | 임의의 기준 모델 | 시계열, 기존 운영 모델 대비 |

### 3.1. Fixed Reference Point

가장 흔한 경우다. Test set 을 평가할 때 분모의 기준을 test 데이터의 평균이 아니라
학습 데이터의 평균 `y_train_bar` 로 고정한다.

$$\mathrm{Ref}\,R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y}_{train})^2}$$

두 가지 이유로 이 형태가 옳다. 첫째, test 데이터의 평균은 평가 시점에 알 수 없는
값이므로 그것을 baseline 에 쓰면 미래 정보를 끌어다 쓰는 셈이 된다. 둘째, test set 을
어떻게 자르느냐에 따라 분모가 달라져 잣대 자체가 흔들린다. 학습 평균으로 고정하면
"훈련 때 알던 평균만 계속 내놓는 모델 대비 얼마나 개선했는가" 라는 질문이 되어,
데이터를 어떻게 잘랐든 같은 잣대가 유지된다.

이 형태는 밖에서 out-of-sample R² 로 통하므로, 외부 보고에서는 `Ref_R2` 대신 그 이름을
쓴다.

### 3.2. Fixed Reference Dispersion

분모를 데이터에서 계산하지 않고 알려진 참조 분산으로 대체한다.

$$\mathrm{Ref}\,R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{N \cdot \sigma_{ref}^2}$$

`sigma_ref` 로는 공정 규격의 허용 산포, 과거 누적 데이터의 분산, 계측 시스템의 기준
분산 같은 도메인 기준을 쓴다. 두께 예측이라면 해당 layer 의 관리 규격 산포를 분모에
넣어, 고정된 잣대 대비 모델의 성능을 재는 지표가 된다.

### 3.3. Baseline Model

분모를 특정 모델의 오차제곱합으로 두면 가장 일반적인 형태가 된다.

$$\mathrm{Ref}\,R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - y_{base,i})^2}$$

시계열에서 baseline 을 직전값으로 두는 persistence baseline, 계절 평균으로 두는 형태,
현재 운영 중인 모델로 두어 교체 가치를 재는 형태가 모두 여기에 속한다. 3.1 은 이 식에서
`y_base_i` 를 상수 `y_train_bar` 로 둔 경우다. 3.2 는 baseline 의 예측값을 정하는 대신
그 baseline 이 낼 오차의 크기를 `N · sigma_ref²` 로 직접 못 박은 형태이므로, 예측값을
경유하지 않는다는 점에서 나머지 둘과 다르다.

세 절의 좌변이 모두 같은 것은 오기가 아니다. 세 변형은 `Ref_R2` 의 분모를 무엇으로
지정했느냐만 다르며, 이것이 2 에서 말한 사실의 표기상 결과다.

## 4. Physical Meaning

분모를 고정하는 진짜 이유는 편의가 아니라 **표준 R² 가 데이터셋 간 비교에 쓸 수 없는
지표**라는 데 있다. 분모가 데이터마다 다시 계산되므로 잣대가 데이터마다 바뀐다.

같은 모델을 세 lot 에 적용해 RMSE 가 세 곳 모두 0.5 nm 로 동일하게 나온 경우를 보면
문제가 분명해진다.

**Table 2. The same model on three lots, RMSE fixed at 0.5 nm**

| Lot | Lot dispersion (nm) | Standard R² | `Ref_R2` against a spec of 1.0 nm |
|---|---|---|---|
| A | 2.0 | 0.9375 | 0.7500 |
| B | 1.2 | 0.8264 | 0.7500 |
| C | 0.6 | 0.3056 | 0.7500 |

모델의 예측 오차는 세 lot 에서 완전히 같은데 표준 R² 는 0.94 에서 0.31 까지 벌어진다.
변한 것은 모델이 아니라 분모다. 산포가 큰 lot 은 baseline 이 원래 못 맞히던 lot 이라
같은 성능이 좋아 보이고, 산포가 작은 lot 은 baseline 도 이미 잘 맞히던 lot 이라 같은
성능이 나빠 보인다. 표준 R² 는 이 사실을 정직하게 반영하고 있을 뿐이며, 이 값으로
lot 을 줄 세우면 모델이 아니라 lot 의 산포를 줄 세우게 된다.

분모를 1.0 nm 로 고정하면 세 lot 모두 0.75 가 되어, 지표가 모델의 성능만 반영한다.
바꿔 말하면 두 지표는 서로 다른 질문에 답한다.

- 표준 R² — 이 데이터셋 안에서 모델이 평균 대비 얼마나 나은가. 데이터셋마다 잣대가 다르다.
- `Ref_R2` — 정해진 잣대 대비 모델이 얼마나 나은가. 데이터셋을 가로질러 비교할 수 있다.

어느 쪽이 옳은지는 질문에 달렸다. "이 lot 을 얼마나 설명했는가" 를 묻는다면 표준 R² 가
맞고, "이 모델을 라인 전체에 깔아도 되는가" 를 묻는다면 `Ref_R2` 가 맞다.

## 5. Reporting

분모를 바꾸면 그 값은 더 이상 "이 데이터셋 안에서 설명된 분산의 비율" 이 아니다.
지정한 baseline 대비 상대 성능으로 의미가 바뀌므로 아래를 지킨다.

- `Ref_R2` 라는 이름을 쓰고 `R²` 기호를 그대로 쓰지 않는다. 같은 기호를 쓰면 읽는 쪽이 표준 R² 로 오해한다.
- 이름만으로는 부족하므로 분모의 기준을 값과 출처까지 함께 적는다. `Ref_R2 = 0.75 (reference: spec tolerance sigma = 1.0 nm)` 처럼 쓰고, 숫자만 적지 않는다.
- 표준 R² 를 함께 보고한다. 두 값의 차이가 곧 그 데이터셋의 산포가 규격 대비 어느 쪽으로 치우쳤는지를 알려준다.
- `sigma_ref` 를 고른 근거를 남긴다. 근거 없이 고른 분모는 지표 전체를 자의적으로 만든다.

`Ref_R2` 는 이 문서에서 정한 이름이므로 문서 밖에서는 통하지 않는다. 외부에 낼 때는 이름
뒤에 정의식을 한 줄 붙이거나, 3.1 처럼 통용되는 이름이 있으면 그쪽을 쓴다.

일반적인 library 함수는 분모를 관측값에서 계산하므로 그대로 쓸 수 없고, 직접 구현해야
한다. 계산 자체는 나눗셈 한 번이다.

```python
# Python
import numpy as np


def referenced_r2(y_true: np.ndarray = None, y_pred: np.ndarray = None,
                  sigma_ref: float = None) -> float:
    """Ref_R2 measured against a fixed reference dispersion instead of the data spread.

    Args:
        y_true: observations, shape (n,).
        y_pred: predictions, shape (n,).
        sigma_ref: reference dispersion in the unit of y, strictly positive.

    Returns:
        The skill relative to a baseline whose error scale is sigma_ref.
        Negative when the model is worse than that baseline.
    """
    if sigma_ref is None or sigma_ref <= 0.0:
        raise ValueError(f"sigma_ref must be a positive dispersion, got {sigma_ref}.")

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_ref = float(y_true.shape[0]) * sigma_ref ** 2
    return 1.0 - ss_res / ss_ref
```

## 6. Cautions

- **음수가 나온다.** 모델이 baseline 보다 못하면 `SS_res` 가 분모를 넘어 값이 0 아래로 간다. 규격 1.0 nm 에 대해 RMSE 가 1.3 nm 이면 −0.69 다. 이는 오류가 아니라 "이 모델을 쓰느니 baseline 을 쓰는 편이 낫다" 는 정상적인 신호이며, out-of-sample 평가에서는 흔히 나온다.
- **분모가 작으면 값이 폭발한다.** 규격 산포가 매우 작은 항목에 이 지표를 쓰면 작은 오차에도 큰 음수가 나온다. 그런 항목은 R² 계열 대신 오차의 절대 크기로 관리하는 편이 낫다.
- **상한은 여전히 1 이다.** 분모를 바꿔도 `SS_res` 는 0 아래로 갈 수 없으므로 값이 1 을 넘지 않는다. 1 에 가까운 값은 모델이 완벽하다는 뜻이 아니라 지정한 baseline 을 압도했다는 뜻이다.
- **분모를 사후에 고르지 않는다.** 결과를 본 뒤 보기 좋은 `sigma_ref` 를 고르면 지표가 아니라 수사가 된다. 기준은 평가 전에 정하고 문서에 남긴다.

## 7. Summary

R² 의 분모는 baseline 의 오차이고, 표준 R² 는 그 baseline 을 데이터의 평균으로 고정한
특수한 경우다. 분모를 학습 평균, 규격 산포, 기존 운영 모델의 오차로 바꾸는 것은 모두
baseline 을 바꾸는 같은 조작이며, 그 결과 지표는 "이 데이터셋 안에서의 설명력" 에서
"지정한 기준 대비 상대 성능" 으로 의미가 옮겨간다. 이렇게 baseline 을 명시적으로 고른
형태를 이 문서는 `Ref_R2` 로 부른다. 이 교체의 실익은 데이터셋마다 달라지던 잣대가
고정되어 lot 과 기간을 가로질러 비교할 수 있게 되는 데 있고, 대가는 표준 R² 로서의
해석을 잃는 것과 기준 선정의 책임을 지는 것이다.

---

## Appendix A. Terminology

- **baseline model** — 비교의 기준이 되는 모델이며, R² 의 분모는 이 모델의 오차제곱합이다.
- **out-of-sample** — 학습에 쓰이지 않은 데이터에서의 평가를 뜻한다.
- **persistence baseline** — 직전 시점의 값을 그대로 다음 시점의 예측으로 쓰는 시계열 baseline 이다.
- **Referenced R² (`Ref_R2`)** — 분모의 baseline 을 밖에서 지정해 계산한 R² 이며, 이 문서에서 정한 이름이다.
- **skill score** — `1 − 모델오차 / baseline오차` 형태의 지표를 통칭하며, R² 는 baseline 을 평균으로 둔 사례다.
- **spec tolerance** — 공정 규격이 허용하는 산포이며, 고정 분모로 자주 쓰인다.
- **SS_res** — 잔차제곱합 `Σ(y_i − y_hat_i)²` 이다.
- **SS_tot** — 총제곱합 `Σ(y_i − y_bar)²` 이며, 표준 R² 의 분모다.
