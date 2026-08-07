# Data Manifest
rev. 2

> Data manifest 는 object storage 에 적재된 데이터가 무엇인지 기록하는 JSON file 의 모음이다.
> 사람이 정한 값과 분석이 판정한 값을 서로 다른 file 에 두어, 판정을 다시 돌릴 때 사람의 결정이 지워지지 않게 한다.

데이터를 받아서 모델에 넣기까지 반복해서 답해야 하는 질문은 세 가지이다. 이 데이터가 어디서 왔고 무엇을 위한 것인가, 열 이름과 형을 어떻게 맞출 것인가, 그리고 각 열이 어떤 성격의 값인가이다. Manifest 는 이 세 질문에 각각 하나의 file 을 대응시키고, 네 번째 file 에 열의 성격을 부르는 이름과 그 판정 규칙을 모아 둔다.

## 1. Scope

### 1.1 Tabular Data

이 문서가 다루는 데이터는 행과 열로 이루어진 table 이다. Cell 하나가 단일 값이 아니라 배열일 수 있으나, 데이터 전체가 행과 열의 틀에 들어간다는 점은 유지된다.

행과 열의 틀에 들어가지 않는 데이터는 이 문서의 대상이 아니다. Image 를 모아 둔 directory 나 layout file 은 catalog 에 항목으로 올릴 수는 있으나, 열이라는 것이 없으므로 column config 와 column profile 의 대상이 되지 않는다.

### 1.2 Source Separation

Manifest 에 들어가는 값은 두 곳에서 온다. 하나는 사람이 이미 알고 있어서 적는 값이고, 다른 하나는 데이터를 읽어서 판정하는 값이다.

두 값을 한 file 에 두면 판정을 다시 돌릴 때 사람이 적은 값이 함께 지워진다. 갱신되는 주기도 다르다. 사람이 적는 값은 상류의 열 이름이 바뀔 때 갱신되고, 판정하는 값은 데이터가 갱신될 때와 판정 기준을 바꿀 때 갱신된다. 그래서 file 을 출처로 나누고, 같은 항목이 양쪽에 있으면 사람이 적은 값을 채택한다. 이 규칙은 7절에서 정한다.

## 2. File Layout

Table 1. Manifest files

| File | Source | Content |
|------|--------|---------|
| `catalog.json` | Human | Dataset 이 어디서 왔고 무엇을 위한 것인지 기록한다 |
| `column-config.json` | Human | 열 이름, 삭제, 형 변환, 결측 표시와 사람이 아는 class 를 기록한다 |
| `column-profile.json` | Analysis | 데이터를 읽어 판정한 class 를 기록한다 |
| `column-class.json` | Definition | Class 의 axis 와 label, 그리고 각 label 의 판정 규칙을 정의한다 |

앞의 세 file 은 모두 object key 를 최상위 key 로 쓴다. 세 file 을 잇는 별도의 참조 항목은 두지 않으며, 같은 key 를 쓰는 것이 곧 연결이다.

```text
object key
    |
    +--> catalog.json          what the dataset is
    +--> column-config.json    what a human declares
    +--> column-profile.json   what the analysis decides

column-class.json              vocabulary shared by the two above
```

`column-profile.json` 은 판정의 산출물이므로 손으로 고치지 않는다. 고쳐도 다음 판정에서 지워진다. 판정 결과를 바꾸고 싶으면 `column-config.json` 에 사람의 값을 적어 덮어쓴다.

## 3. Catalog

Catalog 는 object key 하나마다 그 데이터가 무엇인지를 기록한다. Key 는 항상 같은 dataset 을 가리키고 내용만 갱신되므로, key 자체가 식별자로 성립한다. 같은 key 에 다른 dataset 이 올라오는 일이 없다는 것이 이 전제의 조건이다.

Table 2. Catalog description keys

| Key | Required | Description |
|-----|----------|-------------|
| `project_goal` | Yes | 이 데이터를 어떤 목적으로 적재했는지 한 문장으로 적는다 |
| `provider` | Yes | 데이터를 낸 주체나 system 을 적는다 |
| `date` | Yes | 마지막으로 갱신된 시각을 적는다 |
| `medallion` | Yes | `bronze`, `silver`, `gold` 중 하나를 적는다 |
| `grain` | Yes | 행 하나가 무엇인지 한 문장으로 적는다 |
| `derived_from` | No | 이 데이터를 만들어 낸 상류 object key 를 적고, 원본이면 생략한다 |
| `rows` | No | 행 수를 적는다 |
| `note` | No | 이 데이터를 쓸 때 알아야 할 예외를 적는다 |

`grain` 이 필수인 이유는 행 하나가 무엇인지를 모르면 join 과 집계가 조용히 틀리기 때문이다. `derived_from` 이 있어야 `medallion` 이 이름표에 그치지 않고 상류를 거슬러 올라갈 수 있는 관계가 된다.

```json
// catalog.json
{
  "silver/fdc/etch-ch1.parquet": {
    "project_goal": "etch CD 예측을 위한 chamber trace 확보",
    "provider": "FDC / Etch Bay 3",
    "date": "2026-08-07",
    "medallion": "silver",
    "grain": "wafer 1장과 process step 1개의 조합",
    "derived_from": "bronze/fdc/etch-raw.parquet",
    "rows": 1043221,
    "note": "2026-06 이전 적재분은 pressure 단위가 Pa 이다"
  }
}
```

## 4. Column Config

Column config 는 사람이 적는 file 이다. 상류에서 온 table 을 쓰기 좋은 형태로 바꾸는 조작과, 사람이 이미 알고 있는 class 를 담는다.

Table 3. Column config operation keys

| Key | Description |
|-----|-------------|
| `column_mapping` | 상류의 열 이름을 쓸 이름으로 바꾼다 |
| `columns_to_drop` | 버릴 열을 적는다 |
| `type_conversion` | 열의 형을 바꾼다 |
| `na_values` | 결측을 나타내는 값을 적는다 |
| `class` | 사람이 아는 class label 을 열마다 적는다 |

`na_values` 가 필요한 이유는 계측 데이터가 결측을 `-999` 같은 실수로 표시하는 일이 많고, 이것을 그대로 두면 실제 측정값으로 학습되기 때문이다. 이 값은 데이터를 읽어서는 알 수 없으므로 사람이 적어야 한다.

`class` 에는 사람이 아는 열만 적고 나머지는 생략한다. 생략한 열은 판정 결과가 그대로 쓰인다. 사람이 확실하게 아는 것은 대개 열이 category 라는 사실이다. 어떤 값이 실제로 들어오는지는 데이터를 읽어야 알 수 있으므로 판정이 채운다.

```json
// column-config.json
{
  "silver/fdc/etch-ch1.parquet": {
    "column_mapping": {
      "RF_FORWARD_POWER": "rf_fwd_pwr_w"
    },
    "columns_to_drop": ["reserved_1"],
    "type_conversion": {
      "rf_fwd_pwr_w": "float32"
    },
    "na_values": {
      "chamber_pressure": [-999]
    },
    "class": {
      "bin_code": ["category"],
      "site_thickness": ["vector"]
    }
  }
}
```

조작의 적용 순서는 `column_mapping`, `na_values`, `columns_to_drop`, `type_conversion` 이다. 순서를 JSON 에 담지 않고 고정하는 이유는 JSON object 의 key 순서가 규격상 보장되지 않기 때문이다. 순서가 의미를 가지는 값은 object 가 아니라 array 에 담거나, 이 경우처럼 file 밖에 고정한다.

## 5. Column Profile

Column profile 은 데이터를 읽어 판정한 결과를 담는다. 열마다 class label 과, 그 판정을 뒷받침하는 관측값을 함께 둔다.

Table 4. Column profile keys

| Key | Description |
|-----|-------------|
| `profiled_at` | 판정을 수행한 시각을 적는다 |
| `thresholds` | 판정에 쓴 임계값을 적는다 |
| `columns` | 열마다 판정 결과를 적는다 |

`thresholds` 를 함께 두는 이유는 6절의 판정 규칙 여러 개가 임계값을 필요로 하고, 그 값이 다르면 같은 데이터에서 다른 label 이 나오기 때문이다. 임계값 없이 label 만 남은 profile 은 재현되지 않는다.

Category 로 판정된 열은 관측된 값의 목록을 함께 남긴다. 사람은 그 열이 category 라는 사실만 알고 어떤 값이 들어오는지는 모르므로, 이 목록이 사람과 판정의 역할이 갈리는 지점이다.

```json
// column-profile.json
{
  "silver/fdc/etch-ch1.parquet": {
    "profiled_at": "2026-08-07T09:12:00+09:00",
    "thresholds": {
      "dwell_ratio": 5.0,
      "ramp_fraction": 0.8,
      "acf_peak": 0.6
    },
    "columns": {
      "rf_fwd_pwr_w": {
        "class": ["active", "trace", "qn", "rectangle"],
        "missing_rate": 0.003
      },
      "chamber_pressure": {
        "class": ["active", "trace", "ramp"],
        "missing_rate": 0.0
      },
      "bin_code": {
        "class": ["active", "scalar"],
        "levels": [1, 2, 3, 4, 5, 6, 7, 8],
        "missing_rate": 0.0
      },
      "reserved_1": {
        "class": ["inactive", "scalar"],
        "missing_rate": 0.0
      }
    }
  }
}
```

## 6. Column Class

### 6.1 Axis

Class 는 하나의 목록이 아니라 여러 개의 axis 로 나뉜다. 한 axis 안의 label 은 서로 배타적이고, 한 열은 axis 마다 최대 하나의 label 을 가진다. 이렇게 나누면 서로 무관한 성질을 하나의 label 에 섞지 않아도 되고, 모순된 조합을 규칙으로 걸러낼 수 있다.

Table 5. Class axes

| Axis | Applies to | Source |
|------|------------|--------|
| `activity` | 모든 열 | Analysis |
| `form` | 모든 열 | Human or analysis |
| `trace_quantum` | `form` 이 `trace` 인 열 | Analysis |
| `trace_shape` | `form` 이 `trace` 인 열 | Analysis |

`trace_` 로 시작하는 axis 는 `form` 이 `trace` 일 때만 값을 가진다. 이름에 의존 관계를 넣어 두었으므로 axis 이름만 보고 이 제약을 알 수 있다.

### 6.2 Activity

Activity 는 열에 변화가 있는지를 나눈다. 열을 쓸 것인지 말 것인지의 결정이 아니라 데이터를 읽어 판정하는 사실이다.

판정은 행 사이의 비교로 한다. Cell 을 통째로 하나의 값으로 보므로, cell 이 배열인 열에서는 배열 전체가 같아야 두 행이 같은 값을 가진 것이 된다. Cell 안에서 값이 변하는지는 activity 가 아니라 `trace_quantum` 이 다룬다.

Table 6. Activity labels

| Label | Rule |
|-------|------|
| `active` | 결측을 제외한 행 중 서로 다른 cell 값이 둘 이상 있다 |
| `inactive` | 결측을 제외한 행의 cell 값이 모두 같거나, 모든 행이 결측이다 |

### 6.3 Form

Form 은 cell 하나가 어떤 형태의 값을 담는지를 나눈다.

Table 7. Form labels

| Label | Rule |
|-------|------|
| `category` | 값이 유한한 이름의 집합에서 나오고, 값 사이의 산술 연산이 의미를 갖지 않는다 |
| `scalar` | Cell 하나가 크기를 갖는 단일 수치이다 |
| `vector` | Cell 하나가 길이가 고정된 배열이고, 원소의 순서가 의미를 갖지 않는다 |
| `trace` | Cell 하나가 배열이고, 원소가 시간 순서로 정렬되어 있다 |

`vector` 와 `trace` 를 가르는 것은 배열이라는 사실이 아니라 시간축의 유무이다. Wafer 위 여러 지점에서 잰 두께는 원소를 섞어도 뜻이 같으므로 `vector` 이고, 공정 중에 기록한 압력은 순서가 곧 정보이므로 `trace` 이다.

`category` 만 사람이 정한다. 판정은 고유값이 몇 개인지까지만 알 수 있고, 그 값들이 이름인지 크기인지는 데이터에 들어 있지 않다.

### 6.4 Trace Quantum

Trace 가 연속으로 변하지 않고 몇 개의 level 위에 머무를 때, level 이 몇 개인지를 나눈다. Level 은 baseline 을 포함해서 센다.

Table 8. Trace quantum labels

| Label | Rule |
|-------|------|
| `q1` | Cell 하나 안에서 값이 level 하나 위에만 머문다 |
| `qn` | Cell 하나 안에서 값이 둘 이상의 level 위를 오간다 |

`q1` 은 시간이 지나도 값이 변하지 않는 trace 이므로, 그 trace 가 담은 정보는 수치 하나와 같다. 그래도 열 전체가 뜻을 잃는 것은 아니다. 행마다 그 하나의 값이 다르면 열은 `active` 이고, 모든 행이 같은 값이면 `inactive` 이다. 6.2 절이 activity 를 행 사이의 비교로 정한 것은 이 구분을 위해서이다.

`q1` 인 열은 cell 을 그 하나의 값으로 바꾸어 `scalar` 로 축약할 수 있다. 축약하면 행 사이의 차이는 그대로 남고 시간축만 사라지므로 activity 는 바뀌지 않는다.

### 6.5 Trace Shape

Trace 의 모양을 나눈다. 각 규칙에 나오는 임계값은 판정 configuration 이며, 5절의 `thresholds` 에 함께 기록한다.

Table 9. Trace shape labels

| Label | Rule |
|-------|------|
| `rectangle` | 값이 두 level 사이를 오가고, 한 level 에 머무는 시간이 level 사이를 이동하는 시간보다 정해진 배수 이상 길다 |
| `triangle` | 상승 구간과 하강 구간의 기울기 크기가 서로 비슷하고, 두 구간 사이에 평탄한 구간이 없다 |
| `ramp` | Window 에서 값이 한 방향으로만 변하는 구간이 정해진 비율 이상을 차지한다 |
| `oscillation` | Autocorrelation 에 정해진 크기 이상의 peak 이 일정한 간격으로 나타난다 |

### 6.6 File Format

`column-class.json` 은 axis 를 key 로 두고, 그 아래에 label 과 판정 규칙의 쌍을 둔다. 규칙을 label 옆에 두는 이유는 규칙 없는 label 이 사람마다 다른 뜻으로 쓰이기 때문이다.

```json
// column-class.json
{
  "activity": {
    "active": "결측을 제외한 행 중 서로 다른 cell 값이 둘 이상 있다",
    "inactive": "결측을 제외한 행의 cell 값이 모두 같거나, 모든 행이 결측이다"
  },
  "form": {
    "category": "값이 유한한 이름의 집합에서 나오고, 값 사이의 산술 연산이 의미를 갖지 않는다",
    "scalar": "cell 하나가 크기를 갖는 단일 수치이다",
    "vector": "cell 하나가 길이가 고정된 배열이고, 원소의 순서가 의미를 갖지 않는다",
    "trace": "cell 하나가 배열이고, 원소가 시간 순서로 정렬되어 있다"
  },
  "trace_quantum": {
    "q1": "cell 하나 안에서 값이 level 하나 위에만 머문다",
    "qn": "cell 하나 안에서 값이 둘 이상의 level 위를 오간다"
  },
  "trace_shape": {
    "rectangle": "값이 두 level 사이를 오가고, 한 level 에 머무는 시간이 level 사이를 이동하는 시간보다 dwell_ratio 배 이상 길다",
    "triangle": "상승 구간과 하강 구간의 기울기 크기가 서로 비슷하고, 두 구간 사이에 평탄한 구간이 없다",
    "ramp": "window 에서 값이 한 방향으로만 변하는 구간이 ramp_fraction 이상을 차지한다",
    "oscillation": "autocorrelation 에 acf_peak 이상의 peak 이 일정한 간격으로 나타난다"
  }
}
```

새 label 이 필요하면 이 file 에 규칙과 함께 한 줄을 더한다. 여기에 없는 label 은 8절의 규칙에 따라 거부된다.

## 7. Merge Rule

한 열의 최종 class 는 profile 과 config 를 합쳐서 얻는다.

1. Profile 의 label 을 기본으로 삼는다.
2. Config 에 같은 axis 의 label 이 있으면 그 label 로 바꾼다.
3. 바뀐 항목은 conflict 로 기록한다.

3 이 있는 이유는 사람과 판정이 같은 axis 에서 다른 답을 냈다는 것이 그 자체로 점검할 거리이기 때문이다. 사람이 `scalar` 라고 적었는데 판정이 `trace` 라고 했다면 둘 중 하나가 틀렸고, 어느 쪽이든 데이터나 규칙에 손볼 데가 있다. 출처를 나눈 것만으로 별도의 검사 없이 이 목록이 나온다.

사람이 적지 않은 axis 는 판정 결과가 그대로 남는다. 4절의 예에서 `bin_code` 는 사람이 `category` 를 적었고 판정은 `active` 와 `scalar` 를 냈으므로, 합치면 `active` 와 `category` 가 된다.

## 8. Integrity Rules

Manifest 가 지켜야 하는 조건은 네 가지이다.

Table 10. Integrity rules

| Rule | Condition | Catches |
|------|-----------|---------|
| 1 | 모든 label 은 `column-class.json` 에 있어야 한다 | 오타와 미등록 label |
| 2 | 한 열의 label 은 axis 마다 최대 하나이다 | `scalar` 와 `trace` 를 함께 붙이는 모순 |
| 3 | `form` 이 `trace` 가 아니면 `trace_` 로 시작하는 axis 의 label 을 가질 수 없다 | 의존 관계 위반 |
| 4 | `column-config.json` 과 `column-profile.json` 의 최상위 key 는 `catalog.json` 에 있어야 한다 | Catalog 에 없는 데이터의 설정이 남아 있는 상태 |

이 네 가지는 label 과 key 만 비교하면 확인되므로 데이터를 읽지 않고 검사할 수 있다. 데이터를 읽어야 아는 것, 즉 선언한 형과 실제 값이 맞는지는 판정이 담당하고 7절의 conflict 로 드러난다.

## 9. Non-Tabular Data

1.1 절에서 밝힌 대로 행과 열의 틀에 들어가지 않는 데이터는 catalog 에만 올린다. 이때 `grain` 은 행이 아니라 항목 하나가 무엇인지를 적고, `column-config.json` 과 `column-profile.json` 에는 그 key 를 두지 않는다. 8절의 4 는 catalog 를 기준으로 한 포함 관계이므로 이 경우에도 위반이 되지 않는다.

## Appendix A. Terminology

본문에서 정의하지 않고 쓴 용어를 정리한다.

- **Autocorrelation** 은 신호를 시간축으로 밀어 가며 자기 자신과 곱해 평균한 값이고, 주기 성분이 있으면 그 주기마다 peak 이 나타난다.
- **Baseline** 은 trace 가 아무 동작도 하지 않을 때 머무는 기준 level 이다.
- **Cell** 은 table 에서 행 하나와 열 하나가 만나는 자리이다.
- **Grain** 은 table 의 행 하나가 무엇을 나타내는지를 말한다.
- **Level** 은 값이 연속으로 변하지 않고 몇 개의 값 위에만 머무를 때 그 값 하나를 말한다.
- **Medallion architecture** 는 데이터를 원본에 가까운 bronze, 정제된 silver, 사용 목적에 맞춘 gold 의 세 단계로 나누어 적재하는 방식이다.
- **Object key** 는 object storage 에서 저장된 항목 하나를 가리키는 문자열이다.
- **Object storage** 는 파일을 directory 구조가 아니라 key 로 지시해 저장하는 storage 이다.
- **Window** 는 trace 에서 판정 대상으로 잘라낸 시간 구간이다.
