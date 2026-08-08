# Tabular Data Manifest
rev. 13

> Tabular data manifest 는 object storage 에 적재된 table 이 무엇인지 기록하는 JSON file 의 모음이다.
> 사람이 적는 값과 분석이 판정하는 값을 서로 다른 file 에 두어, 판정을 다시 돌릴 때 사람이 적은 값이 지워지지 않게 한다.

데이터를 받아서 모델에 넣기까지 반복해서 답해야 하는 질문은 세 가지이다. 이 데이터가 어디서 왔고 무엇을 위한 것인가 (provenance), 열 이름과 형을 어떻게 맞출 것인가 (configuration), 그리고 각 열이 어떤 성격의 값인가 (class) 이다. Manifest 는 이 세 질문에 각각 하나의 file 을 대응시키고, 네 번째 file 에 class 를 부르는 이름과 그 판정 규칙을 모아 둔다.

## 1. Scope And Files

### 1.1 Tabular Data

이 문서가 다루는 데이터는 tabular data 이다. Cell 하나가 단일 값이 아니라 배열일 수 있으나, 데이터 전체가 행과 열의 틀에 들어간다는 점은 유지된다. 행과 열의 틀에 들어가지 않는 image directory 나 layout file 은 이 문서의 대상이 아니다.

### 1.2 File Layout

Table 1. Manifest files

| File | Source | Content |
|------|--------|---------|
| `catalog.json` | Human | Dataset 이 어디서 왔고 무엇을 위한 것인지 기록한다 |
| `column-config.json` | Human | 열 이름, 삭제, 형 변환, 결측 표시와 사람이 아는 class 를 기록한다 |
| `column-profile.json` | Analysis | 데이터를 읽어 판정한 class 를 기록한다 |
| `column-class.json` | Definition | Class 의 axis 와 label, 그리고 각 label 의 판정 규칙을 정의한다 |

각 file 이 무엇을 최상위 key 로 쓰는지는 서로 다르다. Object key 는 catalog 에만 나오고, 나머지 file 은 object key 를 쓰지 않는다.

```text
catalog.json           object key  -> what the dataset is
column-config.json     column name -> what a human declares
column-profile.json    column name -> what the analysis decides
column-class.json      axis        -> label and its rule
```

열의 class 는 column profile 에서 정해진다. Column config 의 `class` 는 사람이 아는 것을 판정에 넘겨 주는 입력이고, 판정은 그것을 받아 최종 class 를 column profile 에 쓴다. 그래서 class 를 읽을 곳은 언제나 column profile 하나이며, 두 file 을 합치는 절차는 필요하지 않다.

Column profile 은 판정의 산출물이므로 손으로 고치지 않는다. 고쳐도 다음 판정에서 지워진다. 판정 결과를 바꾸고 싶으면 column config 에 사람의 값을 적어 판정에 넘긴다.

## 2. Catalog

Catalog 는 object key 하나마다 그 데이터가 무엇인지를 기록한다. Key 는 항상 같은 dataset 을 가리키고 내용만 갱신되므로, key 자체가 식별자로 성립한다. 같은 key 에 다른 dataset 이 올라오는 일이 없다는 것이 이 전제의 조건이다.

Table 2. Catalog description keys

| Key | Required | Description |
|-----|----------|-------------|
| `project_goal` | Yes | 이 데이터를 어떤 목적으로 적재했는지 한 문장으로 적는다 |
| `provider` | Yes | 데이터를 낸 주체나 system 을 적는다 |
| `date` | Yes | 마지막으로 갱신된 시각을 적는다 |
| `medallion` | Yes | `bronze`, `silver`, `gold` 중 하나를 적는다 |
| `grain` | Yes | 행 하나가 무엇인지 한 문장으로 적는다 |
| `key` | Yes | Entity 를 묶는 열과 그 안에서 행을 가르는 열을 적는다 |
| `derived_from` | No | 이 데이터를 만들어 낸 상류 object key 를 적고, 원본이면 생략한다 |
| `rows` | No | 행 수를 적는다 |
| `note` | No | 이 데이터를 쓸 때 알아야 할 예외를 적는다 |

`grain` 이 필수인 이유는 행 하나가 무엇인지를 모르면 join 과 집계가 조용히 틀리기 때문이다. `derived_from` 이 있어야 `medallion` 이 이름표에 그치지 않고 상류를 거슬러 올라갈 수 있는 관계가 된다.

`key` 는 행이 서로 어떻게 구별되는지를 두 목록으로 적는다.

- `entity` 는 어느 열로 묶으면 한 entity 가 되는지를 적는다.
- `sequence` 는 한 entity 안에서 행을 가르는 열을 순서대로 적고, 행 하나가 곧 entity 하나이면 비운다.

두 목록을 이은 것이 행마다 유일한 조합이다. **행이 중복인지 아닌지를 명시하는 것이 이 조합이며**, 중복 검사는 `entity` 가 아니라 이 조합으로 한다. 한 wafer 의 trace 를 행마다 한 점씩 펼친 table 에서 `wafer_id` 는 수천 행에 걸쳐 같은 값이지만, `sequence` 의 시간 열이 그 행들을 갈라 놓으므로 중복이 아니다. `sequence` 를 적지 않으면 그 table 을 중복 제거해도 되는지가 기록되지 않는다.

두 목록의 일은 서로 다르다. `entity` 는 묶는 방법을 말하고 `sequence` 는 가르는 방법을 말한다. 행 하나가 wafer 와 process step 의 조합이면 `entity` 는 wafer 까지이고 `sequence` 에 step 이 들어가므로, 이때도 wafer 는 여러 행에 걸쳐 되풀이된다.

같은 데이터를 cell 안의 배열로 담을 수도 있고 행으로 펼칠 수도 있는데, 그 차이는 `key` 가 아니라 열의 class 가 말해 준다. 배열로 담으면 그 열이 `trace` 이고, 행으로 펼치면 `scalar` 가 되면서 시간 열이 `sequence` 에 들어간다.

```json
// catalog.json
{
  "Ulvac/AlPVDPoC/SilverData/#0/V0": {
    "project_goal": "collect chamber trace for etch CD prediction",
    "provider": "FDC / Etch Bay 3",
    "date": "2026-08-07",
    "medallion": "silver",
    "grain": "one wafer and one process step",
    "key": {
      "entity": ["lot_id", "wafer_id"],
      "sequence": ["step_id"]
    },
    "derived_from": "Ulvac/AlPVDPoC/BronzeData/#0/V0",
    "rows": 1043221,
    "note": "pressure is in Pa for the loads before 2026-06"
  }
}
```

## 3. Column Config

Column config 는 사람이 적는 file 이다. 상류에서 온 table 을 쓰기 좋은 형태로 바꾸는 조작과, 사람이 이미 알고 있는 class 를 담는다.

Table 3. Column config operation keys

| Order | Key | Description |
|-------|-----|-------------|
| 1 | `column_mapping` | 상류의 열 이름을 쓸 이름으로 바꾼다 |
| 2 | `na_values` | 결측을 나타내는 값을 결측으로 바꾼다 |
| 3 | `columns_to_drop` | 버릴 열을 적는다 |
| 4 | `type_conversion` | 열의 형을 바꾼다 |
| - | `class` | 사람이 아는 class label 을 열마다 적는다 |

`Order` 는 조작을 적용하는 순서이고, `class` 는 조작이 아니므로 순서를 갖지 않는다. 이름 바꾸기가 첫째인 이유는 뒤의 세 조작이 모두 바뀐 이름으로 열을 지시하게 하기 위해서이다. 형 바꾸기가 마지막인 이유는 결측 표시를 걷어낸 뒤라야 열이 목표한 형으로 들어가기 때문이다. `-999` 가 남아 있는 열을 먼저 정수로 바꾸면 결측이 값으로 굳는다.

순서를 JSON 에 담지 않고 표에 고정하는 이유는 JSON object 의 key 순서가 규격상 보장되지 않기 때문이다. 순서가 의미를 가지는 값은 object 가 아니라 array 에 담거나, 이 경우처럼 file 밖에 고정한다.

`na_values` 가 필요한 이유는 계측 데이터가 결측을 `-999` 같은 실수로 표시하는 일이 많고, 이것을 그대로 두면 실제 측정값으로 학습되기 때문이다. 이 값은 데이터를 읽어서는 알 수 없으므로 사람이 적어야 한다.

`class` 에는 사람이 아는 열만 적고 나머지는 생략한다. 사람이 확실하게 아는 것은 대개 열이 `category` 라는 사실이다. 어떤 값이 실제로 들어오는지는 데이터를 읽어야 알 수 있으므로 판정이 채운다.

```json
// column-config.json
{
  "column_mapping": {
    "RF_FORWARD_POWER": "rf_fwd_pwr_w"
  },
  "na_values": {
    "chamber_pressure": [-999]
  },
  "columns_to_drop": ["reserved_1"],
  "type_conversion": {
    "rf_fwd_pwr_w": "float32"
  },
  "class": {
    "bin_code": ["category"],
    "site_thickness": ["vector"]
  }
}
```

## 4. Column Profile

Column profile 은 데이터를 읽어 판정한 결과를 담는다. 열마다 최종 class label 과, 그 판정을 뒷받침하는 관측값을 함께 둔다.

Table 4. Column profile keys

| Key | Description |
|-----|-------------|
| `profiled_at` | 판정을 수행한 시각을 적는다 |
| `thresholds` | 판정에 쓴 임계값을 적는다 |
| `columns` | 열마다 최종 class 와 관측값을 적는다 |

`thresholds` 를 함께 두는 이유는 5절의 판정 규칙 여러 개가 임계값을 필요로 하고, 그 값이 다르면 같은 데이터에서 다른 label 이 나오기 때문이다. 임계값 없이 label 만 남은 profile 은 재현되지 않는다.

`category` 로 판정된 열은 관측된 값의 목록을 함께 남긴다. 사람은 그 열이 `category` 라는 사실만 알고 어떤 값이 들어오는지는 모르므로, 이 목록이 사람과 판정의 역할이 갈리는 지점이다.

판정 결과가 column config 에 사람이 적은 label 과 다르게 나왔다면 둘 중 하나가 틀린 것이므로, 두 file 을 비교하면 손볼 열이 드러난다.

```json
// column-profile.json
{
  "profiled_at": "2026-08-07T09:12:00+09:00",
  "thresholds": {
    "dwell_ratio": 5.0,
    "ramp_fraction": 0.8,
    "acf_peak": 0.6
  },
  "columns": {
    "rf_fwd_pwr_w": {
      "class": ["active", "numeric", "trace", "fixed", "qn", "rectangle"],
      "missing_rate": 0.003
    },
    "chamber_pressure": {
      "class": ["active", "numeric", "trace", "fixed", "infinite", "ramp"],
      "missing_rate": 0.0
    },
    "bin_code": {
      "class": ["active", "category", "scalar"],
      "levels": [1, 2, 3, 4, 5, 6, 7, 8],
      "missing_rate": 0.0
    },
    "reserved_1": {
      "class": ["inactive", "numeric", "scalar"],
      "missing_rate": 0.0
    }
  }
}
```

## 5. Column Class

### 5.1 Axis

Class 는 하나의 목록이 아니라 여러 개의 axis 로 나뉜다. 한 axis 안의 label 은 서로 배타적이므로, 이렇게 나누면 서로 무관한 성질을 하나의 label 에 섞지 않아도 되고 모순된 조합을 규칙으로 걸러낼 수 있다.

Table 5. Class axes

| Axis | Label | Applies to | Source |
|------|-------|------------|--------|
| `activity` | `active`, `inactive` | 모든 열 | Analysis |
| `value_type` | `category`, `ordinal`, `numeric`, `text`, `datetime` | 모든 열 | Human or analysis |
| `structure` | `scalar`, `vector`, `matrix`, `trace` | 모든 열 | Human or analysis |
| `array_length` | `fixed`, `variable` | `structure` 가 `vector`, `matrix`, `trace` 인 열 | Analysis |
| `trace_quantum` | `q1`, `qn`, `infinite` | `structure` 가 `trace` 인 열 | Analysis |
| `trace_shape` | `flat`, `rectangle`, `triangle`, `ramp`, `oscillation`, `irregular` | `structure` 가 `trace` 인 열 | Analysis |

**열은 자기에게 적용되는 axis 마다 label 을 정확히 하나씩 갖는다.** 적용되지 않는 axis 에는 label 을 갖지 않는다. 그래서 label 이 비어 있는 axis 는 판정이 아직 끝나지 않았다는 뜻이며, 판정이 끝난 열은 어느 axis 를 물어도 답이 하나 나온다.

이 규약 때문에 각 axis 의 label 은 그 axis 가 적용되는 모든 열을 남김없이 받아야 한다. `trace_shape` 의 `irregular` 가 그 자리를 맡는다.

`array_` 와 `trace_` 로 시작하는 axis 는 `structure` 가 각각 배열일 때와 `trace` 일 때만 적용된다. 이름에 의존 관계를 넣어 두었으므로 axis 이름만 보고 이 제약을 알 수 있다.

한 label 은 한 axis 에만 속한다. 열의 class 를 label 의 목록으로 적으므로, 같은 label 이 두 axis 에 있으면 그 label 이 어느 axis 의 값인지 가릴 수 없다.

### 5.2 Activity

Activity 는 열에 변화가 있는지를 나눈다. 열을 쓸 것인지 말 것인지의 결정이 아니라 데이터를 읽어 판정하는 사실이다.

판정은 행 사이의 비교로 한다. Cell 을 통째로 하나의 값으로 보므로, cell 이 배열인 열에서는 배열 전체가 같아야 두 행이 같은 값을 가진 것이 된다. Cell 안에서 값이 변하는지는 activity 가 아니라 `trace_quantum` 이 다룬다.

Table 6. Activity labels

| Label | Rule |
|-------|------|
| `active` | 결측을 제외한 행 중 서로 다른 cell 값이 둘 이상 있다 |
| `inactive` | 결측을 제외한 행의 cell 값이 모두 같거나, 모든 행이 결측이다 |

한 entity 에 여러 행이 놓인 table 에서는 그 행들이 `sequence` 를 따라 서로 다르므로, 이 규칙을 그대로 쓰면 거의 모든 열이 `active` 로 나온다. 그럴 때는 `key` 의 `entity` 로 행을 묶은 뒤 entity 사이를 비교한다.

### 5.3 Value Type

Value type 은 값 하나가 이름인지 수치인지를 나눈다. Cell 이 배열인 열에서는 그 원소 하나를 두고 판정한다.

Table 7. Value type labels

| Label | Rule |
|-------|------|
| `category` | 값이 유한한 이름의 집합에서 나오고, 값 사이에 순서가 없다 |
| `ordinal` | 값이 유한한 이름의 집합에서 나오고, 값 사이에 순서가 있으나 산술 연산은 의미를 갖지 않는다 |
| `numeric` | 값이 크기를 갖는 수치이고, 값 사이의 산술 연산이 의미를 갖는다 |
| `text` | 값이 문자열이고, 값의 집합이 미리 정해져 있지 않다 |
| `datetime` | 값이 시각을 가리키고, 값 사이의 차는 의미를 갖지만 합은 의미를 갖지 않는다 |

`ordinal` 은 `category` 와 `numeric` 사이에 놓인다. 비교는 되고 산술은 되지 않으므로, 등급이나 심각도처럼 값을 줄 세울 수 있는 열이 여기에 속한다. `ordinal` 을 `category` 로 적으면 순서를 잃고, `numeric` 으로 적으면 등급 사이의 간격이 모두 같다고 주장하게 된다.

`category` 와 `text` 는 둘 다 문자열을 담을 수 있고, 값의 집합이 미리 정해져 있는지로 갈린다. 고유값이 수만 개여도 그 집합이 정해져 있으면 `category` 이므로 식별자는 여기에 속하고, FA report 본문처럼 매번 새로 쓰이는 값은 `text` 이다.

`datetime` 과 `numeric` 은 합이 의미를 갖는지로 갈린다. 두 시각을 더한 값은 뜻이 없지만 두 소요 시간을 더한 값은 뜻이 있으므로, queue time 처럼 길이를 재는 열은 `datetime` 이 아니라 `numeric` 이다.

`category` 는 사람이 정한다. 판정은 고유값이 몇 개인지까지만 알 수 있고, 그 값들이 이름인지 크기인지는 데이터에 들어 있지 않다. `bin_code` 의 값 `1` 과 `2` 는 두 종류의 불량을 가리키는 이름이지 크기가 아니지만, 데이터만 보아서는 그것을 알 수 없다.

### 5.4 Structure

Structure 는 cell 하나가 값 하나를 담는지 배열을 담는지를 나눈다.

Table 8. Structure labels

| Label | Rule |
|-------|------|
| `scalar` | Cell 하나가 값 하나를 담는다 |
| `vector` | Cell 하나가 배열을 담고, 원소의 순서가 의미를 갖지 않는다 |
| `matrix` | Cell 하나가 두 축을 갖는 배열을 담고, 두 축 모두 시간축이 아니다 |
| `trace` | Cell 하나가 배열을 담고, 원소가 시간 순서로 정렬되어 있다 |

`vector` 와 `trace` 를 가르는 것은 배열이라는 사실이 아니라 시간축의 유무이다. Wafer 위 여러 지점에서 잰 두께는 원소를 섞어도 뜻이 같으므로 `vector` 이고, 공정 중에 기록한 압력은 순서가 곧 정보이므로 `trace` 이다.

`matrix` 는 원소의 자리가 한 축이 아니라 두 축으로 정해지는 경우이다. Wafer 위 die 마다의 bin code 를 격자로 담은 열이 여기에 해당하며, 같은 값을 자리 정보 없이 늘어놓으면 `vector` 가 되어 이웃 관계를 잃는다.

값의 성격과 cell 의 모양을 두 축으로 나누어 두었으므로 조합이 뜻을 갖는다. 공정 중 압력은 `numeric` 과 `trace` 이고, 장비가 거쳐 간 mode 를 시간순으로 적은 열은 `category` 와 `trace` 이다. 축이 하나뿐이면 이 둘을 가릴 수 없다.

### 5.5 Array Length

Array length 는 배열의 길이가 행마다 같은지를 나눈다. Cell 이 배열인 열에만 적용된다.

Table 9. Array length labels

| Label | Rule |
|-------|------|
| `fixed` | 모든 행에서 배열의 길이가 같고, 축이 둘인 배열에서는 두 축의 길이가 모두 같다 |
| `variable` | 행에 따라 배열의 길이가 다르다 |

Wafer 마다 정해진 자리에서 재는 두께는 자리 수가 늘 같으므로 `fixed` 이고, wafer 마다 검출되는 defect 의 좌표 목록은 개수가 제각각이므로 `variable` 이다. 둘을 갈라 두는 이유는 `fixed` 인 열만 그대로 고정 폭의 feature 로 펼칠 수 있기 때문이다.

### 5.6 Trace Quantum

Trace 의 값이 몇 개의 level 위에 머무는지를 나눈다. Level 은 baseline 을 포함해서 센다.

Table 10. Trace quantum labels

| Label | Rule |
|-------|------|
| `q1` | Cell 하나 안에서 값이 level 하나 위에만 머문다 |
| `qn` | Cell 하나 안에서 값이 셀 수 있는 여러 level 위를 오간다 |
| `infinite` | Cell 하나 안에서 값이 level 위에 머물지 않고 연속으로 변한다 |

세 label 은 level 개수가 하나, 여럿, 무한인 경우이므로 어떤 trace 든 하나에 들어간다. `infinite` 는 양자화되지 않은 아날로그 신호가 앉는 자리이고, 이것이 없으면 매끄럽게 변하는 압력이 `qn` 으로 잘못 적혀 있지도 않은 level 을 주장하게 된다.

`q1` 은 시간이 지나도 값이 변하지 않는 trace 이므로, 그 trace 가 담은 정보는 수치 하나와 같다. 그래도 열 전체가 뜻을 잃는 것은 아니다. 행마다 그 하나의 값이 다르면 열은 `active` 이고, 모든 행이 같은 값이면 `inactive` 이다. 5.2 절이 activity 를 행 사이의 비교로 정한 것은 이 구분을 위해서이다.

`q1` 인 열은 cell 을 그 하나의 값으로 바꾸어 `scalar` 로 축약할 수 있다. 축약하면 행 사이의 차이는 그대로 남고 시간축만 사라지므로 activity 는 바뀌지 않는다.

### 5.7 Trace Shape

Trace 의 모양을 나눈다. 각 규칙에 나오는 임계값은 판정 configuration 이며, 4절의 `thresholds` 에 함께 기록한다.

Table 11. Trace shape labels

| Label | Rule |
|-------|------|
| `flat` | Window 에서 값이 변하지 않는다 |
| `rectangle` | 값이 두 level 사이를 오가고, 한 level 에 머무는 시간이 level 사이를 이동하는 시간보다 정해진 배수 이상 길다 |
| `triangle` | 상승 구간과 하강 구간의 기울기 크기가 서로 비슷하고, 두 구간 사이에 평탄한 구간이 없다 |
| `ramp` | Window 에서 값이 한 방향으로만 변하는 구간이 정해진 비율 이상을 차지한다 |
| `oscillation` | Autocorrelation 에 정해진 크기 이상의 peak 이 일정한 간격으로 나타난다 |
| `irregular` | 위 다섯 규칙을 모두 만족하지 않는다 |

`flat` 과 `trace_quantum` 의 `q1` 은 같은 사실을 두 관점에서 적는다. Level 이 하나뿐인 trace 는 값이 변할 곳이 없으므로 언제나 평탄하다. 따라서 한쪽만 붙어 있는 열은 판정에 오류가 있다는 신호이며, 두 axis 를 맞대어 보는 것으로 확인된다.

`irregular` 는 앞의 다섯이 받지 못한 trace 를 받아, 모든 trace 가 이 axis 에서 label 하나를 갖게 한다. 다른 label 과 성격이 다른 점은 그 뜻이 자기 규칙이 아니라 앞의 다섯 규칙에 매여 있다는 것이다. 임계값을 조정하면 `irregular` 로 판정되는 열의 수가 함께 움직이므로, `thresholds` 를 보지 않고 `irregular` 만 읽으면 그 열이 어떤 trace 인지 알 수 없다.

### 5.8 File Format

`column-class.json` 은 axis 를 key 로 두고, 그 아래에 label 과 판정 규칙의 쌍을 둔다. 규칙을 label 옆에 두는 이유는 규칙 없는 label 이 사람마다 다른 뜻으로 쓰이기 때문이다.

```json
// column-class.json
{
  "activity": {
    "active": "two or more distinct cell values exist among the rows that are not missing",
    "inactive": "every row that is not missing holds the same cell value, or every row is missing"
  },
  "value_type": {
    "category": "the value comes from a finite set of names and the values carry no order",
    "ordinal": "the value comes from a finite set of names that carry an order, while arithmetic between them carries no meaning",
    "numeric": "the value has magnitude and arithmetic between values carries meaning",
    "text": "the value is a string and the set it comes from is not fixed in advance",
    "datetime": "the value points to an instant, and the difference between two values carries meaning while their sum does not"
  },
  "structure": {
    "scalar": "one cell holds a single value",
    "vector": "one cell holds an array whose element order carries no meaning",
    "matrix": "one cell holds an array with two axes and neither axis is time",
    "trace": "one cell holds an array whose elements are ordered in time"
  },
  "array_length": {
    "fixed": "the array has the same length in every row, and an array with two axes has the same length on both",
    "variable": "the array length differs from row to row"
  },
  "trace_quantum": {
    "q1": "within one cell the value stays on a single level",
    "qn": "within one cell the value moves across a countable number of levels",
    "infinite": "within one cell the value changes continuously and rests on no level"
  },
  "trace_shape": {
    "flat": "the value does not change over the window",
    "rectangle": "the value alternates between two levels and the time held on a level is at least dwell_ratio times the time taken to move between them",
    "triangle": "the rising and the falling slope have a similar magnitude and no flat segment lies between them",
    "ramp": "the segments where the value moves in one direction only cover at least ramp_fraction of the window",
    "oscillation": "the autocorrelation shows a peak of at least acf_peak at a regular interval",
    "irregular": "none of the five rules above is satisfied"
  }
}
```

새 label 이 필요하면 이 file 에 규칙과 함께 한 줄을 더한다. 여기에 없는 label 은 6절의 규칙에 따라 거부된다.

## 6. Integrity Rules

Manifest 가 지켜야 하는 조건은 세 가지이다.

Table 12. Integrity rules

| Rule | Condition | Catches |
|------|-----------|---------|
| 1 | 모든 label 은 `column-class.json` 에 있어야 한다 | 오타와 미등록 label |
| 2 | 열은 적용되는 axis 마다 label 을 정확히 하나 갖는다 | `scalar` 와 `trace` 를 함께 붙이는 모순, 그리고 판정이 끝나지 않은 열 |
| 3 | 열은 적용되지 않는 axis 의 label 을 가질 수 없다 | 의존 관계 위반 |

이 세 가지는 label 만 비교하면 확인되므로 데이터를 읽지 않고 검사할 수 있다. 선언한 형과 실제 값이 맞는지처럼 데이터를 읽어야 아는 것은 판정이 담당한다.

## Appendix A. Terminology

본문에서 정의하지 않고 쓴 용어를 정리한다.

- **Autocorrelation** 은 신호를 시간축으로 밀어 가며 자기 자신과 곱해 평균한 값이고, 주기 성분이 있으면 그 주기마다 peak 이 나타난다.
- **Baseline** 은 trace 가 아무 동작도 하지 않을 때 머무는 기준 level 이다.
- **Cell** 은 table 에서 행 하나와 열 하나가 만나는 자리이다.
- **Entity** 는 데이터가 붙는 대상이고, wafer 나 lot 처럼 측정이 귀속되는 단위를 말한다.
- **Grain** 은 table 의 행 하나가 무엇을 나타내는지를 말한다.
- **Level** 은 값이 연속으로 변하지 않고 몇 개의 값 위에만 머무를 때 그 값 하나를 말한다.
- **Medallion architecture** 는 데이터를 원본에 가까운 bronze, 정제된 silver, 사용 목적에 맞춘 gold 의 세 단계로 나누어 적재하는 방식이다.
- **Object key** 는 object storage 에서 저장된 항목 하나를 가리키는 문자열이다.
- **Object storage** 는 파일을 directory 구조가 아니라 key 로 지시해 저장하는 storage 이다.
- **Tabular data** 는 행과 열로 이루어진 데이터이고, 행 하나가 관측 하나에 대응하며 열 하나가 그 관측의 한 항목에 대응한다. Cell 이 단일 값이 아니라 배열이어도 행과 열의 틀이 유지되면 tabular data 이다.
- **Trace** 는 한 대상을 시간에 따라 이어서 기록한 값의 열이고, 값 자체와 값이 놓인 순서가 함께 정보를 이룬다.
- **Window** 는 trace 에서 판정 대상으로 잘라낸 시간 구간이다.
