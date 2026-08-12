# Structured Data Manifest for Semiconductor Machine Data
Rev. 55 | Created: 2026-08-07 | Updated: 2026-08-12 10:50 CDT

데이터를 받아서 모델에 넣기까지 반복해서 답해야 하는 질문은 세 가지이다. 이 데이터가 어디서 왔고 무엇을 위한 것인가 (provenance), 열 이름과 형을 어떻게 맞출 것인가 (configuration), 그리고 각 열이 어떤 성격의 값인가 (class) 이다. Manifest 는 이 세 질문에 각각 하나의 file 을 대응시키고, 네 번째 file 에 class 를 부르는 이름과 그 판정 규칙을 모아 둔다.

Manifest 를 만드는 목적은 column profile 을 얻는 것이다. 사람이 적는 앞의 두 file 은 그 자체가 목적이 아니라 판정을 세울 수 있게 하는 입력이며, 열마다 무엇이 담겨 있는지에 대한 답은 마지막에 나오는 column profile 하나에 모인다.

## 1. Scope And File Order

### 1.1 Structured Data

이 문서가 다루는 데이터는 structured data 이다. Cell 하나가 단일 값이 아니라 배열일 수 있고, 그때에도 데이터 전체는 행과 열의 틀에 들어간다. Manifest 가 기록하는 단위는 열이므로, 열로 나눌 수 없는 image directory 나 layout file 은 이 문서의 대상이 아니다.

### 1.2 File Order

Table 1. Manifest files

| Order | File | Source | Content |
|-------|------|--------|---------|
| 1 | `catalog.json` | Human | Dataset 이 어디서 왔고 무엇을 위한 것인지 기록한다 |
| 2 | `column-config.json` | Human | 열 이름, 삭제, 형 변환, 결측 표시, 단위와 사람이 아는 class 를 기록한다 |
| 3 | `column-profile.json` | Analysis | 데이터를 읽어 판정한 class 를 기록한다 |
| - | `column-class.json` | Reference | Class 의 axis 와 label, 그리고 각 label 의 판정 규칙을 정의한다 |

`Order` 는 file 이 쓰이는 순서이다. Catalog 가 dataset 이 무엇인지 기록하고, column config 가 그 table 을 어떻게 다듬을지 적으며, column profile 은 그 조작을 마친 table 을 읽어 판정한 결과이다. Column class 는 dataset 마다 다시 쓰는 file 이 아니라 project 사이에서 함께 참조하는 어휘이므로 순서를 갖지 않는다.

이 순서 때문에 열 이름의 기준이 file 마다 다르다. Catalog 는 `column_mapping` 보다 먼저 쓰이므로 상류에서 온 그대로의 이름을 적고, column config 의 나머지 항목과 column profile 은 `column_mapping` 을 거친 뒤의 이름을 적는다. 그래서 column profile 이 기술하는 것은 상류에서 온 table 이 아니라 column config 를 적용하고 난 table 이다.

Dataset 을 기술하는 세 file 은 모두 object key 를 최상위 key 로 쓴다. 세 file 을 잇는 참조 항목을 따로 두지 않으며, 같은 key 를 쓰는 것이 곧 연결이다. Column class 는 dataset 에 매이지 않는 어휘이므로 object key 를 쓰지 않는다.

```text
object key
    |
    +--> catalog.json          what the dataset is
    +--> column-config.json    what a human declares
    +--> column-profile.json   what the analysis decides

column-class.json              axis -> label and its rule
```

열의 class 는 column profile 에서 정해진다. Column config 의 `class` 는 사람이 아는 것을 판정에 넘겨 주는 입력이고, 판정은 그것을 받아 최종 class 를 column profile 에 쓴다.

## 2. Catalog

Catalog 는 object key 하나마다 그 데이터가 무엇인지를 기록한다. Key 는 항상 같은 dataset 을 가리키고 내용만 갱신되므로, key 자체가 식별자로 성립한다. 이는 같은 key 에 다른 dataset 이 올라오는 일이 없다는 조건 위에서만 참이다.

Table 2. Catalog description keys

| Key | Required | Description |
|-----|----------|-------------|
| `project_goal` | Yes | 이 데이터를 어떤 목적으로 적재했는지 한 문장으로 적는다 |
| `provider` | Yes | 데이터를 낸 주체나 system 을 적는다 |
| `date` | Yes | 데이터를 제공받은 날짜를 적는다 |
| `file_format` | Yes | 데이터가 어떤 형식으로 적재되어 있는지 적는다 |
| `medallion` | Yes | `bronze`, `silver`, `gold` 중 하나를 적는다 |
| `grain` | Yes | 행 하나가 무엇인지 한 문장으로 적는다 |
| `row_key` | Yes | 행을 유일하게 만드는 열을, 같은 entity 로 묶는 열과 그 entity 안에서 행을 구별하는 열로 나누어 적는다 |
| `derived_from` | No | 이 데이터를 만들어 낸 상류 object key 를 적고, 원본이면 생략한다 |
| `rows` | No | 행 수를 적는다 |
| `note` | No | 이 데이터를 쓸 때 알아야 할 예외를 적는다 |

`grain` 이 필수인 이유는 행 하나가 무엇인지를 모르면 join 과 집계가 조용히 틀리기 때문이다. `derived_from` 이 있어야 `medallion` 이 이름표에 그치지 않고 상류를 거슬러 올라갈 수 있는 사슬의 한 마디가 된다.

`date` 는 데이터를 제공받은 날짜이지 catalog 항목을 손본 날짜가 아니다. 같은 key 에 새 적재가 덮이면 이 날짜도 함께 바뀌므로, 손에 든 데이터가 언제 것인지는 이 값 하나로 답한다.

`file_format` 은 `csv` 나 `parquet` 처럼 적재된 형식을 적는다. Object key 에 확장자가 드러나지 않을 수 있고 형식은 데이터를 열기 전에 알아야 하는 값이므로, 데이터를 읽어 알아내는 대신 사람이 적는다.

`medallion` 은 `bronze`, `silver`, `gold` 세 값만 쓴다. 정제 단계를 부르는 이름이 사람마다 갈리면 `derived_from` 으로 상류를 거슬러 올라가도 그 단계가 무엇인지 알 수 없게 된다.

`row_key` 는 행이 서로 어떻게 구별되는지를 두 목록으로 적는다.

- `entity_columns` 는 값이 같으면 같은 entity 로 보는 열을 적는다.
- `sequence_columns` 는 한 entity 가 여러 행을 가질 때 그 행들을 서로 구별하는 열을 순서대로 적고, 행 하나가 곧 entity 하나이면 빈 목록으로 둔다.

두 목록을 이은 것이 행마다 유일한 조합이다. **행이 중복인지 아닌지를 명시하는 것이 이 조합이며**, 중복 검사는 `entity_columns` 가 아니라 이 조합으로 한다. 한 wafer 의 trace 를 행마다 한 점씩 펼친 table 에서 wafer 번호는 수천 행에 걸쳐 같은 값이지만, `sequence_columns` 의 시간 열이 그 행들을 서로 구별하므로 중복이 아니다. `sequence_columns` 를 적지 않으면 그 table 을 중복 제거해도 되는지가 기록되지 않는다.

행 하나가 wafer 와 process step 의 조합이면 `entity_columns` 는 wafer 까지이고 `sequence_columns` 에 step 이 들어가므로, 이때도 wafer 는 여러 행에 걸쳐 되풀이된다.

두 목록에 적는 이름은 1.2 절이 정한 대로 상류에서 온 그대로의 이름이다.

같은 데이터를 cell 안의 배열로 담을 수도 있고 행으로 펼칠 수도 있는데, 그 차이는 `row_key` 가 아니라 열의 class 가 말해 준다. 배열로 담으면 그 열이 `trace` 이고, 행으로 펼치면 `scalar` 가 되면서 시간 열이 `sequence_columns` 에 들어간다.

```json
// catalog.json
{
  "Ultah/AlPVDPoC/SilverData/#0/V0": {
    "project_goal": "Thin film thickness prediction",
    "provider": "PVD unit",
    "date": "2026-08-07",
    "file_format": "parquet",
    "medallion": "silver",
    "grain": "one wafer and one process step",
    "row_key": {
      "entity_columns": ["LOT_ID", "WAFER_ID"],
      "sequence_columns": ["STEP_ID"]
    },
    "derived_from": "Ultah/AlPVDPoC/BronzeData/#0/V0",
    "rows": 1043221,
    "note": "POC (Proof of Concept)"
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
| - | `unit` | 열의 값이 어떤 단위인지 적는다 |
| - | `class` | 사람이 아는 class label 을 열마다 적는다 |

`Order` 는 조작을 적용하는 순서이고, `unit` 과 `class` 는 조작이 아니므로 순서를 갖지 않는다. 이름 바꾸기가 첫째인 이유는 뒤의 세 조작이 모두 바뀐 이름으로 열을 가리키게 하기 위해서이다. 형 바꾸기가 마지막인 이유는 결측 표시를 걷어낸 뒤라야 열이 목표한 형으로 들어가기 때문이다. `-999` 가 남아 있는 열을 먼저 정수로 바꾸면 결측이 값으로 굳는다.

순서를 JSON 에 담지 않고 표에 고정하는 이유는 JSON object 의 key 순서가 규격상 보장되지 않기 때문이다. 순서가 의미를 가지는 값은 object 가 아니라 array 에 담거나, 이 경우처럼 file 밖에 고정한다.

`na_values` 가 필요한 이유는 계측 데이터가 결측을 `-999` 같은 실수로 표시하는 일이 많고, 이것을 그대로 두면 실제 측정값으로 학습되기 때문이다. 이 값은 데이터를 읽어서는 알 수 없으므로 사람이 적어야 한다.

`unit` 은 값이 어떤 단위로 적혀 있는지를 남긴다. 숫자 `3.3` 이 V 인지 mV 인지는 데이터에 들어 있지 않으므로 판정이 알아낼 수 없고, 적지 않으면 단위는 열 이름에만 남거나 아예 사라진다. 수치를 담는 열에만 적는다.

`class` 에는 사람이 아는 열만 적고 나머지는 생략한다. 사람이 확실하게 아는 것은 대개 열이 `category` 라는 사실이다. 어떤 값이 실제로 들어오는지는 데이터를 읽어야 알 수 있으므로 판정이 채운다.

```json
// column-config.json
{
  "Ultah/AlPVDPoC/SilverData/#0/V0": {
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
    "unit": {
      "rf_fwd_pwr_w": "W",
      "chamber_pressure": "mTorr"
    },
    "class": {
      "bin_code": ["category"],
      "site_thickness": ["vector"]
    }
  }
}
```

## 4. Column Class

판정은 cell 하나 안을 보고 하거나 행 사이를 견주어 한다. 행 사이를 견줄 때 무엇을 한 덩어리로 묶을지는 catalog 의 `row_key` 가 정한다. Axis 마다 어느 쪽인지는 Table 4 의 `Basis` 열에 적는다.

### 4.1 Axis

Class 는 하나의 목록이 아니라 여러 개의 axis 로 나뉜다. 한 axis 안의 label 은 서로 배타적이므로, 이렇게 나누면 서로 무관한 성질을 하나의 label 에 섞지 않아도 되고 모순된 조합을 규칙으로 걸러낼 수 있다.

Table 4. Class axes

| Axis | Label | Applies to | Basis | Source |
|------|-------|------------|-------|--------|
| `activity` | `active`, `inactive` | `all` | `rows or entities` | Analysis |
| `value_type` | `category`, `ordinal`, `numeric`, `text`, `datetime` | `all` | `cell` | Human or analysis |
| `structure` | `scalar`, `vector`, `matrix`, `trace`, `tensor` | `all` | `cell` | Human or analysis |
| `array_length` | `fixed`, `variable` | `non-scalar` | `rows` | Analysis |
| `trace_quantum` | `q1`, `qn`, `infinite` | `trace` | `cell` | Analysis |
| `trace_shape` | `flat`, `step`, `rectangle`, `triangle`, `oscillation`, `irregular` | `trace` | `cell` | Analysis |

`Applies to` 는 그 axis 가 어느 열에 적용되는지를 적고, `Basis` 는 그 axis 를 판정할 때 무엇을 보는지를 적는다. `cell` 은 cell 하나 안만 보므로 다른 행에 무엇이 들어 있든 답이 같다. `rows` 는 그 열의 행을 서로 견준다. `rows or entities` 는 행끼리 견줄 수도 있고 `row_key` 로 묶은 entity 끼리 견줄 수도 있다는 뜻이다.

**열은 자기에게 적용되는 axis 마다 label 을 정확히 하나씩 갖는다.** 적용되지 않는 axis 에는 label 을 갖지 않는다. 그래서 label 이 비어 있는 axis 는 판정이 아직 끝나지 않았다는 뜻이며, 판정이 끝난 열은 어느 axis 를 물어도 답이 하나 나온다.

이 규약 때문에 각 axis 의 label 은 그 axis 가 적용되는 모든 열을 남김없이 받아야 한다. `trace_shape` 의 `irregular` 가 그 자리를 맡는다.

`array_` 와 `trace_` 로 시작하는 axis 는 `structure` 가 각각 배열일 때와 `trace` 일 때만 적용된다. 이름에 의존 관계를 넣어 두었으므로 axis 이름만 보고 이 제약을 알 수 있다.

한 label 은 한 axis 에만 속한다. 열의 class 를 label 의 목록으로 적으므로, 같은 label 이 두 axis 에 있으면 그 label 이 어느 axis 의 값인지 가릴 수 없다.

### 4.2 Activity

Activity 는 열의 값이 행 사이 또는 entity 사이에서 변하는지를 나눈다. 열을 쓸 것인지 말 것인지를 정하는 것이 아니라, 데이터를 읽어 판정한 결과다.

판정은 행 사이의 비교로 한다. Cell 을 통째로 하나의 값으로 보므로, cell 이 배열인 열에서는 배열 전체가 같아야 두 행이 같은 값을 가진 것이 된다. Cell 안에서 값이 변하는지는 activity 가 아니라 `trace_quantum` 이 다룬다.

Table 5. Activity labels

| Label | Rule |
|-------|------|
| `active` | 결측을 제외한 행 중 서로 다른 cell 값이 둘 이상 있다 |
| `inactive` | 결측을 제외한 행의 cell 값이 모두 같거나, 모든 행이 결측이다 |

한 entity 에 여러 행이 놓인 table 에서는 그 행들이 `sequence_columns` 를 따라 서로 다르므로, 이 규칙을 그대로 쓰면 거의 모든 열이 `active` 로 나온다. 그럴 때는 `row_key` 의 `entity_columns` 로 행을 묶은 뒤 entity 사이를 비교한다. 한 entity 가 담은 값 전체를 하나의 값으로 보므로, 두 entity 는 그 값의 묶음이 서로 달라야 다른 값을 가진 것이 된다.

두 비교는 같은 열에 서로 다른 답을 주므로, 어느 쪽으로 판정했는지를 5 절의 `activity_basis` 에 남긴다. 행 사이의 비교는 `row` 이고 entity 사이의 비교는 `entity` 이다. 이 값이 없으면 `active` 와 `inactive` 가 무엇을 비교한 결과인지 알 수 없다.

### 4.3 Value Type

Value type 은 값 하나가 이름인지 수치인지를 나눈다. Cell 이 배열인 열에서는 그 원소 하나를 두고 판정한다.

Table 6. Value type labels

| Label | Rule |
|-------|------|
| `category` | 값이 유한한 이름의 집합에서 나오고, 값 사이에 순서가 없다 |
| `ordinal` | 값이 유한한 이름의 집합에서 나오고, 값 사이에 순서가 있으나 산술 연산은 의미를 갖지 않는다 |
| `numeric` | 값이 크기를 갖는 수치이고, 값 사이의 산술 연산이 의미를 갖는다 |
| `text` | 값이 문자열이고, 값의 집합이 미리 정해져 있지 않다 |
| `datetime` | 값이 시각을 가리키고, 값 사이의 차는 의미를 갖지만 합은 의미를 갖지 않는다 |

`ordinal` 은 `category` 와 `numeric` 사이에 놓인다. 비교는 되고 산술은 되지 않으므로, 등급이나 심각도처럼 값을 줄 세울 수 있는 열이 여기에 속한다. `ordinal` 을 `category` 로 적으면 순서를 잃고, `numeric` 으로 적으면 등급 사이의 간격이 모두 같다고 주장하게 된다.

`category` 와 `text` 는 둘 다 문자열을 담을 수 있고, 값의 집합이 미리 정해져 있는지로 갈린다. 고유값이 수만 개여도 그 집합이 정해져 있으면 `category` 이므로 식별자는 여기에 속하고, 작업자가 그때그때 적어 넣는 설명문처럼 값이 매번 새로 쓰이면 `text` 이다.

`datetime` 과 `numeric` 은 합이 의미를 갖는지로 갈린다. 두 시각을 더한 값은 뜻이 없지만 두 소요 시간을 더한 값은 뜻이 있으므로, queue time 처럼 길이를 재는 열은 `datetime` 이 아니라 `numeric` 이다.

`category` 는 사람이 정한다. 판정은 고유값이 몇 개인지까지만 알 수 있고, 그 값들이 이름인지 크기인지는 데이터에 들어 있지 않다. `bin_code` 의 값 `1` 과 `2` 는 두 종류의 불량을 가리키는 이름이지 크기가 아니지만, 데이터만 보아서는 그것을 알 수 없다.

### 4.4 Structure

Structure 는 행 (wafer) 와 열 (feature) 로 특정된 cell 하나가 값 하나를 담는지 배열을 담는지를 나눈다. Structure label 은 cell structure label 이다.

Table 7. Structure labels

| Cell label | Cell value | Cell dim | Array notation | Data label |
|------------|------------|----------|----------------|------------|
| `scalar` | A single value (tabular data) | 0 | `[wafer, feature]` | matrix |
| `vector` | An array whose elements lie along an axis that is not time | 1 | `[wafer, feature, site]` | tensor |
| `matrix` | An array with two axes, neither of which is time | 2 | `[wafer, feature, die_x, die_y]` | tensor |
| `trace` | An array whose elements are ordered in time | 1 | `[wafer, feature, trace]` | tensor |
| `tensor` | A multidimensional array with three or more axes | >=3 | `[wafer, feature, x, y, z]` | tensor |

Label 마다 cell 에 값이 어떻게 담기는지는 [Appendix B. Structure Example](#appendix-b-structure-example) 에 data example 로 두었다.

Array notation 은 그 label 을 갖는 열만 모은 table 을 배열 하나로 펼쳤을 때의 축 목록이다. 첫 element 를 axis 0 이라 부르는 것이 배열의 표준 표기이므로 이 문서도 그대로 쓰되, 4.1 절의 class axis 와 가르기 위해 array axis 0 처럼 적는다. Array axis 0 은 행이고 array axis 1 은 열이며, array axis 2 부터는 cell 안의 축이다. Array axis 0 을 `wafer` 라고 적은 것은 이 문서의 예가 wafer 단위이기 때문이고, 실제 이름은 catalog 의 `grain` 이 정한 행 단위를 따른다.

Cell 안의 축이 곧 structure label 을 가른다. 모든 열이 `scalar` 인 table 이 tabular data 이고, 나머지는 cell 안에 배열을 담아 그 틀을 넘어선다. `vector` 와 `trace` 를 가르는 것은 배열이라는 사실이 아니라 그 축이 시간인지이다. Wafer 위 여러 지점에서 잰 두께는 원소가 site 자리를 따라 놓이므로 `vector` 이고, 공정 중에 기록한 압력은 원소가 시각을 따라 놓이므로 `trace` 이다.

어느 쪽이든 원소의 자리는 고정되어 있어야 한다. 두께 배열의 세 번째 원소가 행마다 다른 site 를 가리키면 wafer 끼리 비교할 수 없으므로, `vector` 는 순서가 없는 배열이 아니라 시간이 아닌 축을 따라 정렬된 배열이다.

`matrix` 는 원소의 자리가 한 축이 아니라 두 축으로 정해지는 경우이다. Wafer 위 die 마다의 bin code 를 격자로 담은 열이 여기에 해당하며, 같은 값을 자리 정보 없이 늘어놓으면 `vector` 가 되어 이웃 관계를 잃는다.

### 4.5 Array Length

Array length 는 배열의 길이가 행마다 같은지를 나눈다. Cell 이 배열인 열에만 적용된다.

Table 8. Array length labels

| Label | Rule |
|-------|------|
| `fixed` | 모든 행에서 배열의 길이가 같고, `matrix` 처럼 축이 둘이면 두 축이 각각 행마다 같은 길이를 갖는다 |
| `variable` | 행에 따라 배열의 길이가 다르다 |

Wafer 마다 정해진 site 에서 재는 두께는 site 개수가 늘 같으므로 `fixed` 이고, wafer 마다 검출되는 defect 의 좌표 목록은 개수가 제각각이므로 `variable` 이다. 공정 중에 받은 trace 는 recipe 가 같아도 step 이 끝나는 시점이 행마다 조금씩 달라 길이가 흔들리므로 대개 `variable` 이다. 둘을 갈라 두는 이유는 `fixed` 인 열만 그대로 고정 폭의 feature 로 펼칠 수 있기 때문이다.

### 4.6 Trace Quantum

Trace quantum 은 trace 의 값이 몇 개의 level 위에 머무는지를 나눈다. Level 은 baseline 을 포함해서 센다.

Table 9. Trace quantum labels

| Label | Rule |
|-------|------|
| `q1` | Cell 하나 안에서 값이 level 하나 위에만 머문다 |
| `qn` | Cell 하나 안에서 값이 셀 수 있는 여러 level 위를 오간다 |
| `infinite` | Cell 하나 안에서 값이 level 위에 머물지 않고 연속으로 변한다 |

세 label 은 level 개수가 하나, 여럿, 무한인 경우이므로 어떤 trace 든 하나에 들어간다. `infinite` 는 양자화되지 않은 아날로그 신호가 앉는 자리이고, 이것이 없으면 매끄럽게 변하는 압력이 `qn` 으로 잘못 적혀, 있지도 않은 level 을 주장하게 된다.

`q1` 은 시간이 지나도 값이 변하지 않는 trace 이므로, 그 trace 가 담은 정보는 수치 하나와 같다. 그래도 열 전체가 뜻을 잃는 것은 아니다. 행마다 그 하나의 값이 다르면 열은 `active` 이고, 모든 행이 같은 값이면 `inactive` 이다. 4.2 절이 activity 를 행 사이의 비교로 정한 것은 이 구분을 위해서이다.

`q1` 인 열은 cell 을 그 하나의 값으로 바꾸어 `scalar` 로 축약할 수 있다. 축약하면 행 사이의 차이는 그대로 남고 시간축만 사라지므로 activity 는 바뀌지 않는다.

### 4.7 Trace Shape

Trace shape 는 trace 의 모양을 나눈다. 각 규칙에 나오는 임계값은 판정 configuration 이며, 5 절의 `thresholds` 에 함께 기록한다.

[semiconductor-machine-signal-parameterization-shape-taxonomy.md](semiconductor-machine-signal-parameterization-shape-taxonomy.md) 가 같은 label 을 더 자세히 다룬다. 임계값의 기본값, 규칙을 시험하는 순서, 그리고 각 label 이 어떤 형상 모델족에 대응하는지가 그 문서에 있다.

규칙에 나오는 window 는 판정 대상으로 삼는 시간 구간이고, trace 의 전체일 수도 일부일 수도 있다. 따로 정하지 않으면 trace 전체이다. 앞뒤의 대기 구간을 떼어 내는 것처럼 일부만 보아야 할 이유가 있으면 그 구간을 `thresholds` 안의 `window` 에 적는다.

Table 10. Trace shape labels

| Label | Rule |
|-------|------|
| `flat` | Window 에서 값이 변하지 않는다 |
| `step` | 값이 한 준위에 머물다 다른 준위로 옮겨 가 그대로 머무르고, 두 준위의 차가 정해진 값 이상이다 |
| `rectangle` | 값이 두 level 사이를 오가고, 한 level 에 머무는 시간이 level 사이를 이동하는 시간보다 정해진 배수 이상 길다 |
| `triangle` | 상승 구간과 하강 구간의 기울기 크기가 서로 비슷하고, 두 구간 사이에 평탄한 구간이 없다 |
| `oscillation` | Autocorrelation 에 정해진 크기 이상의 peak 이 일정한 간격으로 나타난다 |
| `irregular` | 위 다섯 규칙을 모두 만족하지 않는다 |

`flat` 과 `trace_quantum` 의 `q1` 은 같은 사실을 두 관점에서 적는다. Level 이 하나뿐인 trace 는 값이 변할 곳이 없으므로 언제나 평탄하다. 따라서 한쪽만 붙어 있는 열은 판정에 오류가 있다는 신호이며, 두 axis 를 맞대어 보는 것으로 확인된다.

`irregular` 는 앞의 다섯이 받지 못한 trace 를 받아, 모든 trace 가 이 axis 에서 label 하나를 갖게 한다. 다른 label 과 성격이 다른 점은 그 뜻이 자기 규칙이 아니라 앞의 다섯 규칙에 매여 있다는 것이다. 임계값을 조정하면 `irregular` 로 판정되는 열의 수가 함께 움직이므로, `thresholds` 를 보지 않고 `irregular` 만 읽으면 그 열이 어떤 trace 인지 알 수 없다.

### 4.8 File Format

`column-class.json` 은 project 사이에서 함께 참조하는 file 이다. Dataset 하나에 매이지 않을 뿐 아니라 project 하나에도 매이지 않으므로, project 마다 복사해 고치지 않고 하나를 두고 같이 읽는다. 복사해 고치기 시작하면 같은 label 이 project 마다 다른 규칙으로 판정되어, 열의 class 를 project 를 넘어 비교할 수 없게 된다. `version` 이 dataset 이 아니라 이 어휘에 붙어 있는 이유도 같다.

`column-class.json` 은 `axis` 아래에 axis 이름을 key 로 두고, 그 아래에 label 과 판정 규칙의 쌍을 둔다. Axis 를 한 겹 안에 넣는 이유는 `version` 이 axis 로 잘못 읽히지 않게 하기 위해서이다. `version` 은 label 이나 규칙이 바뀔 때마다 올린다. 규칙을 label 옆에 두는 이유는 규칙 없는 label 이 사람마다 다른 뜻으로 쓰이기 때문이다.

```json
// column-class.json
{
  "version": 3,
  "axis": {
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
      "vector": "one cell holds an array whose elements lie along an axis that is not time",
      "matrix": "one cell holds an array with two axes and neither axis is time",
      "trace": "one cell holds an array whose elements are ordered in time",
      "tensor": "one cell holds a multidimensional array with three or more axes"
    },
    "array_length": {
      "fixed": "the array has the same length in every row, and an array with two axes has the same length on each of its axes in every row",
      "variable": "the array length differs from row to row"
    },
    "trace_quantum": {
      "q1": "within one cell the value stays on a single level",
      "qn": "within one cell the value moves across a countable number of levels",
      "infinite": "within one cell the value changes continuously and rests on no level"
    },
    "trace_shape": {
      "flat": "the value does not change over the window",
      "step": "the value holds one level, moves to another and stays there, and the difference between the two levels is at least step_min",
      "rectangle": "the value alternates between two levels and the time held on a level is at least dwell_ratio times the time taken to move between them",
      "triangle": "the rising and the falling slope have a similar magnitude and no flat segment lies between them",
      "oscillation": "the autocorrelation shows a peak of at least acf_peak at a regular interval",
      "irregular": "none of the five rules above is satisfied"
    }
  }
}
```

새 label 이 필요하면 이 file 에 규칙과 함께 한 줄을 더한다. 여기에 없는 label 은 6 절의 규칙에 따라 거부된다.

## 5. Column Profile

Column profile 은 데이터를 읽어 판정한 결과를 담는다. 열마다 최종 class label 과, 그 판정을 뒷받침하는 관측값을 함께 둔다.

Table 11. Column profile keys

| Key | Description |
|-----|-------------|
| `profiled_at` | 판정을 수행한 시각을 적는다 |
| `class_version` | 판정에 쓴 `column-class.json` 의 version 을 적는다 |
| `activity_basis` | Activity 를 `row` 와 `entity` 중 어느 비교로 판정했는지 적는다 |
| `thresholds` | 판정에 쓴 임계값과 window 를 적는다 |
| `columns` | 열마다 최종 class 와 관측값을 적는다. `class` 외의 key 는 정해 두지 않는다 |

`columns` 는 전수이다. Column config 를 적용하고 난 table 의 모든 열이 여기에 있어야 하고, 버린 열은 그 table 에 없으므로 여기에도 없다. 열이 빠져 있으면 그 열이 판정되지 않은 것인지 대상이 아닌 것인지 가릴 수 없으므로, 빠진 열은 판정이 끝나지 않았다는 뜻이다.

`class_version` 과 `activity_basis` 와 `thresholds` 는 모두 같은 이유로 있다. 어휘에는 label 이 늘 수 있고, activity 는 무엇과 무엇을 비교하느냐에 따라 답이 갈리며, 4 절의 판정 규칙 여러 개가 임계값을 필요로 한다. 셋 중 하나라도 바뀌면 같은 데이터에서 다른 label 이 나오므로, 이 셋이 없는 profile 은 어떤 규칙으로 판정된 것인지 되짚을 수 없다.

`thresholds` 에는 window 도 함께 둔다. Trace 전체를 보았으면 `full` 이고, 구간을 좁혔으면 그 시작과 끝을 적는다. Window 는 임계값이 아니다. 그러나 바꾸면 같은 trace 가 다른 label 을 받는다는 점은 임계값과 같으므로, 판정 configuration 을 한자리에 모아 둔다.

`step_min` 은 무차원이 아니라 센서 분해능의 배수로 적는다. 예시의 `3.0` 은 분해능의 3 배라는 뜻이며, `dwell_ratio` 와 `acf_peak` 처럼 그 자체가 비인 값과 달리 이 단서가 없으면 숫자를 읽을 수 없다.

`category` 로 판정된 열은 관측된 값의 목록을 함께 남긴다. 사람은 그 열이 `category` 라는 사실만 알고 어떤 값이 들어오는지는 모르므로, 이 목록이 사람과 판정의 역할이 갈리는 지점이다.

열 항목에서 정해진 key 는 `class` 하나이다. 나머지는 열어 두어, 판정이 근거로 남기고 싶은 관측값을 자유롭게 더한다. 예시의 `missing_rate` 와 `levels` 가 그렇게 더해진 것이다.

```json
// column-profile.json
{
  "Ultah/AlPVDPoC/SilverData/#0/V0": {
    "profiled_at": "2026-08-07T09:12:00+09:00",
    "class_version": 3,
    "activity_basis": "row",
    "thresholds": {
      "window": "full",
      "dwell_ratio": 5.0,
      "step_min": 3.0,
      "acf_peak": 0.6
    },
    "columns": {
      "LOT_ID": {
        "class": ["active", "category", "scalar"],
        "missing_rate": 0.0
      },
      "WAFER_ID": {
        "class": ["active", "category", "scalar"],
        "missing_rate": 0.0
      },
      "STEP_ID": {
        "class": ["active", "ordinal", "scalar"],
        "levels": [10, 20, 30, 40],
        "missing_rate": 0.0
      },
      "rf_fwd_pwr_w": {
        "class": ["active", "numeric", "trace", "fixed", "qn", "rectangle"],
        "missing_rate": 0.003
      },
      "chamber_pressure": {
        "class": ["active", "numeric", "trace", "fixed", "infinite", "step"],
        "missing_rate": 0.0
      },
      "site_thickness": {
        "class": ["active", "numeric", "vector", "fixed"],
        "missing_rate": 0.0
      },
      "bin_code": {
        "class": ["active", "category", "scalar"],
        "levels": [1, 2, 3, 4, 5, 6, 7, 8],
        "missing_rate": 0.0
      }
    }
  }
}
```

## 6. Integrity Rules

여기의 규칙은 manifest 자체의 형식 검사가 아니라, manifest 를 데이터에 적용할 때 성립해야 하는 조건이다. 아래 넷은 앞의 절들이 세운 설계에서 바로 따라 나오는 최소한의 예시이며, 검사 규칙의 전부를 이 문서에서 정하지 않는다.

Table 12. Integrity rules

| Rule | Condition | Catches |
|------|-----------|---------|
| 1 | 모든 label 은 `column-class.json` 에 있어야 한다 | 오타와 미등록 label |
| 2 | 열은 적용되는 axis 마다 label 을 정확히 하나 갖고, 적용되지 않는 axis 의 label 을 갖지 않는다 | `scalar` 와 `trace` 를 함께 붙이는 모순, 그리고 판정이 끝나지 않은 열 |
| 3 | `column-config.json` 과 `column-profile.json` 의 최상위 key 는 `catalog.json` 에 있어야 한다 | Catalog 에 없는 dataset 의 설정과 판정 |
| 4 | `column-profile.json` 의 `columns` 는 column config 를 적용하고 난 열의 전수여야 한다 | 판정에서 빠진 열과 이미 버린 열에 남은 판정 |

규칙 2 가 보는 class 는 column profile 이 담은 최종 class 이다. Column config 의 `class` 는 사람이 아는 것만 골라 적는 부분 목록이므로 axis 가 비어 있는 것이 정상이고, 이 규칙의 대상이 아니다.

넷 모두 label 과 key 만 비교하면 확인되므로 데이터를 읽지 않고 검사할 수 있다. 선언한 형과 실제 값이 맞는지처럼 데이터를 읽어야 아는 것은 판정이 담당한다.

## Appendix A. Terminology

본문에서 정의하지 않고 쓴 용어를 정리한다.

- **Autocorrelation** 은 신호를 시간축으로 밀어 가며 자기 자신과 곱해 평균한 값이고, 주기 성분이 있으면 그 주기마다 peak 이 나타난다.
- **Baseline** 은 trace 가 아무 동작도 하지 않을 때 머무는 기준 level 이다.
- **Cell** 은 table 에서 행 하나와 열 하나가 만나는 자리이다.
- **Entity** 는 데이터로 관리하는 대상이고, 서로 구별해 저장할 필요가 있는 사람이나 사물이나 개념을 말한다. 관계형 database 에서는 entity 하나가 table 하나로 표현되며, 쇼핑몰이라면 회원과 상품과 주문이 각각 하나의 entity 다. 이 문서에서는 wafer 나 lot 처럼 측정이 귀속되는 단위가 여기에 해당한다.
- **Grain** 은 table 의 행 하나가 무엇을 나타내는지를 말한다.
- **Level** 은 값이 연속으로 변하지 않고 몇 개의 값 위에만 머무를 때 그 값 하나를 말한다.
- **Manifest** 는 저장된 데이터가 무엇인지 기록해 두는 file 의 모음이고, 데이터 자체와 따로 보관된다.
- **Medallion architecture** 는 데이터를 원본에 가까운 bronze, 정제된 silver, 사용 목적에 맞춘 gold 의 세 단계로 나누어 적재하는 방식이다.
- **Object key** 는 object storage 에서 저장된 항목 하나를 가리키는 문자열이다.
- **Object storage** 는 파일을 directory 구조가 아니라 key 로 지시해 저장하는 storage 이다.
- **Semiconductor machine data** 는 반도체 공정 장비와 계측 장비가 남긴 기록이고, 공정 중에 받은 sensor trace, 계측 결과, 그리고 그 측정이 귀속되는 lot 과 wafer 의 식별자가 여기에 속한다.
- **Structured data** 는 미리 정한 schema 를 따르는 데이터이고, 반도체 장비 데이터에서는 그 schema 가 행과 열로 나타난다. 행 하나가 관측 하나에 대응하며 열 하나가 그 관측의 한 항목에 대응하고, cell 이 단일 값이 아니라 배열이어도 그 틀이 유지되면 structured data 이다.
- **Tabular data** 는 cell 하나가 값 하나를 담는 structured data 이다. Cell 이 배열을 담으면 structured data 이기는 하나 tabular data 는 아니다.
- **Trace** 는 한 대상을 시간에 따라 이어서 기록한 값의 열이고, 값 자체와 값이 놓인 순서가 함께 정보를 이룬다.

## Appendix B. Structure Example

4.4 절 Table 7 의 label 마다 cell 하나에 무엇이 담기는지를 wafer 한 행을 두고 보인다.

Table 13. Structure examples

| Label | Column | Cell value | Cell axes |
|-------|--------|------------|-----------|
| `scalar` | `thickness_mean` | `812.4` | 없음 |
| `vector` | `thickness_site` | `[812.4, 809.7, ...]` | `site` |
| `matrix` | `bin_code_map` | `[[3, 3, ...], [3, 1, ...], ...]` | `die_x`, `die_y` |
| `trace` | `chamber_pressure` | `[0.98, 1.02, ...]` | `trace` |
| `tensor` | `thermal_field` | `[[[21.4, 21.6, ...], ...], ...]` | `x`, `y`, `z` |

`...` 은 같은 방식으로 이어지는 원소를 줄인 것이고, 어느 축이든 앞의 두 자리만 적었다. `thickness_site` 에 site 두 곳만 적은 것도 그 생략이며, wafer 에 site 가 스물이면 원소도 스물이다.

`scalar` 만 cell 에 축이 없고 나머지 넷은 cell 안에 축을 갖는다. `vector` 와 `trace` 는 축이 하나여서 값의 나열로 보이지만 그 축이 site 인지 시각인지가 다르고, `matrix` 는 축이 둘이어서 배열이 한 겹 더 중첩되며, `tensor` 는 축이 셋이어서 두 겹 더 중첩된다.
