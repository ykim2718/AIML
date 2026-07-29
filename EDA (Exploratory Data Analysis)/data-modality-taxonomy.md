# Data Modality Taxonomy

> 데이터 모달리티는 "정보가 어떤 형태로 표현되는가"로 가르는 분류다.
> 범용 taxonomy 를 먼저 세우고, 반도체 도메인에 맞게 축을 확장한 형태까지 다룬다.

## 1. General Taxonomy

최상위에서는 구조 기준과 신호 기준 두 축으로 묶이며, 이 두 축은 서로 직교한다.

### 1.1 By Structure

| Type | Description | Example |
|------|-------------|---------|
| Structured | 행과 열 스키마에 맞아떨어진다 | 관계형 테이블, 트랜잭션 로그 |
| Semi-structured | 스키마가 느슨하거나 self-describing 이다 | JSON, XML, YAML, 로그 |
| Unstructured | 정해진 틀이 없는 원본 신호다 | 텍스트, 이미지, 음성, 영상 |

### 1.2 By Signal

머신러닝에서 통상 모달리티라 부르는 축이 이쪽이다.

| Modality | Representation |
|----------|----------------|
| Text | 자연어와 코드, 문서를 토큰 시퀀스로 다룬다 |
| Vision | 이미지와 비디오를 픽셀 격자 또는 프레임 시퀀스로 다룬다 |
| Audio | 음성과 음악, 환경음을 파형 또는 스펙트로그램으로 다룬다 |
| Tabular | 수치와 범주 피처를 행은 샘플, 열은 피처로 다룬다 |
| Time-series | 센서와 금융, IoT 신호를 시간축에 정렬한 값으로 다룬다 |
| Graph | 노드와 엣지 구조로 소셜, 분자, 지식그래프를 다룬다 |
| Geospatial | 위경도와 지도, 위성영상을 좌표 또는 래스터로 다룬다 |
| 3D / Geometric | 포인트클라우드와 메시, depth 를 공간 좌표로 다룬다 |

### 1.3 Composite

- **Multimodal** 은 두 종류 이상을 결합한다. image 와 text 를 함께 쓰는 VQA 가 대표적이다.
- **Cross-modal** 은 한 모달리티를 다른 모달리티로 변환한다. text 에서 image, speech 에서 text 가 여기에 든다.

### 1.4 Tree View

```text
Data Modality
├── By Structure
│   ├── Structured       (tables)
│   ├── Semi-structured  (JSON / XML / logs)
│   └── Unstructured     (raw signals)
│
├── By Signal
│   ├── Text
│   ├── Vision           (image / video)
│   ├── Audio
│   ├── Tabular
│   ├── Time-series
│   ├── Graph
│   ├── Geospatial
│   └── 3D / Geometric
│
└── Composite
    ├── Multimodal       (combine)
    └── Cross-modal      (translate)
```

같은 데이터도 어느 축으로 보는지에 따라 다르게 묶인다. 이미지는 구조 기준으로 unstructured 이고 신호 기준으로 vision 이다. 그래서 taxonomy 를 쓸 때는 어느 축으로 가르는지 먼저 못박아야 한다.

## 2. Scope And Limits

구조 축은 어떤 데이터에도 걸리므로 사실상 전부 적용된다. 반면 신호 축은 열린 목록이어서 새 도메인이 들어오면 항목을 늘려야 한다.

기존 여덟 칸에 억지로 밀어넣으면 왜곡되는 예는 다음과 같다.

- 강화학습의 trajectory 는 state, action, reward 가 엮인 형태다.
- 생물 서열인 DNA 와 protein 은 토큰 시퀀스이지만 어휘와 제약이 자연어와 다르다.
- 화학 표기인 SMILES 는 문자열이면서 실제로는 그래프다.
- 임베딩과 latent vector 는 원본 신호가 아니라 파생 표현이다.
- event stream 은 시각이 비정규적이어서 균일 시계열로 다루면 손실이 생긴다.

또 실제 데이터는 여러 칸을 동시에 밟는 경우가 많다. 따라서 이 taxonomy 는 배타적 분류표가 아니라 태그 집합으로 써야 맞다.

## 3. Semiconductor Taxonomy

반도체 데이터는 범용 모달리티 목록만으로 부족하다. 측정값이 어느 단위에 붙는지가 1차 구분이 되므로 엔티티 계층 축을 하나 더 둔다.

### 3.1 Axis A — Entity Hierarchy

```text
Fab -> Tool(Chamber) -> Lot -> Wafer -> Die/Site -> Device/Structure
```

| Level | Typical data |
|-------|--------------|
| Fab | 라인 단위 capacity 와 WIP 총량 |
| Tool / Chamber | FDC trace, alarm, PM 이력 |
| Lot | route, priority, queue time |
| Wafer | wafer map, metrology 요약값 |
| Die / Site | bin code, CD 와 overlay 실측점 |
| Device | I–V curve, TEM 단면 |

### 3.2 Axis B — Modality

| Modality | Actual data | Tensor shape |
|----------|-------------|--------------|
| Tabular / Contextual | MES 이력, WIP, genealogy, WAT 요약 | `[row, feature]` |
| Equipment trace | FDC sensor 로 받는 온도와 압력, RF, flow | `[time, sensor]` |
| Spatial wafer map | sort bin map, defect map, thickness map | `[x, y]` 또는 `[site, value]` |
| Vector field | overlay 와 distortion 벡터 | `[x, y, (dx, dy)]` |
| Image | ADC defect 이미지, SEM, optical, TEM | `[H, W, C]` |
| Layout / Geometric | GDSII 폴리곤, mask, OPC 패턴 | vector polygon |
| Curve / Functional | I–V, C–V, SIMS depth profile, AFM profile | `[index, value]` |
| Event stream | equipment alarm, MES event, interlock | `(t, event)` 비정규 시각 |
| Recipe / Config | 공정 recipe, step tree | semi-structured tree |
| Graph / Relational | route flow, tool 과 lot 통과 그래프, netlist | node 와 edge |
| Text | FA 리포트, PM 로그, spec 문서 | 토큰 시퀀스 |

### 3.3 Axis C — Lifecycle Stage

문제 정의를 가를 때 세 번째 축으로 덧붙인다.

```text
Design -> Process -> Metrology -> Inspection -> Test -> Reliability
```

| Stage | Representative data |
|-------|---------------------|
| Design | layout, netlist, OPC 패턴 |
| Process | recipe, FDC trace, alarm |
| Metrology | CD, thickness, overlay 실측값 |
| Inspection | defect map, ADC 이미지 |
| Test | WAT 와 PCM, sort bin map |
| Reliability | burn-in, HTOL, FA 리포트 |

### 3.4 Composite Patterns

실제 분석은 단일 칸이 아니라 조합으로 성립한다.

| Pattern | Composition | Use case |
|---------|-------------|----------|
| Multi-scale time-series | intra-wafer trace 와 inter-wafer drift | tool drift, chamber matching |
| Spatio-temporal | wafer map 과 시간 순서 | 결함 패턴 전이 추적 |
| Trace to map | FDC trace 가 원인, bin map 이 결과 | virtual metrology, yield 회귀 |
| Image with tabular | defect 이미지와 좌표, layer 문맥 | ADC 분류 |
| Layout with image | GDSII 와 SEM | hotspot 검출 |

### 3.5 Tree View

```text
Semiconductor Data
├── Axis A — Entity level : Fab / Tool / Lot / Wafer / Die / Device
├── Axis B — Modality
│   ├── Tabular          (MES, WAT)
│   ├── Trace            (FDC time-series)
│   ├── Spatial map      (bin, defect, thickness)
│   ├── Vector field     (overlay)
│   ├── Image            (ADC, SEM, TEM)
│   ├── Layout           (GDSII, mask)
│   ├── Curve            (I-V, depth profile)
│   ├── Event stream     (alarm, MES log)
│   ├── Recipe tree
│   ├── Graph            (route, netlist)
│   └── Text             (FA report)
├── Axis C — Lifecycle stage
│   ├── Design           (layout, netlist, OPC)
│   ├── Process          (recipe, FDC trace, alarm)
│   ├── Metrology        (CD, thickness, overlay)
│   ├── Inspection       (defect map, ADC image)
│   ├── Test             (WAT/PCM, sort bin map)
│   └── Reliability      (burn-in, HTOL, FA report)
└── Composite
    ├── multi-scale time-series
    ├── spatio-temporal
    └── trace -> map (cause -> effect)
```

핵심은 어느 엔티티 단위에 붙은 어떤 형태인지를 함께 적는 것이다. `wafer × spatial map` 이나 `chamber × trace` 처럼 두 값을 같이 표기한다.

## 4. Case Study — Wafer Process Data

wafer 는 처리 순서가 있고 각 wafer 에는 공정 시계열이 딸려 있다. 이 데이터를 3D time-series 라 부르는 것이 맞는지 따져본다.

### 4.1 Verdict

3D time-series 라는 이름은 쓰지 않는다. 텐서 rank 는 3 이 맞지만 라벨이 오해를 부른다.

### 4.2 Why Misleading

3D 는 관례상 x, y, z 공간 geometry 를 뜻한다. 포인트클라우드와 CT volume 이 그 예다. 그런데 wafer 공정 데이터의 세 번째 축은 공간이 아니라 sensor channel 이거나 wafer index 다. 모양만 보고 3D 라 부르면 듣는 쪽은 공간 부피 데이터로 받아들인다.

### 4.3 Two Time Axes

이 데이터의 핵심은 성격이 다른 두 시간 축이 겹쳐 있다는 점이다.

| Axis | Meaning | Character |
|------|---------|-----------|
| Inter-wafer | wafer 와 lot 처리 순서 | 느린 시간이며 tool aging 과 drift 를 담는다 |
| Intra-wafer | 한 공정 step 안의 trace | 빠른 시간이며 FDC sensor 파형을 담는다 |
| Channel | 센서 종류인 온도, 압력, RF, gas flow | 변수 축이며 시간이 아니다 |

텐서로 펴면 `[wafer, time, sensor]` 의 rank-3 이지만, 의미는 여러 multivariate time-series 묶음에 wafer 순서 drift 가 얹힌 형태다.

### 4.4 Correct Naming

| Scope | Recommended name |
|-------|------------------|
| wafer 한 장 안 | multivariate time-series 또는 multichannel time-series |
| wafer 여러 장 누적 | panel time-series 이며 텐서를 강조하면 tensor time-series |
| 두 시간 축 동시 강조 | hierarchical time-series 또는 multi-scale time-series |
| die 위치까지 포함 | spatio-temporal 이며 이때 비로소 진짜 공간 축이 생긴다 |

```text
Wafer process data
├── intra-wafer : multivariate time-series   (time x sensor)
├── inter-wafer : panel / sequence           (drift across wafers)
└── (optional) die map : spatial x-y         -> spatio-temporal
```

### 4.5 Conclusion

die 공간 좌표가 없는 데이터라면 multi-scale multivariate time-series 인 panel 형태가 가장 정확한 명칭이다. 3D 라는 말은 wafer map 처럼 공간 축이 실제로 들어올 때만 아껴 둔다.
