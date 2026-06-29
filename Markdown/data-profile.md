# 🔍 데이터 자동 분석 (Data Profiling)

> 새 데이터를 받으면 사람한테 물어보지 말고 **스크립트로 구조를 자동 파악**한다.
> X/y 위치, shape, labels, dtype, scalar vs category, data length, 결측·불균형까지 한 번에.

## 쓰는 법

```bash
# X와 y가 한 파일에 있고, 타깃 컬럼명을 알 때
python data_profiler.py data.csv --target price

# 타깃을 몰라도 일단 전체 구조만 보고 싶을 때
python data_profiler.py data.csv

# X, y 가 별도 파일일 때
python data_profiler.py X.csv --y y.csv

# 리포트를 .md 파일로 저장
python data_profiler.py data.parquet --target label --out report.md
```

지원 포맷: `.csv .tsv .parquet .json .xlsx .npy`

## 리포트가 자동으로 답해주는 질문

| 영역 | 자동 산출 항목 |
|------|----------------|
| **위치 & 규모** | X/y 파일, data length(N), 컬럼 수, 중복 행, 메모리 |
| **y (타깃)** | y label, y shape, dtype, **scalar(회귀) vs category(분류·클래스 수)**, 결측, 클래스 분포, 불균형 경고 |
| **X (입력)** | X shape, 컬럼별 kind(연속/이산/범주/텍스트/날짜), dtype, nunique, 결측%, 샘플값 |
| **전처리 힌트** | 결측 30%↑, 고카디널리티, 상수 컬럼, 스케일 편차 자동 경고 |

## 판정 규칙 (요약)

- **scalar(회귀)** ↔ 수치형이면서 고유값 > 20
- **category(분류)** ↔ 고유값 ≤ 20 또는 비수치형 → 고유값 2면 binary, 그 이상이면 multiclass
- **numeric(discrete→category?)** ↔ 수치형인데 고유값이 적음 → 인코딩 여부 판단 필요
- 임계값은 스크립트 상단 `CARDINALITY_CAT_LIMIT = 20` 으로 조정

## 출력 예시 (형식)

```text
# 📊 데이터 자동 분석 리포트

## 1. 위치 & 규모
- data length (행 수, N): 10,000
- 전체 컬럼 수: 14
...

## 2. 타깃 y
- y label: `price`
- 문제 유형: 회귀 (scalar, 연속값, nunique=842)
...

## 3. 입력 X
- X shape: (10000, 13)
| column | kind                | dtype  | nunique | missing% | sample |
...

## 4. 전처리 체크
- ⚠️ 고카디널리티 범주형 → target/embedding encoding: ['zipcode']
```
