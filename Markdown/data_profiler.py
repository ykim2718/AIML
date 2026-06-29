"""
data_profiler.py — 새 데이터를 받았을 때 AI/ML 적용 전에 구조를 자동 분석한다.

사용법:
    python data_profiler.py data.csv                 # X/y 합쳐진 파일
    python data_profiler.py data.csv --target price   # 타깃 컬럼 지정
    python data_profiler.py X.csv --y y.csv           # X, y 별도 파일
    python data_profiler.py data.parquet --out report.md   # md 리포트 저장

지원 포맷: .csv .tsv .parquet .json .xlsx .npy
출력: 콘솔에 마크다운 리포트 출력 (--out 주면 파일로도 저장)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 범주형으로 간주할 고유값 상한 (이 이하의 nunique 면 category 후보)
CARDINALITY_CAT_LIMIT = 20


def load_any(path: str) -> pd.DataFrame:
    """확장자에 맞춰 파일을 DataFrame 으로 읽는다."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".csv",):
        return pd.read_csv(p)
    if suf in (".tsv",):
        return pd.read_csv(p, sep="\t")
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if suf in (".json",):
        return pd.read_json(p)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if suf in (".npy",):
        arr = np.load(p)
        cols = [f"f{i}" for i in range(arr.shape[1])] if arr.ndim == 2 else ["value"]
        return pd.DataFrame(arr.reshape(len(arr), -1), columns=cols)
    raise ValueError(f"지원하지 않는 포맷: {suf}")


def infer_kind(s: pd.Series) -> str:
    """컬럼 1개의 데이터 종류를 추론한다."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_bool_dtype(s):
        return "category(bool)"
    if pd.api.types.is_numeric_dtype(s):
        # 수치형이지만 고유값이 적으면 범주형일 수 있음
        if s.nunique(dropna=True) <= CARDINALITY_CAT_LIMIT:
            return "numeric(discrete→category?)"
        return "numeric(continuous)"
    return "category(text)"


def target_problem_type(y: pd.Series) -> str:
    """타깃이 회귀(scalar)인지 분류(category)인지 판정한다."""
    n = y.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(y) and n > CARDINALITY_CAT_LIMIT:
        return f"회귀 (scalar, 연속값, nunique={n})"
    if n == 2:
        return "분류 — 이진 (binary)"
    return f"분류 — 다중 ({n} classes)"


def md_table(df: pd.DataFrame) -> str:
    """DataFrame 을 마크다운 표 문자열로."""
    return df.to_markdown(index=False)


def profile(df: pd.DataFrame, target: str | None, x_src: str, y_src: str | None) -> str:
    lines: list[str] = []
    add = lines.append

    add("# 📊 데이터 자동 분석 리포트\n")

    # 1. 위치 & 규모 ---------------------------------------------------------
    add("## 1. 위치 & 규모 (Where & Scale)\n")
    add(f"- **X 파일**: `{x_src}`")
    add(f"- **y 파일**: `{y_src or x_src}`" + ("" if y_src else " (X와 동일 파일)"))
    add(f"- **data length (행 수, N)**: {len(df):,}")
    add(f"- **전체 컬럼 수**: {df.shape[1]}")
    add(f"- **중복 행**: {df.duplicated().sum():,}")
    add(f"- **전체 메모리**: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB\n")

    # 2. 타깃(y) -------------------------------------------------------------
    feature_cols = list(df.columns)
    add("## 2. 타깃 y (Target)\n")
    if target and target in df.columns:
        y = df[target]
        feature_cols = [c for c in df.columns if c != target]
        add(f"- **y label**: `{target}`")
        add(f"- **y shape**: ({len(y)},)")
        add(f"- **dtype**: {y.dtype}")
        add(f"- **문제 유형**: {target_problem_type(y)}")
        add(f"- **결측치**: {y.isna().sum():,} ({y.isna().mean():.1%})")
        vc = y.value_counts(normalize=True, dropna=False).head(10)
        add("- **분포 (상위 10):**\n")
        dist = pd.DataFrame({"value": vc.index.astype(str), "ratio": vc.values.round(4)})
        add(md_table(dist) + "\n")
        # 불균형 경고
        if "분류" in target_problem_type(y):
            mn = y.value_counts(normalize=True).min()
            if mn < 0.1:
                add(f"> ⚠️ **클래스 불균형**: 최소 클래스 비율 {mn:.1%} → resampling/class_weight 고려\n")
    else:
        add("- ⚠️ 타깃 컬럼 미지정 (`--target` 으로 지정). 아래는 X 전체 기준 분석.\n")

    # 3. 입력 X --------------------------------------------------------------
    X = df[feature_cols]
    add("## 3. 입력 X (Features)\n")
    add(f"- **X shape**: ({X.shape[0]}, {X.shape[1]})")
    num = X.select_dtypes(include=np.number).columns
    add(f"- **수치형 컬럼 수**: {len(num)}  /  **비수치형**: {X.shape[1] - len(num)}\n")

    rows = []
    for c in X.columns:
        s = X[c]
        rows.append({
            "column": c,
            "kind": infer_kind(s),
            "dtype": str(s.dtype),
            "nunique": s.nunique(dropna=True),
            "missing%": round(s.isna().mean() * 100, 1),
            "sample": str(s.dropna().iloc[0]) if s.notna().any() else "-",
        })
    add("### 컬럼별 요약\n")
    add(md_table(pd.DataFrame(rows)) + "\n")

    # 4. 전처리 힌트 ---------------------------------------------------------
    add("## 4. 전처리 체크 (Auto Hints)\n")
    high_miss = [r["column"] for r in rows if r["missing%"] > 30]
    high_card = [r["column"] for r in rows
                 if "category" in r["kind"] and r["nunique"] > 50]
    const_cols = [r["column"] for r in rows if r["nunique"] <= 1]
    if high_miss:
        add(f"- ⚠️ **결측 30%↑** → imputation/drop 검토: {high_miss}")
    if high_card:
        add(f"- ⚠️ **고카디널리티 범주형** → target/embedding encoding: {high_card}")
    if const_cols:
        add(f"- ⚠️ **상수 컬럼 (정보 없음)** → drop: {const_cols}")
    if len(num) and X[num].std().max() / max(X[num].std().min(), 1e-9) > 100:
        add("- ⚠️ **피처 간 스케일 편차 큼** → 정규화/표준화 필요")
    if not (high_miss or high_card or const_cols):
        add("- ✅ 명백한 위험 신호 없음 (세부 EDA 권장)")
    add("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="새 데이터 구조 자동 분석 → 마크다운 리포트")
    ap.add_argument("data", help="X (또는 X+y 합쳐진) 데이터 파일")
    ap.add_argument("--target", help="타깃(y) 컬럼명 (X와 같은 파일일 때)")
    ap.add_argument("--y", dest="y_file", help="y 가 별도 파일일 때 경로")
    ap.add_argument("--out", help="마크다운 리포트 저장 경로 (.md)")
    args = ap.parse_args()

    # Windows 콘솔(cp949)에서도 이모지/유니코드가 깨지지 않도록 UTF-8 강제
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    df = load_any(args.data)
    y_src = None
    if args.y_file:
        ydf = load_any(args.y_file)
        ycol = ydf.columns[0]
        df = df.reset_index(drop=True)
        df[ycol] = ydf[ycol].reset_index(drop=True)
        args.target = ycol
        y_src = args.y_file

    report = profile(df, args.target, args.data, y_src)
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\n[저장됨] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
