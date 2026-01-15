"""실패 분석 Streamlit 페이지들을 위한 공유 유틸리티

- 메인 페이지와 날짜별 페이지 간 스키마/정제 로직 일관성 유지
- 모듈 레벨 데이터 처리 방지 (안전한 import)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -------------------------------------------------------------------------
# Streamlit 페이지 설정 (부모 Streamlit 앱에서 import 시 안전)
_PAGE_CONFIG = dict(page_title="실패분석", page_icon="🛣️", layout="wide")

def _maybe_set_page_config() -> None:
    """가능하면 set_page_config를 호출하되, 이미 부모 앱에서 설정된 경우 무시"""
    try:
        st.set_page_config(**_PAGE_CONFIG)
    except Exception:
        # Streamlit은 페이지 설정이 이미 설정되었거나 너무 늦게 호출되면 예외를 발생시킴
        pass


# -------------------------------------------------------------------------
# 표준 컬럼 (고정 스키마)

TS_COL = "Timestamp"                   # 선택 (ms 또는 ISO 문자열)
QUALITY_COL = "Lane Quality Score"     # 0~100
MASK_RATIO_COL = "Mask White Ratio"    # 0~1 (흰 픽셀 / 마스크 픽셀)

ERROR_COL = "Lane Error"               # 부호 있음
ABS_ERROR_COL = "Abs Lane Error"

PROC_COL = "Processing Time (ms)"      # 선택
WEATHER_COL = "Weather"                # 선택
TOD_COL = "Time of Day"                # 선택
MODE_COL = "Mode"                      # 선택

# Fixed schema (업로드 로그는 아래 8개 컬럼명을 '그대로' 사용한다고 가정)
FIXED_LOG_COLS = [
    TS_COL,
    WEATHER_COL,
    TOD_COL,
    MASK_RATIO_COL,
    QUALITY_COL,
    ERROR_COL,
    PROC_COL,
    MODE_COL,
]

# 합성 ID (로드/병합 시 생성)
RUN_ID_COL = "Run ID"
ROW_IN_RUN_COL = "Row In Run"
EVENT_ID_COL = "Event ID"


# -------------------------------------------------------------------------
# 핵심 정제 / ID 생성

def _select_fixed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """고정 컬럼을 안정적인 순서로 선택하고, 누락된 컬럼이 있으면 즉시 실패"""
    missing = [c for c in FIXED_LOG_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {', '.join(missing)}")
    return df[FIXED_LOG_COLS].copy()


def _coerce_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    """모든 페이지에서 사용하는 최소한의 데이터 타입 정규화 (비파괴적)"""
    d = df.copy()

    # 타임스탬프
    if TS_COL in d.columns:
        d[TS_COL] = pd.to_numeric(d[TS_COL], errors="coerce")

    # 숫자 필드
    if ERROR_COL in d.columns:
        d[ERROR_COL] = pd.to_numeric(d[ERROR_COL], errors="coerce")
    if PROC_COL in d.columns:
        d[PROC_COL] = pd.to_numeric(d[PROC_COL], errors="coerce")

    # 텍스트 필드
    for c in [WEATHER_COL, TOD_COL, MODE_COL]:
        if c in d.columns:
            d[c] = d[c].astype("string").fillna("Unknown")

    return d


def _ensure_fields(df: pd.DataFrame) -> pd.DataFrame:
    """주요 파생 컬럼이 존재하는지 확인하고 핵심 범위를 정규화"""
    d = _coerce_and_fill(df)

    if QUALITY_COL in d.columns:
        q = pd.to_numeric(d[QUALITY_COL], errors="coerce")
        d[QUALITY_COL] = q.clip(0, 100)

    if MASK_RATIO_COL in d.columns:
        r = pd.to_numeric(d[MASK_RATIO_COL], errors="coerce")
        # 0~100(%) 가능성 처리
        r = np.where(r > 1.5, r / 100.0, r)
        d[MASK_RATIO_COL] = pd.to_numeric(r, errors="coerce").clip(0, 1)

    if ERROR_COL in d.columns and ABS_ERROR_COL not in d.columns:
        e = pd.to_numeric(d[ERROR_COL], errors="coerce")
        d[ABS_ERROR_COL] = e.abs()

    # 선택적 기본값 (이전 CSV와의 호환성 유지)
    if WEATHER_COL not in d.columns:
        d[WEATHER_COL] = "Unknown"
    if TOD_COL not in d.columns:
        d[TOD_COL] = "Unknown"
    if MODE_COL not in d.columns:
        d[MODE_COL] = "Unknown"

    return d


def _add_event_ids_per_run(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    d = df.copy()
    d[RUN_ID_COL] = run_id
    d[ROW_IN_RUN_COL] = np.arange(len(d), dtype=int)
    d[EVENT_ID_COL] = d[RUN_ID_COL].astype(str) + "_" + d[ROW_IN_RUN_COL].astype(str).str.zfill(6)
    return d


def _make_tooltip(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    """df에 실제로 존재하는 툴팁 컬럼 반환 (순서 유지)"""
    return [c for c in wanted if c in df.columns]


# -------------------------------------------------------------------------
# 진단 / 통계 헬퍼

def _describe_missing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df.columns:
            rows.append({"column": c, "present": False, "missing_rate": 1.0, "dtype": "N/A"})
        else:
            miss = df[c].isnull().mean()
            rows.append({"column": c, "present": True, "missing_rate": float(miss), "dtype": str(df[c].dtype)})

    res = pd.DataFrame(rows)
    if not res.empty:
        res["missing_%"] = (res["missing_rate"] * 100).round(2)
        res = res.drop(columns=["missing_rate"])
    return res


def perform_linear_regression(df: pd.DataFrame, x_col: str, y_col: str, sigma_threshold: float) -> pd.DataFrame:
    clean_df = df.dropna(subset=[x_col, y_col]).copy()
    if clean_df.empty:
        clean_df["Status"] = "In Range"
        return clean_df

    x = clean_df[x_col].to_numpy()
    y = clean_df[y_col].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)
    predictions = (slope * x) + intercept
    residuals = y - predictions
    std_dev = float(np.std(residuals)) if len(residuals) else 0.0

    upper_bound = predictions + (sigma_threshold * std_dev)
    lower_bound = predictions - (sigma_threshold * std_dev)

    clean_df["Predicted"] = predictions
    clean_df["Upper Bound"] = upper_bound
    clean_df["Lower Bound"] = lower_bound
    clean_df["Status"] = np.where(
        (clean_df[y_col] > upper_bound) | (clean_df[y_col] < lower_bound),
        "Outlier",
        "In Range",
    )
    return clean_df


def draw_histogram(df: pd.DataFrame, metric_name: str, bins: int = 20, height: int = 220) -> None:
    clean_df = df.dropna(subset=[metric_name])
    if clean_df.empty:
        st.info(f"No data for {metric_name}")
        return

    st.altair_chart(
        alt.Chart(clean_df, height=height)
        .mark_bar(binSpacing=0)
        .encode(
            alt.X(metric_name, type="quantitative").bin(maxbins=bins),
            alt.Y("count()").axis(None),
        ),
        use_container_width=True,
    )


# -------------------------------------------------------------------------
# 고정 CSV 로딩 (날짜별 페이지에서 사용)

def _find_csv_file(filename: str, base_dir: Path | None = None) -> Path:
    """고정 CSV 파일의 일반적인 위치를 검색하고 첫 번째로 발견된 경로 반환"""
    base_dir = base_dir or Path.cwd()
    candidates: list[Path] = [
        Path("data") / filename,
        Path(filename),
        base_dir / "data" / filename,
        base_dir / filename,
    ]
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        raise FileNotFoundError(
            f"고정 데이터 파일을 찾을 수 없습니다: {filename}\n"
            "다음 위치 중 하나에 CSV를 두세요: ./data/, 현재 작업폴더, 또는 이 스크립트와 같은 폴더/그 하위 data 폴더."
        )
    return found


def load_fixed_csv(filename: str, *, run_id: str, base_dir: Path | None = None) -> pd.DataFrame:
    """고정 스키마 CSV를 로드하고 정규화한 뒤 합성 ID 추가"""
    p = _find_csv_file(filename, base_dir=base_dir)
    d = pd.read_csv(p)
    d = _select_fixed_columns(d)
    d = _ensure_fields(d)
    d = _add_event_ids_per_run(d, run_id=run_id)
    return d


def try_load_fixed_csv(filename: str, *, run_id: str, base_dir: Path | None = None) -> pd.DataFrame:
    """load_fixed_csv와 유사하지만, 파일이 없으면 빈 DataFrame 반환"""
    try:
        return load_fixed_csv(filename, run_id=run_id, base_dir=base_dir)
    except FileNotFoundError:
        return pd.DataFrame(columns=list(FIXED_LOG_COLS) + [ABS_ERROR_COL, RUN_ID_COL, ROW_IN_RUN_COL, EVENT_ID_COL])
