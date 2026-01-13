"""Shared utilities for failure_analysis Streamlit pages.

- Keeps schema/cleaning logic consistent across main + date-specific pages
- Avoids module-level data processing (safe to import)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -------------------------------------------------------------------------
# Streamlit page config (safe when imported by a parent Streamlit app)
_PAGE_CONFIG = dict(page_title="실패분석", page_icon="🛣️", layout="wide")

def _maybe_set_page_config() -> None:
    """Call set_page_config if possible; ignore if already set by parent app."""
    try:
        st.set_page_config(**_PAGE_CONFIG)
    except Exception:
        # Streamlit raises if page config is already set or called too late.
        pass


# -------------------------------------------------------------------------
# Canonical columns (FIXED SCHEMA)

TS_COL = "Timestamp"                   # optional (ms or ISO string)
QUALITY_COL = "Lane Quality Score"     # 0~100
MASK_RATIO_COL = "Mask White Ratio"    # 0~1 (white pixels / mask pixels)

ERROR_COL = "Lane Error"               # signed
ABS_ERROR_COL = "Abs Lane Error"

PROC_COL = "Processing Time (ms)"      # optional
WEATHER_COL = "Weather"                # optional
TOD_COL = "Time of Day"                # optional
MODE_COL = "Mode"                      # optional

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

# Synthetic IDs (created at load/merge time)
RUN_ID_COL = "Run ID"
ROW_IN_RUN_COL = "Row In Run"
EVENT_ID_COL = "Event ID"


# -------------------------------------------------------------------------
# Core cleaning / ids

def _select_fixed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select fixed columns in a stable order and fail fast if any are missing."""
    missing = [c for c in FIXED_LOG_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {', '.join(missing)}")
    return df[FIXED_LOG_COLS].copy()


def _coerce_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal dtype normalization used by all pages (non-breaking)."""
    d = df.copy()

    # Timestamp
    if TS_COL in d.columns:
        d[TS_COL] = pd.to_numeric(d[TS_COL], errors="coerce")

    # Numeric fields
    if ERROR_COL in d.columns:
        d[ERROR_COL] = pd.to_numeric(d[ERROR_COL], errors="coerce")
    if PROC_COL in d.columns:
        d[PROC_COL] = pd.to_numeric(d[PROC_COL], errors="coerce")

    # Text fields
    for c in [WEATHER_COL, TOD_COL, MODE_COL]:
        if c in d.columns:
            d[c] = d[c].astype("string").fillna("Unknown")

    return d


def _ensure_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure key derived columns exist and normalize core ranges."""
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

    # Optional defaults (keep compatibility with older CSVs)
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
    """Return tooltip columns that actually exist in df (keeps order)."""
    return [c for c in wanted if c in df.columns]


# -------------------------------------------------------------------------
# Diagnostics / stats helpers

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
# Fixed CSV loading (used by date pages)

def _find_csv_file(filename: str, base_dir: Path | None = None) -> Path:
    """Search common locations for a fixed CSV file and return the first existing path."""
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
    """Load + normalize a fixed-schema CSV and add synthetic IDs."""
    p = _find_csv_file(filename, base_dir=base_dir)
    d = pd.read_csv(p)
    d = _select_fixed_columns(d)
    d = _ensure_fields(d)
    d = _add_event_ids_per_run(d, run_id=run_id)
    return d


def try_load_fixed_csv(filename: str, *, run_id: str, base_dir: Path | None = None) -> pd.DataFrame:
    """Like load_fixed_csv, but returns an empty DataFrame if file is missing."""
    try:
        return load_fixed_csv(filename, run_id=run_id, base_dir=base_dir)
    except FileNotFoundError:
        return pd.DataFrame(columns=list(FIXED_LOG_COLS) + [ABS_ERROR_COL, RUN_ID_COL, ROW_IN_RUN_COL, EVENT_ID_COL])
