import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Page config (safe when imported by a parent Streamlit app)
_PAGE_CONFIG = dict(page_title="실패분석", page_icon="🛣️", layout="wide")

def _maybe_set_page_config() -> None:
    """Call set_page_config if possible; ignore if already set by parent app."""
    try:
        st.set_page_config(**_PAGE_CONFIG)
    except Exception:
        # Streamlit raises if page config is already set or called too late.
        pass

# Canonical columns (FIXED SCHEMA)
TS_COL = "Timestamp"                   # optional (ms or ISO string)

QUALITY_COL = "Lane Quality Score"     # 0~100 (현재 logger 구현에서는 ratio의 스케일링일 수 있음)
MASK_RATIO_COL = "Mask White Ratio"    # 0~1 (white pixels / mask pixels)

ERROR_COL = "Lane Error"               # signed 
ABS_ERROR_COL = "Abs Lane Error"

PROC_COL = "Processing Time (ms)"      # optional
WEATHER_COL = "Weather"                # optional
TOD_COL = "Time of Day"                # optional
MODE_COL = "Mode"                    # optional 


# Synthetic IDs (created at load/merge time)
RUN_ID_COL = "Run ID"
ROW_IN_RUN_COL = "Row In Run"
EVENT_ID_COL = "Event ID"

# -----------------------------------------------------------------------------
_FIXED_FILENAMES = ["0112_11_log.csv"]

_FIXED_COLUMNS = [
    TS_COL,            # "Timestamp"
    WEATHER_COL,       # "Weather"
    TOD_COL,           # "Time of Day"
    MASK_RATIO_COL,    # "Mask White Ratio"
    QUALITY_COL,       # "Lane Quality Score"
    ERROR_COL,         # "Lane Error"
    PROC_COL,          # "Processing Time (ms)"
    MODE_COL,          # "Mode"
]

@st.cache_data(show_spinner=False)
def _load_fixed_data() -> pd.DataFrame:
    """Load the fixed 0112 dataset (fixed schema; no column-name preprocessing)."""
    base = Path(__file__).resolve().parent
    candidates: list[Path] = []
    for name in _FIXED_FILENAMES:
        candidates.extend([
            Path("data") / name,
            Path(name),
            base / "data" / name,
            base / name,
        ])

    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        raise FileNotFoundError(
            "0112 고정 데이터 파일을 찾을 수 없습니다. "
            "다음 위치 중 하나에 '0112_11_log.csv'를 두세요: "
            "./data/, 현재 폴더, 또는 이 스크립트와 같은 폴더/그 하위 data 폴더."
        )

    d = pd.read_csv(found)

    # 1) 고정 컬럼 존재 체크 (누락이면 바로 에러)
    missing = [c for c in _FIXED_COLUMNS if c not in d.columns]
    if missing:
        raise ValueError(
            "CSV에 필요한 고정 컬럼이 없습니다.\n"
            f"- 누락: {', '.join(missing)}\n"
            f"- 필요: {', '.join(_FIXED_COLUMNS)}"
        )

    # 2) 컬럼 순서/필요 컬럼만 유지
    d = d[_FIXED_COLUMNS].copy()

    # 3) 타입/값 최소 정리 (컬럼명 전처리는 제거했지만, 분석에 필요한 값 정리는 유지)
    # Timestamp (가능하면 숫자화)
    d[TS_COL] = pd.to_numeric(d[TS_COL], errors="coerce")

    # Mask White Ratio: 0~1로 정규화 (0~100 들어오면 /100)
    r = pd.to_numeric(d[MASK_RATIO_COL], errors="coerce")
    r = np.where(r > 1.5, r / 100.0, r)  # 0~100(%) 가능성 처리
    d[MASK_RATIO_COL] = pd.to_numeric(r, errors="coerce").clip(0, 1)

    # Lane Quality Score: 0~100 클립
    q = pd.to_numeric(d[QUALITY_COL], errors="coerce")
    d[QUALITY_COL] = q.clip(0, 100)

    # Lane Error + Abs Lane Error 생성 (Part I에서 필요)
    e = pd.to_numeric(d[ERROR_COL], errors="coerce")
    d[ERROR_COL] = e
    d[ABS_ERROR_COL] = e.abs()

    # Processing Time (ms)
    d[PROC_COL] = pd.to_numeric(d[PROC_COL], errors="coerce")

    # 문자열 컬럼 결측 채움
    for c in [WEATHER_COL, TOD_COL, MODE_COL]:
        d[c] = d[c].astype("string").fillna("Unknown")

    # 4) Event ID 생성 (기존 로직 유지)
    d = _add_event_ids_per_run(d, run_id="run_0112_fixed")
    return d


# -----------------------------------------------------------------------------
# 0109 baseline (for Part X comparison)
_BASELINE_0109_FILENAMES = ["0109_11_log.csv"]

@st.cache_data(show_spinner=False)
def _load_fixed_data_0109_baseline() -> pd.DataFrame:
    """Load the fixed 0109 dataset (same fixed schema) for comparison in Part X.

    If the baseline CSV is not present, Part X will simply skip the comparison section.
    """
    base_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    for name in _BASELINE_0109_FILENAMES:
        candidates.extend([
            Path("data") / name,
            Path(name),
            base_dir / "data" / name,
            base_dir / name,
        ])

    found = next((pp for pp in candidates if pp.exists()), None)
    if found is None:
        raise FileNotFoundError(
            "0109 비교용 데이터 파일을 찾을 수 없습니다. "
            "Part X에서 0109 대비 비교를 하려면 '0109_11_log.csv'를 "
            "./data/, 현재 폴더, 또는 이 스크립트와 같은 폴더/그 하위 data 폴더에 두세요."
        )

    d = pd.read_csv(found)

    # 1) 고정 컬럼 존재 체크
    missing = [c for c in _FIXED_COLUMNS if c not in d.columns]
    if missing:
        raise ValueError(
            "0109 CSV에 필요한 고정 컬럼이 없습니다.\n"
            f"- 누락: {', '.join(missing)}\n"
            f"- 필요: {', '.join(_FIXED_COLUMNS)}"
        )

    # 2) 컬럼 순서/필요 컬럼만 유지
    d = d[_FIXED_COLUMNS].copy()

    # 3) 타입/값 최소 정리 (0112과 동일한 규칙)
    d[TS_COL] = pd.to_numeric(d[TS_COL], errors="coerce")

    r = pd.to_numeric(d[MASK_RATIO_COL], errors="coerce")
    r = np.where(r > 1.5, r / 100.0, r)
    d[MASK_RATIO_COL] = pd.to_numeric(r, errors="coerce").clip(0, 1)

    q = pd.to_numeric(d[QUALITY_COL], errors="coerce")
    d[QUALITY_COL] = q.clip(0, 100)

    e = pd.to_numeric(d[ERROR_COL], errors="coerce")
    d[ERROR_COL] = e
    d[ABS_ERROR_COL] = e.abs()

    d[PROC_COL] = pd.to_numeric(d[PROC_COL], errors="coerce")

    for c in [WEATHER_COL, TOD_COL, MODE_COL]:
        d[c] = d[c].astype("string").fillna("Unknown")

    d = _add_event_ids_per_run(d, run_id="run_0109_fixed")
    return d

from pathlib import Path
import re
import textwrap

# =============================================================================
# Helpers

def _make_tooltip(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    """Return tooltip columns that actually exist in df (keeps order)."""
    return [c for c in wanted if c in df.columns]


def _add_event_ids_per_run(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    d = df.copy()
    d[RUN_ID_COL] = run_id
    d[ROW_IN_RUN_COL] = np.arange(len(d), dtype=int)
    d[EVENT_ID_COL] = d[RUN_ID_COL].astype(str) + "_" + d[ROW_IN_RUN_COL].astype(str).str.zfill(6)
    return d


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
        "In Range"
    )
    return clean_df


def draw_histogram(df: pd.DataFrame, metric_name: str, bins: int = 20, height: int = 220):
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


# =============================================================================

def render() -> None:
    _maybe_set_page_config()
    # UI

    st.title("0112 자율주행 실패 분석")
    st.caption("Mask White Ratio / Lane Error / Processing Time (ms)를 중심으로 분석합니다.")

    df = pd.DataFrame()
    try:
        df = _load_fixed_data()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"CSV 로드 실패: {e}")
        st.stop()

    # Required checks
    missing = [c for c in [MASK_RATIO_COL] if c not in df.columns]
    if missing:
        st.error(
            "필수 컬럼이 없습니다. (이 버전은 Mask White Ratio 중심으로 분석합니다.)\n\n"
            f"- 필수: {MASK_RATIO_COL}\n"
            f"- 누락: {', '.join(missing)}\n\n"
            "※ ratio는 0~1 또는 0~100(%) 모두 허용되며 자동 정규화됩니다."
        )
        st.stop()

    # Timestamp is recommended for time-based interpretation
    if TS_COL not in df.columns:
        st.warning("Timestamp 컬럼이 없습니다. 이벤트 식별은 Event ID로 가능하지만, 시간 기반 해석(구간/추세)은 제한될 수 있습니다.")

    # Column config
    COLUMN_CONFIG = {
        TS_COL: st.column_config.NumberColumn(format='%.0f'),
        QUALITY_COL: st.column_config.ProgressColumn(min_value=0, max_value=100, format="compact", width=130),
        MASK_RATIO_COL: st.column_config.NumberColumn(format="%.4f"),
        ERROR_COL: st.column_config.NumberColumn(format="%.2f"),
        ABS_ERROR_COL: st.column_config.NumberColumn(format="%.2f"),
        PROC_COL: st.column_config.NumberColumn(format="%.1f ms"),
        WEATHER_COL: st.column_config.TextColumn(),
        TOD_COL: st.column_config.TextColumn(),
    }

    def _column_config_for(df_or_cols) -> dict:
        """Filter COLUMN_CONFIG to only columns that exist (avoids errors when optional cols are missing)."""
        cols = df_or_cols.columns if hasattr(df_or_cols, "columns") else list(df_or_cols)
        return {k: v for k, v in COLUMN_CONFIG.items() if k in cols}

    # =============================================================================
    # ===============================================
    # Part 0: sanity checks

    st.divider()
    st.subheader("Part 0: 컬럼/결측/커버리지 확인")

    # 핵심: 이 버전은 Mask White Ratio를 중심으로 보며, Lane Error/Processing Time은 있으면 더 강한 판단이 가능합니다.
    check_cols = [TS_COL, WEATHER_COL, TOD_COL, MASK_RATIO_COL, ERROR_COL, PROC_COL, QUALITY_COL]
    st.dataframe(_describe_missing(df, check_cols), hide_index=True, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        r_med = pd.to_numeric(df[MASK_RATIO_COL], errors="coerce").median()
        st.metric("Mask Ratio (median)", f"{float(r_med):.4f}" if pd.notna(r_med) else "N/A")
    with c2:
        if ERROR_COL in df.columns:
            cov = 1.0 - float(df[ERROR_COL].isna().mean())
            st.metric("Lane Error coverage", f"{cov*100:.1f}%")
        else:
            st.metric("Lane Error coverage", "N/A")
    with c3:
        if PROC_COL in df.columns:
            p_med = pd.to_numeric(df[PROC_COL], errors="coerce").median()
            st.metric("Proc Time (median)", f"{float(p_med):.1f} ms" if pd.notna(p_med) else "N/A")
        else:
            st.metric("Proc Time (median)", "N/A")


    # ---- Lane Error missingness patterns (data-driven)
    st.markdown("### Lane Error 결측 패턴 분석 (데이터 기반)")

    if ERROR_COL not in df.columns:
        st.info("Lane Error 컬럼이 없어 결측(NA) 패턴을 분석할 수 없습니다.")
    else:
        err_missing = df[ERROR_COL].isna()
        miss_rate = float(err_missing.mean())
        pres_rate = 1.0 - miss_rate

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Lane Error missing", f"{miss_rate*100:.1f}%")
        with m2:
            st.metric("Lane Error present", f"{pres_rate*100:.1f}%")
        with m3:
            if MODE_COL in df.columns:
                st.metric("Mode 종류 수", f"{df[MODE_COL].nunique(dropna=True)}")
            else:
                st.metric("Mode 종류 수", "N/A")

        tab_overview, tab_bins, tab_env = st.tabs(["개요", "Mask Ratio 구간", "환경"])

        with tab_overview:
            # Compare mask ratio distributions by missingness
            if MASK_RATIO_COL in df.columns:
                tmp = df[[MASK_RATIO_COL]].copy()
                tmp[MASK_RATIO_COL] = pd.to_numeric(tmp[MASK_RATIO_COL], errors="coerce").clip(0, 1)
                tmp["Error Recorded"] = np.where(err_missing, "Missing", "Present")
                tmp = tmp.dropna(subset=[MASK_RATIO_COL])

                st.altair_chart(
                    alt.Chart(tmp)
                    .mark_bar(opacity=0.65)
                    .encode(
                        x=alt.X(f"{MASK_RATIO_COL}:Q", bin=alt.Bin(maxbins=40), title="Mask White Ratio"),
                        y=alt.Y("count()", title="Frames"),
                        color=alt.Color("Error Recorded:N"),
                        tooltip=[
                            alt.Tooltip("Error Recorded:N"),
                            alt.Tooltip("count():Q", title="frames"),
                        ],
                    )
                    .properties(height=260, title="Mask White Ratio 분포 (Lane Error 기록 여부별)"),
                    use_container_width=True,
                )

                # Quick quantiles for missing vs present
                q = (
                    tmp.groupby("Error Recorded")[MASK_RATIO_COL]
                    .quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
                    .unstack()
                    .reset_index()
                )
                q.columns = ["Error Recorded", "p01", "p05", "p25", "p50", "p75", "p95", "p99"]
                st.dataframe(q, hide_index=True, use_container_width=True)
            else:
                st.info("Mask White Ratio 컬럼이 없어 분포 비교를 생략합니다.")

        with tab_bins:
            st.caption("Mask White Ratio 구간별로 Lane Error 결측률(%)을 계산합니다.")
            preset = st.selectbox(
                "구간 프리셋",
                options=["기본", "세밀", "간단"],
                index=0,
                help="세밀: 저비율 영역을 더 촘촘히 나눔 / 간단: 큰 구간으로만 요약",
            )
            if preset == "세밀":
                edges = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0]
            elif preset == "간단":
                edges = [0.0, 0.01, 0.05, 0.1, 0.2, 1.0]
            else:
                edges = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0]

            # Threshold highlight (matches earlier analysis style)
            low_th = st.slider("저비율 강조 임계값", min_value=0.0, max_value=0.2, value=0.01, step=0.001)

            r = pd.to_numeric(df[MASK_RATIO_COL], errors="coerce").clip(0, 1)
            b = pd.cut(r, bins=edges, include_lowest=True)
            bdf = pd.DataFrame({"ratio_bin": b, "err_missing": err_missing})

            bin_summary = (
                bdf.groupby("ratio_bin", dropna=False)
                .agg(frames=("err_missing", "size"), missing=("err_missing", "sum"))
                .reset_index()
            )
            bin_summary["missing_%"] = (bin_summary["missing"] / bin_summary["frames"] * 100).replace([np.inf, -np.inf], np.nan)
            bin_summary["missing_%"] = bin_summary["missing_%"].round(2)

            st.altair_chart(
                alt.Chart(bin_summary.dropna(subset=["ratio_bin"]))
                .mark_bar()
                .encode(
                    x=alt.X("ratio_bin:N", title="Mask White Ratio bin", sort=None),
                    y=alt.Y("missing_%:Q", title="Lane Error missing (%)", scale=alt.Scale(domain=[0, 100])),
                    tooltip=[
                        alt.Tooltip("ratio_bin:N", title="bin"),
                        alt.Tooltip("frames:Q", title="frames"),
                        alt.Tooltip("missing:Q", title="missing"),
                        alt.Tooltip("missing_%:Q", title="missing %"),
                    ],
                )
                .properties(height=320, title="Mask White Ratio 구간별 Lane Error 결측률"),
                use_container_width=True,
            )
            st.dataframe(bin_summary, hide_index=True, use_container_width=True)

            # Highlight: missing rate under low_th
            low_mask = r.le(low_th)
            if low_mask.any():
                st.write(
                    f"- ratio ≤ {low_th:.3f} 구간: frames={int(low_mask.sum())}, missing={int((err_missing & low_mask).sum())}, "
                    f"missing%={(float((err_missing & low_mask).mean())*100):.1f}%"
                )
            else:
                st.write(f"- ratio ≤ {low_th:.3f} 구간: 해당 프레임이 없습니다.")

        with tab_env:
            env_groups = [g for g in [WEATHER_COL, TOD_COL] if g in df.columns]
            if not env_groups:
                st.info("Weather/Time of Day 컬럼이 없어 환경별 결측률 분석을 생략합니다.")
            else:
                gcol = st.selectbox("환경 그룹 기준", options=env_groups, index=0)
                ge = (
                    df.assign(err_missing=err_missing)
                    .groupby(gcol, dropna=False)["err_missing"]
                    .agg(frames="size", missing="sum")
                    .reset_index()
                )
                ge["missing_%"] = (ge["missing"] / ge["frames"] * 100).round(2)
                st.altair_chart(
                    alt.Chart(ge)
                    .mark_bar()
                    .encode(
                        x=alt.X(f"{gcol}:N", sort="-y", axis=alt.Axis(labelAngle=-20)),
                        y=alt.Y("missing_%:Q", scale=alt.Scale(domain=[0, 100])),
                        tooltip=[gcol, "frames:Q", "missing:Q", "missing_%:Q"],
                    )
                    .properties(height=300, title=f"{gcol}별 Lane Error 결측률"),
                    use_container_width=True,
                )
                st.dataframe(ge.sort_values("missing_%", ascending=False), hide_index=True, use_container_width=True)


    # =============================================================================
    # Part I: Mask White Ratio ↔ Abs Lane Error

    st.divider()
    st.markdown(f"""
    ## Part I: {MASK_RATIO_COL} ↔ {ABS_ERROR_COL}

    - **Mask White Ratio(0~1)**: 마스크에서 흰 픽셀(0이 아닌 픽셀)이 차지하는 비율(= 검출량/가시성 신호)
    - **Abs Lane Error**: 화면 중앙 대비 목표 지점의 오차 크기(픽셀)

    여기서의 판단(원인 단정 X):
    - **ratio↓ & error↑**: 검출량이 부족한 구간에서 오차가 커지는 패턴
    - **ratio↑ & error↑**: 마스크는 잡히는데 오차가 큰 “불일치” 후보(중심 추정/노이즈/가정값 영향 가능)
    """)

    if ABS_ERROR_COL not in df.columns:
        st.info(f"'{ERROR_COL}'(또는 '{ABS_ERROR_COL}') 컬럼이 없어서 Part I의 Error 기반 분석은 생략됩니다.")
    else:
        model_df = perform_linear_regression(df, MASK_RATIO_COL, ABS_ERROR_COL, sigma_threshold=2.0)

        c1, c2 = st.columns([0.7, 0.3])
        with c1:
            st.altair_chart(
                alt.Chart(model_df)
                .mark_point(filled=True, opacity=0.5)
                .encode(
                    x=alt.X(MASK_RATIO_COL, type="quantitative", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y(ABS_ERROR_COL, type="quantitative", scale=alt.Scale(zero=True)),
                    color=alt.Color("Status:N").legend(None),
                    shape=alt.Shape("Status:N").scale(range=["circle", "cross"]).legend(None),
                    tooltip=_make_tooltip(model_df, [EVENT_ID_COL, RUN_ID_COL, TS_COL, WEATHER_COL, TOD_COL, MASK_RATIO_COL, ABS_ERROR_COL, "Status"]),
                )
                .properties(height=420),
                use_container_width=True,
            )
        with c2:
            draw_histogram(df, MASK_RATIO_COL)
            draw_histogram(df, ABS_ERROR_COL)

        st.caption("※ Outlier는 단순 회귀 기준(기본 2σ)으로 회귀선 대비 크게 벗어난 프레임을 뜻합니다.")

        a, b = st.columns(2)
        with a:
            st.caption("High-error frames (Abs Error 상위 20)")
            show_cols = [TS_COL, WEATHER_COL, TOD_COL, MASK_RATIO_COL, ABS_ERROR_COL]
            if PROC_COL in df.columns:
                show_cols.append(PROC_COL)
            st.dataframe(
                df.dropna(subset=[ABS_ERROR_COL]).sort_values(ABS_ERROR_COL, ascending=False)[show_cols].head(20),
                column_config=_column_config_for(show_cols),
                height=360,
            )

        with b:
            st.caption("Low-ratio frames (Mask Ratio 하위 20)")
            show_cols = [TS_COL, WEATHER_COL, TOD_COL, MASK_RATIO_COL]
            if ABS_ERROR_COL in df.columns:
                show_cols.append(ABS_ERROR_COL)
            if PROC_COL in df.columns:
                show_cols.append(PROC_COL)
            st.dataframe(
                df.sort_values(MASK_RATIO_COL)[show_cols].head(20),
                column_config=_column_config_for(show_cols),
                height=360,
            )

    # =============================================================================
    # Part II: Mask White Ratio ↔ Processing Time (ms)

    st.divider()
    st.markdown(f"""
    ## Part II: {MASK_RATIO_COL} ↔ {PROC_COL}

    처리시간(지연)과 Mask White Ratio가 함께 변하는지 확인합니다.
    - ratio가 극단(매우 낮음/높음)인 구간에서 처리시간이 튀는지
    - 특정 run/조건에서만 지연이 반복되는지
    """)

    if PROC_COL not in df.columns:
        st.info(f"'{PROC_COL}' 컬럼이 없어서 Part II(처리시간 분석)는 생략됩니다.")
    else:
        model_df2 = perform_linear_regression(df, MASK_RATIO_COL, PROC_COL, sigma_threshold=2.0)

        c1, c2 = st.columns([0.7, 0.3])
        with c1:
            st.altair_chart(
                alt.Chart(model_df2)
                .mark_point(filled=True, opacity=0.5)
                .encode(
                    x=alt.X(MASK_RATIO_COL, type="quantitative", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y(PROC_COL, type="quantitative", scale=alt.Scale(zero=False)),
                    color=alt.Color("Status:N").legend(None),
                    shape=alt.Shape("Status:N").scale(range=["circle", "cross"]).legend(None),
                    tooltip=_make_tooltip(model_df2, [EVENT_ID_COL, RUN_ID_COL, TS_COL, WEATHER_COL, TOD_COL, MASK_RATIO_COL, PROC_COL, "Status"]),
                )
                .properties(height=420),
                use_container_width=True,
            )
        with c2:
            draw_histogram(df, PROC_COL)
            draw_histogram(df, MASK_RATIO_COL)

        a, b = st.columns(2)
        with a:
            st.caption("High processing-time frames (상위 20)")
            show_cols = [TS_COL, WEATHER_COL, TOD_COL, MASK_RATIO_COL, PROC_COL]
            if ABS_ERROR_COL in df.columns:
                show_cols.append(ABS_ERROR_COL)
            st.dataframe(
                df.dropna(subset=[PROC_COL]).sort_values(PROC_COL, ascending=False)[show_cols].head(20),
                column_config=_column_config_for(show_cols),
                height=360,
            )

        with b:
            st.caption("Low processing-time frames (하위 20)")
            show_cols = [TS_COL, WEATHER_COL, TOD_COL, MASK_RATIO_COL, PROC_COL]
            if ABS_ERROR_COL in df.columns:
                show_cols.append(ABS_ERROR_COL)
            st.dataframe(
                df.dropna(subset=[PROC_COL]).sort_values(PROC_COL, ascending=True)[show_cols].head(20),
                column_config=_column_config_for(show_cols),
                height=360,
            )


    # Part III: Outlier Candidates (rule-first)

    st.divider()
    st.markdown(textwrap.dedent("""
    ## Part III: 이상치 후보(자동 기준 + 민감도 1개)

    이 파트는 “원인 확정”이 아니라, **확인 우선순위를 정하기 위한 후보 추출**입니다.

    - **Mask White Ratio**: 하위 꼬리(매우 낮음) / 상위 꼬리(매우 높음)
    - **Abs Lane Error**: 상위 꼬리(오차 과다) *(컬럼이 있을 때만)*
    - **Processing Time (ms)**: 상위 꼬리(지연 과다) *(컬럼이 있을 때만)*
    - **Lane Error 결측(NA)**: 오차가 기록되지 않은 프레임 *(컬럼이 있을 때만)*

    조절값은 **민감도 1개(%)**만 사용하며, 나머지 임계값은 데이터 분위수로 자동 결정됩니다.
    """))

    st.markdown("#### 민감도 설정")
    # ---- single control (in-body)
    pctl = st.slider(
        "이상치 후보 민감도(%)",
        min_value=80,
        max_value=99,
        value=95,
        step=1,
        key="outlier_sensitivity_pctl_0112",
        help="높을수록 더 극단(상위/하위 꼬리)만 후보로 잡습니다. (오차/처리시간: 상위 pctl, 마스크비율: 하위(100-pctl) 및 상위 pctl)"
    )

    d = df.copy()

    # Normalize numeric columns safely
    d[MASK_RATIO_COL] = pd.to_numeric(d[MASK_RATIO_COL], errors="coerce").clip(0, 1)

    r = d[MASK_RATIO_COL].dropna()
    if r.empty:
        st.warning("Mask White Ratio가 전부 결측이라 후보 탐지가 불가능합니다.")
        cand = pd.DataFrame()
    else:
        low_ratio_th = float(r.quantile((100 - pctl) / 100.0))
        high_ratio_th = float(r.quantile(pctl / 100.0))

        mask_low = d[MASK_RATIO_COL].le(low_ratio_th)
        mask_high = d[MASK_RATIO_COL].ge(high_ratio_th)

        err_missing = pd.Series(False, index=d.index)
        if ERROR_COL in d.columns:
            err_missing = pd.to_numeric(d[ERROR_COL], errors="coerce").isna()

        err_high = pd.Series(False, index=d.index)
        err_th_value = None
        if ABS_ERROR_COL in d.columns:
            ae = pd.to_numeric(d[ABS_ERROR_COL], errors="coerce")
            if ae.notna().any():
                err_th_value = float(ae.quantile(pctl / 100.0))
                err_high = ae.ge(err_th_value)

        proc_high = pd.Series(False, index=d.index)
        proc_th_value = None
        if PROC_COL in d.columns:
            pr = pd.to_numeric(d[PROC_COL], errors="coerce")
            if pr.notna().any():
                proc_th_value = float(pr.quantile(pctl / 100.0))
                proc_high = pr.ge(proc_th_value)

        # candidate tags
        def _join_tags(row) -> str:
            tags = []
            if row.get("mask_low", False):
                tags.append("마스크 비율 매우 낮음")
            if row.get("mask_high", False):
                tags.append("마스크 비율 매우 높음")
            if row.get("err_high", False):
                tags.append("오차 과다")
            if row.get("proc_high", False):
                tags.append("처리시간 과다")
            if row.get("err_missing", False):
                tags.append("오차 기록 누락")
            return ", ".join(tags)

        d["mask_low"] = mask_low
        d["mask_high"] = mask_high
        d["err_high"] = err_high
        d["proc_high"] = proc_high
        d["err_missing"] = err_missing

        d["Candidate Tags"] = d.apply(_join_tags, axis=1)

        # primary tag (priority)
        d["Primary Tag"] = "Normal"
        d.loc[d["mask_low"], "Primary Tag"] = "마스크 비율 매우 낮음"
        d.loc[(d["Primary Tag"] == "Normal") & d["mask_high"], "Primary Tag"] = "마스크 비율 매우 높음"
        d.loc[(d["Primary Tag"] == "Normal") & d["err_high"], "Primary Tag"] = "오차 과다"
        d.loc[(d["Primary Tag"] == "Normal") & d["proc_high"], "Primary Tag"] = "처리시간 과다"
        d.loc[(d["Primary Tag"] == "Normal") & d["err_missing"], "Primary Tag"] = "오차 기록 누락"

        cand = d[d["Candidate Tags"].astype(str).str.len() > 0].copy()

        # show thresholds
        th_lines = [
            f"- Mask Ratio 하한(하위 {100 - pctl}% 분위): **{low_ratio_th:.4f}**",
            f"- Mask Ratio 상한(상위 {100 - pctl}% 분위): **{high_ratio_th:.4f}**",
        ]
        if err_th_value is not None:
            th_lines.append(f"- Abs Lane Error 임계(상위 {100 - pctl}% 분위): **{err_th_value:.2f}**")
        if proc_th_value is not None:
            th_lines.append(f"- Proc Time 임계(상위 {100 - pctl}% 분위): **{proc_th_value:.1f} ms**")

        st.markdown("#### 자동 임계값(현재 데이터 기준)")
        st.markdown("\n".join(th_lines))

    st.subheader("후보 요약(Primary Tag)")
    if cand.empty:
        st.info("현재 민감도 설정에서 후보가 없습니다.")
    else:
        summary = cand["Primary Tag"].value_counts(dropna=False).reset_index()
        summary.columns = ["Primary Tag", "count"]
        summary["%"] = (summary["count"] / len(df) * 100).round(2)
        st.dataframe(summary, hide_index=True, use_container_width=True, height=220)

        # Charts (fixed axes, no extra controls)
        tabs = []
        tabs.append("Mask Ratio ↔ Abs Error" if ABS_ERROR_COL in cand.columns else "Mask Ratio ↔ Proc Time")
        if (ABS_ERROR_COL in cand.columns) and (PROC_COL in cand.columns):
            tabs.append("Mask Ratio ↔ Proc Time")

        t = st.tabs(tabs)

        # Helper for sampling
        def _sample_for_plot(x: pd.DataFrame, n: int = 3000) -> pd.DataFrame:
            if len(x) <= n:
                return x
            return x.sample(n, random_state=7)

        if ABS_ERROR_COL in cand.columns:
            with t[0]:
                plot_df = cand.dropna(subset=[MASK_RATIO_COL, ABS_ERROR_COL]).copy()
                plot_df = _sample_for_plot(plot_df, n=min(3000, len(plot_df)))
                st.altair_chart(
                    alt.Chart(plot_df)
                    .mark_point(filled=True, opacity=0.55)
                    .encode(
                        x=alt.X(MASK_RATIO_COL, type="quantitative", scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y(ABS_ERROR_COL, type="quantitative", scale=alt.Scale(zero=True)),
                        color=alt.Color("Primary Tag:N").legend(title="Primary Tag"),
                        tooltip=_make_tooltip(plot_df, [EVENT_ID_COL, RUN_ID_COL, TS_COL, WEATHER_COL, TOD_COL, MASK_RATIO_COL, ABS_ERROR_COL, PROC_COL, "Candidate Tags", "Primary Tag"]),
                    )
                    .properties(height=420),
                    use_container_width=True,
                )
        else:
            with t[0]:
                if PROC_COL not in cand.columns:
                    st.info("Abs Lane Error / Processing Time 컬럼이 없어 산점도를 표시할 수 없습니다.")
                else:
                    plot_df = cand.dropna(subset=[MASK_RATIO_COL, PROC_COL]).copy()
                    plot_df = _sample_for_plot(plot_df, n=min(3000, len(plot_df)))
                    st.altair_chart(
                        alt.Chart(plot_df)
                        .mark_point(filled=True, opacity=0.55)
                        .encode(
                            x=alt.X(MASK_RATIO_COL, type="quantitative", scale=alt.Scale(domain=[0, 1])),
                            y=alt.Y(PROC_COL, type="quantitative", scale=alt.Scale(zero=False)),
                            color=alt.Color("Primary Tag:N").legend(title="Primary Tag"),
                            tooltip=_make_tooltip(plot_df, [EVENT_ID_COL, RUN_ID_COL, TS_COL, WEATHER_COL, TOD_COL, MASK_RATIO_COL, PROC_COL, "Candidate Tags", "Primary Tag"]),
                        )
                        .properties(height=420),
                        use_container_width=True,
                    )

        if (ABS_ERROR_COL in cand.columns) and (PROC_COL in cand.columns):
            with t[1]:
                plot_df = cand.dropna(subset=[MASK_RATIO_COL, PROC_COL]).copy()
                plot_df = _sample_for_plot(plot_df, n=min(3000, len(plot_df)))
                st.altair_chart(
                    alt.Chart(plot_df)
                    .mark_point(filled=True, opacity=0.55)
                    .encode(
                        x=alt.X(MASK_RATIO_COL, type="quantitative", scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y(PROC_COL, type="quantitative", scale=alt.Scale(zero=False)),
                        color=alt.Color("Primary Tag:N").legend(title="Primary Tag"),
                        tooltip=_make_tooltip(plot_df, [EVENT_ID_COL, RUN_ID_COL, TS_COL, WEATHER_COL, TOD_COL, MASK_RATIO_COL, PROC_COL, ABS_ERROR_COL, "Candidate Tags", "Primary Tag"]),
                    )
                    .properties(height=420),
                    use_container_width=True,
                )

        st.caption("※ 후보는 분위수 기반 자동 임계값으로 추출됩니다. 후보 해석은 Part 0의 결측 패턴/분포와 함께 보세요.")


    # Part IV: Top low-mask-ratio frames

    st.divider()
    st.markdown("## Part IV: 최저 Mask White Ratio 프레임 Top 20 요약")

    top = df[df[MASK_RATIO_COL] != 0].sort_values(MASK_RATIO_COL).head(20).copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Avg Mask Ratio", f"{float(pd.to_numeric(top[MASK_RATIO_COL], errors='coerce').mean()):.4f}")
    with c2:
        if ABS_ERROR_COL in top.columns:
            st.metric("Avg Abs Error", f"{float(pd.to_numeric(top[ABS_ERROR_COL], errors='coerce').mean()):.2f}")
        else:
            st.metric("Avg Abs Error", "N/A")
    with c3:
        if PROC_COL in top.columns:
            st.metric("Avg Proc Time", f"{float(pd.to_numeric(top[PROC_COL], errors='coerce').mean()):.1f} ms")
        else:
            st.metric("Avg Proc Time", "N/A")

    show = [TS_COL, WEATHER_COL, TOD_COL, MASK_RATIO_COL]
    if ABS_ERROR_COL in top.columns:
        show.append(ABS_ERROR_COL)
    if PROC_COL in top.columns:
        show.append(PROC_COL)
    if ERROR_COL in top.columns:
        show.append(ERROR_COL)
    if QUALITY_COL in top.columns:
        show.append(QUALITY_COL)

    show = [c for c in show if c in top.columns]
    st.dataframe(top[show], column_config=_column_config_for(show), height=360)

    st.divider()
    st.markdown("## Part V: 전체 로그 보기")
    st.dataframe(df.drop(["Run ID", "Row In Run", "Event ID"], axis=1))

    _render_part_x(df=df, pctl=pctl, cand=cand)

# =============================================================================
# Part X: Improvements (analysis-driven)


def _render_part_x(df: pd.DataFrame, pctl: int, cand: 'pd.DataFrame | None' = None) -> None:
    st.divider()
    st.markdown("## Part X: 자율주행 개선사항 (분석 기반)")

    # -------------------------------------------------------------------------
    # Metric helpers (same rules for 0112 + 0109 so we can compare apples-to-apples)
    def _to_num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    def _metrics(xdf: pd.DataFrame) -> dict:
        m: dict = {}
        m["n_total"] = int(len(xdf))

        # Mask ratio
        ratio = _to_num(xdf.get(MASK_RATIO_COL)).clip(0, 1)
        rv = ratio.dropna()
        if rv.empty:
            m.update({
                "ratio_p05": None, "ratio_p50": None, "ratio_p95": None,
                "low_th": None, "high_th": None, "low_rate": None, "high_rate": None,
            })
        else:
            m["ratio_p05"] = float(rv.quantile(0.05))
            m["ratio_p50"] = float(rv.quantile(0.50))
            m["ratio_p95"] = float(rv.quantile(0.95))
            m["low_th"] = float(rv.quantile((100 - pctl) / 100.0))
            m["high_th"] = float(rv.quantile(pctl / 100.0))

            valid = ratio.notna()
            m["low_rate"] = float((ratio[valid] <= m["low_th"]).mean() * 100.0)
            m["high_rate"] = float((ratio[valid] >= m["high_th"]).mean() * 100.0)

        # Lane Error missingness
        if ERROR_COL in xdf.columns:
            m["err_missing_rate"] = float(_to_num(xdf[ERROR_COL]).isna().mean() * 100.0)
        else:
            m["err_missing_rate"] = None

        # Abs error
        if ABS_ERROR_COL in xdf.columns:
            ae = _to_num(xdf[ABS_ERROR_COL])
            aev = ae.dropna()
            if aev.empty:
                m.update({"abs_p95": None, "abs_p99": None, "abs_tail_th": None, "abs_tail_rate": None})
            else:
                m["abs_p95"] = float(aev.quantile(0.95))
                m["abs_p99"] = float(aev.quantile(0.99))
                m["abs_tail_th"] = float(aev.quantile(pctl / 100.0))
                m["abs_tail_rate"] = float((aev >= m["abs_tail_th"]).mean() * 100.0)
        else:
            m.update({"abs_p95": None, "abs_p99": None, "abs_tail_th": None, "abs_tail_rate": None})

        # Processing time
        if PROC_COL in xdf.columns:
            pr = _to_num(xdf[PROC_COL])
            prv = pr.dropna()
            if prv.empty:
                m.update({"proc_p95": None, "proc_p99": None, "proc_max": None, "proc_tail_th": None, "proc_tail_rate": None})
            else:
                m["proc_p95"] = float(prv.quantile(0.95))
                m["proc_p99"] = float(prv.quantile(0.99))
                m["proc_max"] = float(prv.max())
                m["proc_tail_th"] = float(prv.quantile(pctl / 100.0))
                m["proc_tail_rate"] = float((prv >= m["proc_tail_th"]).mean() * 100.0)
        else:
            m.update({"proc_p95": None, "proc_p99": None, "proc_max": None, "proc_tail_th": None, "proc_tail_rate": None})

        return m

    def _fmt(v, fmt: str) -> str:
        if v is None:
            return "N/A"
        try:
            if pd.isna(v):
                return "N/A"
        except Exception:
            pass
        return fmt.format(v)

    cur = _metrics(df)

    # -------------------------------------------------------------------------
    # Current summary
    st.markdown("### 핵심 지표 요약 (0112)")
    summary_rows = [
        {"항목": "총 프레임 수", "값": f"{cur['n_total']:,}", "의미": "분석 대상 전체 행 수"},
        {"항목": "Mask Ratio p05 / p50 / p95", "값": f"{_fmt(cur['ratio_p05'], '{:.4f}')} / {_fmt(cur['ratio_p50'], '{:.4f}')} / {_fmt(cur['ratio_p95'], '{:.4f}')}", "의미": "마스크 검출량의 하위/중앙/상위 수준(0~1)"},
    ]

    if cur["low_th"] is not None:
        summary_rows.append({"항목": f"Mask Ratio 하한(하위 {100 - pctl}% 분위)", "값": _fmt(cur["low_th"], "{:.4f}"), "의미": "매우 낮은 검출량(저가시성/미검출 후보) 기준"})
        summary_rows.append({"항목": "Mask Ratio 하한 이하 비율", "값": _fmt(cur["low_rate"], "{:.2f}%"), "의미": "꼬리 구간 비중(원인 단정 불가)"})
        summary_rows.append({"항목": f"Mask Ratio 상한(상위 {100 - pctl}% 분위)", "값": _fmt(cur["high_th"], "{:.4f}"), "의미": "매우 높은 검출량(과검출 후보) 기준"})
        summary_rows.append({"항목": "Mask Ratio 상한 이상 비율", "값": _fmt(cur["high_rate"], "{:.2f}%"), "의미": "꼬리 구간 비중(원인 단정 불가)"})

    if cur["err_missing_rate"] is not None:
        summary_rows.append({"항목": "Lane Error 결측률", "값": _fmt(cur["err_missing_rate"], "{:.2f}%"), "의미": "오차 기반 분석/모니터링이 불가한 구간 비중"})

    if cur["abs_p95"] is not None:
        summary_rows.append({"항목": "Abs Error p95 / p99", "값": f"{_fmt(cur['abs_p95'], '{:.2f}')} / {_fmt(cur['abs_p99'], '{:.2f}')}", "의미": "오차 분포 상위 꼬리(픽셀 단위) 크기"})
        summary_rows.append({"항목": f"Abs Error 상위 {100 - pctl}% 기준", "값": f"{_fmt(cur['abs_tail_th'], '{:.2f}')} (≈ {_fmt(cur['abs_tail_rate'], '{:.2f}%')})", "의미": "Part III 후보(오차 과다) 임계값과 동일"})

    if cur["proc_p95"] is not None:
        summary_rows.append({"항목": "Proc Time p95 / p99 / max", "값": f"{_fmt(cur['proc_p95'], '{:.1f}')} / {_fmt(cur['proc_p99'], '{:.1f}')} / {_fmt(cur['proc_max'], '{:.1f}')} ms", "의미": "지연의 상위 꼬리(outlier) 크기"})
        summary_rows.append({"항목": f"Proc Time 상위 {100 - pctl}% 기준", "값": f"{_fmt(cur['proc_tail_th'], '{:.1f}')} ms (≈ {_fmt(cur['proc_tail_rate'], '{:.2f}%')})", "의미": "Part III 후보(처리시간 과다) 임계값과 동일"})

    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    # -------------------------------------------------------------------------
    # Candidate distribution (current)
    st.markdown("### 후보(Part III) 분포 요약 (0112)")
    if isinstance(cand, pd.DataFrame) and (not cand.empty) and ("Primary Tag" in cand.columns):
        vc = cand["Primary Tag"].value_counts(dropna=False)
        total_cand = int(len(cand))
        lines = []
        for k, v in vc.head(6).items():
            lines.append(f"- {k}: {int(v)}건 (전체 대비 {(v / cur['n_total'] * 100.0):.2f}%, 후보 내 {(v / total_cand * 100.0):.2f}%)")
        st.markdown("\n".join(lines))
    else:
        st.info("현재 민감도 설정에서 후보가 없거나, 후보 태그를 만들 수 없는 구성입니다.")

    # -------------------------------------------------------------------------
    # 0109 baseline comparison (optional)
    st.markdown("### 0109 대비 변화 (0112 - 0109)")
    baseline_metrics = None
    try:
        base_df = _load_fixed_data_0109_baseline()
        baseline_metrics = _metrics(base_df)
    except FileNotFoundError:
        st.info("0109 비교용 파일(0109_11_log.csv)이 없어 비교를 생략합니다. (파일을 두면 자동 비교됩니다.)")
    except Exception as e:
        st.warning(f"0109 비교 로드/계산 중 오류로 비교를 생략합니다: {e}")

    def _delta(cur_v, base_v):
        if (cur_v is None) or (base_v is None):
            return None
        try:
            if pd.isna(cur_v) or pd.isna(base_v):
                return None
        except Exception:
            pass
        return float(cur_v) - float(base_v)

    def _judge(delta_v: float | None, better_when: str) -> str:
        if delta_v is None:
            return "N/A"
        # small epsilon to avoid noise-driven wording
        eps = 1e-9
        if abs(delta_v) <= eps:
            return "거의 동일"
        if better_when == "lower":
            return "개선(↓)" if delta_v < 0 else "악화(↑)"
        if better_when == "higher":
            return "개선(↑)" if delta_v > 0 else "악화(↓)"
        return "참고"

    if baseline_metrics is not None:
        cmp_rows = []

        # Mask ratio: use p05/p50 as more meaningful than tail rates (tail rates are quantile-defined)
        for name, key, better, fmt in [
            ("Mask Ratio p05", "ratio_p05", "higher", "{:.4f}"),
            ("Mask Ratio p50(중앙값)", "ratio_p50", "higher", "{:.4f}"),
        ]:
            b = baseline_metrics.get(key)
            c = cur.get(key)
            dlt = _delta(c, b)
            cmp_rows.append({
                "지표": name,
                "0109": None if b is None else float(b),
                "0112": None if c is None else float(c),
                "Δ(0112-0109)": dlt,
                "판정": _judge(dlt, better),
                "설명": "높을수록 '최저/평균 가시성'이 좋아졌을 가능성(추측) — 영상 확인 필요",
            })

        # Missing rate / error / proc: lower is better (more stable)
        for name, key, better, note in [
            ("Lane Error 결측률(%)", "err_missing_rate", "lower", "낮을수록 오차 기반 모니터링/튜닝이 쉬움"),
            ("Abs Error p95", "abs_p95", "lower", "낮을수록 상위 꼬리 오차가 줄어듦"),
            ("Abs Error p99", "abs_p99", "lower", "낮을수록 극단 오차가 줄어듦"),
            ("Proc Time p95(ms)", "proc_p95", "lower", "낮을수록 지연 상위 꼬리가 줄어듦"),
            ("Proc Time p99(ms)", "proc_p99", "lower", "낮을수록 극단 지연이 줄어듦"),
            ("Proc Time max(ms)", "proc_max", "lower", "낮을수록 최악 지연이 줄어듦"),
        ]:
            b = baseline_metrics.get(key)
            c = cur.get(key)
            dlt = _delta(c, b)
            cmp_rows.append({
                "지표": name,
                "0109": None if b is None else float(b),
                "0112": None if c is None else float(c),
                "Δ(0112-0109)": dlt,
                "판정": _judge(dlt, better),
                "설명": note,
            })

        cmp_df = pd.DataFrame(cmp_rows)
        st.dataframe(cmp_df, hide_index=True, use_container_width=True)

        # One-paragraph narrative (keep it non-assertive)
        bullets = []
        d1 = _delta(cur.get("err_missing_rate"), baseline_metrics.get("err_missing_rate"))
        if d1 is not None:
            bullets.append(f"- Lane Error 결측률: {_fmt(baseline_metrics.get('err_missing_rate'), '{:.2f}%')} → {_fmt(cur.get('err_missing_rate'), '{:.2f}%')} (Δ{d1:+.2f}pp)")
        d2 = _delta(cur.get("abs_p95"), baseline_metrics.get("abs_p95"))
        if d2 is not None:
            bullets.append(f"- Abs Error p95: {_fmt(baseline_metrics.get('abs_p95'), '{:.2f}')} → {_fmt(cur.get('abs_p95'), '{:.2f}')} (Δ{d2:+.2f})")
        d3 = _delta(cur.get("proc_p95"), baseline_metrics.get("proc_p95"))
        if d3 is not None:
            bullets.append(f"- Proc Time p95: {_fmt(baseline_metrics.get('proc_p95'), '{:.1f}')}ms → {_fmt(cur.get('proc_p95'), '{:.1f}')}ms (Δ{d3:+.1f}ms)")
        if bullets:
            st.markdown("#### 요약(핵심 변화)")
            st.markdown("\n".join(bullets))

    # -------------------------------------------------------------------------
    # Recommendations (kept actionable, avoid asserting root causes)
    st.markdown("### 개선사항(권장 우선순위)")
    recos: list[str] = []

    if cur["low_rate"] is not None and cur["low_rate"] >= 5.0:
        recos.append(
            f"1) **선이 잘 안 보이는 구간 대비**: Mask Ratio가 아주 낮은 프레임이 {cur['low_rate']:.2f}% 있음(≤ {cur['low_th']:.4f}). "
            "→ (a) 밝기/대비 보정, (b) ROI/전처리 조정, (c) 선이 안 잡히면 감속/안전 동작으로 전환 같은 규칙을 정하고 테스트하기."
        )

    if cur["high_rate"] is not None and cur["high_rate"] >= 5.0:
        recos.append(
            f"2) **과하게 잡히는 경우 줄이기**: Mask Ratio가 아주 높은 프레임이 {cur['high_rate']:.2f}% 있음(≥ {cur['high_th']:.4f}). "
            "→ 반사/밝은 노이즈/테이프가 선으로 잡힐 가능성(추측)이 있으니, 작은 잡음 제거/모양 필터/좌우 균형 체크 같은 후처리를 검토."
        )

    if cur["abs_p95"] is not None and cur["abs_p95"] > 0:
        extra = ""
        if baseline_metrics is not None and baseline_metrics.get("abs_p95") is not None:
            dlt = _delta(cur["abs_p95"], baseline_metrics["abs_p95"])
            if dlt is not None:
                extra = f" (0109 대비 p95 Δ{dlt:+.2f})"
        recos.append(
            f"3) **조향 튀는 현상 완화**: Abs Error가 큰 구간(p95={cur['abs_p95']:.2f}px, p99={_fmt(cur['abs_p99'], '{:.2f}') }px){extra}. "
            "→ (a) 오차/조향각을 시간적으로 부드럽게(평균/필터), (b) 조향 변화량 상한 설정, "
            "(c) 한쪽 차선만 보일 때 쓰는 가정값(차선 폭 등)이 문제를 키우지 않는지 점검."
        )

    if cur["err_missing_rate"] is not None and cur["err_missing_rate"] >= 10.0:
        extra = ""
        if baseline_metrics is not None and baseline_metrics.get("err_missing_rate") is not None:
            dlt = _delta(cur["err_missing_rate"], baseline_metrics["err_missing_rate"])
            if dlt is not None:
                extra = f" (0109 대비 Δ{dlt:+.2f}pp)"
        recos.append(
            f"4) **로그 품질 개선**: Lane Error 결측률이 {cur['err_missing_rate']:.2f}%로 높음{extra}. "
            "→ NA가 나오는 상황도 '차선 미검출/한쪽만 검출/모드' 같은 상태를 함께 저장하고, 어떤 조건에서 NA가 많이 생기는지 요약해보기."
        )

    if cur["proc_p95"] is not None:
        extra = ""
        if baseline_metrics is not None and baseline_metrics.get("proc_p95") is not None:
            dlt = _delta(cur["proc_p95"], baseline_metrics["proc_p95"])
            if dlt is not None:
                extra = f" (0109 대비 p95 Δ{dlt:+.1f}ms)"
        recos.append(
            f"5) **처리시간 튐 감소**: 처리시간 p95={cur['proc_p95']:.1f}ms, p99={_fmt(cur['proc_p99'], '{:.1f}') }ms, max={_fmt(cur['proc_max'], '{:.1f}') }ms{extra}. "
            "→ (a) 단계별 시간 측정으로 병목 찾기, (b) 해상도/ROI 줄이기, (c) 모델/후처리 경량화 검토."
        )

    if not recos:
        st.info("현재 데이터 기준으로는 즉시 추천할 항목이 없습니다. (민감도/컬럼/데이터 범위를 확인해 주세요.)")
    else:
        for line in recos:
            st.markdown(f"- {line}")


if __name__ == "__main__":
        render()

