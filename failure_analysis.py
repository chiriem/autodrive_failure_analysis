import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from pathlib import Path
import re

st.set_page_config(page_title="실패분석", page_icon="🛣️", layout="wide")

# =============================================================================
# Canonical columns (you can rename your CSV columns to match, or rely on auto-rename)

TS_COL = "Timestamp"                   # optional (ms or ISO string)

QUALITY_COL = "Lane Quality Score"     # 0~100 (higher is better)  [OPTIONAL] (현재 logger 구현에서는 ratio의 스케일링일 수 있음)
MASK_RATIO_COL = "Mask White Ratio"    # 0~1 (white pixels / mask pixels) [REQUIRED]

ERROR_COL = "Lane Error"               # signed (e.g., pixels)
ABS_ERROR_COL = "Abs Lane Error"

PROC_COL = "Processing Time (ms)"      # optional
WEATHER_COL = "Weather"                # optional
TOD_COL = "Time of Day"                # optional
MODE_COL = "Mode"                    # optional (e.g., auto/manual)



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

def _select_fixed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select fixed columns in a stable order and fail fast if any are missing."""
    missing = [c for c in FIXED_LOG_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {', '.join(missing)}")
    return df[FIXED_LOG_COLS].copy()

# Synthetic IDs (created at load/merge time)
RUN_ID_COL = "Run ID"
ROW_IN_RUN_COL = "Row In Run"
EVENT_ID_COL = "Event ID"

# =============================================================================
# Helpers

def _ensure_fields(df: pd.DataFrame) -> pd.DataFrame:
    if QUALITY_COL in df.columns:
        q = pd.to_numeric(df[QUALITY_COL], errors="coerce") # 수치형 데이터로 변환, 불가능 시 NaN 값 반환
        df[QUALITY_COL] = q.clip(0, 100)

    if MASK_RATIO_COL in df.columns:
        r = pd.to_numeric(df[MASK_RATIO_COL], errors="coerce")
        df[MASK_RATIO_COL] = np.where(r > 1.5, r / 100.0, r)
        df[MASK_RATIO_COL] = pd.to_numeric(df[MASK_RATIO_COL], errors="coerce").clip(0, 1)

    if ERROR_COL in df.columns and ABS_ERROR_COL not in df.columns:
        e = pd.to_numeric(df[ERROR_COL], errors="coerce")
        df[ABS_ERROR_COL] = e.abs()

    # Optional defaults
    if WEATHER_COL not in df.columns:
        df[WEATHER_COL] = "Unknown"
    if TOD_COL not in df.columns:
        df[TOD_COL] = "Unknown"

    return df


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
# UI

analysis_mode = st.sidebar.radio(
    "분석 모드",
    ["파일 업로드 (기본)", "0109 고정 데이터 분석", "0112 고정 데이터 분석"],
    index=0
)

df = pd.DataFrame()

if analysis_mode == "0109 고정 데이터 분석":
    # 0109 고정 데이터 분석은 별도 모듈(failure_analysis_0109.py)로 위임합니다.
    import failure_analysis_0109 as page_0109
    page_0109.render()
    st.stop()

if analysis_mode == "0112 고정 데이터 분석":
    # 0112 고정 데이터 분석은 별도 모듈(failure_analysis_0112.py)로 위임합니다.
    import failure_analysis_0112 as page_0112
    page_0112.render()
    st.stop()

else:
    st.title("자율주행 실패 분석 (OpenCV 로그: Mask Ratio + Error 중심)")
    st.caption("Mask White Ratio / Lane Error / Processing Time (ms)를 중심으로 분석합니다.")

    if "uploader_count" not in st.session_state:
        st.session_state.uploader_count = 1

    uploaded_files = []
    for i in range(st.session_state.uploader_count):
        f = st.file_uploader(f"주행 로그 CSV 업로드 {i+1}", type=["csv"], key=f"uploader_{i}")
        if f is not None:
            uploaded_files.append(f)

    if st.session_state.uploader_count < 5:
        if st.button("➕ CSV 추가 업로드 공간 만들기"):
            st.session_state.uploader_count += 1
            st.rerun()

    use_demo = st.toggle("데모 데이터 사용", value=(not uploaded_files))

    def _generate_demo(n: int = 1200) -> pd.DataFrame:
        np.random.seed(7)

        weather = np.random.choice(["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy"], n)
        tod = np.random.choice(["Day", "Night"], n, p=[0.75, 0.25])

        # Mask ratio 0~1 (very low when lane is barely visible; can be noisy high in glare)
        base_ratio = np.random.beta(3, 25, n)  # mostly small ratios
        base_ratio[(weather == "Foggy") | (weather == "Snowy")] *= 0.7
        base_ratio[tod == "Night"] *= 0.8
        mask_ratio = np.clip(base_ratio + np.random.normal(0, 0.01, n), 0, 1)

        # Quality 0~100: not identical to ratio (so you can see divergence cases)
        quality = np.clip((mask_ratio * 180) + np.random.normal(0, 8, n), 0, 100)
        # inject false positives (ratio high but quality low)
        fp = np.random.rand(n) < 0.03
        quality[fp] = np.clip(quality[fp] - 50, 0, 100)

        # Error grows when quality low
        abs_err = np.clip((100 - quality) * 0.9 + np.random.normal(0, 6, n), 0, None)
        err = abs_err * np.random.choice([-1, 1], n)


        proc = np.random.normal(28, 5, n)
        mode = np.random.choice(["AUTO", "MANUAL"], n, p=[0.9, 0.1])
        df = pd.DataFrame({
            TS_COL: np.arange(n) * 100,  # ms
            WEATHER_COL: weather,
            TOD_COL: tod,
            MASK_RATIO_COL: mask_ratio,
            QUALITY_COL: quality,
            ERROR_COL: err,
            PROC_COL: proc,
            MODE_COL: mode,
        })
        df[ABS_ERROR_COL] = df[ERROR_COL].abs()
        return df

    if uploaded_files:
        dfs = []
        for f in uploaded_files:
            try:
                d = pd.read_csv(f)

                # 고정 스키마: 컬럼명은 이미 정해져 있으므로 전처리(리네임/중복 컬럼 병합)는 생략
                d = _select_fixed_columns(d)
                d = _ensure_fields(d)

                safe_name = re.sub(r"[^0-9A-Za-z가-힣_\-]+", "_", Path(f.name).stem)[:40]
                run_id = f"run_{len(dfs)+1:03d}_{safe_name}"
                d = _add_event_ids_per_run(d, run_id=run_id)
                dfs.append(d)
            except Exception as e:
                st.error(f"CSV 로드 실패 ({f.name}): {e}")

        if dfs:
            try:
                df = pd.concat(dfs, ignore_index=True)
            except Exception as e:
                st.error(f"파일 병합 실패: {e}")
                st.stop()
    elif use_demo:
        df = _generate_demo()
        df = _ensure_fields(df)
        df = _add_event_ids_per_run(df, run_id="demo")

    if df.empty:
        st.info("CSV를 업로드하거나 '데모 데이터 사용'을 켜세요.")
        st.stop()

# Required checks (고정 스키마 기준)
missing = [c for c in FIXED_LOG_COLS if c not in df.columns]
if missing:
    st.error(
        "필수 컬럼이 없습니다. (이 버전은 고정 컬럼 스키마를 사용합니다.)\n\n"
        f"- 필수: {', '.join(FIXED_LOG_COLS)}\n"
        f"- 누락: {', '.join(missing)}\n\n"
        "※ Mask White Ratio는 0~1 또는 0~100(%) 모두 허용되며 자동 정규화됩니다."
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
    MODE_COL: st.column_config.TextColumn(),
    RUN_ID_COL: st.column_config.TextColumn(),
    ROW_IN_RUN_COL: st.column_config.NumberColumn(format="%.0f"),
    EVENT_ID_COL: st.column_config.TextColumn(),
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

# =============================================================================
# Part III: Environment summary (optional)

if analysis_mode == "파일 업로드 (기본)":
    st.divider()
    st.markdown("## Part III: 환경(날씨/시간대)별 비교 (선택)")

    def _group_summary(data: pd.DataFrame, group_col: str, metric_col: str, include_unknown: bool) -> pd.DataFrame:
        d = data.dropna(subset=[group_col, metric_col])
        if not include_unknown:
            d = d[d[group_col] != "Unknown"]
        return (
            d.groupby(group_col)[metric_col]
            .agg(
                num_frames='count',
                median='median',
                mean='mean',
                p25=lambda x: x.quantile(0.25),
                p75=lambda x: x.quantile(0.75)
            )
            .reset_index()
            .sort_values("median", ascending=False)
        )

    def _bar_chart(summary: pd.DataFrame, group_col: str, metric_label: str, domain=None, height: int = 320):
        if summary.empty:
            st.info("그룹 비교에 사용할 데이터가 없습니다.")
            return
        st.altair_chart(
            alt.Chart(summary, height=height)
            .mark_bar()
            .encode(
                x=alt.X(f"{group_col}:N", sort="-y", axis=alt.Axis(labelAngle=-20)),
                y=alt.Y("median:Q", scale=alt.Scale(domain=domain) if domain else alt.Scale()),
                tooltip=[
                    group_col,
                    alt.Tooltip("num_frames:Q", title="frames"),
                    alt.Tooltip("median:Q", format=".4f"),
                    alt.Tooltip("mean:Q", format=".4f"),
                    alt.Tooltip("p25:Q", format=".4f"),
                    alt.Tooltip("p75:Q", format=".4f"),
                ],
            )
            .properties(title=f"Median of {metric_label} by {group_col}"),
            use_container_width=True,
        )

    # Decide whether Part III is meaningful (avoid awkward 'Unknown-only' charts)
    group_candidates = []
    for g in [WEATHER_COL, TOD_COL]:
        if g in df.columns:
            uniq_non_unknown = df[df[g] != "Unknown"][g].nunique()
            if uniq_non_unknown >= 2:
                group_candidates.append(g)

    if not group_candidates:
        st.info("날씨/시간대 데이터가 없거나 값이 1종류뿐이라 Part III 그룹 비교를 생략합니다.")
    else:
        include_unknown = st.checkbox("Unknown 포함", value=False)
        group_col = st.selectbox("그룹 기준", options=group_candidates, index=0)

        metric_options = [
            (MASK_RATIO_COL, "Mask White Ratio (0~1)", [0, 1]),
        ]
        if ABS_ERROR_COL in df.columns:
            metric_options.append((ABS_ERROR_COL, "Abs Lane Error", None))
        if PROC_COL in df.columns:
            metric_options.append((PROC_COL, "Processing Time (ms)", None))

        metric_names = [m[1] for m in metric_options]
        metric_idx = st.selectbox("비교 지표(중앙값)", options=list(range(len(metric_names))), format_func=lambda i: metric_names[i], index=0)
        metric_col, metric_label, domain = metric_options[int(metric_idx)]
        
        summary = _group_summary(df, group_col, metric_col, include_unknown)
        
        # If user includes Unknown and it dominates, still keep it—but warn if it's the only group.
        if len(summary) <= 1:
            st.info("선택한 그룹 기준에서 비교 가능한 범주가 1개뿐입니다(대부분 Unknown일 수 있음).")
        _bar_chart(summary, group_col, metric_label, domain=domain)
        
        st.caption("표는 median/mean과 IQR(p25~p75)을 함께 제공합니다. 프레임 수가 적은 그룹은 해석에 주의하세요.")
        st.dataframe(summary, hide_index=True, use_container_width=True)

# =============================================================================
# Part IV: Outlier Candidates (rule-first)

st.divider()
st.markdown("""
## Part IV: 이상치 후보(자동 기준 + 민감도 1개)

이 파트는 “원인 확정”이 아니라, **확인 우선순위를 정하기 위한 후보 추출**입니다.

- **Mask White Ratio**: 하위 꼬리(매우 낮음) / 상위 꼬리(매우 높음)
- **Abs Lane Error**: 상위 꼬리(오차 과다) *(컬럼이 있을 때만)*
- **Processing Time (ms)**: 상위 꼬리(지연 과다) *(컬럼이 있을 때만)*
- **Lane Error 결측(NA)**: 오차가 기록되지 않은 프레임 *(컬럼이 있을 때만)*

조절값은 **민감도 1개(%)**만 사용하며, 나머지 임계값은 데이터 분위수로 자동 결정됩니다.
""")

st.markdown("#### 민감도 설정")
# ---- single control (in-body)
pctl = st.slider(
    "이상치 후보 민감도(%)",
    min_value=80,
    max_value=99,
    value=95,
    step=1,
    key="outlier_sensitivity_pctl_main",
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


# Part V: Top low-mask-ratio frames

st.divider()
st.markdown("## Part V: 최저 Mask White Ratio 프레임 Top 20 요약")

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

# =============================================================================
# Part VI: Browse

st.divider()
st.markdown("## Part VI: 전체 로그 보기")
st.dataframe(df.drop(["Run ID", "Row In Run", "Event ID"], axis=1))