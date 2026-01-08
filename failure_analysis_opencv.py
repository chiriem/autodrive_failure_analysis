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

QUALITY_COL = "Lane Quality Score"     # 0~100 (higher is better)  [REQUIRED]
MASK_RATIO_COL = "Mask White Ratio"    # 0~1 (white pixels / mask pixels) [REQUIRED]

ERROR_COL = "Lane Error"               # signed (e.g., pixels)
ABS_ERROR_COL = "Abs Lane Error"

PROC_COL = "Processing Time (ms)"      # optional
WEATHER_COL = "Weather"                # optional
TOD_COL = "Time of Day"                # optional

# Synthetic IDs (created at load/merge time)
RUN_ID_COL = "Run ID"
ROW_IN_RUN_COL = "Row In Run"
EVENT_ID_COL = "Event ID"

# =============================================================================
# Helpers

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common variants to canonical column names."""
    rename_map = {}

    # Frame / timestamp
    if TS_COL not in df.columns:
        for c in ["timestamp", "ts", "time", "t", "Time", "datetime"]:
            if c in df.columns:
                rename_map[c] = TS_COL
                break

    # Quality
    if QUALITY_COL not in df.columns:
        for c in ["lane_quality_score", "quality", "LaneQuality", "lane_quality", "Lane Quality"]:
            if c in df.columns:
                rename_map[c] = QUALITY_COL
                break

    # Mask ratio
    if MASK_RATIO_COL not in df.columns:
        for c in ["mask_white_ratio", "lane_white_ratio", "white_ratio", "mask_ratio", "MaskWhiteRatio"]:
            if c in df.columns:
                rename_map[c] = MASK_RATIO_COL
                break

    # Error
    if ERROR_COL not in df.columns:
        for c in ["error", "lane_error", "LaneError", "target_error", "center_error"]:
            if c in df.columns:
                rename_map[c] = ERROR_COL
                break

    # Processing time
    if PROC_COL not in df.columns:
        for c in ["processing_time_ms", "proc_ms", "latency_ms", "inference_ms", "Processing Time"]:
            if c in df.columns:
                rename_map[c] = PROC_COL
                break

    # Environment / mode
    if WEATHER_COL not in df.columns:
        for c in ["weather", "Weather Condition"]:
            if c in df.columns:
                rename_map[c] = WEATHER_COL
                break
    if TOD_COL not in df.columns:
        for c in ["time_of_day", "tod", "day_night", "DayNight"]:
            if c in df.columns:
                rename_map[c] = TOD_COL
                break

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _ensure_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Fill convenience columns and sanitize ratio/quality ranges."""
    # Required columns check happens later, but we can sanitize if present:
    if QUALITY_COL in df.columns:
        q = pd.to_numeric(df[QUALITY_COL], errors="coerce")
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


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If df has duplicate column names, coalesce them into a single column (left-to-right fillna)."""
    if df.columns.is_unique:
        return df

    out = df.copy()
    seen = set()

    for col in list(out.columns):
        if col in seen:
            continue

        matches = [i for i, c in enumerate(out.columns) if c == col]
        if len(matches) <= 1:
            seen.add(col)
            continue

        block = out.loc[:, col]  # DataFrame when duplicates exist
        base = block.iloc[:, 0]
        for j in range(1, block.shape[1]):
            base = base.fillna(block.iloc[:, j])
        out[col] = base

        drop_positions = matches[1:]
        out = out.drop(out.columns[drop_positions], axis=1)
        seen.add(col)

    return out


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

st.title("자율주행 실패 분석 (OpenCV 로그: Lane Quality + Mask Ratio 중심)")
st.caption("Lane Quality Score / Mask White Ratio를 기준으로 분석합니다.")

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
    df = pd.DataFrame({
        TS_COL: np.arange(n) * 100,  # ms
        WEATHER_COL: weather,
        TOD_COL: tod,
        MASK_RATIO_COL: mask_ratio,
        QUALITY_COL: quality,
        ERROR_COL: err,
        PROC_COL: proc,
    })
    df[ABS_ERROR_COL] = df[ERROR_COL].abs()
    return df


df = pd.DataFrame()
if uploaded_files:
    dfs = []
    for f in uploaded_files:
        try:
            d = pd.read_csv(f)
            d = _coalesce_duplicate_columns(d)
            d = _standardize_columns(d)
            d = _coalesce_duplicate_columns(d)
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
            df = _coalesce_duplicate_columns(df)
        except Exception as e:
            st.error(f"파일 병합 실패: {e}")
            st.stop()
elif use_demo:
    df = _generate_demo()
    df = _standardize_columns(df)
    df = _ensure_fields(df)
    df = _add_event_ids_per_run(df, run_id="demo")

if df.empty:
    st.info("CSV를 업로드하거나 '데모 데이터 사용'을 켜세요.")
    st.stop()

# Required checks
missing = [c for c in [QUALITY_COL, MASK_RATIO_COL] if c not in df.columns]
if missing:
    st.error(
        "필수 컬럼이 없습니다.\n\n"
        f"- 필수: {QUALITY_COL}, {MASK_RATIO_COL}\n"
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
    RUN_ID_COL: st.column_config.TextColumn(),
    ROW_IN_RUN_COL: st.column_config.NumberColumn(format="%.0f"),
    EVENT_ID_COL: st.column_config.TextColumn(),
}

# =============================================================================
# Part 0: sanity checks

st.divider()
st.subheader("Part 0: 컬럼/결측 확인")
check_cols = [QUALITY_COL, MASK_RATIO_COL, ERROR_COL, PROC_COL, WEATHER_COL, TOD_COL]
st.dataframe(_describe_missing(df, check_cols), hide_index=True, use_container_width=True)

# =============================================================================
# Part I: Quality vs Error

st.divider()
st.markdown("""
## Part I: Lane Quality Score ↔ Lane Error

- **좋은 품질인데 오차가 큰 프레임**: 제어/캘리브레이션/트랙폭 가정 문제 가능
- **품질이 낮고 오차도 큰 프레임**: 인식 품질 붕괴(조명/바닥색/가림) 가능
""")

if ABS_ERROR_COL not in df.columns:
    st.info(f"'{ERROR_COL}' 컬럼이 없어서 Part I의 Error 기반 분석은 생략됩니다.")
else:
    model_df = perform_linear_regression(df, QUALITY_COL, ABS_ERROR_COL, sigma_threshold=2.0)

    c1, c2 = st.columns([0.7, 0.3])
    with c1:
        st.altair_chart(
            alt.Chart(model_df)
            .mark_point(filled=True, opacity=0.5)
            .encode(
                x=alt.X(QUALITY_COL, type="quantitative", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y(ABS_ERROR_COL, type="quantitative", scale=alt.Scale(zero=True)),
                color=alt.Color("Status:N").legend(None),
                shape=alt.Shape("Status:N").scale(range=["circle", "cross"]).legend(None),
                tooltip=_make_tooltip(model_df, [EVENT_ID_COL, RUN_ID_COL, TS_COL, WEATHER_COL, TOD_COL, QUALITY_COL, ABS_ERROR_COL, "Status"]),
            )
            .properties(height=420),
            use_container_width=True,
        )
    with c2:
        draw_histogram(df, QUALITY_COL)
        draw_histogram(df, ABS_ERROR_COL)

    st.subheader("우선 확인 목록")
    a, b = st.columns(2, border=True)

    with a:
        st.caption("High-quality but high-error (제어/보정 의심)")
        st.dataframe(
            model_df.sort_values(ABS_ERROR_COL, ascending=False)[
                [EVENT_ID_COL, RUN_ID_COL, TS_COL, WEATHER_COL, TOD_COL, QUALITY_COL, ABS_ERROR_COL] +
                ([PROC_COL] if PROC_COL in model_df.columns else [])
            ].head(20),
            column_config=COLUMN_CONFIG,
            height=360,
        )

    with b:
        st.caption("Low-quality frames (인식 품질 붕괴 의심)")
        st.dataframe(
            df.sort_values(QUALITY_COL)[
                [EVENT_ID_COL, RUN_ID_COL, TS_COL, WEATHER_COL, TOD_COL, QUALITY_COL] +
                ([ABS_ERROR_COL] if ABS_ERROR_COL in df.columns else []) +
                ([PROC_COL] if PROC_COL in df.columns else [])
            ].head(20),
            column_config=COLUMN_CONFIG,
            height=360,
        )

# =============================================================================
# Part II: Mask White Ratio ↔ Quality (consistency check)

st.divider()
st.markdown("""
## Part II: Mask White Ratio ↔ Lane Quality Score

- ratio는 **마스크에서 흰 픽셀이 차지하는 비율(0~1)**입니다.
- quality는 ratio뿐 아니라 **오차/안정성 등을 반영해 만든 최종 점수**라서, 둘이 완전히 같지 않은 것이 정상입니다.
""")

c1, c2 = st.columns([0.7, 0.3])
with c1:
    st.altair_chart(
        alt.Chart(df.dropna(subset=[MASK_RATIO_COL, QUALITY_COL]))
        .mark_point(filled=True, opacity=0.45)
        .encode(
            x=alt.X(MASK_RATIO_COL, type="quantitative", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(QUALITY_COL, type="quantitative", scale=alt.Scale(domain=[0, 100])),
            tooltip=_make_tooltip(df, [EVENT_ID_COL, RUN_ID_COL, TS_COL, WEATHER_COL, TOD_COL, MASK_RATIO_COL, QUALITY_COL]),
        )
        .properties(height=420),
        use_container_width=True,
    )
with c2:
    draw_histogram(df, MASK_RATIO_COL)
    draw_histogram(df, QUALITY_COL)

st.caption("Tip: ratio가 높은데 quality가 낮으면(산점도 오른쪽-아래), 바닥 반사/노이즈/한쪽 차선만 검출 같은 케이스를 의심해볼 수 있습니다.")

# =============================================================================
# Part III: Environment summary (optional)

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

    metric_options = [(QUALITY_COL, "Lane Quality Score (0~100)", [0, 100]),
                      (MASK_RATIO_COL, "Mask White Ratio (0~1)", [0, 1])]
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
# Part IV: Outlier Candidates (rule-first, safer interpretation)

st.divider()
st.markdown("## Part IV: 이상치 후보 탐색")
st.caption("이 파트는 이상치 후보를 탐색하기 위한 용도입니다. 특정 프레임이 이상치로 표시되더라도 실패 원인으로 단정할 수 없습니다.")

required_for_part4 = [QUALITY_COL, MASK_RATIO_COL]
missing_part4 = [c for c in required_for_part4 if c not in df.columns]
if missing_part4:
    st.info("필수 컬럼이 부족해서 Part IV는 생략됩니다: " + ", ".join(missing_part4))
else:
    cfg1, cfg2, cfg3, cfg4 = st.columns([1, 1, 1, 1])
    with cfg1:
        quality_low_th = st.slider("품질 점수 임계값(이하)", min_value=0, max_value=100, value=30, step=1)
    with cfg2:
        ratio_low_th = st.number_input("마스크 비율 하한", min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.3f")
    with cfg3:
        ratio_high_th = st.number_input("마스크 비율 상한", min_value=0.0, max_value=1.0, value=0.25, step=0.01, format="%.2f")
    with cfg4:
        sample_n_part4 = st.number_input("차트 샘플 수", min_value=500, max_value=200000, value=5000, step=500)

    df4 = df.copy()

    df4[QUALITY_COL] = pd.to_numeric(df4[QUALITY_COL], errors="coerce")
    df4[MASK_RATIO_COL] = pd.to_numeric(df4[MASK_RATIO_COL], errors="coerce")

    if ABS_ERROR_COL in df4.columns:
        df4[ABS_ERROR_COL] = pd.to_numeric(df4[ABS_ERROR_COL], errors="coerce")
    if ERROR_COL in df4.columns:
        df4[ERROR_COL] = pd.to_numeric(df4[ERROR_COL], errors="coerce")

    low_quality = df4[QUALITY_COL] <= float(quality_low_th)
    low_ratio = df4[MASK_RATIO_COL] <= float(ratio_low_th)
    high_ratio = df4[MASK_RATIO_COL] >= float(ratio_high_th)

    abs_err_thresh = None
    high_error = pd.Series(False, index=df4.index)
    inconsistent = pd.Series(False, index=df4.index)

    if ABS_ERROR_COL in df4.columns and pd.api.types.is_numeric_dtype(df4[ABS_ERROR_COL]):
        pctl = st.slider("절대오차 분위수 임계값", min_value=80, max_value=99, value=95, step=1)
        vals = df4[ABS_ERROR_COL].to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            abs_err_thresh = float(np.percentile(vals, pctl))
            st.caption(f"절대오차 임계값은 상위 {pctl}% 기준으로 {abs_err_thresh:.2f} 입니다.")
            high_error = df4[ABS_ERROR_COL] >= abs_err_thresh

            hi_q = st.slider("불일치 후보: 품질 높음 기준", min_value=0, max_value=100, value=70, step=1)
            inconsistent = (df4[QUALITY_COL] >= float(hi_q)) & high_error

    tag_cols = {
        "마스크 비율 매우 낮음": low_ratio,
        "품질 낮음": low_quality,
        "오차 과다": high_error,
        "마스크 비율 매우 높음": high_ratio,
        "불일치(품질 높음+오차 큼)": inconsistent,
    }

    any_candidate = None
    for mask in tag_cols.values():
        if any_candidate is None:
            any_candidate = mask.copy()
        else:
            any_candidate = any_candidate | mask

    df4["Candidate Tags"] = ""
    for tag_name, mask in tag_cols.items():
        if mask is None:
            continue
        df4.loc[mask.fillna(False), "Candidate Tags"] = df4.loc[mask.fillna(False), "Candidate Tags"].where(
            df4.loc[mask.fillna(False), "Candidate Tags"] == "",
            df4.loc[mask.fillna(False), "Candidate Tags"] + ", "
        ) + tag_name

    priority = [
        "마스크 비율 매우 낮음",
        "불일치(품질 높음+오차 큼)",
        "품질 낮음",
        "오차 과다",
        "마스크 비율 매우 높음",
    ]

    df4["Primary Tag"] = "정상 범위"
    for tag_name in priority:
        mask = tag_cols.get(tag_name)
        if mask is None:
            continue
        df4.loc[mask.fillna(False), "Primary Tag"] = tag_name

    candidates = df4[any_candidate.fillna(False)].copy()
    st.subheader("후보 요약")
    left, right = st.columns([1, 1])
    with left:
        counts = candidates["Primary Tag"].value_counts(dropna=False).reset_index()
        counts.columns = ["Primary Tag", "count"]
        st.dataframe(counts, use_container_width=True, height=240)
    with right:
        if WEATHER_COL in candidates.columns:
            st.caption("Weather 분포(후보만)")
            w_counts = candidates[WEATHER_COL].value_counts(dropna=False).reset_index()
            w_counts.columns = [WEATHER_COL, "count"]
            st.dataframe(w_counts, use_container_width=True, height=240)

    st.subheader("후보 산점도")
    default_x = MASK_RATIO_COL if MASK_RATIO_COL in df4.columns else QUALITY_COL
    default_y = QUALITY_COL if QUALITY_COL in df4.columns else MASK_RATIO_COL

    numeric_for_scatter = []
    for c in [MASK_RATIO_COL, QUALITY_COL, ABS_ERROR_COL, ERROR_COL, PROC_COL]:
        if c in df4.columns and pd.api.types.is_numeric_dtype(df4[c]):
            numeric_for_scatter.append(c)

    c1, c2 = st.columns(2)
    with c1:
        x_col4 = st.selectbox("X (탐색)", options=numeric_for_scatter, index=numeric_for_scatter.index(default_x) if default_x in numeric_for_scatter else 0)
    with c2:
        y_col4 = st.selectbox("Y (탐색)", options=numeric_for_scatter, index=numeric_for_scatter.index(default_y) if default_y in numeric_for_scatter else min(1, len(numeric_for_scatter)-1))

    plot_df = df4.copy()
    if len(plot_df) > int(sample_n_part4):
        plot_df = plot_df.sample(int(sample_n_part4), random_state=42)

    tooltip_cols = []
    for c in [EVENT_ID_COL, RUN_ID_COL, TS_COL, WEATHER_COL, TOD_COL, x_col4, y_col4, "Primary Tag", "Candidate Tags"]:
        if c in plot_df.columns:
            tooltip_cols.append(c)

    st.altair_chart(
        alt.Chart(plot_df)
        .mark_point(filled=True, opacity=0.5)
        .encode(
            x=alt.X(x_col4, scale=alt.Scale(zero=False)),
            y=alt.Y(y_col4, scale=alt.Scale(zero=False)),
            color=alt.Color("Primary Tag:N"),
            tooltip=tooltip_cols,
        )
        .properties(height=420),
        use_container_width=True,
    )

    st.subheader("후보 목록")
    show_cols = []
    for c in [TS_COL, WEATHER_COL, TOD_COL, QUALITY_COL, MASK_RATIO_COL, ABS_ERROR_COL, ERROR_COL, PROC_COL, "Primary Tag", "Candidate Tags"]:
        if c in candidates.columns:
            show_cols.append(c)
    st.dataframe(candidates[show_cols], column_config=COLUMN_CONFIG, use_container_width=True, height=360)

    # with st.expander("고급: 회귀 기반 이상치(참고용)"):
    #     st.caption("선형 회귀 기반 이상치는 가정에 민감합니다. 탐색 참고용으로만 사용해 주세요.")
    #     numeric_cols = []
    #     for c in [QUALITY_COL, MASK_RATIO_COL, ABS_ERROR_COL, ERROR_COL, PROC_COL]:
    #         if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
    #             numeric_cols.append(c)

    #     if len(numeric_cols) < 2:
    #         st.info("숫자형 컬럼이 부족해서 회귀 기반 탐색은 생략됩니다.")
    #     else:
    #         col1, col2, col3 = st.columns([1, 1, 1])
    #         with col1:
    #             x_col = st.selectbox("X Axis (predictor)", options=numeric_cols, index=0, key="p4_reg_x")
    #         with col2:
    #             y_col = st.selectbox("Y Axis (target)", options=numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="p4_reg_y")
    #         with col3:
    #             sigma_threshold = st.slider("잔차 임계값(표준편차 배수)", min_value=1.0, max_value=6.0, value=3.0, step=0.5, key="p4_reg_sigma")

    #         model_df = perform_linear_regression(df, x_col=x_col, y_col=y_col, sigma_threshold=float(sigma_threshold))
    #         outliers = model_df[model_df["Status"] == "Outlier"].copy()

    #         tooltip_reg = []
    #         for c in [TS_COL, WEATHER_COL, TOD_COL, x_col, y_col, "Status"]:
    #             if c in model_df.columns:
    #                 tooltip_reg.append(c)

    #         st.altair_chart(
    #             alt.Chart(model_df)
    #             .mark_point(filled=True, opacity=0.5)
    #             .encode(
    #                 x=alt.X(x_col, scale=alt.Scale(zero=False)),
    #                 y=alt.Y(y_col, scale=alt.Scale(zero=False)),
    #                 color=alt.Color("Status:N").legend(None),
    #                 shape=alt.Shape("Status:N").scale(range=["circle", "cross"]).legend(None),
    #                 tooltip=tooltip_reg,
    #             ).properties(height=420),
    #             use_container_width=True,
    #         )

    #         st.subheader("Detected Outliers (회귀 기반)")
    #         show_cols_reg = []
    #         for c in [TS_COL, WEATHER_COL, TOD_COL, QUALITY_COL, MASK_RATIO_COL, ABS_ERROR_COL, ERROR_COL, PROC_COL]:
    #             if c in outliers.columns:
    #                 show_cols_reg.append(c)
    #         st.dataframe(outliers[show_cols_reg], column_config=COLUMN_CONFIG, use_container_width=True, height=360)

# =============================================================================
# Part V: Top low-quality frames

st.divider()
st.markdown("## Part V: 최저 품질 프레임 Top 20 요약")

top = df.sort_values(QUALITY_COL).head(20)

mcols = st.columns(4)
mcols[0].metric("Avg Quality (Top 20)", f"{float(top[QUALITY_COL].mean()):.1f}")
mcols[1].metric("Avg Mask Ratio (Top 20)", f"{float(top[MASK_RATIO_COL].mean()):.4f}")
if ABS_ERROR_COL in top.columns:
    mcols[2].metric("Avg Abs Error", f"{float(top[ABS_ERROR_COL].mean()):.1f}")
else:
    mcols[2].metric("Avg Abs Error", "N/A")
if PROC_COL in top.columns:
    mcols[3].metric("Avg Proc Time", f"{float(top[PROC_COL].mean()):.1f} ms")
else:
    mcols[3].metric("Avg Proc Time", "N/A")

show = [EVENT_ID_COL, RUN_ID_COL, TS_COL, WEATHER_COL, TOD_COL, QUALITY_COL, MASK_RATIO_COL]
for c in [ABS_ERROR_COL, PROC_COL]:
    if c in top.columns:
        show.append(c)

st.dataframe(top[[c for c in show if c in top.columns]], column_config=COLUMN_CONFIG, height=360)

# =============================================================================
# Part VI: Browse

st.divider()
st.markdown("## Part VI: 전체 로그 보기")
st.dataframe(df, height=560, column_config=COLUMN_CONFIG)