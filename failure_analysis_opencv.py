import streamlit as st
import altair as alt
import polars as pl
import numpy as np

st.set_page_config(page_title="실패분석", page_icon="🛣️", layout="wide")

# =============================================================================
# Canonical columns (you can rename your CSV columns to match, or rely on auto-rename)

FRAME_COL = "Frame ID"                 # str/int
TS_COL = "Timestamp"                   # optional (ms or ISO string)

QUALITY_COL = "Lane Quality Score"     # 0~100 (higher is better)  [REQUIRED]
MASK_RATIO_COL = "Mask White Ratio"    # 0~1 (white pixels / mask pixels) [REQUIRED]

ERROR_COL = "Lane Error"               # signed (e.g., pixels)
ABS_ERROR_COL = "Abs Lane Error"

LEFT_SPEED_COL = "Left Speed"          # optional
RIGHT_SPEED_COL = "Right Speed"        # optional

PROC_COL = "Processing Time (ms)"      # optional
WEATHER_COL = "Weather"                # optional
TOD_COL = "Time of Day"                # optional

# =============================================================================
# Helpers

def _standardize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Rename common variants to canonical column names."""
    rename_map = {}

    # Frame / timestamp
    if FRAME_COL not in df.columns:
        for c in ["frame_id", "frame", "Frame", "id", "index", "idx"]:
            if c in df.columns:
                rename_map[c] = FRAME_COL
                break
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

    # Speeds
    if LEFT_SPEED_COL not in df.columns:
        for c in ["left_speed", "l_speed", "motor_left", "LeftMotor", "left_pwm"]:
            if c in df.columns:
                rename_map[c] = LEFT_SPEED_COL
                break
    if RIGHT_SPEED_COL not in df.columns:
        for c in ["right_speed", "r_speed", "motor_right", "RightMotor", "right_pwm"]:
            if c in df.columns:
                rename_map[c] = RIGHT_SPEED_COL
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
        df = df.rename(rename_map)
    return df


def _ensure_fields(df: pl.DataFrame) -> pl.DataFrame:
    """Fill convenience columns and sanitize ratio/quality ranges."""
    # Frame fallback
    if FRAME_COL not in df.columns:
        df = df.with_row_index(name="__idx").with_columns(pl.col("__idx").alias(FRAME_COL)).drop("__idx")

    # Required columns check happens later, but we can sanitize if present:
    if QUALITY_COL in df.columns:
        df = df.with_columns(pl.col(QUALITY_COL).cast(pl.Float64).clip(0, 100))

    if MASK_RATIO_COL in df.columns:
        r = pl.col(MASK_RATIO_COL).cast(pl.Float64)
        # accept 0~1 or 0~100
        df = df.with_columns(
            pl.when(r > 1.5).then((r / 100.0)).otherwise(r).clip(0, 1).alias(MASK_RATIO_COL)
        )

    if ERROR_COL in df.columns and ABS_ERROR_COL not in df.columns:
        df = df.with_columns(pl.col(ERROR_COL).cast(pl.Float64).abs().alias(ABS_ERROR_COL))

    # Optional defaults
    if WEATHER_COL not in df.columns:
        df = df.with_columns(pl.lit("Unknown").alias(WEATHER_COL))
    if TOD_COL not in df.columns:
        df = df.with_columns(pl.lit("Unknown").alias(TOD_COL))

    return df


def _describe_missing(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    rows = []
    n = df.height
    for c in cols:
        if c not in df.columns:
            rows.append({"column": c, "present": False, "missing_rate": 1.0, "dtype": "N/A"})
        else:
            miss = df.select(pl.col(c).is_null().mean()).item()
            rows.append({"column": c, "present": True, "missing_rate": float(miss), "dtype": str(df[c].dtype)})
    return pl.DataFrame(rows).with_columns(
        (pl.col("missing_rate") * 100).round(2).alias("missing_%")
    ).drop("missing_rate")


def perform_linear_regression(df: pl.DataFrame, x_col: str, y_col: str, sigma_threshold: float) -> pl.DataFrame:
    clean_df = df.drop_nulls([x_col, y_col])
    if clean_df.is_empty():
        return clean_df.with_columns(pl.lit("In Range").alias("Status"))

    x = clean_df[x_col].to_numpy()
    y = clean_df[y_col].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)
    predictions = (slope * x) + intercept
    residuals = y - predictions
    std_dev = float(np.std(residuals)) if len(residuals) else 0.0

    upper_bound = predictions + (sigma_threshold * std_dev)
    lower_bound = predictions - (sigma_threshold * std_dev)

    return clean_df.with_columns(
        [
            pl.Series("Predicted", predictions),
            pl.Series("Upper Bound", upper_bound),
            pl.Series("Lower Bound", lower_bound),
            pl.when(
                (pl.col(y_col) > pl.Series(upper_bound)) | (pl.col(y_col) < pl.Series(lower_bound))
            ).then(pl.lit("Outlier")).otherwise(pl.lit("In Range")).alias("Status"),
        ]
    )


def draw_histogram(df: pl.DataFrame, metric_name: str, bins: int = 20, height: int = 220):
    clean_df = df.drop_nulls(subset=[metric_name])
    if clean_df.is_empty():
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
st.caption("YOLO Confidence/IoU 없이, 직접 계산한 Lane Quality Score / Mask White Ratio를 기준으로 분석합니다.")

uploaded = st.file_uploader("주행 로그 CSV 업로드", type=["csv"])
use_demo = st.toggle("데모 데이터 사용", value=(uploaded is None))

def _generate_demo(n: int = 1200) -> pl.DataFrame:
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

    base_speed = 0.5
    kp = 0.01
    steer = err * kp
    left_speed = np.clip(base_speed + steer, -1.0, 1.0)
    right_speed = np.clip(base_speed - steer, -1.0, 1.0)

    proc = np.random.normal(28, 5, n)
    df = pl.DataFrame({
        FRAME_COL: np.arange(n),
        TS_COL: np.arange(n) * 100,  # ms
        WEATHER_COL: weather,
        TOD_COL: tod,
        MASK_RATIO_COL: mask_ratio,
        QUALITY_COL: quality,
        ERROR_COL: err,
        PROC_COL: proc,
        LEFT_SPEED_COL: left_speed,
        RIGHT_SPEED_COL: right_speed,
    }).with_columns(pl.col(ERROR_COL).abs().alias(ABS_ERROR_COL))
    return df


if uploaded is not None:
    try:
        df = pl.read_csv(uploaded, infer_schema_length=1000)
    except Exception as e:
        st.error(f"CSV 로드 실패: {e}")
        st.stop()
else:
    df = _generate_demo() if use_demo else pl.DataFrame()

if df.is_empty():
    st.info("CSV를 업로드하거나 '데모 데이터 사용'을 켜세요.")
    st.stop()

df = _standardize_columns(df)
df = _ensure_fields(df)

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

# Column config
COLUMN_CONFIG = {
    FRAME_COL: st.column_config.TextColumn(pinned=True),
    TS_COL: st.column_config.TextColumn(),
    QUALITY_COL: st.column_config.ProgressColumn(min_value=0, max_value=100, format="compact", width=130),
    MASK_RATIO_COL: st.column_config.NumberColumn(format="%.4f"),
    ERROR_COL: st.column_config.NumberColumn(format="%.2f"),
    ABS_ERROR_COL: st.column_config.NumberColumn(format="%.2f"),
    LEFT_SPEED_COL: st.column_config.NumberColumn(format="%.3f"),
    RIGHT_SPEED_COL: st.column_config.NumberColumn(format="%.3f"),
    PROC_COL: st.column_config.NumberColumn(format="%.1f ms"),
    WEATHER_COL: st.column_config.TextColumn(),
    TOD_COL: st.column_config.TextColumn(),
}

# =============================================================================
# Part 0: sanity checks

st.divider()
st.subheader("Part 0: 컬럼/결측 확인")
check_cols = [QUALITY_COL, MASK_RATIO_COL, ERROR_COL, LEFT_SPEED_COL, RIGHT_SPEED_COL, PROC_COL, WEATHER_COL, TOD_COL]
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
                tooltip=[FRAME_COL, TS_COL, WEATHER_COL, TOD_COL, QUALITY_COL, ABS_ERROR_COL, "Status"],
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
            model_df.sort(ABS_ERROR_COL, descending=True).select(
                [FRAME_COL, TS_COL, WEATHER_COL, TOD_COL, QUALITY_COL, ABS_ERROR_COL] +
                ([PROC_COL] if PROC_COL in model_df.columns else [])
            ).head(20),
            column_config=COLUMN_CONFIG,
            height=360,
        )

    with b:
        st.caption("Low-quality frames (인식 품질 붕괴 의심)")
        st.dataframe(
            df.sort(QUALITY_COL).select(
                [FRAME_COL, TS_COL, WEATHER_COL, TOD_COL, QUALITY_COL] +
                ([ABS_ERROR_COL] if ABS_ERROR_COL in df.columns else []) +
                ([PROC_COL] if PROC_COL in df.columns else [])
            ).head(20),
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
        alt.Chart(df.drop_nulls([MASK_RATIO_COL, QUALITY_COL]))
        .mark_point(filled=True, opacity=0.45)
        .encode(
            x=alt.X(MASK_RATIO_COL, type="quantitative", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(QUALITY_COL, type="quantitative", scale=alt.Scale(domain=[0, 100])),
            tooltip=[FRAME_COL, TS_COL, WEATHER_COL, TOD_COL, MASK_RATIO_COL, QUALITY_COL],
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

def _group_summary(data: pl.DataFrame, group_col: str, metric_col: str, include_unknown: bool) -> pl.DataFrame:
    d = data.drop_nulls(subset=[group_col, metric_col])
    if not include_unknown:
        d = d.filter(pl.col(group_col) != "Unknown")
    return (
        d.group_by(group_col)
        .agg(
            pl.len().alias("num_frames"),
            pl.col(metric_col).median().alias("median"),
            pl.col(metric_col).mean().alias("mean"),
            pl.col(metric_col).quantile(0.25).alias("p25"),
            pl.col(metric_col).quantile(0.75).alias("p75"),
        )
        .sort("median", descending=True)
    )

def _bar_chart(summary: pl.DataFrame, group_col: str, metric_label: str, domain=None, height: int = 320):
    if summary.is_empty():
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
        uniq_non_unknown = df.filter(pl.col(g) != "Unknown").select(pl.col(g)).unique().height
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
    if summary.height <= 1:
        st.info("선택한 그룹 기준에서 비교 가능한 범주가 1개뿐입니다(대부분 Unknown일 수 있음).")
    _bar_chart(summary, group_col, metric_label, domain=domain)

    st.caption("표는 median/mean과 IQR(p25~p75)을 함께 제공합니다. 프레임 수가 적은 그룹은 해석에 주의하세요.")
    st.dataframe(summary, hide_index=True, use_container_width=True)

# =============================================================================
# Part IV: Outlier Explorer


st.divider()
st.markdown("## Part IV: Outlier Explorer (회귀 기반)")

numeric_cols = []
for c in [QUALITY_COL, MASK_RATIO_COL, ABS_ERROR_COL, ERROR_COL, PROC_COL, LEFT_SPEED_COL, RIGHT_SPEED_COL]:
    if c in df.columns and df[c].dtype in (pl.Int64, pl.Int32, pl.Float32, pl.Float64):
        numeric_cols.append(c)

if len(numeric_cols) < 2:
    st.info("숫자형 컬럼이 부족해서 Part IV는 생략됩니다.")
else:
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X Axis (predictor)", options=numeric_cols, index=0)
    with col2:
        y_col = st.selectbox("Y Axis (target)", options=numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

    sigma_val = st.slider("Confidence interval (sigma)", min_value=0.5, max_value=4.0, value=2.0, step=0.1)

    model_df = perform_linear_regression(df, x_col, y_col, sigma_val)
    outliers = model_df.filter(pl.col("Status") == "Outlier")

    st.altair_chart(
        alt.Chart(model_df)
        .mark_point(filled=True, opacity=0.5)
        .encode(
            x=alt.X(x_col, scale=alt.Scale(zero=False)),
            y=alt.Y(y_col, scale=alt.Scale(zero=False)),
            color=alt.Color("Status:N").legend(None),
            shape=alt.Shape("Status:N").scale(range=["circle", "cross"]).legend(None),
            tooltip=[FRAME_COL, TS_COL, WEATHER_COL, TOD_COL, x_col, y_col, "Status"],
        ).properties(height=420),
        use_container_width=True,
    )

    st.subheader("Detected Outliers")
    show_cols = [FRAME_COL, TS_COL, WEATHER_COL, TOD_COL, x_col, y_col]
    for extra in [QUALITY_COL, MASK_RATIO_COL, PROC_COL]:
        if extra in df.columns and extra not in show_cols:
            show_cols.append(extra)
    st.dataframe(outliers.select([c for c in show_cols if c in outliers.columns]), column_config=COLUMN_CONFIG, height=360)

# =============================================================================
# Part V: Top low-quality frames

st.divider()
st.markdown("## Part V: 최저 품질 프레임 Top 20 요약")

top = df.sort(QUALITY_COL).head(20)

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

show = [FRAME_COL, TS_COL, WEATHER_COL, TOD_COL, QUALITY_COL, MASK_RATIO_COL]
for c in [ABS_ERROR_COL, PROC_COL, LEFT_SPEED_COL, RIGHT_SPEED_COL]:
    if c in top.columns:
        show.append(c)

st.dataframe(top.select([c for c in show if c in top.columns]), column_config=COLUMN_CONFIG, height=360)

# =============================================================================
# Part VI: Browse

st.divider()
st.markdown("## Part VI: 전체 로그 보기")
st.dataframe(df, height=560, column_config=COLUMN_CONFIG)
