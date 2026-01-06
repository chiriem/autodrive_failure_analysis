import os
from pathlib import Path

import altair as alt
import numpy as np
import time
import polars as pl
import streamlit as st


st.set_page_config(page_title="실패분석", page_icon=":clapper:", layout="wide")


FRAME_COL = "Frame ID"
WEATHER_COL = "Weather"
TIME_COL = "Time of Day"
PROC_COL = "Processing Time (ms)"
FAIL_TYPE_COL = "Failure Type"
STEER_COL = "Steering Angle"
CONF_COL = "Model Confidence"
IMG_COL = "Image Path"
TS_COL = "Timestamp (ms)"


PLOT_H = 320
MARK_SIZE = 60


def _col_exists(df: pl.DataFrame, col: str) -> bool:
    return col in df.columns


def _is_all_null(df: pl.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return True
    s = df.get_column(col)
    return s.null_count() == df.height


@st.cache_data(show_spinner=False)
def load_csv_to_pl(file_bytes: bytes) -> pl.DataFrame:
    return pl.read_csv(file_bytes, try_parse_dates=True, ignore_errors=True)


@st.cache_data(show_spinner=False)
def load_csv_path_to_pl(csv_path: str) -> pl.DataFrame:
    return pl.read_csv(csv_path, try_parse_dates=True, ignore_errors=True)


@st.cache_data(show_spinner=False)
def generate_demo_df(
    n_rows: int,
    weather: str,
    time_of_day: str,
    include_confidence: bool,
    include_images: bool,
) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    start_ts = int(time.time() * 1000)

    frame_ids = np.arange(n_rows, dtype=np.int64)
    ts = start_ts + (frame_ids * 33)

    base_steer = 90 + rng.normal(0, 4, n_rows)
    steer = base_steer.copy()

    proc = rng.normal(30, 6, n_rows)
    proc = np.clip(proc, 10, 120)

    failure_type = np.array(["missed_detection"] * n_rows, dtype=object)

    idx_unstable = rng.choice(n_rows, size=max(5, n_rows // 25), replace=False)
    steer[idx_unstable] += rng.normal(30, 10, len(idx_unstable))
    proc[idx_unstable] += rng.normal(15, 8, len(idx_unstable))
    failure_type[idx_unstable] = "unstable_steering"

    idx_slow = rng.choice(n_rows, size=max(5, n_rows // 30), replace=False)
    proc[idx_slow] += rng.normal(40, 10, len(idx_slow))
    failure_type[idx_slow] = "slow_inference"

    failure_type = np.where(proc > 80, "slow_inference", failure_type)

    conf = None
    if include_confidence:
        conf = rng.normal(6.5, 1.2, n_rows)
        conf = np.clip(conf, 0, 10)
        conf = np.where(failure_type == "missed_detection", rng.normal(2.5, 1.0, n_rows), conf)
        conf = np.clip(conf, 0, 10)

    img_paths = None
    if include_images:
        img_paths = np.array([None] * n_rows, dtype=object)
        for i in idx_unstable[: min(20, len(idx_unstable))]:
            img_paths[i] = f"/home/pi/AI_CAR/logs/fail_images/demo_fail_{int(i):06d}.jpg"

    df = pl.DataFrame(
        {
            FRAME_COL: frame_ids,
            WEATHER_COL: [weather] * n_rows,
            TIME_COL: [time_of_day] * n_rows,
            FAIL_TYPE_COL: failure_type,
            PROC_COL: proc.astype(float),
            STEER_COL: steer.astype(float),
            TS_COL: ts,
        }
    )

    if include_confidence:
        df = df.with_columns(pl.Series(CONF_COL, conf.astype(float)))

    if include_images:
        df = df.with_columns(pl.Series(IMG_COL, img_paths))

    return df


def clean_and_cast(df: pl.DataFrame) -> pl.DataFrame:
    required = [FRAME_COL, WEATHER_COL, TIME_COL, PROC_COL, FAIL_TYPE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"필수 컬럼이 없습니다: {missing}")
        st.stop()

    df = df.with_columns(
        [
            pl.col(FRAME_COL).cast(pl.Int64, strict=False),
            pl.col(WEATHER_COL).cast(pl.Utf8, strict=False).fill_null("Unknown"),
            pl.col(TIME_COL).cast(pl.Utf8, strict=False).fill_null("Unknown"),
            pl.col(FAIL_TYPE_COL).cast(pl.Utf8, strict=False).fill_null("Unknown"),
            pl.col(PROC_COL).cast(pl.Float64, strict=False),
        ]
    )

    df = df.with_columns(
        pl.when(pl.col(PROC_COL).is_finite()).then(pl.col(PROC_COL)).otherwise(None).alias(PROC_COL)
    ).with_columns(
        pl.col(PROC_COL).fill_null(0.0).alias(PROC_COL)
    )

    if _col_exists(df, STEER_COL):
        df = df.with_columns(pl.col(STEER_COL).cast(pl.Float64, strict=False))
        df = df.with_columns(
            pl.when(pl.col(STEER_COL).is_finite()).then(pl.col(STEER_COL)).otherwise(None).alias(STEER_COL)
        ).with_columns(pl.col(STEER_COL).fill_null(0.0).alias(STEER_COL))

    if _col_exists(df, CONF_COL):
        df = df.with_columns(pl.col(CONF_COL).cast(pl.Float64, strict=False))
        df = df.with_columns(
            pl.when(pl.col(CONF_COL).is_finite()).then(pl.col(CONF_COL)).otherwise(None).alias(CONF_COL)
        )

    if _col_exists(df, IMG_COL):
        df = df.with_columns(pl.col(IMG_COL).cast(pl.Utf8, strict=False))

    if _col_exists(df, TS_COL):
        df = df.with_columns(pl.col(TS_COL).cast(pl.Int64, strict=False))

    if df.is_empty():
        st.error("데이터가 비어 있습니다.")
        st.stop()

    return df


@st.cache_data(show_spinner=False)
def add_control_stability_features(df: pl.DataFrame) -> pl.DataFrame:
    base = df

    if TS_COL in base.columns:
        base = base.sort(TS_COL)
        base = base.with_columns(pl.col(TS_COL).diff().alias("dt_ms"))
    else:
        base = base.sort(FRAME_COL)
        base = base.with_columns(pl.lit(None).alias("dt_ms"))

    if STEER_COL in base.columns:
        base = base.with_columns(pl.col(STEER_COL).diff().abs().alias("delta_steer"))
    else:
        base = base.with_columns(pl.lit(None).alias("delta_steer"))

    base = base.with_columns(
        pl.when((pl.col("dt_ms").is_not_null()) & (pl.col("dt_ms") > 0) & (pl.col("delta_steer").is_not_null()))
        .then(pl.col("delta_steer") / (pl.col("dt_ms") / 1000.0))
        .otherwise(None)
        .alias("steer_rate_per_s")
    )

    return base


def sample_for_charts(df: pl.DataFrame, n: int) -> pl.DataFrame:
    if df.height <= n:
        return df
    return df.sample(n=n, seed=42, shuffle=True)


def data_health_panel(df: pl.DataFrame):
    st.subheader("데이터 상태")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("행 수", f"{df.height:,}")
    c2.metric("Failure Type 수", str(df.get_column(FAIL_TYPE_COL).n_unique()))
    c3.metric("Steering Angle 존재", "있음" if STEER_COL in df.columns else "없음")
    c4.metric("Image Path 존재", "있음" if IMG_COL in df.columns else "없음")

    st.caption("GT가 없으므로 IoU 기반 평가는 제외되고, 이벤트 및 조향 안정성 중심으로 분석합니다.")


def render_failure_overview(df: pl.DataFrame):
    st.header("Part I: 실패 개요")

    st.markdown(
        """
실패 로그 중심 분석입니다.
- Failure Type별 빈도
- Weather, Time of Day 조건별 실패 패턴
"""
    )

    fail_counts = (
        df.group_by(FAIL_TYPE_COL)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    chart = (
        alt.Chart(fail_counts)
        .mark_bar()
        .encode(
            x=alt.X(f"{FAIL_TYPE_COL}:N", sort="-y", title="Failure Type"),
            y=alt.Y("count:Q", title="Count"),
            tooltip=[FAIL_TYPE_COL, "count"],
        )
        .properties(height=PLOT_H)
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("조건별 실패 분포")
    gcols = st.columns(2)

    with gcols[0]:
        by_weather = (
            df.group_by([WEATHER_COL, FAIL_TYPE_COL])
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        st.altair_chart(
            alt.Chart(by_weather)
            .mark_bar()
            .encode(
                x=alt.X(f"{WEATHER_COL}:N", title="Weather"),
                y=alt.Y("count:Q", title="Count"),
                color=alt.Color(f"{FAIL_TYPE_COL}:N", title="Failure Type"),
                tooltip=[WEATHER_COL, FAIL_TYPE_COL, "count"],
            )
            .properties(height=PLOT_H),
            use_container_width=True,
        )

    with gcols[1]:
        by_time = (
            df.group_by([TIME_COL, FAIL_TYPE_COL])
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        st.altair_chart(
            alt.Chart(by_time)
            .mark_bar()
            .encode(
                x=alt.X(f"{TIME_COL}:N", title="Time of Day"),
                y=alt.Y("count:Q", title="Count"),
                color=alt.Color(f"{FAIL_TYPE_COL}:N", title="Failure Type"),
                tooltip=[TIME_COL, FAIL_TYPE_COL, "count"],
            )
            .properties(height=PLOT_H),
            use_container_width=True,
        )


def render_control_stability(df: pl.DataFrame, delta_thresh: float, sample_n: int, use_sampling: bool):
    st.header("Part II: 조향 안정성 분석")

    if STEER_COL not in df.columns:
        st.info("Steering Angle 컬럼이 없어서 조향 안정성 분석을 수행할 수 없습니다.")
        return

    st.markdown(
        """
조향 안정성은 주행 품질과 직접 연결되는 지표입니다.
- 조향각 자체 분포
- 프레임 간 조향각 변화량(delta_steer)
- 임계값을 넘는 급변 프레임 탐지
"""
    )

    view_df = sample_for_charts(df, sample_n) if use_sampling else df

    st.subheader("조향각 분포")
    steer_hist = (
        alt.Chart(view_df.drop_nulls([STEER_COL]))
        .mark_bar(binSpacing=0)
        .encode(
            x=alt.X(f"{STEER_COL}:Q", bin=alt.Bin(maxbins=30), title="Steering Angle"),
            y=alt.Y("count()", title="Count"),
            tooltip=[alt.Tooltip("count()", title="Count")],
        )
        .properties(height=PLOT_H)
    )
    st.altair_chart(steer_hist, use_container_width=True)

    st.subheader("조향각 변화량(delta_steer) 분포")
    delta_hist = (
        alt.Chart(view_df.drop_nulls(["delta_steer"]))
        .mark_bar(binSpacing=0)
        .encode(
            x=alt.X("delta_steer:Q", bin=alt.Bin(maxbins=30), title="delta_steer"),
            y=alt.Y("count()", title="Count"),
            tooltip=[alt.Tooltip("count()", title="Count")],
        )
        .properties(height=PLOT_H)
    )
    st.altair_chart(delta_hist, use_container_width=True)

    st.subheader("급변 프레임 탐지")
    st.caption("delta_steer 임계값을 넘는 프레임을 조향 불안정 후보로 분류합니다. 이 기준은 분석 목적의 휴리스틱이며 절대 기준이 아닙니다.")

    unstable = df.filter(pl.col("delta_steer").is_not_null() & (pl.col("delta_steer") >= float(delta_thresh)))
    st.metric("unstable 후보 프레임 수", f"{unstable.height:,}")

    cols = [FRAME_COL, WEATHER_COL, TIME_COL, FAIL_TYPE_COL, PROC_COL, STEER_COL, "delta_steer"]
    if CONF_COL in df.columns:
        cols.append(CONF_COL)
    if IMG_COL in df.columns:
        cols.append(IMG_COL)
    if TS_COL in df.columns:
        cols.append(TS_COL)

    show_df = unstable.select([c for c in cols if c in unstable.columns]).sort("delta_steer", descending=True)

    st.dataframe(show_df.head(50).to_pandas(), height=360, use_container_width=True)

    if IMG_COL in df.columns:
        with st.expander("실패 이미지 미리보기", expanded=False):
            n_preview = st.slider("미리보기 개수", min_value=1, max_value=20, value=8, step=1)
            paths = [p for p in show_df.head(int(n_preview)).get_column(IMG_COL).to_list() if p]

            if not paths:
                st.info("표시할 이미지 경로가 없습니다.")
            else:
                for p in paths:
                    st.write(p)
                    try:
                        st.image(p, use_container_width=True)
                    except Exception:
                        st.warning("이미지를 불러오지 못했습니다. 경로 접근 권한 또는 파일 존재 여부를 확인해 주세요.")


def render_latency(df: pl.DataFrame, sample_n: int, use_sampling: bool):
    st.header("Part III: 처리시간 분석")

    st.markdown(
        """
실시간 주행에서는 처리시간이 길어질수록 안전 여유가 줄어듭니다.
- 처리시간 분포
- 처리시간과 조향 급변 간 관계
"""
    )

    view_df = sample_for_charts(df, sample_n) if use_sampling else df

    hist = (
        alt.Chart(view_df)
        .mark_bar(binSpacing=0)
        .encode(
            x=alt.X(f"{PROC_COL}:Q", bin=alt.Bin(maxbins=30), title="Processing Time (ms)"),
            y=alt.Y("count()", title="Count"),
            tooltip=[alt.Tooltip("count()", title="Count")],
        )
        .properties(height=PLOT_H)
    )
    st.altair_chart(hist, use_container_width=True)

    if STEER_COL in df.columns and "delta_steer" in df.columns:
        scatter_df = view_df.drop_nulls([PROC_COL, "delta_steer"])
        if scatter_df.height > 0:
            sc = (
                alt.Chart(scatter_df)
                .mark_point(size=MARK_SIZE, opacity=0.4)
                .encode(
                    x=alt.X(f"{PROC_COL}:Q", title="Processing Time (ms)"),
                    y=alt.Y("delta_steer:Q", title="delta_steer"),
                    color=alt.Color(f"{FAIL_TYPE_COL}:N", title="Failure Type"),
                    tooltip=[FRAME_COL, FAIL_TYPE_COL, PROC_COL, "delta_steer"],
                )
                .properties(height=PLOT_H)
            )
            st.altair_chart(sc, use_container_width=True)


def render_confidence_optional(df: pl.DataFrame, sample_n: int, use_sampling: bool):
    st.header("Part IV: Model Confidence (선택)")

    if CONF_COL not in df.columns or _is_all_null(df, CONF_COL):
        st.info("Model Confidence 값이 없어서 이 파트는 생략합니다.")
        return

    st.markdown(
        """
confidence는 실패의 원인을 단정하는 지표가 아니라, 실패 상황에서의 경향을 보기 위한 보조 신호입니다.
"""
    )

    view_df = sample_for_charts(df, sample_n) if use_sampling else df
    clean = view_df.drop_nulls([CONF_COL])

    hist = (
        alt.Chart(clean)
        .mark_bar(binSpacing=0)
        .encode(
            x=alt.X(f"{CONF_COL}:Q", bin=alt.Bin(maxbins=30), title="Model Confidence"),
            y=alt.Y("count()", title="Count"),
            tooltip=[alt.Tooltip("count()", title="Count")],
        )
        .properties(height=PLOT_H)
    )
    st.altair_chart(hist, use_container_width=True)

    box = (
        alt.Chart(clean)
        .mark_boxplot()
        .encode(
            x=alt.X(f"{FAIL_TYPE_COL}:N", title="Failure Type"),
            y=alt.Y(f"{CONF_COL}:Q", title="Model Confidence"),
            color=alt.Color(f"{FAIL_TYPE_COL}:N", legend=None),
        )
        .properties(height=PLOT_H)
    )
    st.altair_chart(box, use_container_width=True)


def _load_df_from_ui(
    uploaded,
    local_path: str,
    demo_clicked: bool,
    demo_rows: int,
    demo_conf: bool,
    demo_images: bool,
) -> pl.DataFrame:
    if demo_clicked:
        return generate_demo_df(
            n_rows=int(demo_rows),
            weather="Sunny",
            time_of_day="Day",
            include_confidence=bool(demo_conf),
            include_images=bool(demo_images),
        )

    if uploaded is not None:
        return load_csv_to_pl(uploaded.getvalue())

    csv_path = local_path.strip()
    if not csv_path:
        st.info("CSV를 업로드하거나, CSV 경로를 입력하거나, 데모 데이터를 생성해 주세요.")
        st.stop()

    if not Path(csv_path).exists():
        st.error("입력한 경로에 파일이 없습니다.")
        st.stop()

    return load_csv_path_to_pl(csv_path)


def main():
    st.title("자율주행 차선 인식 실패 분석")
    st.caption("버튼을 누르면 임시 데이터를 생성해서 즉시 동작 여부를 확인할 수 있습니다.")

    with st.sidebar:
        st.header("데이터 로드")

        with st.expander("데모 데이터 생성", expanded=True):
            demo_rows = st.number_input("데모 행 수", min_value=100, max_value=20000, value=1500, step=100)
            st.caption("Weather와 Time of Day는 실제 주행 시 라즈베리파이에서 입력되어 CSV에 저장됩니다. 데모 데이터는 임시 고정값을 사용합니다.")
            demo_conf = st.checkbox("Model Confidence 포함", value=True)
            demo_images = st.checkbox("Image Path 포함", value=False)
            demo_clicked = st.button("데모 데이터로 실행")

        st.divider()
        uploaded = st.file_uploader("라즈베리파이 실패 로그 CSV 업로드", type=["csv"])
        local_path = st.text_input("또는 로컬 CSV 경로", value="")
        st.caption("데모 버튼을 누르면 업로드/경로 입력 없이도 실행됩니다.")

        st.divider()
        st.header("성능 옵션")
        use_sampling = st.checkbox("차트 샘플링 사용", value=True)
        sample_n = st.number_input("차트 샘플 개수", min_value=300, max_value=50000, value=5000, step=500)

        st.divider()
        st.header("조향 불안정 기준")
        delta_thresh = st.slider("delta_steer 임계값", min_value=1.0, max_value=90.0, value=15.0, step=1.0)

    raw_df = _load_df_from_ui(
        uploaded=uploaded,
        local_path=local_path,
        demo_clicked=demo_clicked,
        demo_rows=int(demo_rows),        demo_conf=bool(demo_conf),
        demo_images=bool(demo_images),
    )

    df = clean_and_cast(raw_df)
    df = add_control_stability_features(df)

    data_health_panel(df)

    st.divider()
    render_failure_overview(df)

    st.divider()
    render_control_stability(df, delta_thresh=float(delta_thresh), sample_n=int(sample_n), use_sampling=bool(use_sampling))

    st.divider()
    render_latency(df, sample_n=int(sample_n), use_sampling=bool(use_sampling))

    st.divider()
    render_confidence_optional(df, sample_n=int(sample_n), use_sampling=bool(use_sampling))

    st.divider()
    st.subheader("원본 로그")
    with st.expander("원본 테이블 보기", expanded=False):
        st.dataframe(df.to_pandas(), use_container_width=True, height=420)


if __name__ == "__main__":
    main()
