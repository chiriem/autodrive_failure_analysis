import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import textwrap
from pathlib import Path

from fa_utils import (
    _maybe_set_page_config,
    TS_COL, QUALITY_COL, MASK_RATIO_COL, ERROR_COL, ABS_ERROR_COL,
    PROC_COL, WEATHER_COL, TOD_COL, MODE_COL,
    RUN_ID_COL, ROW_IN_RUN_COL, EVENT_ID_COL,
    load_fixed_csv, try_load_fixed_csv,
    _make_tooltip, _describe_missing, perform_linear_regression, draw_histogram,
)

_FIXED_FILENAME = "0112_11_log.csv"
_BASELINE_FILENAME = "0109_18_log.csv"

MASK_RATIO_PLOT_MAX = 0.30  # x축을 이 범위로 고정(빈 공간 축소)

@st.cache_data(show_spinner=False)
def _load_fixed_data() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    return load_fixed_csv(_FIXED_FILENAME, run_id="run_0112_fixed", base_dir=base_dir)

@st.cache_data(show_spinner=False)
def _load_fixed_data_0109_baseline() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    return try_load_fixed_csv(_BASELINE_FILENAME, run_id="run_0109_baseline", base_dir=base_dir)


# =============================================================================
# Timestamp 기반 이상치(후보) 구간 시각화 (x축은 Timestamp 숫자 그대로 사용)

def _render_timestamp_outlier_windows_from_flags(df_flags: pd.DataFrame, pctl=None) -> None:
    """Part III에서 계산된 후보 플래그(mask_low/mask_high/err_high/proc_high)를
    Timestamp 구간으로 집계하여 '이상치(후보) 밀도'를 시각화합니다.

    - x축: Timestamp(숫자 그대로)
    - y축: 구간별 outlier count(개수)
    """
    if TS_COL not in df_flags.columns:
        st.info("Timestamp 컬럼이 없어 시간축 기반 구간 시각화를 생략합니다.")
        return

    ts_num = pd.to_numeric(df_flags[TS_COL], errors="coerce")
    if ts_num.notna().sum() == 0:
        st.info("Timestamp 값이 전부 결측이라 시간축 기반 구간 시각화를 생략합니다.")
        return

    dfv = df_flags.copy()
    dfv[TS_COL] = ts_num

    x_col = TS_COL
    x_type = "Q"
    x_title = "Timestamp"

    def _agg_timeline(flag_series: pd.Series):
        tmp = pd.DataFrame({
            "ts": pd.to_numeric(dfv[x_col], errors="coerce"),
            "is_out": flag_series.fillna(False).astype(bool),
        }).dropna(subset=["ts"])
        if tmp.empty:
            return pd.DataFrame(), ""

        n_bins = int(min(250, max(80, round(tmp.shape[0] / 30))))
        try:
            tmp["_bin"] = pd.cut(tmp["ts"], bins=n_bins)
        except Exception:
            # timestamp 값이 거의 상수/비정상일 때의 fallback
            tmp["_bin"] = pd.cut(np.arange(len(tmp)), bins=n_bins)

        g = (
            tmp.groupby("_bin", observed=True)
            .agg(ts_mid=("ts", "mean"), frames=("is_out", "size"), outliers=("is_out", "sum"))
            .reset_index(drop=True)
        )
        g = g.rename(columns={"ts_mid": "ts"})
        g["outlier_rate"] = (g["outliers"] / g["frames"] * 100.0).replace([np.inf, -np.inf], np.nan).round(2)
        return g, f"{n_bins} bins"

    def _plot_one(flag_cols_needed: list[str], flag_expr, metric_title: str) -> None:
        for c in flag_cols_needed:
            if c not in dfv.columns:
                st.info(f"{metric_title}: 필요한 플래그({c})가 없어 시각화를 생략합니다.")
                return

        flag = flag_expr(dfv)
        g, freq_txt = _agg_timeline(flag)
        if g.empty:
            st.info(f"{metric_title}: 유효한 timestamp 구간이 없어 시각화를 생략합니다.")
            return

        base = alt.Chart(g)
        chart = base.mark_line(point=True).encode(
            x=alt.X(f"ts:{x_type}", title=x_title),
            y=alt.Y("outliers:Q", title="Outliers (count)", scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip(f"ts:{x_type}", title="timestamp"),
                alt.Tooltip("frames:Q", title="frames"),
                alt.Tooltip("outliers:Q", title="outliers"),
            ],
        ).properties(height=210, title=metric_title)

        st.altair_chart(chart, use_container_width=True)

        top5 = g.sort_values(["outliers", "frames"], ascending=False).head(5).copy()
        top5["구간"] = top5["ts"].astype(str)

        with st.expander(f"{metric_title} 이상치(후보) 구간 Top 5", expanded=False):
            cap = f"집계 단위: {freq_txt}"
            if pctl is not None:
                cap += f" / 민감도(pctl): {pctl}"
            st.caption(cap)
            st.dataframe(top5[["구간", "frames", "outliers"]], hide_index=True, use_container_width=True)

    _plot_one(
        ["mask_low", "mask_high"],
        lambda d: d["mask_low"].astype(bool) | d["mask_high"].astype(bool),
        "Mask White Ratio 이상치(하위/상위 꼬리) 개수",
    )
    _plot_one(
        ["err_high"],
        lambda d: d["err_high"].astype(bool),
        "Abs Lane Error 이상치(상위 꼬리) 개수",
    )
    _plot_one(
        ["proc_high"],
        lambda d: d["proc_high"].astype(bool),
        "Processing Time 이상치(상위 꼬리) 개수",
    )

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

                present_df = tmp[tmp["Error Recorded"] == "Present"]
                missing_df = tmp[tmp["Error Recorded"] == "Missing"]

                enc = dict(
                    x=alt.X(f"{MASK_RATIO_COL}:Q", bin=alt.Bin(maxbins=30), title="Mask White Ratio"),
                    y=alt.Y("count():Q", title="Frames"),
                    tooltip=[alt.Tooltip("count():Q", title="frames")],
                )

                present = (
                    alt.Chart(present_df)
                    .mark_bar(opacity=0.75)
                    .encode(**enc)
                    .properties(height=140, title="Present")
                )

                missing = (
                    alt.Chart(missing_df)
                    .mark_bar(opacity=0.75)
                    .encode(**enc)
                    .properties(height=140, title="Missing")
                )

                chart = alt.vconcat(present, missing, spacing=10).resolve_scale(x="shared", y="shared")
                st.altair_chart(chart, use_container_width=True)

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
                    x=alt.X(MASK_RATIO_COL, type="quantitative", scale=alt.Scale(domain=[0, MASK_RATIO_PLOT_MAX])),
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
                    x=alt.X(MASK_RATIO_COL, type="quantitative", scale=alt.Scale(domain=[0, MASK_RATIO_PLOT_MAX])),
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
                        x=alt.X(MASK_RATIO_COL, type="quantitative", scale=alt.Scale(domain=[0, MASK_RATIO_PLOT_MAX])),
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
                            x=alt.X(MASK_RATIO_COL, type="quantitative", scale=alt.Scale(domain=[0, MASK_RATIO_PLOT_MAX])),
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
                        x=alt.X(MASK_RATIO_COL, type="quantitative", scale=alt.Scale(domain=[0, MASK_RATIO_PLOT_MAX])),
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
    st.markdown("## Part V: Timestamp 기반 이상치 구간 보기")
    st.caption("Part III에서 계산된 후보 플래그(mask_low/mask_high/err_high/proc_high)를 Timestamp 구간으로 집계합니다.")

    if TS_COL not in d.columns:
        st.info("Timestamp 컬럼이 없어 시간축 기반 구간 시각화를 생략합니다.")
    else:
        need_any = any(c in d.columns for c in ["mask_low", "mask_high", "err_high", "proc_high"])
        if not need_any:
            st.info("후보 플래그 컬럼이 없어(Part III 후보 탐지 불가) 시간축 기반 구간 시각화를 생략합니다.")
        else:
            _render_timestamp_outlier_windows_from_flags(d, pctl=pctl)

    st.divider()
    st.markdown("## Part VI: 전체 로그 보기")
    st.dataframe(df.drop(["Run ID", "Row In Run", "Event ID"], axis=1))

    _render_part_x(df=df, pctl=pctl, cand=cand)

# =============================================================================
# Part X: Improvements (analysis-driven)


def _render_part_x(df: pd.DataFrame, pctl: int, cand: 'pd.DataFrame | None' = None) -> None:
    st.divider()
    st.markdown("## Part X: 자율주행 개선사항 (분석 기반)")

    # -------------------------------------------------------------------------
    # Helpers
    def _to_num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    def _fmt(v, fmt: str) -> str:
        if v is None:
            return "N/A"
        try:
            if pd.isna(v):
                return "N/A"
        except Exception:
            pass
        return fmt.format(v)

    def _hist_with_rules(series: pd.Series, colname: str, title: str, rules: list[tuple[float, str]] | None = None,
                         maxbins: int = 40, height: int = 220, domain: tuple[float, float] | None = None):
        s = series.dropna()
        if s.empty:
            st.info(f"{colname} 데이터가 없어 시각화를 생략합니다.")
            return

        dd = pd.DataFrame({colname: s})
        x_enc = alt.X(f"{colname}:Q", bin=alt.Bin(maxbins=maxbins), title=colname)
        if domain is not None:
            x_enc = alt.X(f"{colname}:Q", bin=alt.Bin(maxbins=maxbins), title=colname, scale=alt.Scale(domain=list(domain)))

        base = (
            alt.Chart(dd)
            .mark_bar(binSpacing=0, opacity=0.7)
            .encode(
                x=x_enc,
                y=alt.Y("count():Q", title="Frames"),
                tooltip=[alt.Tooltip("count():Q", title="frames")],
            )
            .properties(height=height, title=title)
        )

        if not rules:
            st.altair_chart(base, use_container_width=True)
            return

        rule_df = pd.DataFrame([{"x": float(v), "label": str(label)} for v, label in rules if v is not None and not pd.isna(v)])
        if rule_df.empty:
            st.altair_chart(base, use_container_width=True)
            return

        rules_layer = (
            alt.Chart(rule_df)
            .mark_rule(strokeDash=[4, 2])
            .encode(
                x=alt.X("x:Q"),
                tooltip=[alt.Tooltip("label:N"), alt.Tooltip("x:Q", format=".4f")],
            )
        )

        st.altair_chart(base + rules_layer, use_container_width=True)

    def _metrics(xdf: pd.DataFrame) -> dict:
        m: dict = {}
        m["n_total"] = int(len(xdf))

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
            m["low_rate"] = float((ratio.le(m["low_th"]) & valid).mean() * 100.0) if valid.any() else None
            m["high_rate"] = float((ratio.ge(m["high_th"]) & valid).mean() * 100.0) if valid.any() else None

        abs_err = _to_num(xdf.get(ABS_ERROR_COL))
        ae = abs_err.dropna()
        if ae.empty:
            m.update({
                "abs_p95": None, "abs_p99": None, "abs_max": None,
                "abs_tail_th": None, "abs_tail_rate": None,
            })
        else:
            m["abs_p95"] = float(ae.quantile(0.95))
            m["abs_p99"] = float(ae.quantile(0.99))
            m["abs_max"] = float(ae.max())
            m["abs_tail_th"] = float(ae.quantile(pctl / 100.0))
            m["abs_tail_rate"] = float(ae.ge(m["abs_tail_th"]).mean() * 100.0)

        proc = _to_num(xdf.get(PROC_COL))
        pr = proc.dropna()
        if pr.empty:
            m.update({
                "proc_p95": None, "proc_p99": None, "proc_max": None,
                "proc_tail_th": None, "proc_tail_rate": None,
            })
        else:
            m["proc_p95"] = float(pr.quantile(0.95))
            m["proc_p99"] = float(pr.quantile(0.99))
            m["proc_max"] = float(pr.max())
            m["proc_tail_th"] = float(pr.quantile(pctl / 100.0))
            m["proc_tail_rate"] = float(pr.ge(m["proc_tail_th"]).mean() * 100.0)

        m["err_missing_rate"] = None
        if ERROR_COL in xdf.columns:
            m["err_missing_rate"] = float(_to_num(xdf[ERROR_COL]).isna().mean() * 100.0)

        return m

    # -------------------------------------------------------------------------
    # Current + baseline
    cur = _metrics(df)
    base_df = None
    base = None
    try:
        base_df = _load_fixed_data_0109_baseline()
        if (base_df is None) or (getattr(base_df, "empty", False)):
            base_df = None
            base = None
            st.info(f"0109 비교용 파일({_BASELINE_FILENAME})이 없어 비교(그래프/표 일부)를 생략합니다. (파일을 두면 자동 비교됩니다.)")
        else:
            base = _metrics(base_df)
    except Exception as e:
        st.warning(f"0109 비교 로드/계산 중 오류로 비교를 생략합니다: {e}")

    # -------------------------------------------------------------------------
    # 0) Insight summary + prioritized actions (auto)
    st.markdown("### 요약")

    n_total = int(cur.get("n_total") or len(df))

    # Recompute boolean flags for overlaps (safe)
    _ratio = _to_num(df.get(MASK_RATIO_COL)).clip(0, 1)
    _abs = _to_num(df.get(ABS_ERROR_COL))
    _proc = _to_num(df.get(PROC_COL))

    low_th = cur.get("low_th")
    high_th = cur.get("high_th")
    abs_tail_th = cur.get("abs_tail_th")
    proc_tail_th = cur.get("proc_tail_th")

    _mask_low = (_ratio.le(low_th)) if (low_th is not None) else pd.Series(False, index=df.index)
    _mask_high = (_ratio.ge(high_th)) if (high_th is not None) else pd.Series(False, index=df.index)
    _mask_ext = _mask_low | _mask_high
    _abs_high = (_abs.ge(abs_tail_th)) if (abs_tail_th is not None) else pd.Series(False, index=df.index)
    _proc_high = (_proc.ge(proc_tail_th)) if (proc_tail_th is not None) else pd.Series(False, index=df.index)

    _err_missing = pd.Series(False, index=df.index)
    if ERROR_COL in df.columns:
        _err_missing = _to_num(df[ERROR_COL]).isna()

    def _cnt(mask: pd.Series) -> int:
        try:
            return int(mask.fillna(False).sum())
        except Exception:
            return 0

    def _pct(cnt: int) -> float:
        return round((cnt / n_total) * 100.0, 2) if n_total > 0 else 0.0

    facts = [
        {"항목": "Lane Error 결측", "프레임수": _cnt(_err_missing), "비율(%)": _pct(_cnt(_err_missing)), "기준/메모": "ERROR_COL 결측"},
        {"항목": f"Mask Ratio 하위 꼬리(≤ {low_th:.4f})" if low_th is not None else "Mask Ratio 하위 꼬리", "프레임수": _cnt(_mask_low), "비율(%)": _pct(_cnt(_mask_low)), "기준/메모": f"하위 {100-pctl}% 분위"},
        {"항목": f"Mask Ratio 상위 꼬리(≥ {high_th:.4f})" if high_th is not None else "Mask Ratio 상위 꼬리", "프레임수": _cnt(_mask_high), "비율(%)": _pct(_cnt(_mask_high)), "기준/메모": f"상위 {pctl}% 분위"},
        {"항목": f"Abs Lane Error 상위 꼬리(≥ {abs_tail_th:.4f})" if abs_tail_th is not None else "Abs Lane Error 상위 꼬리", "프레임수": _cnt(_abs_high), "비율(%)": _pct(_cnt(_abs_high)), "기준/메모": f"상위 {pctl}% 분위"},
        {"항목": f"Processing Time 상위 꼬리(≥ {proc_tail_th:.2f} ms)" if proc_tail_th is not None else "Processing Time 상위 꼬리", "프레임수": _cnt(_proc_high), "비율(%)": _pct(_cnt(_proc_high)), "기준/메모": f"상위 {pctl}% 분위"},
        {"항목": "Mask 극단 & Abs Error 꼬리(교집합)", "프레임수": _cnt(_mask_ext & _abs_high), "비율(%)": _pct(_cnt(_mask_ext & _abs_high)), "기준/메모": "동시 발생"},
        {"항목": "Abs Error 꼬리 & Proc 꼬리(교집합)", "프레임수": _cnt(_abs_high & _proc_high), "비율(%)": _pct(_cnt(_abs_high & _proc_high)), "기준/메모": "동시 발생"},
        {"항목": "Mask 극단 & Abs Error & Proc(3중 교집합)", "프레임수": _cnt(_mask_ext & _abs_high & _proc_high), "비율(%)": _pct(_cnt(_mask_ext & _abs_high & _proc_high)), "기준/메모": "동시 발생"},
    ]
    st.dataframe(pd.DataFrame(facts), use_container_width=True)

    # Optional: baseline delta (if available)
    if base:
        def _delta(a, b):
            if a is None or b is None:
                return None
            try:
                return float(a) - float(b)
            except Exception:
                return None

        delta_rows = []
        for key, label in [
            ("low_rate", "Mask 하위 꼬리 비율(%)"),
            ("high_rate", "Mask 상위 꼬리 비율(%)"),
            ("abs_tail_rate", "Abs Error 꼬리 비율(%)"),
            ("proc_tail_rate", "Proc 꼬리 비율(%)"),
            ("err_missing_rate", "Lane Error 결측 비율(%)"),
        ]:
            dv = _delta(cur.get(key), base.get(key))
            if dv is None:
                continue
            delta_rows.append({"항목": label, "변화(현재-기준, %p)": round(dv, 2)})

        if delta_rows:
            st.markdown("#### 기준일 대비 변화(현재-기준)")
            st.dataframe(pd.DataFrame(delta_rows), use_container_width=True)

    # Prioritized actions (speculative)
    st.markdown("### 다음 액션 (권장, 추측)")
    issues = [
        {"_score": _cnt(_err_missing), "신호": "Lane Error 결측", "권장 액션(추측)": "로그/파이프라인에서 Lane Error 기록 경로 점검(컬럼 생성, 타입 변환, 저장 시점)", "검증": "결측 구간의 원본 로그/코드 경로 확인"},
        {"_score": _cnt(_mask_low), "신호": "Mask Ratio 하위 꼬리", "권장 액션(추측)": "저가시성(역광/야간/오염) 프레임 표본 확인 → 전처리(감마/대비) 또는 데이터 보강", "검증": "하위 꼬리 프레임 20~50개 샘플링 확인"},
        {"_score": _cnt(_mask_high), "신호": "Mask Ratio 상위 꼬리", "권장 액션(추측)": "과검출(반사/표지/노이즈) 여부 확인 → 후처리(연결성/폭 제약) 또는 필터 강화", "검증": "상위 꼬리 프레임 표본 확인"},
        {"_score": _cnt(_abs_high), "신호": "Abs Lane Error 상위 꼬리", "권장 액션(추측)": "오차 큰 구간의 환경/모드/날씨 교차 확인 → 실패 조건 라벨링 및 재현 테스트", "검증": "tail 구간의 Weather/Time of Day 분포 비교"},
        {"_score": _cnt(_proc_high), "신호": "Processing Time 상위 꼬리", "권장 액션(추측)": "지연 구간에서 모델/전처리 병목 확인 → 프로파일링 후 캐시/리사이즈/배치 전략 점검", "검증": "tail 구간 프레임에서 단계별 시간 로깅"},
        {"_score": _cnt(_mask_ext & _abs_high), "신호": "Mask 극단 & Abs Error 동시", "권장 액션(추측)": "차선 미검출/과검출 시 오차 증가 가능성 → 안전 규칙(감속/정지)·fallback 로직 후보 검토", "검증": "교집합 프레임에서 실패 원인 유형 분류"},
    ]
    issues = [x for x in issues if x["_score"] > 0]
    issues = sorted(issues, key=lambda x: x["_score"], reverse=True)[:4]
    if not issues:
        st.info("상대적으로 두드러진 꼬리/결측 신호가 크지 않습니다. (추측) 기준일을 늘려 추세로 보는 편이 유리합니다.")
    else:
        for i, it in enumerate(issues, start=1):
            it["우선순위"] = i
            it["프레임수"] = it.pop("_score")
        st.dataframe(pd.DataFrame(issues)[["우선순위", "신호", "프레임수", "권장 액션(추측)", "검증"]], use_container_width=True)

# -------------------------------------------------------------------------
    # 1) Summary + visuals
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
    if cur["abs_p95"] is not None:
        summary_rows.append({"항목": f"Abs Error 상위 {100 - pctl}% 분위 임계", "값": _fmt(cur["abs_tail_th"], "{:.2f}"), "의미": "Part III(오차 과다) 기준과 동일"})
        summary_rows.append({"항목": "Abs Error tail 비율", "값": _fmt(cur["abs_tail_rate"], "{:.2f}%"), "의미": "상위 꼬리 비중"})
        summary_rows.append({"항목": "Abs Error p95 / p99 / max", "값": f"{_fmt(cur['abs_p95'], '{:.2f}')} / {_fmt(cur['abs_p99'], '{:.2f}')} / {_fmt(cur['abs_max'], '{:.2f}')}", "의미": "오차 과다(outlier) 수준"})
    if cur["proc_p95"] is not None:
        summary_rows.append({"항목": f"Proc Time 상위 {100 - pctl}% 분위 임계", "값": _fmt(cur["proc_tail_th"], "{:.1f} ms"), "의미": "Part III(처리시간 과다) 기준과 동일"})
        summary_rows.append({"항목": "Proc Time tail 비율", "값": _fmt(cur["proc_tail_rate"], "{:.2f}%"), "의미": "지연 꼬리 비중"})
        summary_rows.append({"항목": "Proc Time p95 / p99 / max", "값": f"{_fmt(cur['proc_p95'], '{:.1f}')} / {_fmt(cur['proc_p99'], '{:.1f}')} / {_fmt(cur['proc_max'], '{:.1f}')} ms", "의미": "지연의 극단(outlier) 크기"})
    if cur["err_missing_rate"] is not None:
        summary_rows.append({"항목": "Lane Error 결측률", "값": _fmt(cur["err_missing_rate"], "{:.2f}%"), "의미": "오차 기록 누락 비중(원인 단정 불가)"})

    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    ratio_cur = _to_num(df.get(MASK_RATIO_COL)).clip(0, 1)
    abs_cur = _to_num(df.get(ABS_ERROR_COL))
    proc_cur = _to_num(df.get(PROC_COL))

    c1, c2, c3 = st.columns(3)
    with c1:
        _hist_with_rules(
            ratio_cur,
            colname=MASK_RATIO_COL,
            title="Mask White Ratio 분포(0112 요약)",
            rules=[(cur.get("low_th"), f"low_th (p{100 - pctl})"), (cur.get("high_th"), f"high_th (p{pctl})")],
            domain=(0.0, 1.0),
            height=240,
        )
    with c2:
        if ABS_ERROR_COL in df.columns:
            _hist_with_rules(abs_cur, ABS_ERROR_COL, "Abs Lane Error 분포(0112 요약)", rules=[(cur.get("abs_tail_th"), f"tail_th (p{pctl})")], height=240)
        else:
            st.info(f"'{ABS_ERROR_COL}' 컬럼이 없어 오차 분포 시각화를 생략합니다.")
    with c3:
        if PROC_COL in df.columns:
            _hist_with_rules(proc_cur, PROC_COL, "Processing Time 분포(0112 요약)", rules=[(cur.get("proc_tail_th"), f"tail_th (p{pctl})")], height=240)
        else:
            st.info(f"'{PROC_COL}' 컬럼이 없어 처리시간 분포 시각화를 생략합니다.")

    # -------------------------------------------------------------------------
    # 2) Candidate distribution visuals
    st.markdown("### 후보(Part III) 분포 요약 (0112)")
    if isinstance(cand, pd.DataFrame) and (not cand.empty) and ("Primary Tag" in cand.columns):
        # n_total (전체 프레임 수) - 일부 코드에서 n_total 변수를 직접 참조할 수 있어 명시적으로 둡니다.
        n_total = int(cur.get("n_total", len(df)))

        tag_df = (
            cand["Primary Tag"]
            .value_counts(dropna=False)
            .rename_axis("Primary Tag")
            .reset_index(name="count")
        )
        # count가 문자열로 꼬이는 케이스 방지
        tag_df["count"] = pd.to_numeric(tag_df["count"], errors="coerce")
        tag_df["% (전체)"] = (tag_df["count"] / n_total * 100.0).round(2)
        tag_df["% (후보 내)"] = (tag_df["count"] / len(cand) * 100.0).round(2)

        st.altair_chart(
            alt.Chart(tag_df)
            .mark_bar()
            .encode(
                y=alt.Y("Primary Tag:N", sort="-x", title=None),
                x=alt.X("count:Q", title="Candidate frames"),
                tooltip=["Primary Tag:N", "count:Q", "% (전체):Q", "% (후보 내):Q"],
            )
            .properties(height=260, title="후보(Primary Tag) 카운트(0112)"),
            use_container_width=True,
        )
        st.dataframe(tag_df, hide_index=True, use_container_width=True)
    else:
        st.info("후보 데이터(cand)가 없거나, 'Primary Tag' 컬럼이 없어 분포 시각화를 생략합니다.")

    # -------------------------------------------------------------------------
    # 3) Baseline comparison visuals
    st.markdown("### 0109 대비 변화 (0112 - 0109)")
    if base is not None:
        comp_rows = [
            ("Mask Ratio p50", cur["ratio_p50"], base["ratio_p50"], "중앙값(전반적 검출량)"),
            (f"Mask Ratio low-rate(≤p{100 - pctl})", cur["low_rate"], base["low_rate"], "저가시성 꼬리 비중(%)"),
            (f"Mask Ratio high-rate(≥p{pctl})", cur["high_rate"], base["high_rate"], "과검출 꼬리 비중(%)"),
            ("Lane Error missing(%)", cur["err_missing_rate"], base["err_missing_rate"], "오차 기록 누락 비중(%)"),
            (f"Abs Error tail-rate(≥p{pctl})", cur["abs_tail_rate"], base["abs_tail_rate"], "오차 과다 꼬리 비중(%)"),
            (f"Proc Time tail-rate(≥p{pctl})", cur["proc_tail_rate"], base["proc_tail_rate"], "지연 꼬리 비중(%)"),
        ]
        comp = pd.DataFrame(comp_rows, columns=["metric", "0112", "0109", "의미"])
        comp["delta(0112-0109)"] = comp.apply(lambda r: None if (r["0112"] is None or r["0109"] is None) else float(r["0112"]) - float(r["0109"]), axis=1)
        st.dataframe(comp, hide_index=True, use_container_width=True)

        # Chart: grouped bars
        chart_df = comp.melt(id_vars=["metric", "의미"], value_vars=["0112", "0109"], var_name="dataset", value_name="value").dropna(subset=["value"])
        # 핵심 지표 비교(0112 vs 0109)
        # - 기존에는 두 데이터셋 막대가 같은 위치에 겹쳐 '한 줄로 합쳐진 것'처럼 보일 수 있어,
        #   (Altair 버전에 따라 xOffset이 무시되거나, 수평 막대에서 offset이 기대대로 동작하지 않는 경우가 있음)
        #   yOffset(가능한 경우)을 사용해 항목별로 0109 → 0112 순서로 막대를 분리합니다.
        # - yOffset이 지원되지 않는 Altair 버전에서는 row facet으로 안전하게 분리합니다.

        core_bar = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                y=alt.Y("metric:N", sort=None, title=None),
                yOffset=alt.YOffset("dataset:N", sort=["0109", "0112"]),
                x=alt.X("value:Q", title="percentage(%)"),
                color=alt.Color("dataset:N", sort=["0109", "0112"], legend=alt.Legend(orient="top")),
                tooltip=["metric:N", "dataset:N", alt.Tooltip("value:Q"), "의미:N"],
            )
            .properties(height=320, title="핵심 지표 비교(0112 vs 0109)")
        )

        st.altair_chart(core_bar, use_container_width=True)

        # Small compare hist (facet) for ratio only
    comb_ratio = pd.concat([
        pd.DataFrame({'dataset': '0112', MASK_RATIO_COL: pd.to_numeric(ratio_cur, errors='coerce').dropna()}),
        pd.DataFrame({'dataset': '0109', MASK_RATIO_COL: pd.to_numeric(base_df[MASK_RATIO_COL], errors='coerce').dropna()}) if base_df is not None else pd.DataFrame(),
    ], ignore_index=True)
    if not comb_ratio.empty:
        # x축 빈 공간을 줄이기 위해 0~MASK_RATIO_PLOT_MAX 범위로 고정합니다.
        n_clip_0112 = int((pd.to_numeric(ratio_cur, errors='coerce') > MASK_RATIO_PLOT_MAX).sum())
        n_clip_0109 = int((pd.to_numeric(base_df[MASK_RATIO_COL], errors='coerce') > MASK_RATIO_PLOT_MAX).sum()) if base_df is not None else 0
        if n_clip_0112 or n_clip_0109:
            st.caption(f"※ Mask White Ratio > {MASK_RATIO_PLOT_MAX:.2f} 구간은 비교 시각화에서 제외했습니다 (0112: {n_clip_0112}프레임, 0109: {n_clip_0109}프레임).")
        comb_ratio_plot = comb_ratio[comb_ratio[MASK_RATIO_COL] <= MASK_RATIO_PLOT_MAX].copy()

        # 데이터가 0~0.30 범위에 몰려있더라도 실제 분포가 0.05~0.15처럼 더 좁으면,
        # 우측 빈 공간이 커집니다. 비교 공정성을 위해 0112/0109 각각의 상위 분위(99.5%)를
        # 계산한 뒤, 둘 중 큰 값을 기준으로 x축 상한을 자동 축소합니다(최대 0.30, 최소 0.12).
        def _safe_q(s: pd.Series, q: float):
            s2 = pd.to_numeric(s, errors="coerce").dropna()
            if len(s2) == 0:
                return None
            return float(s2.quantile(q))

        q_0112 = _safe_q(comb_ratio_plot.loc[comb_ratio_plot["dataset"] == "0112", MASK_RATIO_COL], 0.995)
        q_0109 = _safe_q(comb_ratio_plot.loc[comb_ratio_plot["dataset"] == "0109", MASK_RATIO_COL], 0.995)
        q_candidates = [v for v in [q_0112, q_0109] if v is not None]
        x_max = MASK_RATIO_PLOT_MAX if not q_candidates else min(MASK_RATIO_PLOT_MAX, max(0.12, max(q_candidates) + 0.01))
        q_0112_str = f"{q_0112:.3f}" if q_0112 is not None else "NA"
        q_0109_str = f"{q_0109:.3f}" if q_0109 is not None else "NA"
        st.caption(f"Mask White Ratio 분포 비교 x축: 0 ~ {x_max:.2f} (0112 q99.5={q_0112_str} / 0109 q99.5={q_0109_str}, 최대 {MASK_RATIO_PLOT_MAX:.2f})")

        col_hist, col_info = st.columns([2, 1], gap="large")
        with col_hist:
            st.altair_chart(
                alt.Chart(comb_ratio_plot)
                .mark_bar(opacity=0.65, binSpacing=0)
                .encode(
                    x=alt.X(
                        f"{MASK_RATIO_COL}:Q",
                        bin=alt.Bin(step=0.01),
                        scale=alt.Scale(domain=[0, x_max]),
                        title="Mask White Ratio (binned)",
                    ),
                    y=alt.Y("count():Q", title="Frames"),
                    row=alt.Row("dataset:N", title=None),
                    tooltip=[alt.Tooltip("count():Q", title="frames")],
                )
                .properties(height=120, title="Mask Ratio 분포 비교(0112 vs 0109)"),
                use_container_width=True,
            )
    else:
        st.info("0109 비교용 데이터가 없어, 변화(비교) 시각화는 생략됩니다.")

    # -------------------------------------------------------------------------
    # 4) Recommendations + visuals per explanation
    st.markdown("### 개선사항(권장 우선순위)")
    st.caption("아래는 데이터 패턴 기반의 권장사항이며, 원인 단정이 아닙니다. 각 항목에 근거 시각화를 함께 제공합니다.")

    # Use current metrics
    if (cur.get("low_rate") is not None) and (cur.get("low_th") is not None) and (cur.get("low_rate") >= 5.0):
        with st.expander("1) 선이 잘 안 보이는 구간 대비 (Mask Ratio 아주 낮음)"):
            st.markdown(
                f"- Mask Ratio ≤ **{cur['low_th']:.4f}** 구간이 **{cur['low_rate']:.2f}%** 입니다. "
                "저가시성(어둠/그림자/역광 등)에서 검출량이 급감할 수 있습니다."
            )
            _hist_with_rules(ratio_cur, MASK_RATIO_COL, "Mask Ratio 분포(저가시성 꼬리 확인)", rules=[(cur["low_th"], "low_th")], domain=(0.0, 1.0), height=260)
            st.markdown("- 권장: (a) 밝기/대비/감마 보정, (b) ROI/전처리 조정, (c) 미검출 시 감속/정지 등 안전 규칙과 테스트")

    if (cur.get("high_rate") is not None) and (cur.get("high_th") is not None) and (cur.get("high_rate") >= 5.0):
        with st.expander("2) 과검출/노이즈 구간 대비 (Mask Ratio 아주 높음)"):
            st.markdown(
                f"- Mask Ratio ≥ **{cur['high_th']:.4f}** 구간이 **{cur['high_rate']:.2f}%** 입니다. "
                "노면 반사/표지/노이즈로 흰 영역이 과도하게 잡히면 중심 추정이 흔들릴 수 있습니다."
            )
            _hist_with_rules(ratio_cur, MASK_RATIO_COL, "Mask Ratio 분포(과검출 꼬리 확인)", rules=[(cur["high_th"], "high_th")], domain=(0.0, 1.0), height=260)
            st.markdown("- 권장: (a) 이진화 임계/후처리 조정, (b) 차선 형태 제약(폭/연결성), (c) 차선 후보 필터 강화")

    if (cur.get("abs_tail_rate") is not None) and (cur.get("abs_tail_th") is not None) and (cur.get("abs_tail_rate") >= 5.0):
        with st.expander("3) 오차 과다 프레임 원인 후보 점검 (Abs Error tail)"):
            st.markdown(f"- Abs Lane Error ≥ **{cur['abs_tail_th']:.2f}** 구간이 **{cur['abs_tail_rate']:.2f}%** 입니다.")
            _hist_with_rules(abs_cur, ABS_ERROR_COL, "Abs Lane Error 분포(꼬리 확인)", rules=[(cur["abs_tail_th"], "tail_th")], height=260)
            scat = pd.DataFrame({MASK_RATIO_COL: ratio_cur, ABS_ERROR_COL: abs_cur}).dropna()
            if not scat.empty:
                scat["is_tail"] = scat[ABS_ERROR_COL].ge(cur["abs_tail_th"])
                st.altair_chart(
                    alt.Chart(scat)
                    .mark_point(filled=True, opacity=0.55)
                    .encode(
                        x=alt.X(f"{MASK_RATIO_COL}:Q", scale=alt.Scale(domain=[0, MASK_RATIO_PLOT_MAX])),
                        y=alt.Y(f"{ABS_ERROR_COL}:Q"),
                        shape=alt.Shape("is_tail:N", title="AbsError tail"),
                        tooltip=[alt.Tooltip(MASK_RATIO_COL, format=".4f"), alt.Tooltip(ABS_ERROR_COL, format=".2f")],
                    )
                    .properties(height=320, title="Mask Ratio ↔ Abs Error (tail 프레임 표시)"),
                    use_container_width=True,
                )
            st.markdown("- 권장: (a) 차선 중심 추정 로직 점검, (b) 한쪽만 검출 시 fallback, (c) 곡률/차선폭 제약 도입")

    if (cur.get("err_missing_rate") is not None) and (cur.get("err_missing_rate") >= 5.0):
        with st.expander("4) 오차 기록 누락(NA) 원인 로깅 강화"):
            st.markdown(f"- Lane Error 결측률이 **{cur['err_missing_rate']:.2f}%** 입니다. NA 사유를 함께 남겨야 재현성이 올라갑니다.")
            if ERROR_COL in df.columns:
                miss = _to_num(df[ERROR_COL]).isna()
                mdf = pd.DataFrame({MASK_RATIO_COL: ratio_cur, "missing": miss}).dropna(subset=[MASK_RATIO_COL])
                mdf["ratio_bin"] = pd.cut(mdf[MASK_RATIO_COL], bins=[0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0], include_lowest=True)
                ms = mdf.groupby("ratio_bin", dropna=False)["missing"].agg(frames="size", missing="sum").reset_index()
                ms["missing_%"] = (ms["missing"] / ms["frames"] * 100.0).round(2)
                st.altair_chart(
                    alt.Chart(ms.dropna(subset=["ratio_bin"]))
                    .mark_bar()
                    .encode(
                        x=alt.X("ratio_bin:N", sort=None, title="Mask Ratio bin"),
                        y=alt.Y("missing_%:Q", scale=alt.Scale(domain=[0, 100]), title="Lane Error missing (%)"),
                        tooltip=["ratio_bin:N", "frames:Q", "missing:Q", "missing_%:Q"],
                    )
                    .properties(height=300, title="Mask Ratio 구간별 Lane Error 결측률(요약)"),
                    use_container_width=True,
                )
            st.markdown("- 권장: (a) NA 사유 코드(미검출/부분검출/모드/센서) 추가, (b) ratio/환경/후보 태그 동시 저장")

    if (cur.get("proc_tail_rate") is not None) and (cur.get("proc_tail_th") is not None) and (cur.get("proc_tail_rate") >= 5.0):
        with st.expander("5) 처리시간 튐 감소 (Proc Time tail)"):
            st.markdown(
                f"- Processing Time ≥ **{cur['proc_tail_th']:.1f} ms** 구간이 **{cur['proc_tail_rate']:.2f}%** 입니다. "
                "프레임 드랍/조향 지연에 영향 가능성이 있습니다."
            )
            _hist_with_rules(proc_cur, PROC_COL, "Processing Time 분포(꼬리 확인)", rules=[(cur["proc_tail_th"], "tail_th")], height=260)
            scat2 = pd.DataFrame({MASK_RATIO_COL: ratio_cur, PROC_COL: proc_cur}).dropna()
            if not scat2.empty:
                scat2["is_tail"] = scat2[PROC_COL].ge(cur["proc_tail_th"])
                st.altair_chart(
                    alt.Chart(scat2)
                    .mark_point(filled=True, opacity=0.55)
                    .encode(
                        x=alt.X(f"{MASK_RATIO_COL}:Q", scale=alt.Scale(domain=[0, MASK_RATIO_PLOT_MAX])),
                        y=alt.Y(f"{PROC_COL}:Q", title="Processing Time (ms)"),
                        shape=alt.Shape("is_tail:N", title="Proc tail"),
                        tooltip=[alt.Tooltip(MASK_RATIO_COL, format=".4f"), alt.Tooltip(PROC_COL, format=".1f")],
                    )
                    .properties(height=320, title="Mask Ratio ↔ Processing Time (tail 프레임 표시)"),
                    use_container_width=True,
                )
            st.markdown("- 권장: (a) 단계별 시간측정으로 병목 찾기, (b) 해상도/ROI 축소, (c) 모델/후처리 경량화 검토")

    if (cur.get("low_rate") is None) and (cur.get("high_rate") is None) and (cur.get("abs_tail_rate") is None) and (cur.get("err_missing_rate") is None) and (cur.get("proc_tail_rate") is None):
        st.info("현재 데이터/컬럼 기준으로는 Part X에서 제시할 패턴이 충분하지 않습니다. (민감도/결측/범위를 확인해 주세요.)")

if __name__ == "__main__":
        render()
