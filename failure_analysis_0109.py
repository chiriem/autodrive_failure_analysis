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
    load_fixed_csv,
    _make_tooltip, _describe_missing, perform_linear_regression, draw_histogram,
)

_FIXED_FILENAME = "0109_18_log.csv"

@st.cache_data(show_spinner=False)
def _load_fixed_data() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    return load_fixed_csv(_FIXED_FILENAME, run_id="run_0109_fixed", base_dir=base_dir)

def render() -> None:
    _maybe_set_page_config()
    # UI

    st.title("0109 자율주행 실패 분석")
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
                        y=alt.Y("count()", title="Frames", stack=None),
                        xOffset=alt.XOffset("Error Recorded:N", sort=["Present", "Missing"]),
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
        key="outlier_sensitivity_pctl_0109",
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

    # -------------------------------------------------------------------------
    # Compute key stats (same thresholds as Part III narrative)
    n_total = int(len(df))

    ratio = _to_num(df.get(MASK_RATIO_COL)).clip(0, 1)
    rv = ratio.dropna()
    low_th = high_th = None
    low_rate = high_rate = None
    ratio_p05 = ratio_p50 = ratio_p95 = None
    if not rv.empty:
        ratio_p05 = float(rv.quantile(0.05))
        ratio_p50 = float(rv.quantile(0.50))
        ratio_p95 = float(rv.quantile(0.95))
        low_th = float(rv.quantile((100 - pctl) / 100.0))
        high_th = float(rv.quantile(pctl / 100.0))

        valid = ratio.notna()
        if valid.any():
            low_rate = float((ratio.le(low_th) & valid).mean() * 100.0)
            high_rate = float((ratio.ge(high_th) & valid).mean() * 100.0)

    abs_err = _to_num(df.get(ABS_ERROR_COL))
    ae = abs_err.dropna()
    abs_p95 = abs_p99 = abs_max = None
    abs_tail_th = abs_tail_rate = None
    if not ae.empty:
        abs_p95 = float(ae.quantile(0.95))
        abs_p99 = float(ae.quantile(0.99))
        abs_max = float(ae.max())
        abs_tail_th = float(ae.quantile(pctl / 100.0))
        abs_tail_rate = float(ae.ge(abs_tail_th).mean() * 100.0)

    proc = _to_num(df.get(PROC_COL))
    pr = proc.dropna()
    proc_p95 = proc_p99 = proc_max = None
    proc_tail_th = proc_tail_rate = None
    if not pr.empty:
        proc_p95 = float(pr.quantile(0.95))
        proc_p99 = float(pr.quantile(0.99))
        proc_max = float(pr.max())
        proc_tail_th = float(pr.quantile(pctl / 100.0))
        proc_tail_rate = float(pr.ge(proc_tail_th).mean() * 100.0)

    err_missing_rate = None
    if ERROR_COL in df.columns:
        err_missing_rate = float(_to_num(df[ERROR_COL]).isna().mean() * 100.0)

    # -------------------------------------------------------------------------
    # 1) Summary + visuals
    st.markdown("### 핵심 지표 요약 (0109)")
    summary_rows = [
        {"항목": "총 프레임 수", "값": f"{n_total:,}", "의미": "분석 대상 전체 행 수"},
        {"항목": "Mask Ratio p05 / p50 / p95", "값": f"{_fmt(ratio_p05, '{:.4f}')} / {_fmt(ratio_p50, '{:.4f}')} / {_fmt(ratio_p95, '{:.4f}')}", "의미": "검출량의 하위/중앙/상위 수준(0~1)"},
    ]
    if low_th is not None:
        summary_rows.append({"항목": f"Mask Ratio 하한(하위 {100 - pctl}% 분위)", "값": _fmt(low_th, "{:.4f}"), "의미": "매우 낮은 검출량(저가시성/미검출 후보) 기준"})
        summary_rows.append({"항목": "Mask Ratio 하한 이하 비율", "값": _fmt(low_rate, "{:.2f}%"), "의미": "꼬리 구간 비중(원인 단정 불가)"})
        summary_rows.append({"항목": f"Mask Ratio 상한(상위 {100 - pctl}% 분위)", "값": _fmt(high_th, "{:.4f}"), "의미": "매우 높은 검출량(과검출 후보) 기준"})
        summary_rows.append({"항목": "Mask Ratio 상한 이상 비율", "값": _fmt(high_rate, "{:.2f}%"), "의미": "꼬리 구간 비중(원인 단정 불가)"})
    if abs_p95 is not None:
        summary_rows.append({"항목": f"Abs Error 상위 {100 - pctl}% 분위 임계", "값": _fmt(abs_tail_th, "{:.2f}"), "의미": "Part III(오차 과다) 기준과 동일"})
        summary_rows.append({"항목": "Abs Error tail 비율", "값": _fmt(abs_tail_rate, "{:.2f}%"), "의미": "상위 꼬리 비중"})
        summary_rows.append({"항목": "Abs Error p95 / p99 / max", "값": f"{_fmt(abs_p95, '{:.2f}')} / {_fmt(abs_p99, '{:.2f}')} / {_fmt(abs_max, '{:.2f}')}", "의미": "오차 과다(outlier) 수준"})
    if proc_p95 is not None:
        summary_rows.append({"항목": f"Proc Time 상위 {100 - pctl}% 분위 임계", "값": _fmt(proc_tail_th, "{:.1f} ms"), "의미": "Part III(처리시간 과다) 기준과 동일"})
        summary_rows.append({"항목": "Proc Time tail 비율", "값": _fmt(proc_tail_rate, "{:.2f}%"), "의미": "지연 꼬리 비중"})
        summary_rows.append({"항목": "Proc Time p95 / p99 / max", "값": f"{_fmt(proc_p95, '{:.1f}')} / {_fmt(proc_p99, '{:.1f}')} / {_fmt(proc_max, '{:.1f}')} ms", "의미": "지연의 극단(outlier) 크기"})
    if err_missing_rate is not None:
        summary_rows.append({"항목": "Lane Error 결측률", "값": _fmt(err_missing_rate, "{:.2f}%"), "의미": "오차 기록 누락 비중(원인 단정 불가)"})

    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        _hist_with_rules(
            ratio,
            colname=MASK_RATIO_COL,
            title="Mask White Ratio 분포(요약)",
            rules=[
                (low_th, f"low_th (p{100 - pctl})"),
                (high_th, f"high_th (p{pctl})"),
            ],
            domain=(0.0, 1.0),
            height=240,
        )
        if low_th is not None:
            st.caption(f"low_th={low_th:.4f} / high_th={high_th:.4f} (민감도 {pctl}%)")
    with c2:
        if ABS_ERROR_COL in df.columns:
            _hist_with_rules(
                abs_err,
                colname=ABS_ERROR_COL,
                title="Abs Lane Error 분포(요약)",
                rules=[(abs_tail_th, f"tail_th (p{pctl})")],
                height=240,
            )
        else:
            st.info(f"'{ABS_ERROR_COL}' 컬럼이 없어 오차 분포 시각화를 생략합니다.")
    with c3:
        if PROC_COL in df.columns:
            _hist_with_rules(
                proc,
                colname=PROC_COL,
                title="Processing Time 분포(요약)",
                rules=[(proc_tail_th, f"tail_th (p{pctl})")],
                height=240,
            )
        else:
            st.info(f"'{PROC_COL}' 컬럼이 없어 처리시간 분포 시각화를 생략합니다.")

    # Ratio bin → 평균 오차/지연 (설명 근거용)
    st.markdown("#### Mask Ratio 구간별 평균 변화(근거 시각화)")
    tmp = pd.DataFrame({
        MASK_RATIO_COL: ratio,
        ABS_ERROR_COL: abs_err,
        PROC_COL: proc,
        WEATHER_COL: df.get(WEATHER_COL),
        TOD_COL: df.get(TOD_COL),
    })
    tmp = tmp.dropna(subset=[MASK_RATIO_COL])
    try:
        tmp["ratio_bin"] = pd.qcut(tmp[MASK_RATIO_COL], q=10, duplicates="drop")
        tmp["ratio_mid"] = tmp["ratio_bin"].apply(lambda x: float(x.mid) if hasattr(x, "mid") else np.nan)
        tmp["ratio_mid"] = pd.to_numeric(tmp["ratio_mid"], errors="coerce")
    except Exception:
        tmp["ratio_bin"] = pd.cut(tmp[MASK_RATIO_COL], bins=[0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0], include_lowest=True)
        tmp["ratio_mid"] = tmp["ratio_bin"].apply(lambda x: float(getattr(x, "mid", np.nan)))
        tmp["ratio_mid"] = pd.to_numeric(tmp["ratio_mid"], errors="coerce")

    g = tmp.groupby("ratio_bin", dropna=False).agg(
        ratio_mid=("ratio_mid", "mean"),
        frames=(MASK_RATIO_COL, "size"),
        mean_abs=(ABS_ERROR_COL, "mean"),
        mean_proc=(PROC_COL, "mean"),
    ).reset_index()

    cc1, cc2 = st.columns(2)
    with cc1:
        if g["mean_abs"].notna().any():
            st.altair_chart(
                alt.Chart(g.dropna(subset=["ratio_mid"]))
                .mark_line(point=True)
                .encode(
                    x=alt.X("ratio_mid:Q", title="Mask Ratio (bin mid)"),
                    y=alt.Y("mean_abs:Q", title="Mean Abs Lane Error"),
                    tooltip=["ratio_bin:N", "frames:Q", alt.Tooltip("mean_abs:Q", format=".2f")],
                )
                .properties(height=260),
                use_container_width=True,
            )
        else:
            st.info("오차 값이 없어(또는 결측이 많아) 구간별 평균 오차 시각화를 생략합니다.")
    with cc2:
        if g["mean_proc"].notna().any():
            st.altair_chart(
                alt.Chart(g.dropna(subset=["ratio_mid"]))
                .mark_line(point=True)
                .encode(
                    x=alt.X("ratio_mid:Q", title="Mask Ratio (bin mid)"),
                    y=alt.Y("mean_proc:Q", title="Mean Processing Time (ms)"),
                    tooltip=["ratio_bin:N", "frames:Q", alt.Tooltip("mean_proc:Q", format=".1f")],
                )
                .properties(height=260),
                use_container_width=True,
            )
        else:
            st.info("처리시간 값이 없어(또는 결측이 많아) 구간별 평균 처리시간 시각화를 생략합니다.")

    # -------------------------------------------------------------------------
    # 2) Candidate distribution visuals
    st.markdown("### 후보(Part III) 분포 요약")
    if isinstance(cand, pd.DataFrame) and (not cand.empty) and ("Primary Tag" in cand.columns):
        tag_df = cand["Primary Tag"].value_counts(dropna=False).reset_index()
        tag_df.columns = ["Primary Tag", "count"]
        tag_df["count"] = pd.to_numeric(tag_df["count"], errors="coerce")
        tag_df["count"] = tag_df["count"].fillna(0)
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
            .properties(height=260, title="후보(Primary Tag) 카운트"),
            use_container_width=True,
        )
        st.dataframe(tag_df, hide_index=True, use_container_width=True)
    else:
        st.info("후보 데이터(cand)가 없거나, 'Primary Tag' 컬럼이 없어 분포 시각화를 생략합니다.")

    # -------------------------------------------------------------------------
    # 3) Recommendations + visuals per explanation
    st.markdown("### 개선사항(권장 우선순위)")
    st.caption("아래는 데이터 패턴 기반의 권장사항이며, 원인 단정이 아닙니다. 각 항목에 근거 시각화를 함께 제공합니다.")

    # 1) Low ratio
    if (low_rate is not None) and (low_th is not None) and (low_rate >= 5.0):
        with st.expander("1) 선이 잘 안 보이는 구간 대비 (Mask Ratio 아주 낮음)"):
            st.markdown(
                f"- Mask Ratio ≤ **{low_th:.4f}** 구간이 **{low_rate:.2f}%** 입니다. "
                "저가시성(어둠/그림자/역광/흰선 훼손 등)에서 검출량이 급감할 수 있습니다."
            )
            _hist_with_rules(ratio, MASK_RATIO_COL, "Mask Ratio 분포(저가시성 꼬리 확인)", rules=[(low_th, "low_th")], domain=(0.0, 1.0), height=260)
            # environment breakdown
            env_cols = [c for c in [WEATHER_COL, TOD_COL] if c in df.columns]
            if env_cols:
                ecol = st.selectbox("저가시성 프레임 환경 분해 기준", options=env_cols, index=0, key="partx0109_low_env")
                low_df = df.loc[ratio.le(low_th)].copy()
                ed = low_df[ecol].astype("string").fillna("Unknown").value_counts().reset_index()
                ed.columns = [ecol, "frames"]
                st.altair_chart(
                    alt.Chart(ed)
                    .mark_bar()
                    .encode(x=alt.X(f"{ecol}:N", sort="-y", axis=alt.Axis(labelAngle=-20)), y="frames:Q", tooltip=[ecol, "frames:Q"])
                    .properties(height=240, title=f"저가시성(≤low_th) 프레임의 {ecol} 분포"),
                    use_container_width=True,
                )

            st.markdown("- 권장: (a) 밝기/대비/감마 보정, (b) ROI/전처리 조정, (c) 미검출 시 감속/정지 등 안전 규칙과 테스트")

    # 2) High ratio
    if (high_rate is not None) and (high_th is not None) and (high_rate >= 5.0):
        with st.expander("2) 과검출/노이즈 구간 대비 (Mask Ratio 아주 높음)"):
            st.markdown(
                f"- Mask Ratio ≥ **{high_th:.4f}** 구간이 **{high_rate:.2f}%** 입니다. "
                "흰 영역이 과도하게 잡히면(노면 반사/표지/노이즈) 중심 추정이 흔들릴 수 있습니다."
            )
            _hist_with_rules(ratio, MASK_RATIO_COL, "Mask Ratio 분포(과검출 꼬리 확인)", rules=[(high_th, "high_th")], domain=(0.0, 1.0), height=260)
            st.markdown("- 권장: (a) 이진화 임계/후처리(모폴로지) 조정, (b) 차선 형태 제약(폭/연결성) 추가, (c) 차선 후보 필터 강화")

    # 3) High abs error
    if (abs_tail_rate is not None) and (abs_tail_th is not None) and (abs_tail_rate >= 5.0):
        with st.expander("3) 오차 과다 프레임 원인 후보 점검 (Abs Error tail)"):
            st.markdown(
                f"- Abs Lane Error ≥ **{abs_tail_th:.2f}** (상위 {100 - pctl}% 꼬리) 구간이 **{abs_tail_rate:.2f}%** 입니다."
            )
            _hist_with_rules(abs_err, ABS_ERROR_COL, "Abs Lane Error 분포(꼬리 확인)", rules=[(abs_tail_th, "tail_th")], height=260)

            # Scatter: ratio vs abs error (candidate highlight if available)
            scat = pd.DataFrame({MASK_RATIO_COL: ratio, ABS_ERROR_COL: abs_err})
            scat = scat.dropna()
            if not scat.empty:
                scat["is_tail"] = scat[ABS_ERROR_COL].ge(abs_tail_th)
                st.altair_chart(
                    alt.Chart(scat)
                    .mark_point(filled=True, opacity=0.55)
                    .encode(
                        x=alt.X(f"{MASK_RATIO_COL}:Q", scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y(f"{ABS_ERROR_COL}:Q"),
                        shape=alt.Shape("is_tail:N", title="AbsError tail"),
                        tooltip=[alt.Tooltip(MASK_RATIO_COL, format=".4f"), alt.Tooltip(ABS_ERROR_COL, format=".2f")],
                    )
                    .properties(height=320, title="Mask Ratio ↔ Abs Error (tail 프레임 표시)"),
                    use_container_width=True,
                )

            st.markdown("- 권장: (a) 차선 중심 추정 로직(가정/평균/fit) 점검, (b) 한쪽 차선만 검출 시 fallback, (c) 곡률/차선폭 제약 도입")

    # 4) Lane Error missingness
    if (err_missing_rate is not None) and (err_missing_rate >= 5.0):
        with st.expander("4) 오차 기록 누락(NA) 원인 로깅 강화"):
            st.markdown(f"- Lane Error 결측률이 **{err_missing_rate:.2f}%** 입니다. NA는 '미검출/한쪽만 검출/모드' 등의 상태를 함께 남겨야 재현성이 올라갑니다.")
            if ERROR_COL in df.columns:
                miss = _to_num(df[ERROR_COL]).isna()
                mdf = pd.DataFrame({MASK_RATIO_COL: ratio, "missing": miss}).dropna(subset=[MASK_RATIO_COL])
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
            st.markdown("- 권장: (a) NA 사유 코드(미검출/부분검출/모드/센서) 추가, (b) 같은 프레임에서 ratio/후보 태그/환경 함께 저장")

    # 5) Processing time spikes
    if (proc_tail_rate is not None) and (proc_tail_th is not None) and (proc_tail_rate >= 5.0):
        with st.expander("5) 처리시간 튐 감소 (Proc Time tail)"):
            st.markdown(
                f"- Processing Time ≥ **{proc_tail_th:.1f} ms** (상위 {100 - pctl}% 꼬리) 구간이 **{proc_tail_rate:.2f}%** 입니다. "
                "프레임 드랍/조향 지연에 직접 영향 가능성이 있습니다."
            )
            _hist_with_rules(proc, PROC_COL, "Processing Time 분포(꼬리 확인)", rules=[(proc_tail_th, "tail_th")], height=260)
            scat2 = pd.DataFrame({MASK_RATIO_COL: ratio, PROC_COL: proc}).dropna()
            if not scat2.empty:
                scat2["is_tail"] = scat2[PROC_COL].ge(proc_tail_th)
                st.altair_chart(
                    alt.Chart(scat2)
                    .mark_point(filled=True, opacity=0.55)
                    .encode(
                        x=alt.X(f"{MASK_RATIO_COL}:Q", scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y(f"{PROC_COL}:Q", title="Processing Time (ms)"),
                        shape=alt.Shape("is_tail:N", title="Proc tail"),
                        tooltip=[alt.Tooltip(MASK_RATIO_COL, format=".4f"), alt.Tooltip(PROC_COL, format=".1f")],
                    )
                    .properties(height=320, title="Mask Ratio ↔ Processing Time (tail 프레임 표시)"),
                    use_container_width=True,
                )
            st.markdown("- 권장: (a) 단계별 시간측정으로 병목 찾기, (b) 해상도/ROI 축소, (c) 모델/후처리 경량화(정수화/프루닝 등) 검토")

    # Fallback when nothing triggered
    if (low_rate is None) and (high_rate is None) and (abs_tail_rate is None) and (err_missing_rate is None) and (proc_tail_rate is None):
        st.info("현재 데이터/컬럼 기준으로는 Part X에서 제시할 패턴이 충분하지 않습니다. (민감도/결측/범위를 확인해 주세요.)")

