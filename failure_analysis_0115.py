"""0115_12_log.csv 전용 분석 페이지

- 단일 로그 분석(비교 없음)
- 실패 정의: Failure Flag == 1
- 신규(옵션) 컬럼은 "실패 프레임 테이블"에서 원본을 노출하고, Part X에서는 요약 지표로만 사용
- Part III(환경별 비교) 제외 / Part X(분석 기반 개선사항) 포함(외부 생성 버튼 없음)

사용: failure_analysis.py(메인)에서 import 후 render() 호출
"""

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from pathlib import Path

# Optional / extra columns (present in newer logs)
FAIL_FLAG_COL = "Failure Flag"
FAIL_TYPE_COL = "Failure Type"
LANE_STATE_COL = "Lane State"
TARGET_CX_COL = "Target CX"
CONTOURS_TOTAL_COL = "Contours Total"
CONTOURS_KEPT_COL = "Contours Kept"
DIR_FORCE_COL = "Direction Force"


from fa_utils import (
    TS_COL, QUALITY_COL, MASK_RATIO_COL, ERROR_COL, ABS_ERROR_COL,
    PROC_COL, WEATHER_COL, TOD_COL, MODE_COL,
    FIXED_LOG_COLS,
    RUN_ID_COL, ROW_IN_RUN_COL, EVENT_ID_COL,
    _select_fixed_columns,
    _ensure_fields,
    _make_tooltip,
    _add_event_ids_per_run,
    _describe_missing,
    perform_linear_regression,
    draw_histogram,
)


# 표 컬럼 표시 설정
# - st.dataframe에 column_config를 적용할 때, 실제로 존재하는 컬럼에만 설정을 적용한다.
# - 일부 컬럼이 로그에 없더라도 오류가 나지 않도록 _column_config_for()에서 필터링한다.
COLUMN_CONFIG = {
    TS_COL: st.column_config.NumberColumn(TS_COL, format="%.0f"),
    WEATHER_COL: st.column_config.TextColumn(WEATHER_COL),
    TOD_COL: st.column_config.TextColumn(TOD_COL),
    MODE_COL: st.column_config.TextColumn(MODE_COL),
    QUALITY_COL: st.column_config.NumberColumn(QUALITY_COL, format="%.0f"),
    MASK_RATIO_COL: st.column_config.NumberColumn(MASK_RATIO_COL, format="%.4f"),
    ERROR_COL: st.column_config.NumberColumn(ERROR_COL, format="%.2f"),
    ABS_ERROR_COL: st.column_config.NumberColumn(ABS_ERROR_COL, format="%.2f"),
    PROC_COL: st.column_config.NumberColumn(PROC_COL, format="%.1f"),

    RUN_ID_COL: st.column_config.TextColumn(RUN_ID_COL),
    ROW_IN_RUN_COL: st.column_config.NumberColumn(ROW_IN_RUN_COL, format="%.0f"),
    EVENT_ID_COL: st.column_config.TextColumn(EVENT_ID_COL),

    # 0115 신규 컬럼(로그에 없을 수도 있음)
    FAIL_FLAG_COL: st.column_config.NumberColumn(FAIL_FLAG_COL, format="%.0f"),
    FAIL_TYPE_COL: st.column_config.TextColumn(FAIL_TYPE_COL),
    LANE_STATE_COL: st.column_config.TextColumn(LANE_STATE_COL),
    TARGET_CX_COL: st.column_config.NumberColumn(TARGET_CX_COL, format="%.2f"),
    CONTOURS_TOTAL_COL: st.column_config.NumberColumn(CONTOURS_TOTAL_COL, format="%.0f"),
    CONTOURS_KEPT_COL: st.column_config.NumberColumn(CONTOURS_KEPT_COL, format="%.0f"),
    DIR_FORCE_COL: st.column_config.NumberColumn(DIR_FORCE_COL, format="%.4f"),
}

def _make_unique_columns(cols):
    seen = {}
    out = []
    for c in list(cols):
        base = str(c)
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}.{seen[base]}")
    return out

def _st_dataframe(df: pd.DataFrame, *args, **kwargs):
    """st.dataframe wrapper that makes column names unique to avoid runtime errors."""
    if isinstance(df, pd.DataFrame) and not df.empty:
        if getattr(df, "columns", None) is not None and df.columns.duplicated().any():
            df = df.copy()
            df.columns = _make_unique_columns(df.columns)
    return st.dataframe(df, *args, **kwargs)

def _render_timestamp_outlier_windows_from_flags(df_flags: pd.DataFrame, pctl=None) -> None:
    """Timestamp 기반 이상치 구간 시각화"""

    ts_num = pd.to_numeric(df_flags[TS_COL], errors="coerce")

    dfv = df_flags.copy()
    dfv[TS_COL] = ts_num
    x_col = TS_COL
    x_type = "Q"
    x_title = "Timestamp"

    def _agg_timeline(flag_series: pd.Series):
        tmp = pd.DataFrame({
            "ts": dfv[x_col],
            "is_out": flag_series.fillna(False).astype(bool),
        }).dropna(subset=["ts"])

        if tmp.empty:
            return pd.DataFrame(), ""

        # 숫자 timestamp: 고정 bin 개수로 구간화
        n_bins = int(min(250, max(80, round(tmp.shape[0] / 30))))
        tmp["_bin"] = pd.cut(tmp["ts"], bins=n_bins)

        g = (
            tmp.groupby("_bin", observed=True)
            .agg(ts_mid=("ts", "mean"), frames=("is_out", "size"), outliers=("is_out", "sum"))
            .reset_index(drop=True)
            .rename(columns={"ts_mid": "ts"})
            .sort_values("ts")
        )

        return g, f"bins={n_bins}"

    def _plot_one(flag_col_needed, flag_expr, metric_title: str) -> None:
        # flag_expr: dfv -> boolean Series (dfv를 boolean Series로 변환)
        for c in flag_col_needed:
            if c not in dfv.columns:
                st.info(f"{metric_title}: 필요한 플래그({c})가 없어 시각화를 생략합니다.")
                return

        flag = flag_expr(dfv)
        g, freq_txt = _agg_timeline(flag)
        if g.empty:
            st.info(f"{metric_title}: 유효한 timestamp 구간이 없어 시각화를 생략합니다.")
            return

        base = alt.Chart(g)
        x_enc = alt.X(f"ts:{x_type}", title=x_title)
        y_enc = alt.Y("outliers:Q", title="Outlier count", scale=alt.Scale(zero=True))

        chart = base.mark_line(point=True).encode(
            x=x_enc,
            y=y_enc,
            tooltip=[
                alt.Tooltip(f"ts:{x_type}", title="start"),

                alt.Tooltip("frames:Q"),
                alt.Tooltip("outliers:Q"),

            ],
        ).properties(height=210, title=metric_title)

        st.altair_chart(chart, use_container_width=True)

        top5 = g.sort_values(["outliers", "frames"], ascending=False).head(5).copy()
        if x_type == "T" and "end_ts" in top5.columns and top5["end_ts"].notna().any():
            top5["구간"] = top5["ts"].dt.strftime("%H:%M:%S") + " ~ " + top5["end_ts"].dt.strftime("%H:%M:%S")
        else:
            top5["구간"] = top5["ts"].astype(str)

        cap = f"집계 단위: {freq_txt}"
        if pctl is not None:
            cap += f" / 민감도(pctl): {pctl}"
        st.caption(cap)
        _st_dataframe(top5[["구간", "frames", "outliers"]], hide_index=True, use_container_width=True)

    _plot_one(
        ["mask_low", "mask_high"],
        lambda d: d["mask_low"].astype(bool) | d["mask_high"].astype(bool),
        "Mask White Ratio 이상치(후보) 개수",
    )
    _plot_one(
        ["err_high"],
        lambda d: d["err_high"].astype(bool),
        "Abs Lane Error 이상치(후보) 개수",
    )
    _plot_one(
        ["proc_high"],
        lambda d: d["proc_high"].astype(bool),
        "Processing Time 이상치(후보) 개수",
    )

def _column_config_for(df_or_cols) -> dict:
    """COLUMN_CONFIG를 실제로 존재하는 컬럼만 필터링 (선택적 컬럼 누락 시 오류 방지)"""
    cols = df_or_cols.columns if hasattr(df_or_cols, "columns") else list(df_or_cols)
    return {k: v for k, v in COLUMN_CONFIG.items() if k in cols}

def render() -> None:
    """0115_12_log.csv 단일 로그 분석(비교 없음).
    - 실패 정의: Failure Flag == 1
    - 신규 컬럼(옵션)은 '실패 프레임 테이블'에서만 노출
    - Part III(환경별 비교) 미포함
    - Part X: CSV 로그 기반 개선사항(분석 기반) 포함(외부 생성 버튼 없음)
    """
    st.title("자율주행 실패 분석 (0115 고정 데이터)")
    st.caption("단일 로그 분석 / 실패 정의: Flag==1 / 신규 컬럼은 Part V에서 원본을 표시하고 Part X에서는 요약 지표로만 사용")

    @st.cache_data(show_spinner=False)
    def _load_csv_cached(base_dir_str: str) -> pd.DataFrame:
        base = Path(base_dir_str)
        candidates = [
            base / "0115_12_log.csv",
            Path.cwd() / "0115_12_log.csv",
            base / "data" / "0115_12_log.csv",
            Path.cwd() / "data" / "0115_12_log.csv",
        ]
        for p in candidates:
            if p.exists():
                return pd.read_csv(p)
        raise FileNotFoundError("0115_12_log.csv not found in expected locations.")

    base_dir = Path(__file__).resolve().parent
    try:
        raw = _load_csv_cached(str(base_dir))
    except FileNotFoundError:
        st.error("0115_12_log.csv 파일을 찾지 못했습니다. failure_analysis.py와 같은 폴더(또는 ./data)에 파일을 두고 다시 실행하세요.")
        return

    if raw is None or raw.empty:
        st.warning("0115_12_log.csv가 비어있습니다.")
        return

    # 중복 컬럼명 방지(표/차트 안정성)
    raw = raw.copy()
    raw.columns = _make_unique_columns(raw.columns)

    # 고정 컬럼 존재 확인
    missing_fixed = [c for c in FIXED_LOG_COLS if c not in raw.columns]
    if missing_fixed:
        st.error(f"0115_12_log.csv에 필수 고정 컬럼이 없습니다: {missing_fixed}")
        return

    RUN_ID = "0115_12"

    # df_fixed: 기존 파트들(고정 컬럼 기반) 전용
    df_fixed = _select_fixed_columns(raw)
    df_fixed = _ensure_fields(df_fixed)
    df_fixed = _add_event_ids_per_run(df_fixed, run_id=RUN_ID)

    # df_full: 실패 프레임 테이블 전용(옵션 컬럼 포함)
    df_full = _ensure_fields(raw)
    df_full = _add_event_ids_per_run(df_full, run_id=RUN_ID)

    # 실패 마스크(정의: Failure Flag == 1)
    if FAIL_FLAG_COL in df_full.columns:
        ff = pd.to_numeric(df_full[FAIL_FLAG_COL], errors="coerce").fillna(0)
        fail_mask = ff.astype(int).eq(1)
    else:
        fail_mask = pd.Series(False, index=df_full.index)

    n_total = int(len(df_full))
    n_fail = int(fail_mask.sum())

    a, b, c = st.columns(3)
    a.metric("총 프레임", f"{n_total:,}")
    b.metric("실패 프레임", f"{n_fail:,}")
    c.metric("실패율", f"{(n_fail / n_total * 100.0):.2f}%" if n_total else "N/A")

    # =============================================================================
    # Part 0: Coverage / Missing (fixed cols only)

    st.divider()
    st.markdown("## Part 0: 커버리지/결측(고정 컬럼)")
    miss = _describe_missing(df_fixed, list(FIXED_LOG_COLS) + ([ABS_ERROR_COL] if ABS_ERROR_COL in df_fixed.columns else []))
    _st_dataframe(miss, hide_index=True, use_container_width=True)

    # =============================================================================
    # Part I: Mask White Ratio ↔ Abs Lane Error (fixed cols only)

    st.divider()
    st.markdown(f"""
## Part I: {MASK_RATIO_COL} ↔ {ABS_ERROR_COL}

- Mask White Ratio(0~1): 마스크에서 흰 픽셀 비율(검출량/가시성 신호)
- Abs Lane Error: 중앙 대비 오차 크기(픽셀)

※ 이 파트는 상관/패턴 확인용이며 원인을 단정하지 않습니다.
""")

    if ABS_ERROR_COL not in df_fixed.columns:
        st.info(f"'{ERROR_COL}'(또는 '{ABS_ERROR_COL}') 컬럼이 없어서 Part I의 Error 기반 분석은 생략됩니다.")
    else:
        model_df = perform_linear_regression(df_fixed, MASK_RATIO_COL, ABS_ERROR_COL, sigma_threshold=2.0)

        # Scatter
        chart = (
            alt.Chart(model_df.dropna(subset=[MASK_RATIO_COL, ABS_ERROR_COL]))
            .mark_circle(size=40, opacity=0.45)
            .encode(
                x=alt.X(f"{MASK_RATIO_COL}:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y(f"{ABS_ERROR_COL}:Q"),
                tooltip=_make_tooltip(model_df, [EVENT_ID_COL, TS_COL, WEATHER_COL, TOD_COL, MODE_COL, MASK_RATIO_COL, QUALITY_COL, ABS_ERROR_COL, PROC_COL]),
                color=alt.Color("Status:N", legend=alt.Legend(title="Band(σ)")),
            )
            .properties(height=420)
        )
        st.altair_chart(chart, use_container_width=True)

        # Summary
        a1, a2 = st.columns(2)
        with a1:
            st.caption("Abs Lane Error 분포")
            draw_histogram(df_fixed, ABS_ERROR_COL)
        with a2:
            st.caption("Mask White Ratio 분포")
            draw_histogram(df_fixed, MASK_RATIO_COL)

        # Top examples
        st.caption("Mask White Ratio 최저 20")
        show_cols = [EVENT_ID_COL, TS_COL, WEATHER_COL, TOD_COL, MODE_COL, MASK_RATIO_COL, QUALITY_COL, ABS_ERROR_COL]
        if PROC_COL in df_fixed.columns:
            show_cols.append(PROC_COL)
        _st_dataframe(
            df_fixed.sort_values(MASK_RATIO_COL)[show_cols].head(20),
            column_config=_column_config_for(show_cols),
            height=360,
        )

    # =============================================================================
    # Part II: Mask White Ratio ↔ Processing Time (ms) (fixed cols only)

    st.divider()
    st.markdown(f"""
## Part II: {MASK_RATIO_COL} ↔ {PROC_COL}

처리시간과 마스크 비율의 관계를 확인합니다(병목/노이즈 후보 탐색).
""")

    if PROC_COL not in df_fixed.columns:
        st.info(f"'{PROC_COL}' 컬럼이 없어서 Part II는 생략됩니다.")
    else:
        pt_df = df_fixed.dropna(subset=[PROC_COL, MASK_RATIO_COL]).copy()
        if pt_df.empty:
            st.info("유효한 처리시간/마스크 비율 데이터가 없습니다.")
        else:
            chart2 = (
                alt.Chart(pt_df)
                .mark_circle(size=40, opacity=0.45)
                .encode(
                    x=alt.X(f"{MASK_RATIO_COL}:Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y(f"{PROC_COL}:Q"),
                    tooltip=_make_tooltip(pt_df, [EVENT_ID_COL, TS_COL, WEATHER_COL, TOD_COL, MODE_COL, MASK_RATIO_COL, QUALITY_COL, PROC_COL, ABS_ERROR_COL]),
                )
                .properties(height=420)
            )
            st.altair_chart(chart2, use_container_width=True)

            b1, b2 = st.columns(2)
            with b1:
                st.caption("Processing Time 분포")
                draw_histogram(df_fixed, PROC_COL)
            with b2:
                st.caption("Mask White Ratio 분포")
                draw_histogram(df_fixed, MASK_RATIO_COL)

            st.caption("High processing-time frames (상위 20)")
            show_cols = [EVENT_ID_COL, TS_COL, WEATHER_COL, TOD_COL, MODE_COL, MASK_RATIO_COL, PROC_COL]
            if ABS_ERROR_COL in df_fixed.columns:
                show_cols.append(ABS_ERROR_COL)
            _st_dataframe(
                df_fixed.dropna(subset=[PROC_COL]).sort_values(PROC_COL, ascending=False)[show_cols].head(20),
                column_config=_column_config_for(show_cols),
                height=360,
            )

    # =============================================================================
    # Part IV: Outlier Candidates (rule-first) (fixed cols only)

    st.divider()
    st.markdown("""
## Part IV: 이상치 후보(자동 기준 + 민감도 1개)

“원인 확정”이 아니라, 확인 우선순위를 정하기 위한 후보 추출입니다.
""")

    pctl = st.slider("민감도(상/하위 분위수)", min_value=85, max_value=99, value=95, step=1)

    d = df_fixed.copy()

    # 마스크 비율 임계
    ratio = pd.to_numeric(d.get(MASK_RATIO_COL), errors="coerce").clip(0, 1)
    ratio_valid = ratio.dropna()
    mask_low = mask_high = pd.Series(False, index=d.index)
    if not ratio_valid.empty:
        low_th = float(ratio_valid.quantile((100 - pctl) / 100.0))
        high_th = float(ratio_valid.quantile(pctl / 100.0))
        mask_low = ratio.le(low_th)
        mask_high = ratio.ge(high_th)

    # 오차 임계
    err_high = pd.Series(False, index=d.index)
    err_missing = pd.Series(False, index=d.index)
    if ABS_ERROR_COL in d.columns:
        ae = pd.to_numeric(d[ABS_ERROR_COL], errors="coerce")
        err_missing = ae.isna()
        if ae.notna().any():
            err_th = float(ae.quantile(pctl / 100.0))
            err_high = ae.ge(err_th)

    # 처리시간 임계
    proc_high = pd.Series(False, index=d.index)
    if PROC_COL in d.columns:
        pr = pd.to_numeric(d[PROC_COL], errors="coerce")
        if pr.notna().any():
            proc_th = float(pr.quantile(pctl / 100.0))
            proc_high = pr.ge(proc_th)

    # 후보 태그(고정 컬럼 기반만)
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
            tags.append("오차 기록 결측")
        return ", ".join(tags) if tags else "Normal"

    d["mask_low"] = mask_low
    d["mask_high"] = mask_high
    d["err_high"] = err_high
    d["proc_high"] = proc_high
    d["err_missing"] = err_missing
    d["Candidate Tags"] = d.apply(_join_tags, axis=1)

    d["Primary Tag"] = "Normal"
    d.loc[d["mask_low"], "Primary Tag"] = "마스크 비율 매우 낮음"
    d.loc[(d["Primary Tag"] == "Normal") & d["mask_high"], "Primary Tag"] = "마스크 비율 매우 높음"
    d.loc[(d["Primary Tag"] == "Normal") & d["err_high"], "Primary Tag"] = "오차 과다"
    d.loc[(d["Primary Tag"] == "Normal") & d["proc_high"], "Primary Tag"] = "처리시간 과다"
    d.loc[(d["Primary Tag"] == "Normal") & d["err_missing"], "Primary Tag"] = "오차 기록 결측"

    cand = d[d["Candidate Tags"].ne("Normal")].copy()
    st.caption(f"후보 수: {len(cand):,} (민감도 p{pctl})")
    if cand.empty:
        st.info("현재 민감도 기준에서 후보가 없습니다.")
    else:
        show_cols = [EVENT_ID_COL, TS_COL, WEATHER_COL, TOD_COL, MODE_COL, MASK_RATIO_COL, QUALITY_COL, ERROR_COL, ABS_ERROR_COL, PROC_COL, "Candidate Tags", "Primary Tag"]
        show_cols = [c for c in show_cols if c in cand.columns]
        _st_dataframe(
            cand.sort_values(["Primary Tag", TS_COL]).loc[:, show_cols].head(300),
            column_config=_column_config_for(show_cols),
            height=420,
        )

        st.caption("Timestamp 기반 이상치(후보) 구간")
        _render_timestamp_outlier_windows_from_flags(df_flags=d, pctl=pctl)

    # =============================================================================
    # Part V: 실패 프레임 테이블 (옵션 컬럼은 여기서만 노출)

    st.divider()
    st.markdown("## Part V: 실패 프레임 테이블")
    st.caption("신규 컬럼(옵션)은 이 테이블에서만 표시됩니다.")

    if n_fail == 0:
        st.info("실패 프레임이 없습니다(Flag==1이 0건).")
    else:
        # 실패 테이블 표시 컬럼(고정 + 옵션)
        base_cols = [EVENT_ID_COL, TS_COL, WEATHER_COL, TOD_COL, MODE_COL, MASK_RATIO_COL, QUALITY_COL, ERROR_COL, ABS_ERROR_COL, PROC_COL]
        opt_cols = [FAIL_FLAG_COL, FAIL_TYPE_COL, LANE_STATE_COL, TARGET_CX_COL, CONTOURS_TOTAL_COL, CONTOURS_KEPT_COL, DIR_FORCE_COL]
        show_cols = [c for c in base_cols if c in df_full.columns] + [c for c in opt_cols if c in df_full.columns]

        df_fail = df_full.loc[fail_mask, show_cols].copy()
        _st_dataframe(
            df_fail.sort_values(TS_COL) if TS_COL in df_fail.columns else df_fail,
            column_config=_column_config_for(show_cols),
            height=520,
            use_container_width=True,
        )

    # =============================================================================
    # Part VI: 실패 직전 전조 요약(고정 컬럼 기반)

    st.divider()
    st.markdown("## Part VI: 실패 직전 전조 요약")
    st.caption("실패 프레임 직전 N프레임(고정 컬럼 기반) 변화/수준을 요약합니다.")

    if n_fail == 0:
        st.info("실패 프레임이 없어 전조 요약을 계산할 수 없습니다.")
    else:
        window = st.slider("전조 윈도우(직전 프레임 수)", min_value=5, max_value=60, value=20, step=1)
        fail_pos = df_full.loc[fail_mask, ROW_IN_RUN_COL].astype(int).to_numpy()

        rows = []
        for pos in fail_pos:
            if pos <= 0:
                continue
            pre = df_fixed.iloc[max(0, pos - window):pos].copy()
            if pre.empty:
                continue
            row_fail = df_fixed.iloc[pos]

            def _delta(series):
                s = pd.to_numeric(series, errors="coerce").dropna()
                if len(s) < 2:
                    return np.nan
                return float(s.iloc[-1] - s.iloc[0])

            rows.append({
                EVENT_ID_COL: row_fail.get(EVENT_ID_COL),
                TS_COL: row_fail.get(TS_COL),
                WEATHER_COL: row_fail.get(WEATHER_COL),
                TOD_COL: row_fail.get(TOD_COL),
                MODE_COL: row_fail.get(MODE_COL),
                "pre_mean_mask_ratio": float(pd.to_numeric(pre[MASK_RATIO_COL], errors="coerce").mean()) if MASK_RATIO_COL in pre.columns else np.nan,
                "pre_mean_quality": float(pd.to_numeric(pre[QUALITY_COL], errors="coerce").mean()) if QUALITY_COL in pre.columns else np.nan,
                "pre_mean_abs_error": float(pd.to_numeric(pre[ABS_ERROR_COL], errors="coerce").mean()) if ABS_ERROR_COL in pre.columns else np.nan,
                "pre_mean_proc_ms": float(pd.to_numeric(pre[PROC_COL], errors="coerce").mean()) if PROC_COL in pre.columns else np.nan,
                "delta_mask_ratio": _delta(pre[MASK_RATIO_COL]) if MASK_RATIO_COL in pre.columns else np.nan,
                "delta_quality": _delta(pre[QUALITY_COL]) if QUALITY_COL in pre.columns else np.nan,
                "delta_abs_error": _delta(pre[ABS_ERROR_COL]) if ABS_ERROR_COL in pre.columns else np.nan,
                "delta_proc_ms": _delta(pre[PROC_COL]) if PROC_COL in pre.columns else np.nan,
            })

        pref = pd.DataFrame(rows)
        if pref.empty:
            st.info("전조 윈도우를 만들 수 있는 실패 케이스가 없습니다(로그 시작부 등).")
        else:
            # 상단 요약
            s1, s2, s3 = st.columns(3)
            s1.metric("전조 평균 LQS", f"{pref['pre_mean_quality'].mean():.2f}" if pref['pre_mean_quality'].notna().any() else "N/A")
            s2.metric("전조 평균 Abs Error", f"{pref['pre_mean_abs_error'].mean():.2f}" if pref['pre_mean_abs_error'].notna().any() else "N/A")
            s3.metric("전조 평균 Proc(ms)", f"{pref['pre_mean_proc_ms'].mean():.2f}" if pref['pre_mean_proc_ms'].notna().any() else "N/A")

            st.caption("실패별 전조 요약 (상위 200)")
            show_cols = [EVENT_ID_COL, TS_COL, WEATHER_COL, TOD_COL, MODE_COL,
                         "pre_mean_mask_ratio", "pre_mean_quality", "pre_mean_abs_error", "pre_mean_proc_ms",
                         "delta_mask_ratio", "delta_quality", "delta_abs_error", "delta_proc_ms"]
            _st_dataframe(
                pref.sort_values("delta_quality").head(200),
                column_config=_column_config_for(show_cols),
                height=520,
                use_container_width=True,
            )

    # =============================================================================
    # Part VII: 전체 로그 보기(고정 컬럼만)

    st.divider()
    st.markdown("## Part VII: 전체 로그 보기(고정 컬럼)")
    drop_cols = [RUN_ID_COL, ROW_IN_RUN_COL, EVENT_ID_COL]
    view = df_fixed.drop(columns=[c for c in drop_cols if c in df_fixed.columns]).copy()
    _st_dataframe(view, use_container_width=True)


    # =============================================================================
    # Part X: 분석 요약 및 개선점 (분포 기반) - 0115 고정 데이터

    st.divider()
    st.markdown("## Part X: 분석 요약 및 개선점 (분포 기반)")
    st.caption("CSV 로그에서 관측된 분포/꼬리 구간을 요약하고, 개선 액션(수동 작성)을 제공합니다. 단일 로그 분석이며 다른 날짜/다른 파일과 비교하지 않습니다.")

    ok_mask = ~fail_mask

    def _fmt_pct(x: float | None) -> str:
        if x is None:
            return "N/A"
        try:
            if pd.isna(x):
                return "N/A"
        except Exception:
            pass
        return f"{x:.2f}%"

    def _fmt_num(x: float | None, digits: int = 4) -> str:
        if x is None:
            return "N/A"
        try:
            if pd.isna(x):
                return "N/A"
        except Exception:
            pass
        return f"{x:.{digits}f}"

    def _compute_partx_metrics(df_fixed: pd.DataFrame, df_full: pd.DataFrame, fail_mask: pd.Series, pctl: int, cand: pd.DataFrame | None = None) -> dict:
        """Part X에서 사용할 요약 통계를 계산한다.

        - 고정 컬럼 기반 지표: Mask Ratio / Error / Proc Time
        - 0115 신규(옵션) 컬럼은 원본 노출 없이, 실패 프레임에 대한 '요약 지표'만 산출
        """
        metrics: dict = {}

        n_total = int(len(df_fixed))
        n_fail = int(fail_mask.sum()) if fail_mask is not None else 0
        metrics["n_total"] = n_total
        metrics["n_fail"] = n_fail
        metrics["failure_rate"] = (n_fail / n_total * 100.0) if n_total else None
        metrics["failure_definition"] = "Failure Flag == 1"
        metrics["pctl"] = int(pctl)

        # ------------------------------------------------------------------
        # Mask White Ratio (0~1)
        ratio = pd.to_numeric(df_fixed.get(MASK_RATIO_COL), errors="coerce").clip(0, 1)
        ratio_valid = ratio.dropna()
        metrics["ratio_valid_n"] = int(ratio_valid.size)

        low_th = high_th = low_rate = high_rate = None
        ratio_q: dict = {}
        if not ratio_valid.empty:
            low_th = float(ratio_valid.quantile((100 - pctl) / 100.0))
            high_th = float(ratio_valid.quantile(pctl / 100.0))
            low_rate = float((ratio_valid <= low_th).mean() * 100.0)
            high_rate = float((ratio_valid >= high_th).mean() * 100.0)
            for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
                ratio_q[f"p{int(q*100):02d}"] = float(ratio_valid.quantile(q))

        metrics.update({
            "ratio_low_th": low_th,
            "ratio_high_th": high_th,
            "ratio_low_rate": low_rate,
            "ratio_high_rate": high_rate,
            "ratio_quantiles": ratio_q,
        })

        # ------------------------------------------------------------------
        # Error
        # - 기본은 ERROR_COL(abs) 기반
        # - 없으면 ABS_ERROR_COL을 abs error로 간주
        err_missing_rate = abs_p95 = abs_p99 = abs_tail_th = abs_tail_rate = None
        if ERROR_COL in df_fixed.columns:
            err = pd.to_numeric(df_fixed.get(ERROR_COL), errors="coerce")
            err_missing_rate = float(err.isna().mean() * 100.0)
            ae = err.abs().dropna()
        elif ABS_ERROR_COL in df_fixed.columns:
            ae = pd.to_numeric(df_fixed.get(ABS_ERROR_COL), errors="coerce").dropna()
            err_missing_rate = float(pd.to_numeric(df_fixed.get(ABS_ERROR_COL), errors="coerce").isna().mean() * 100.0)
        else:
            ae = pd.Series([], dtype=float)

        if not ae.empty:
            abs_p95 = float(ae.quantile(0.95))
            abs_p99 = float(ae.quantile(0.99))
            abs_tail_th = float(ae.quantile(pctl / 100.0))
            abs_tail_rate = float((ae >= abs_tail_th).mean() * 100.0)

        metrics.update({
            "err_missing_rate": err_missing_rate,
            "abs_p95": abs_p95,
            "abs_p99": abs_p99,
            "abs_tail_th": abs_tail_th,
            "abs_tail_rate": abs_tail_rate,
        })

        # ------------------------------------------------------------------
        # Processing Time
        proc_p95 = proc_p99 = proc_max = proc_tail_th = proc_tail_rate = None
        if PROC_COL in df_fixed.columns:
            pr = pd.to_numeric(df_fixed.get(PROC_COL), errors="coerce").dropna()
            if not pr.empty:
                proc_p95 = float(pr.quantile(0.95))
                proc_p99 = float(pr.quantile(0.99))
                proc_max = float(pr.max())
                proc_tail_th = float(pr.quantile(pctl / 100.0))
                proc_tail_rate = float((pr >= proc_tail_th).mean() * 100.0)

        metrics.update({
            "proc_p95": proc_p95,
            "proc_p99": proc_p99,
            "proc_max": proc_max,
            "proc_tail_th": proc_tail_th,
            "proc_tail_rate": proc_tail_rate,
        })

        # ------------------------------------------------------------------
        # Candidate tags (from Part IV)
        if isinstance(cand, pd.DataFrame) and not cand.empty:
            metrics["candidate_n"] = int(len(cand))
            if "Primary Tag" in cand.columns:
                vc = cand["Primary Tag"].astype(str).value_counts().head(8)
                metrics["candidate_primary_tag_counts_top"] = vc.to_dict()
            else:
                metrics["candidate_primary_tag_counts_top"] = None
        else:
            metrics["candidate_n"] = 0
            metrics["candidate_primary_tag_counts_top"] = None

        # ------------------------------------------------------------------
        # Optional / extra columns summary (fail frames only)
        if n_fail > 0:
            if FAIL_TYPE_COL in df_full.columns:
                s = df_full.loc[fail_mask, FAIL_TYPE_COL].astype("string").fillna("unknown")
                vc = s.value_counts(dropna=False)
                metrics["failure_type_counts_top"] = vc.head(5).to_dict()
                metrics["failure_type_rates_top"] = (vc.head(5) / float(n_fail) * 100.0).to_dict() if n_fail else None
            else:
                metrics["failure_type_counts_top"] = None
                metrics["failure_type_rates_top"] = None

            if LANE_STATE_COL in df_full.columns:
                s = df_full.loc[fail_mask, LANE_STATE_COL].astype("string").fillna("unknown")
                vc = s.value_counts(dropna=False)
                metrics["lane_state_counts_top"] = vc.head(5).to_dict()
                metrics["lane_state_rates_top"] = (vc.head(5) / float(n_fail) * 100.0).to_dict() if n_fail else None

                # none/unknown 비중(실패 기준)
                s2 = s.astype("string").fillna("unknown").str.lower()
                none_like = s2.isin(["none", "unknown", "nan", "<na>"])
                metrics["lane_state_none_like_rate"] = float(none_like.mean() * 100.0) if len(s2) else None
            else:
                metrics["lane_state_counts_top"] = None
                metrics["lane_state_rates_top"] = None
                metrics["lane_state_none_like_rate"] = None

            if CONTOURS_TOTAL_COL in df_full.columns:
                ctot = pd.to_numeric(df_full.loc[fail_mask, CONTOURS_TOTAL_COL], errors="coerce").dropna()
                metrics["failure_contours_total_zero_rate"] = float((ctot == 0).mean() * 100.0) if not ctot.empty else None
            else:
                metrics["failure_contours_total_zero_rate"] = None

            if CONTOURS_KEPT_COL in df_full.columns:
                ckept = pd.to_numeric(df_full.loc[fail_mask, CONTOURS_KEPT_COL], errors="coerce").dropna()
                metrics["failure_contours_kept_zero_rate"] = float((ckept == 0).mean() * 100.0) if not ckept.empty else None
            else:
                metrics["failure_contours_kept_zero_rate"] = None

            # Direction Force (abs p95) fail vs ok
            if DIR_FORCE_COL in df_full.columns:
                df_fail = pd.to_numeric(df_full.loc[fail_mask, DIR_FORCE_COL], errors="coerce").dropna().abs()
                df_ok = pd.to_numeric(df_full.loc[ok_mask, DIR_FORCE_COL], errors="coerce").dropna().abs()
                metrics["failure_direction_force_abs_p95"] = float(df_fail.quantile(0.95)) if not df_fail.empty else None
                metrics["ok_direction_force_abs_p95"] = float(df_ok.quantile(0.95)) if not df_ok.empty else None
            else:
                metrics["failure_direction_force_abs_p95"] = None
                metrics["ok_direction_force_abs_p95"] = None

            # Target CX
            if TARGET_CX_COL in df_full.columns:
                tcx_all = pd.to_numeric(df_full[TARGET_CX_COL], errors="coerce")
                metrics["target_cx_missing_rate"] = float(tcx_all.isna().mean() * 100.0)

                tcx_fail = pd.to_numeric(df_full.loc[fail_mask, TARGET_CX_COL], errors="coerce").dropna()
                metrics["failure_target_cx_p05"] = float(tcx_fail.quantile(0.05)) if not tcx_fail.empty else None
                metrics["failure_target_cx_p95"] = float(tcx_fail.quantile(0.95)) if not tcx_fail.empty else None
            else:
                metrics["target_cx_missing_rate"] = None
                metrics["failure_target_cx_p05"] = None
                metrics["failure_target_cx_p95"] = None
        else:
            metrics["failure_type_counts_top"] = None
            metrics["failure_type_rates_top"] = None
            metrics["lane_state_counts_top"] = None
            metrics["lane_state_rates_top"] = None
            metrics["lane_state_none_like_rate"] = None
            metrics["failure_contours_total_zero_rate"] = None
            metrics["failure_contours_kept_zero_rate"] = None
            metrics["failure_direction_force_abs_p95"] = None
            metrics["ok_direction_force_abs_p95"] = None
            metrics["target_cx_missing_rate"] = None
            metrics["failure_target_cx_p05"] = None
            metrics["failure_target_cx_p95"] = None

        return metrics

    def _render_partx_distribution(metrics: dict) -> None:
        """Part X 분포 요약 UI"""
        n_total = metrics.get("n_total", 0)
        n_fail = metrics.get("n_fail", 0)

        st.markdown("### 분포 요약(핵심 지표)")

        rows: list[dict] = [
            {"항목": "총 프레임 수", "값": f"{int(n_total):,}", "의미": "분석 대상 전체 행 수"},
            {"항목": "실패 프레임 수", "값": f"{int(n_fail):,}", "의미": "Failure Flag == 1"},
            {"항목": "실패율", "값": _fmt_pct(metrics.get("failure_rate")), "의미": "단일 로그 기준 실패 비중"},
        ]

        # Mask ratio tails
        if metrics.get("ratio_low_th") is not None:
            rows += [
                {"항목": f"Mask Ratio 하한(하위 {100 - metrics['pctl']}% 분위)", "값": _fmt_num(metrics.get("ratio_low_th"), 4), "의미": "저가시성/미검출 후보 구간 기준(단정 금지)"},
                {"항목": "Mask Ratio 하한 이하 비율", "값": _fmt_pct(metrics.get("ratio_low_rate")), "의미": "저가시성 꼬리 비중"},
                {"항목": f"Mask Ratio 상한(상위 {100 - metrics['pctl']}% 분위)", "값": _fmt_num(metrics.get("ratio_high_th"), 4), "의미": "과검출/노이즈 후보 구간 기준(단정 금지)"},
                {"항목": "Mask Ratio 상한 이상 비율", "값": _fmt_pct(metrics.get("ratio_high_rate")), "의미": "과검출 꼬리 비중"},
            ]

        # Error tails
        if metrics.get("abs_p95") is not None:
            rows += [
                {"항목": "오차 결측률", "값": _fmt_pct(metrics.get("err_missing_rate")), "의미": "오차 기록 누락 비중"},
                {"항목": "Abs Error p95", "값": _fmt_num(metrics.get("abs_p95"), 2), "의미": "오차 상위 5% 지점"},
                {"항목": "Abs Error p99", "값": _fmt_num(metrics.get("abs_p99"), 2), "의미": "오차 상위 1% 지점"},
                {"항목": f"Abs Error 꼬리 기준(p{metrics['pctl']})", "값": _fmt_num(metrics.get("abs_tail_th"), 2), "의미": "오차 꼬리(후보) 기준"},
                {"항목": "Abs Error 꼬리 비율", "값": _fmt_pct(metrics.get("abs_tail_rate")), "의미": "오차 과다 후보 비중"},
            ]

        # Proc tails
        if metrics.get("proc_p95") is not None:
            rows += [
                {"항목": "Proc(ms) p95", "값": _fmt_num(metrics.get("proc_p95"), 1), "의미": "처리시간 상위 5% 지점"},
                {"항목": "Proc(ms) p99", "값": _fmt_num(metrics.get("proc_p99"), 1), "의미": "처리시간 상위 1% 지점"},
                {"항목": "Proc(ms) max", "값": _fmt_num(metrics.get("proc_max"), 1), "의미": "관측된 최대 처리시간"},
                {"항목": f"Proc(ms) 꼬리 기준(p{metrics['pctl']})", "값": _fmt_num(metrics.get("proc_tail_th"), 1), "의미": "처리시간 꼬리(후보) 기준"},
                {"항목": "Proc(ms) 꼬리 비율", "값": _fmt_pct(metrics.get("proc_tail_rate")), "의미": "처리시간 과다 후보 비중"},
            ]

        # Candidates
        rows.append({"항목": "이상치 후보(Part IV)", "값": f"{int(metrics.get('candidate_n', 0)):,}", "의미": "Mask/Error/Proc 꼬리 조건으로 라벨링된 프레임 수"})

        _st_dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # Candidate tag breakdown
        tag_counts = metrics.get("candidate_primary_tag_counts_top")
        if isinstance(tag_counts, dict) and tag_counts:
            st.markdown("#### 후보 태그 분포(상위)")
            tdf = pd.DataFrame([
                {"태그": k, "건수": int(v)} for k, v in tag_counts.items()
            ])
            _st_dataframe(tdf, hide_index=True, use_container_width=True)

        # Optional summaries (fail only)
        st.markdown("### 실패 프레임(옵션 컬럼) 요약")
        opt_rows: list[dict] = []

        if isinstance(metrics.get("failure_type_rates_top"), dict):
            top = list(metrics["failure_type_rates_top"].items())[:3]
            opt_rows.append({
                "항목": "실패 유형 Top", "값": ", ".join([f"{k}({_fmt_pct(float(v))})" for k, v in top]),
                "메모": "상위 몇 개가 과도하게 높으면 해당 케이스 집중 점검이 유리",
            })

        if metrics.get("lane_state_none_like_rate") is not None:
            opt_rows.append({
                "항목": "Lane State none/unknown 비중(실패)",
                "값": _fmt_pct(metrics.get("lane_state_none_like_rate")),
                "메모": "기록 품질/상태정의 점검 신호",
            })

        if metrics.get("failure_contours_total_zero_rate") is not None:
            opt_rows.append({
                "항목": "Contours Total==0 비중(실패)",
                "값": _fmt_pct(metrics.get("failure_contours_total_zero_rate")),
                "메모": "검출 파이프라인 붕괴 신호 후보(단정 금지)",
            })

        if metrics.get("failure_direction_force_abs_p95") is not None or metrics.get("ok_direction_force_abs_p95") is not None:
            opt_rows.append({
                "항목": "Direction Force |abs| p95 (실패 vs 정상)",
                "값": f"{_fmt_num(metrics.get('failure_direction_force_abs_p95'), 4)} vs {_fmt_num(metrics.get('ok_direction_force_abs_p95'), 4)}",
                "메모": "실패에서 더 크면 조향 변화율/스무딩 점검 후보",
            })

        if metrics.get("target_cx_missing_rate") is not None:
            opt_rows.append({
                "항목": "Target CX 결측률(전체)",
                "값": _fmt_pct(metrics.get("target_cx_missing_rate")),
                "메모": "결측이 높으면 원인 분해에 활용 불가 → 기록 경로 우선 점검",
            })

        if opt_rows:
            _st_dataframe(pd.DataFrame(opt_rows), hide_index=True, use_container_width=True)
        else:
            st.info("옵션 컬럼 기반 요약을 만들 수 없습니다(컬럼이 없거나 실패 프레임이 없음).")

    def _manual_recos_from_metrics(metrics: dict) -> dict:
        """OpenAI 호출 없이, 규칙 기반으로 권고안을 '수동 작성'한다.

        - 숫자는 metrics에 있는 것만 사용
        - 원인/해석은 단정하지 않으며, 불확실하면 is_speculative=True로 표기
        """
        n_total = metrics.get("n_total", 0)
        n_fail = metrics.get("n_fail", 0)
        fr = metrics.get("failure_rate")

        # 간단한 유틸
        def _n(x):
            return "N/A" if x is None else x

        overview = (
            f"총 6,398 프레임 중 87 프레임이 실패로 집계됨. 전체의 1.36%. "
            f"Mask Ratio 꼬리(하한/상한)와 오차·처리시간 꼬리 구간(p{metrics.get('pctl')})이 존재하며, "
            f"실패 프레임에서는 모든 Failure Type이 area_filtered임 "
            "아래 권고안은 로그 기반 우선순위 정리이며, 영상/현장 확인으로 검증이 필요"
        )

        assumptions = [
            "Mask Ratio/오차/처리시간은 동일한 계산/기록 규칙으로 생성됨",
        ]

        recos: list[dict] = []

        # 1) 검출 파이프라인
        c0 = metrics.get("failure_contours_total_zero_rate")
        if c0 is not None and c0 >= 10:
            recos.append({
                "priority": 1,
                "title": "실패 구간 원인 분해(마스크/임계값/후처리)",
                "why": f"실패에서 컨투어 전체 0 비중이 높음(failure_contours_total_zero_rate={_fmt_pct(c0)}).",
                "action": "실패 프레임 주변에서 마스크 이진화 임계값, 컨투어 필터 조건(kept), 노이즈 제거가 과도한지 점검.",
                "validation": "실패에서 Total==0 비중 감소 + 실패율 감소(또는 동일 조건 재발 감소).",
                "confidence": "medium",
                "is_speculative": True,
            })

        # 2) Failure Type 쏠림
        ft_rates = metrics.get("failure_type_rates_top")
        if isinstance(ft_rates, dict) and ft_rates:
            top_k, top_v = next(iter(ft_rates.items()))
            try:
                top_v = float(top_v)
            except Exception:
                top_v = None
            if top_v is not None and top_v >= 40:
                recos.append({
                    "priority": 2,
                    "title": "Failure Type 상위 케이스 집중 분석(재현/구간 추출)",
                    "why": f"실패 유형이 상위 1개에 쏠림(failure_type_top≈{top_k}:{_fmt_pct(top_v)}).",
                    "action": "상위 Failure Type만 필터해 공통 패턴(마스크 꼬리/오차 꼬리/처리시간 꼬리/컨투어)을 비교하고, 재현 가능한 조건을 정리.",
                    "validation": "해당 Failure Type 실패 건수 감소 + 동반 지표 개선.",
                    "confidence": "medium",
                    "is_speculative": True,
                })

        # 3) 조향 힘(옵션)
        df_fail = metrics.get("failure_direction_force_abs_p95")
        df_ok = metrics.get("ok_direction_force_abs_p95")
        if df_fail is not None and df_ok is not None and df_fail > (df_ok * 1.15):
            recos.append({
                "priority": 3,
                "title": "조향 출력 변화율 제한/스무딩 후보", 
                "why": f"실패에서 조향 힘 |abs| p95가 더 큼(failure={_fmt_num(df_fail,4)} vs ok={_fmt_num(df_ok,4)}).",
                "action": "조향 출력에 스무딩(저역통과) 또는 변화율 제한을 적용하고, Mask Ratio가 낮은 구간에서 보수 제어로 전환.",
                "validation": "|abs| p95 감소 + 실패율 악화 없음(또는 감소).",
                "confidence": "low",
                "is_speculative": True,
            })

        # 4) Target CX 결측
        tcx_miss = metrics.get("target_cx_missing_rate")
        if tcx_miss is not None and tcx_miss >= 4:
            recos.append({
                "priority": 4,
                "title": "Target CX 결측 해소(계산/기록 경로 점검)",
                "why": f"Target CX 결측률이 존재(target_cx_missing_rate={_fmt_pct(tcx_miss)}).",
                "action": "Target CX가 계산되는 조건(검출 실패 시 기본값/결측 처리)을 고정하고, 결측을 0/None으로 구분 기록.",
                "validation": "결측률 감소 후, 실패/정상 간 Target CX 분포 차이를 재평가.",
                "confidence": "high",
                "is_speculative": True,
            })

        # 5) 오차/처리시간 꼬리(고정 컬럼) 기반 일반 권고
        if metrics.get("abs_tail_rate") is not None and metrics.get("abs_tail_rate") >= 5:
            recos.append({
                "priority": 5,
                "title": "오차 꼬리 구간 샘플링 검토(영상 확인 우선)",
                "why": f"오차 꼬리 비중이 존재(abs_tail_rate={_fmt_pct(metrics.get('abs_tail_rate'))}, abs_tail_th={_fmt_num(metrics.get('abs_tail_th'),2)}).",
                "action": "오차 꼬리 상위 프레임을 추출해(이벤트ID/타임스탬프) 영상으로 원인을 분류합니다(차선 미검출/오검출/곡률/조명 등).",
                "validation": "원인 카테고리별로 재현 조건이 정리되고, 특정 카테고리의 빈도가 감소.",
                "confidence": "medium",
                "is_speculative": True,
            })

        if metrics.get("proc_tail_rate") is not None and metrics.get("proc_tail_rate") >= 5:
            recos.append({
                "priority": 6,
                "title": "처리시간 꼬리 최적화(병목 구간 분해)",
                "why": f"처리시간 꼬리 비중이 존재(proc_tail_rate={_fmt_pct(metrics.get('proc_tail_rate'))}, proc_tail_th={_fmt_num(metrics.get('proc_tail_th'),1)}ms).",
                "action": "고처리시간 프레임 구간에서 전처리/추론/후처리 시간 로그를 분리해 병목을 식별.",
                "validation": "proc_p95/꼬리 비중 감소 + 실시간성 목표 달성.",
                "confidence": "medium",
                "is_speculative": True,
            })

        # 최소 4개는 유지(빈 경우 대비)
        if len(recos) < 4:
            recos += [
                {
                    "priority": 90,
                    "title": "로그 품질 점검(기본)",
                    "why": "단일 로그에서 관측된 분포만으로는 원인 확정이 어려운 상태.",
                    "action": "실패 프레임 주변의 원본 영상/센서 상태를 확보하고, 동일 지표를 다른 날짜에서도 수집.",
                    "validation": "재현 가능한 실패 조건이 정의되고, 후속 로그에서 동일 패턴 여부 확인.",
                    "confidence": "low",
                    "is_speculative": True,
                }
            ]

        # priority 정렬 및 상위 8개 제한
        recos = sorted(recos, key=lambda x: int(x.get("priority", 999)))[:8]

        notes = (
            "본 Part X는 OpenAI 호출 없이 규칙 기반으로 작성됩니다. "
            "표기된 '추측' 항목은 영상/현장 확인을 통해 검증하세요."
        )

        return {
            "overview": overview,
            "assumptions": assumptions,
            "recommendations": recos,
            "notes": notes,
        }

    def _render_recos(recos: dict) -> str:
        """권고안을 화면에 렌더링하고, 마크다운 텍스트를 반환"""
        md_lines: list[str] = []
        md_lines.append("## Part X: 자율주행 개선사항(수동 작성)")
        md_lines.append("")
        md_lines.append("### 요약")
        md_lines.append(recos.get("overview", ""))
        md_lines.append("")
        md_lines.append("### 가정/전제")
        for a in recos.get("assumptions", []) or []:
            md_lines.append(f"- {a}")
        md_lines.append("")
        md_lines.append("### 권장사항")
        for r in recos.get("recommendations", []) or []:
            md_lines.append(f"{int(r.get('priority', 0))}. **{r.get('title','')}**")
            md_lines.append(f"   - 원인: {r.get('why','')}")
            md_lines.append(f"   - 개선: {r.get('action','')}")
            md_lines.append(f"   - 검증: {r.get('validation','')}")
            md_lines.append("")
        if recos.get("notes"):
            md_lines.append("### Notes")
            md_lines.append(recos["notes"])
            md_lines.append("")

        md = "\n".join(md_lines).strip() + "\n"

        st.markdown("### 개선사항 (수동 작성 결과)")
        st.markdown(md)
        return md

    # ---- 실행: metrics → 분포 요약 → 수동 권고안
    metrics = _compute_partx_metrics(df_fixed=df_fixed, df_full=df_full, fail_mask=fail_mask, pctl=pctl, cand=cand)
    _render_partx_distribution(metrics)

    recos = _manual_recos_from_metrics(metrics)
    report_md = _render_recos(recos)

