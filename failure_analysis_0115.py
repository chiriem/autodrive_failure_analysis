"""0115_12_log.csv 전용 분석 페이지

- 단일 로그 분석(비교 없음)
- 실패 정의: Failure Flag == 1
- 신규(옵션) 컬럼은 "실패 프레임 테이블"에서만 노출
- Part III(환경별 비교), Part X(AI 개선방안) 제외

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

COLUMN_CONFIG = {
    TS_COL: st.column_config.NumberColumn(format="%.0f"),
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

    # 신규(옵션) 컬럼 — 실패 프레임 테이블에서만 노출되지만 포맷은 여기서 정의
    FAIL_FLAG_COL: st.column_config.NumberColumn(format="%.0f"),
    FAIL_TYPE_COL: st.column_config.TextColumn(),
    LANE_STATE_COL: st.column_config.TextColumn(),
    TARGET_CX_COL: st.column_config.NumberColumn(format="%.2f"),
    CONTOURS_TOTAL_COL: st.column_config.NumberColumn(format="%.0f"),
    CONTOURS_KEPT_COL: st.column_config.NumberColumn(format="%.0f"),
    DIR_FORCE_COL: st.column_config.NumberColumn(format="%.2f"),
}

def _column_config_for(df_or_cols) -> dict:
    """COLUMN_CONFIG를 실제로 존재하는 컬럼만 필터링 (선택적 컬럼 누락 시 오류 방지)"""
    cols = df_or_cols.columns if hasattr(df_or_cols, "columns") else list(df_or_cols)
    return {k: v for k, v in COLUMN_CONFIG.items() if k in cols}

def render() -> None:
    """0115_12_log.csv 단일 로그 분석(비교 없음).
    - 실패 정의: Failure Flag == 1
    - 신규 컬럼(옵션)은 '실패 프레임 테이블'에서만 노출
    - Part III(환경별 비교), Part X(AI 개선방안) 미포함
    """
    st.title("자율주행 실패 분석 (0115 고정 데이터)")
    st.caption("단일 로그 분석 / 실패 정의: Flag==1 / 신규 컬럼은 '실패 프레임 테이블'에서만 표시")

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
