import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from pathlib import Path
import re
import json

# -----------------------------------------------------------------------------
# Optional drive-log column names (used when present).
# Keep as constants to avoid typos and enable consistent metrics/LLM prompts.
FAIL_FLAG_COL = "Failure Flag"
FAIL_TYPE_COL = "Failure Type"
FAIL_FLAG_FIXED_COL = "Failure Flag (fixed)"
LANE_STATE_COL = "Lane State"
TARGET_CX_COL = "Target CX"
CONTOURS_TOTAL_COL = "Contours Total"
CONTOURS_KEPT_COL = "Contours Kept"
DIR_FORCE_COL = "Direction Force"
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Helpers: ensure unique column names for safe display (avoids Streamlit/Altair errors)
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
# -----------------------------------------------------------------------------

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

st.set_page_config(page_title="로그분석", page_icon="🛣️", layout="wide")


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


# =============================================================================
# UI

analysis_mode = st.sidebar.radio(
    "분석 모드",
    ["파일 업로드 (기본)", "0109 고정 데이터 분석", "0112 고정 데이터 분석", "0113 고정 데이터 분석", "0114 고정 데이터 분석", "0115 고정 데이터 분석"],
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

if analysis_mode == "0113 고정 데이터 분석":
    # 0113 고정 데이터 분석은 별도 모듈(failure_analysis_0113.py)로 위임합니다.
    import failure_analysis_0113 as page_0113
    page_0113.render()
    st.stop()

if analysis_mode == "0114 고정 데이터 분석":
    # 0114 고정 데이터 분석은 별도 모듈(failure_analysis_0114.py)로 위임합니다.
    import failure_analysis_0114 as page_0114
    page_0114.render()
    st.stop()    

if analysis_mode == "0115 고정 데이터 분석":
    # 0115 고정 데이터 분석은 별도 모듈(failure_analysis_0115.py)로 위임합니다.
    import failure_analysis_0115 as page_0115
    page_0115.render()
    st.stop()   

else:
    st.title("자율주행 로그 분석")
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

    st.subheader("이 페이지는 https://github.com/chiriem/autodrive_failure_analysis/tree/main/data 의 0115_12_log.csv를 업로드하는 것을 기준으로 제작되었습니다.")

    def _generate_demo(n: int = 1200) -> pd.DataFrame:
        np.random.seed(7)
    
        weather = np.random.choice(["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy"], n)
        tod = np.random.choice(["Day", "Night"], n, p=[0.75, 0.25])
    
        # Mask White Ratio: 0~1
        base_ratio = np.random.beta(3, 25, n)
        base_ratio[(weather == "Foggy") | (weather == "Snowy")] *= 0.7
        base_ratio[tod == "Night"] *= 0.8
        mask_ratio = np.clip(base_ratio + np.random.normal(0, 0.01, n), 0, 1)
    
        # Lane Quality Score: 0~100
        quality = np.clip((mask_ratio * 180) + np.random.normal(0, 8, n), 0, 100)
        fp = np.random.rand(n) < 0.03
        quality[fp] = np.clip(quality[fp] - 50, 0, 100)
    
        abs_err = np.clip((100 - quality) * 0.9 + np.random.normal(0, 6, n), 0, None)
        err = abs_err * np.random.choice([-1, 1], n)
    
        proc = np.random.normal(28, 5, n)
        mode = np.random.choice(["AUTO", "MANUAL"], n, p=[0.9, 0.1])
    
        # Failure Flag: about 10% are failures
        n_fail = max(1, int(round(n * 0.10)))
        fail_idx = np.random.choice(np.arange(n), size=n_fail, replace=False)
        fail_flag = np.zeros(n, dtype=int)
        fail_flag[fail_idx] = 1
    
        # 실패 케이스에만 생성되는 값들
        lane_state = np.array([np.nan] * n, dtype=object)
        failure_type = np.array([np.nan] * n, dtype=object)
        target_cx = np.full(n, np.nan, dtype=float)
        contours_total = np.full(n, np.nan, dtype=float)
        contours_kept = np.full(n, np.nan, dtype=float)
        dir_force = np.full(n, np.nan, dtype=float)
    
        lane_state[fail_idx] = "none"
        failure_type[fail_idx] = np.random.choice(["mask_empty", "area_filtered", "lane_lost"], size=n_fail)

        _ct_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28]
        _ct_counts = [1, 3, 5, 2, 4, 2, 2, 2, 4, 1, 4, 3, 2, 2, 2, 6, 1, 8, 3, 7, 1, 8, 4, 8, 2]
        _ct_p = np.array(_ct_counts, dtype=float)
        _ct_p = _ct_p / _ct_p.sum()
        contours_total[fail_idx] = np.random.choice(_ct_vals, size=n_fail, p=_ct_p).astype(float)
    
        contours_kept[fail_idx] = 0.0
        dir_force[fail_idx] = 0.0
    
        target_cx[fail_idx] = np.clip(np.random.normal(160, 30, n_fail), 0, 320)

        err[fail_idx] = np.nan
    
        df = pd.DataFrame({
            TS_COL: np.arange(n) * 100,  # ms
            WEATHER_COL: weather,
            TOD_COL: tod,
            MASK_RATIO_COL: mask_ratio,
            QUALITY_COL: quality,
            ERROR_COL: err,
            PROC_COL: proc,
            MODE_COL: mode,
            FAIL_FLAG_COL: fail_flag,
            FAIL_TYPE_COL: failure_type,
            LANE_STATE_COL: lane_state,
            TARGET_CX_COL: target_cx,
            CONTOURS_TOTAL_COL: contours_total,
            CONTOURS_KEPT_COL: contours_kept,
            DIR_FORCE_COL: dir_force,
        })
    
        df[ABS_ERROR_COL] = pd.to_numeric(df[ERROR_COL], errors="coerce").abs()
        return df
    if uploaded_files:
        dfs = []
        for f in uploaded_files:
            try:
                d = pd.read_csv(f)
                _ff = d['Failure Flag'].to_numpy(copy=True) if 'Failure Flag' in d.columns else None
                d = _select_fixed_columns(d)
                if _ff is not None and 'Failure Flag' not in d.columns:
                    if len(d) == len(_ff):
                        d['Failure Flag'] = _ff
                    else:
                        d['Failure Flag'] = pd.Series(_ff).reindex(d.index).to_numpy()
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


# Abs Lane Error는 Lane Error의 절대값으로 정의 (부호는 무의미)
df[ABS_ERROR_COL] = pd.to_numeric(df[ERROR_COL], errors="coerce").abs()

# 필수 컬럼 검증
missing = [c for c in FIXED_LOG_COLS if c not in df.columns]
if missing:
    st.error(
        "필수 컬럼이 없습니다. (이 버전은 고정 컬럼 스키마를 사용합니다.)\n\n"
        f"- 필수: {', '.join(FIXED_LOG_COLS)}\n"
        f"- 누락: {', '.join(missing)}\n\n"
        "※ Mask White Ratio는 0~1 또는 0~100(%) 모두 허용되며 자동 정규화됩니다."
    )
    st.stop()

if TS_COL not in df.columns:
    st.warning("Timestamp 컬럼이 없습니다. 이벤트 식별은 Event ID로 가능하지만, 시간 기반 해석(구간/추세)은 제한될 수 있습니다.")

# 컬럼 설정
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
    """COLUMN_CONFIG를 실제로 존재하는 컬럼만 필터링 (선택적 컬럼 누락 시 오류 방지)"""
    cols = df_or_cols.columns if hasattr(df_or_cols, "columns") else list(df_or_cols)
    return {k: v for k, v in COLUMN_CONFIG.items() if k in cols}

# =============================================================================
# Part 0: 컬럼/결측/커버리지 확인

st.divider()
st.subheader("Part 0: 컬럼/결측/커버리지 확인")
check_cols = [TS_COL, WEATHER_COL, TOD_COL, MASK_RATIO_COL, ERROR_COL, PROC_COL, QUALITY_COL]
_st_dataframe(_describe_missing(df, check_cols), hide_index=True, use_container_width=True)

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


st.markdown("### Lane Error 결측 패턴 분석")

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
        # 결측 여부별 마스크 비율 분포 비교
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


            # 결측 vs 존재에 대한 빠른 분위수
            q = (
                tmp.groupby("Error Recorded")[MASK_RATIO_COL]
                .quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
                .unstack()
                .reset_index()
            )
            q.columns = ["Error Recorded", "p01", "p05", "p25", "p50", "p75", "p95", "p99"]
            _st_dataframe(q, hide_index=True, use_container_width=True)
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

        # 임계값 강조 (이전 분석 스타일과 일치)
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
        _st_dataframe(bin_summary, hide_index=True, use_container_width=True)

        # 강조: low_th 이하의 결측률
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
            _st_dataframe(ge.sort_values("missing_%", ascending=False), hide_index=True, use_container_width=True)



# =============================================================================
# Part I: 실패 프레임 테이블 (Failure Flag == 1)

st.divider()
st.markdown("## Part I — 실패 프레임 테이블")

if FAIL_FLAG_COL not in df.columns:
    st.warning(f"'{FAIL_FLAG_COL}' 컬럼이 없어 실패 프레임 테이블을 생성할 수 없습니다.")
else:
    # Failure 정의: Failure Flag == 1
    fail_s = pd.to_numeric(df[FAIL_FLAG_COL], errors="coerce").fillna(0)
    fail_mask = fail_s.eq(1)

    n_total = int(len(df))
    n_fail = int(fail_mask.sum())
    pct = (n_fail / n_total * 100.0) if n_total > 0 else 0.0

    st.caption(f"실패 프레임: {n_fail} / {n_total} ({pct:.2f}%)")

    fail_df = df.loc[fail_mask].copy()

    # 기본 컬럼(다른 파트에서도 흔히 쓰는 컬럼) + 나머지(신규/추가 컬럼)는 이 테이블에서만 노출
    base_cols = [
        TS_COL,
        RUN_ID_COL, ROW_IN_RUN_COL, EVENT_ID_COL,
        FAIL_FLAG_COL, FAIL_TYPE_COL,
        WEATHER_COL, TOD_COL, MODE_COL,
        QUALITY_COL,
        MASK_RATIO_COL, ERROR_COL, ABS_ERROR_COL, PROC_COL,
    ]
    base_cols_present = [c for c in base_cols if c in fail_df.columns]
    extra_cols = [c for c in fail_df.columns if c not in base_cols_present]
    show_cols = base_cols_present + extra_cols

    # 정렬(가능하면 Timestamp 기준)
    if TS_COL in fail_df.columns:
        try:
            fail_df = fail_df.sort_values(TS_COL, ascending=True)
        except Exception:
            # 정렬 실패 시 원본 순서 유지
            pass

    if fail_df.empty:
        st.info("Failure Flag == 1 프레임이 없습니다.")
    else:
        _st_dataframe(
            fail_df[show_cols],
            column_config=_column_config_for(show_cols),
            hide_index=True,
            use_container_width=True,
            height=420,
        )


# =============================================================================
# Part II: 실패 요약 카드 (Failure Flag == 1)

st.divider()
st.markdown("## Part II — 실패 요약 카드")
st.caption("Failure Flag == 1 프레임만 대상으로, 핵심 분포/빈도 지표를 빠르게 요약합니다.")

if FAIL_FLAG_COL not in df.columns:
    st.warning(f"'{FAIL_FLAG_COL}' 컬럼이 없어 실패 요약 카드를 생성할 수 없습니다.")
else:
    # Failure 정의: Failure Flag == 1
    _fail_s = pd.to_numeric(df[FAIL_FLAG_COL], errors="coerce").fillna(0)
    _fail_mask = _fail_s.eq(1)

    _n_total = int(len(df))
    _n_fail = int(_fail_mask.sum())
    _fail_rate = (_n_fail / _n_total * 100.0) if _n_total > 0 else 0.0

    if _n_fail == 0:
        st.info("Failure Flag == 1 프레임이 없습니다.")
    else:
        _fail_df = df.loc[_fail_mask].copy()

        def _num_series(col: str) -> "pd.Series | None":
            if col not in _fail_df.columns:
                return None
            s = pd.to_numeric(_fail_df[col], errors="coerce").dropna()
            return s if len(s) else None

        def _top_value(col: str) -> tuple[str, int] | None:
            if col not in _fail_df.columns:
                return None
            vc = _fail_df[col].astype(str).fillna("nan").value_counts()
            if vc.empty:
                return None
            return (str(vc.index[0]), int(vc.iloc[0]))

        # --- 핵심 분포 지표(실패 프레임 기준)
        abs_s = _num_series(ABS_ERROR_COL)
        ratio_s = _num_series(MASK_RATIO_COL)
        proc_s = _num_series(PROC_COL)

        abs_med = float(abs_s.median()) if abs_s is not None else None
        abs_p95 = float(abs_s.quantile(0.95)) if abs_s is not None else None

        ratio_mean = float(ratio_s.mean()) if ratio_s is not None else None

        proc_mean = float(proc_s.mean()) if proc_s is not None else None

        # Failure 프레임에서 Lane Error(원값) 기록 여부(결측) 요약
        _err_raw = pd.to_numeric(_fail_df.get(ERROR_COL), errors="coerce") if ERROR_COL in _fail_df.columns else pd.Series(dtype=float)
        _err_present_n = int(_err_raw.notna().sum()) if len(_fail_df) else 0
        _err_missing_rate = float(_err_raw.isna().mean() * 100.0) if len(_fail_df) else None

        ft_top = _top_value(FAIL_TYPE_COL)
        # --- 1행 카드
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("실패 프레임 수", f"{_n_fail:,}")
        with c2:
            st.metric("실패율", f"{_fail_rate:.2f}%")
        with c3:
            if ft_top is None:
                st.metric(f"최다 {FAIL_TYPE_COL}", "N/A")
            else:
                st.metric(f"최다 {FAIL_TYPE_COL}", ft_top[0], delta=f"n={ft_top[1]:,}")

        # --- 2행 카드
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            if abs_med is None:
                st.metric(f"{ERROR_COL} 기록 프레임", f"{_err_present_n:,}")
            else:
                st.metric(f"{ABS_ERROR_COL} 중앙값", f"{abs_med:.2f}")
        with c6:
            if abs_p95 is None:
                st.metric(f"{ERROR_COL} 결측률", "N/A" if _err_missing_rate is None else f"{_err_missing_rate:.1f}%")
            else:
                st.metric(f"{ABS_ERROR_COL} p95", f"{abs_p95:.2f}")
        with c7:
            st.metric(f"{MASK_RATIO_COL} 평균", "N/A" if ratio_mean is None else f"{ratio_mean:.4f}")
        with c8:
            st.metric(f"{PROC_COL} 평균", "N/A" if proc_mean is None else f"{proc_mean:.1f} ms")

        if abs_s is None:
            st.caption(f"참고: 현재 실패 프레임에서 '{ERROR_COL}' 값이 비어 있어 '{ABS_ERROR_COL}' 분위수를 계산할 수 없습니다. 대신 '{ERROR_COL}' 결측 정보를 표시합니다.")

        # --- 상위 분포(빈도) 표(선택)
        colA, colB = st.columns(2)

        with colA:
            if FAIL_TYPE_COL not in _fail_df.columns:
                st.info(f"'{FAIL_TYPE_COL}' 컬럼이 없어 실패 타입 빈도 표를 만들 수 없습니다.")
            else:
                st.markdown(f"### {FAIL_TYPE_COL} 상위")

                # Failure Type 상위 8개
                _ft_s = _fail_df[FAIL_TYPE_COL].astype(str).fillna("nan")
                vc = _ft_s.value_counts().head(8)

                t = pd.DataFrame({
                    FAIL_TYPE_COL: vc.index,
                    "count": vc.values,
                    "rate(%)": (vc.values / _n_fail * 100.0).round(2),
                })

                # Failure Type별 Processing Time 평균(ms) 추가
                if PROC_COL in _fail_df.columns:
                    _pt_s = pd.to_numeric(_fail_df[PROC_COL], errors="coerce")
                    _pt_mean_by_ft = (
                        pd.DataFrame({FAIL_TYPE_COL: _ft_s, "_pt": _pt_s})
                        .groupby(FAIL_TYPE_COL, dropna=False)["_pt"]
                        .mean()
                    )
                    t["Processing Time 평균(ms)"] = t[FAIL_TYPE_COL].map(_pt_mean_by_ft).round(1)
                else:
                    t["Processing Time 평균(ms)"] = np.nan

                _st_dataframe(t, use_container_width=True, hide_index=True)
        # --- Processing Time 평균 비교(정상 vs 실패) - table
        if PROC_COL in df.columns:
            _proc_all = pd.to_numeric(df[PROC_COL], errors="coerce")

            _case_rows = []
            for _case_name, _mask in [("정상", ~_fail_mask), ("실패", _fail_mask)]:
                _n_case = int(_mask.sum())
                _s = _proc_all.loc[_mask]
                _n_valid = int(_s.notna().sum())
                _mean_ms = float(_s.mean()) if _n_valid > 0 else np.nan
                _missing_rate = float((1 - (_n_valid / _n_case)) * 100.0) if _n_case > 0 else np.nan

                _case_rows.append({
                    "case": _case_name,
                    "프레임 수": _n_case,
                    "Processing Time 평균": (round(_mean_ms, 1) if _n_valid > 0 else np.nan),
                })

            st.markdown("### Processing Time 평균 비교")
            _cmp_df = pd.DataFrame(_case_rows)
            _st_dataframe(_cmp_df, use_container_width=True, hide_index=True)
            st.caption("Processing Time의 평균을 정상/실패 프레임으로 비교합니다.")
        else:
            st.info(f"'{PROC_COL}' 컬럼이 없어 Processing Time 평균 비교 테이블을 만들 수 없습니다.")




# =============================================================================
# Reference (collapsed): deviation regression (legacy Part I & II)


def _render_reference_deviation_regression(df: pd.DataFrame) -> None:
    # =============================================================================
    # Part I: Mask White Ratio ↔ Abs Lane Error

    st.divider()
    st.markdown(f"""
    ## Reference I: {MASK_RATIO_COL} ↔ {ABS_ERROR_COL}

    - **Mask White Ratio(0~1)**: 마스크에서 흰 픽셀(0이 아닌 픽셀)이 차지하는 비율(= 검출량/가시성 신호)
    - **Abs Lane Error**: 화면 중앙 대비 목표 지점의 오차 크기(픽셀)
    - 둘 사이의 회귀를 진행. 초록색 +로 표기된 데이터는 이상값으로 추정

    가능한 판단:
    - **ratio↓ & error↑**: 검출량이 부족한 구간에서 조향각이 커지는 패턴
    - **ratio↑ & error↑**:흰 선이 지나치게 잘 잡히면서 조향각도 큰 패턴(빛 반사/노이즈 영향 가능)
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

        st.caption("※ Outlier는 회귀 기준(2σ)으로 회귀선 대비 크게 벗어난 프레임입니다.")

        a, b = st.columns(2)
        with a:
            st.caption("High-error frames (Abs Error 상위 20)")
            show_cols = [TS_COL, WEATHER_COL, TOD_COL, MASK_RATIO_COL, ABS_ERROR_COL]
            if PROC_COL in df.columns:
                show_cols.append(PROC_COL)
            _st_dataframe(
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
            _st_dataframe(
                df.sort_values(MASK_RATIO_COL)[show_cols].head(20),
                column_config=_column_config_for(show_cols),
                height=360,
            )

    # =============================================================================
    # Part II: Mask White Ratio ↔ Processing Time (ms)

    st.divider()
    st.markdown(f"""
    ## Reference II: {MASK_RATIO_COL} ↔ {PROC_COL}

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
            _st_dataframe(
                df.dropna(subset=[PROC_COL]).sort_values(PROC_COL, ascending=False)[show_cols].head(20),
                column_config=_column_config_for(show_cols),
                height=360,
            )

        with b:
            st.caption("Low processing-time frames (하위 20)")
            show_cols = [TS_COL, WEATHER_COL, TOD_COL, MASK_RATIO_COL, PROC_COL]
            if ABS_ERROR_COL in df.columns:
                show_cols.append(ABS_ERROR_COL)
            _st_dataframe(
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

    # Part III가 의미 있는지 결정 (어색한 'Unknown만' 차트 방지)
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
        
        # 사용자가 Unknown을 포함하고 그것이 지배적이더라도 유지하되, 유일한 그룹이면 경고
        if len(summary) <= 1:
            st.info("선택한 그룹 기준에서 비교 가능한 범주가 1개뿐입니다(대부분 Unknown일 수 있음).")
        _bar_chart(summary, group_col, metric_label, domain=domain)
        
        st.caption("표는 median/mean과 IQR(p25~p75)을 함께 제공합니다. 프레임 수가 적은 그룹은 해석에 주의하세요.")
        _st_dataframe(summary, hide_index=True, use_container_width=True)

# =============================================================================
# Part IV: Outlier Candidates (rule-first)

st.divider()
st.markdown("""
## Part IV: 이상치 후보(자동 기준 + 민감도 1개)

이 파트는 **확인 우선순위를 정하기 위한 후보 추출**입니다.

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

    # 숫자 컬럼을 안전하게 정규화
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

    # 후보 태그
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

    # 주요 태그 (우선순위)
    d["Primary Tag"] = "Normal"
    d.loc[d["mask_low"], "Primary Tag"] = "마스크 비율 매우 낮음"
    d.loc[(d["Primary Tag"] == "Normal") & d["mask_high"], "Primary Tag"] = "마스크 비율 매우 높음"
    d.loc[(d["Primary Tag"] == "Normal") & d["err_high"], "Primary Tag"] = "오차 과다"
    d.loc[(d["Primary Tag"] == "Normal") & d["proc_high"], "Primary Tag"] = "처리시간 과다"
    d.loc[(d["Primary Tag"] == "Normal") & d["err_missing"], "Primary Tag"] = "오차 기록 누락"

    cand = d[d["Candidate Tags"].astype(str).str.len() > 0].copy()

    # 임계값 표시
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
    _st_dataframe(summary, hide_index=True, use_container_width=True, height=220)

    # 차트 (고정 축, 추가 컨트롤 없음)
    tabs = []
    tabs.append("Mask Ratio ↔ Abs Error" if ABS_ERROR_COL in cand.columns else "Mask Ratio ↔ Proc Time")
    if (ABS_ERROR_COL in cand.columns) and (PROC_COL in cand.columns):
        tabs.append("Mask Ratio ↔ Proc Time")

    t = st.tabs(tabs)

    # 샘플링을 위한 헬퍼
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

top = df.sort_values(MASK_RATIO_COL).head(20).copy()

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
_st_dataframe(top[show], column_config=_column_config_for(show), height=360)

# Part VI: Timestamp 기반 이상치(후보) 구간
st.markdown("## Part VI: Timestamp 기반 이상치 구간")
st.caption("이상치가 '몰리는 시간 구간'을 찾습니다.")
_render_timestamp_outlier_windows_from_flags(df_flags=d if 'd' in locals() else df, pctl=pctl if 'pctl' in locals() else None)

# Part VII: 전체 로그 보기
st.divider()
st.markdown("## Part VII: 전체 로그 보기")
drop_cols = [RUN_ID_COL, ROW_IN_RUN_COL, EVENT_ID_COL]
_st_dataframe(df.drop(columns=[c for c in drop_cols if c in df.columns]))

# =============================================================================
# Part X: 분석 요약 및 개선점

# 분포 기반 요약 지표를 한번에 계산해 metrics 딕셔너리로 반환
def _compute_partx_metrics(df: pd.DataFrame, pctl: int, cand: "pd.DataFrame | None" = None) -> dict:
    """Part X의 요약 메트릭 계산. UI 및 LLM 프롬프팅에 사용되는 딕셔너리 반환"""
    metrics: dict = {}
    n_total = int(len(df))
    metrics["n_total"] = n_total

    # 마스크 비율
    ratio = pd.to_numeric(df.get(MASK_RATIO_COL), errors="coerce").clip(0, 1)
    ratio_valid = ratio.dropna()
    metrics["ratio_valid_n"] = int(ratio_valid.size)

    low_th = high_th = low_rate = high_rate = None
    ratio_q = {}
    if not ratio_valid.empty:
        low_th = float(ratio_valid.quantile((100 - pctl) / 100.0))
        high_th = float(ratio_valid.quantile(pctl / 100.0))
        low_rate = float((ratio_valid <= low_th).mean() * 100.0)
        high_rate = float((ratio_valid >= high_th).mean() * 100.0)
        for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            ratio_q[f"p{int(q*100):02d}"] = float(ratio_valid.quantile(q))
    metrics.update({
        "ratio_low_th": low_th,
        "ratio_high_th": high_th,
        "ratio_low_rate": low_rate,
        "ratio_high_rate": high_rate,
        "ratio_quantiles": ratio_q,
    })

    # LLM/보고서용 scalar 분위수
    metrics["ratio_p05"] = ratio_q.get("p05")
    metrics["ratio_p50"] = ratio_q.get("p50")
    metrics["ratio_p95"] = ratio_q.get("p95")
    metrics["ratio_p99"] = ratio_q.get("p99")

    # 오차
    err_missing_rate = abs_p95 = abs_p99 = abs_tail_th = abs_tail_rate = None
    if ERROR_COL in df.columns:
        err = pd.to_numeric(df.get(ERROR_COL), errors="coerce")
        err_missing_rate = float(err.isna().mean() * 100.0)
        ae = err.abs().dropna()
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

    # 처리 시간
    proc_p95 = proc_p99 = proc_max = proc_tail_th = proc_tail_rate = None
    if PROC_COL in df.columns:
        pr = pd.to_numeric(df.get(PROC_COL), errors="coerce").dropna()
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

    # 후보 태그 분포 (사용 가능한 경우)
    tag_counts = None
    if cand is not None and isinstance(cand, pd.DataFrame) and len(cand) > 0 and "Primary Tag" in cand.columns:
        vc = cand["Primary Tag"].astype(str).value_counts()
        tag_counts = {k: int(v) for k, v in vc.items()}
    metrics["candidate_tag_counts"] = tag_counts
    metrics["candidate_n"] = int(len(cand)) if isinstance(cand, pd.DataFrame) else None

    # ---------------------------------------------------------------------
    # Failure-related metrics (Failure Flag == 1 only)
    if FAIL_FLAG_COL in df.columns:
        raw_flag = pd.to_numeric(df[FAIL_FLAG_COL], errors="coerce").fillna(0).astype(int)
        fail_mask = (raw_flag == 1)
        failure_n = int(fail_mask.sum())

        metrics["failure_definition"] = "Failure Flag==1"
        metrics["failure_n"] = failure_n
        metrics["failure_rate"] = float(fail_mask.mean() * 100.0) if n_total else None

        if failure_n > 0:
            # Failure Type / Lane State distributions (fail frames only)
            if FAIL_TYPE_COL in df.columns:
                ft = df.loc[fail_mask, FAIL_TYPE_COL].astype(str).fillna("NA")
                vc = ft.value_counts().head(8)
                metrics["failure_type_counts_top"] = {k: int(v) for k, v in vc.items()}
                metrics["failure_type_rates_top"] = {k: float(v / failure_n * 100.0) for k, v in vc.items()}
            else:
                metrics["failure_type_counts_top"] = None
                metrics["failure_type_rates_top"] = None

            if "Lane State" in df.columns:
                stv = df.loc[fail_mask, "Lane State"].astype(str).fillna("NA")
                vc = stv.value_counts().head(6)
                metrics["lane_state_counts_top"] = {k: int(v) for k, v in vc.items()}
                metrics["lane_state_rates_top"] = {k: float(v / failure_n * 100.0) for k, v in vc.items()}
            else:
                metrics["lane_state_counts_top"] = None
                metrics["lane_state_rates_top"] = None

            # Contours: zero-rate signals (fail frames only)
            if "Contours Total" in df.columns:
                ctot = pd.to_numeric(df.loc[fail_mask, "Contours Total"], errors="coerce").dropna()
                metrics["failure_contours_total_zero_rate"] = float((ctot == 0).mean() * 100.0) if not ctot.empty else None
            else:
                metrics["failure_contours_total_zero_rate"] = None

            if "Contours Kept" in df.columns:
                ckept = pd.to_numeric(df.loc[fail_mask, "Contours Kept"], errors="coerce").dropna()
                metrics["failure_contours_kept_zero_rate"] = float((ckept == 0).mean() * 100.0) if not ckept.empty else None
            else:
                metrics["failure_contours_kept_zero_rate"] = None

            # Direction Force (abs p95) (fail frames only)
            if "Direction Force" in df.columns:
                dfc = pd.to_numeric(df.loc[fail_mask, "Direction Force"], errors="coerce").dropna().abs()
                metrics["failure_direction_force_abs_p95"] = float(dfc.quantile(0.95)) if not dfc.empty else None
            else:
                metrics["failure_direction_force_abs_p95"] = None

            # Target CX quantiles (fail frames only)
            if "Target CX" in df.columns:
                tcx = pd.to_numeric(df.loc[fail_mask, "Target CX"], errors="coerce").dropna()
                metrics["failure_target_cx_p05"] = float(tcx.quantile(0.05)) if not tcx.empty else None
                metrics["failure_target_cx_p95"] = float(tcx.quantile(0.95)) if not tcx.empty else None
            else:
                metrics["failure_target_cx_p05"] = None
                metrics["failure_target_cx_p95"] = None
        else:
            metrics["failure_type_counts_top"] = None
            metrics["failure_type_rates_top"] = None
            metrics["lane_state_counts_top"] = None
            metrics["lane_state_rates_top"] = None
            metrics["failure_contours_total_zero_rate"] = None
            metrics["failure_contours_kept_zero_rate"] = None
            metrics["failure_direction_force_abs_p95"] = None
            metrics["failure_target_cx_p05"] = None
            metrics["failure_target_cx_p95"] = None
    else:
        metrics["failure_definition"] = "Failure Flag column missing"
        metrics["failure_n"] = None
        metrics["failure_rate"] = None
        metrics["failure_type_counts_top"] = None
        metrics["failure_type_rates_top"] = None
        metrics["lane_state_counts_top"] = None
        metrics["lane_state_rates_top"] = None
        metrics["failure_contours_total_zero_rate"] = None
        metrics["failure_contours_kept_zero_rate"] = None
        metrics["failure_direction_force_abs_p95"] = None
        metrics["failure_target_cx_p05"] = None
        metrics["failure_target_cx_p95"] = None
    # ---------------------------------------------------------------------


    return metrics


def _render_part_x_distribution(df: pd.DataFrame, pctl: int, cand: "pd.DataFrame | None" = None) -> dict:
    """Part X의 분포 요약 UI 렌더링. 계산된 메트릭 딕셔너리 반환"""
    st.divider()
    st.markdown("## Part X: 분석 요약 및 개선점 (분포 기반)")

    metrics = _compute_partx_metrics(df, pctl, cand=cand)
    n_total = metrics["n_total"]

    # 요약 테이블
    st.markdown("### 분포 요약(핵심 지표)")
    rows = [{"항목": "총 프레임 수", "값": f"{n_total:,}", "의미": "분석 대상 전체 행 수"}]

    low_th = metrics["ratio_low_th"]
    high_th = metrics["ratio_high_th"]
    low_rate = metrics["ratio_low_rate"]
    high_rate = metrics["ratio_high_rate"]

    if low_th is not None:
        rows.append({"항목": f"Mask Ratio 하한(하위 {100 - pctl}% 분위)", "값": f"{low_th:.4f}", "의미": "차선 픽셀량이 매우 낮은 꼬리 구간 기준"})
        rows.append({"항목": "Mask Ratio 하한 이하 비율", "값": f"{low_rate:.2f}%", "의미": "저가시성/미검출 후보 비중"})
        rows.append({"항목": f"Mask Ratio 상한(상위 {100 - pctl}% 분위)", "값": f"{high_th:.4f}", "의미": "차선 픽셀량이 매우 높은 꼬리 구간 기준"})
        rows.append({"항목": "Mask Ratio 상한 이상 비율", "값": f"{high_rate:.2f}%", "의미": "과검출 후보 비중"})

    if metrics["err_missing_rate"] is not None:
        rows.append({"항목": "Lane Error 결측률", "값": f"{metrics['err_missing_rate']:.2f}%", "의미": "오차 기반 분석/모니터링이 불가한 구간 비중"})
    if metrics["abs_p95"] is not None:
        rows.append({"항목": "Abs Error p95 / p99", "값": f"{metrics['abs_p95']:.2f} / {metrics['abs_p99']:.2f}", "의미": "오차 분포의 상위 꼬리 크기(픽셀 단위)"})
        rows.append({"항목": f"Abs Error 상위 {100 - pctl}% 기준", "값": f"≥ {metrics['abs_tail_th']:.2f} px (≈ {metrics['abs_tail_rate']:.2f}%)", "의미": "Part V 후보(오차 과다) 기준과 동일"})
    if metrics["proc_p95"] is not None:
        rows.append({"항목": "Proc Time p95 / p99 / max", "값": f"{metrics['proc_p95']:.1f} / {metrics['proc_p99']:.1f} / {metrics['proc_max']:.1f} ms", "의미": "지연의 꼬리(outlier) 크기"})
        rows.append({"항목": f"Proc Time 상위 {100 - pctl}% 기준", "값": f"≥ {metrics['proc_tail_th']:.1f} ms (≈ {metrics['proc_tail_rate']:.2f}%)", "의미": "Part V 후보(처리시간 과다) 기준과 동일"})

    _st_dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 분위수 테이블
    qrows = []
    rq = metrics.get("ratio_quantiles") or {}
    if rq:
        for k in ["p01","p05","p10","p25","p50","p75","p90","p95","p99"]:
            if k in rq:
                qrows.append({"지표": "Mask White Ratio", "분위": k, "값": f"{rq[k]:.4f}"})
    if qrows:
        _st_dataframe(pd.DataFrame(qrows), use_container_width=True, hide_index=True)
    else:
        st.info("분위수 계산에 필요한 컬럼이 부족합니다.")

    # Candidate tag mix
    st.markdown("### 후보(이상 구간) 분포 요약")
    if metrics.get("candidate_tag_counts"):
        tag_counts = metrics["candidate_tag_counts"]
        total_cand = metrics.get("candidate_n") or 0
        lines = [f"- 전체 후보 수: **{total_cand:,}** / 전체 대비 **{(total_cand / n_total * 100.0 if n_total else 0):.2f}%**"]
        # 상위 6개 표시
        for k, v in list(tag_counts.items())[:6]:
            lines.append(f"- {k}: {v:,}건 (전체 대비 {(v / n_total * 100.0):.2f}%")
        st.markdown("\n".join(lines))
    else:
        st.info("후보가 없거나(민감도 높음), 후보 태그를 만들 수 없는 구성입니다.")

    return metrics


def _openai_generate_recos_from_metrics(metrics: dict, pctl: int) -> dict:
    """
    계산된 지표(metrics)를 바탕으로 OpenAI Responses API를 호출해 개선사항(권고안)을 생성한다.

    반환값은 (가능한 한) 아래 스키마를 따르는 dict 형태여야 한다:
    {
      "overview": str,                      # 요약(한국어)
      "assumptions": [str],                 # 가정/전제(한국어)
      "recommendations": [                  # 권고안 목록(한국어)
         {"priority": int,                  # 우선순위(1이 가장 높음)
          "title": str,                     # 제목
          "why": str,                       # 근거/이유(가능하면 metrics 기반)
          "action": str,                    # 실행 방안(구체적)
          "validation": str,                # 검증 방법(지표/테스트 기준)
          "confidence": "high|medium|low",  # 확신 수준
          "is_speculative": bool}           # 추측 여부(근거 부족 시 True)
      ],
      "notes": str                          # 추가 메모(선택)
    }

    주의:
    - 모든 문자열 값은 한국어로 작성한다(단위/약어: ms, p95 등은 예외).
    - metrics에 근거가 부족한 내용은 반드시 추측으로 표시한다(is_speculative=True).
    """
    import os, json
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"openai 패키지 import 실패: {e}")

    # API 키 호출
    api_key = None
    if hasattr(st, "secrets"):
        api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai_api_key")
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. (st.secrets 또는 환경변수)")

    client = OpenAI(api_key=api_key)

    # 모델 설정
    model = "gpt-4o-mini"
    max_out = 700
    temperature = 0.2

    def _to_jsonable(x):

        if x is None:
            return None
        if pd is not None and hasattr(pd, "isna"):
            try:
                if pd.isna(x):
                    return None
            except Exception:
                pass
        if np is not None:
            if isinstance(x, (np.integer, np.floating, np.bool_)):
                return x.item()
        if isinstance(x, (int, float, bool, str)):
            return x
        if isinstance(x, dict):
            return {str(k): _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_to_jsonable(v) for v in x]
        return str(x)

    safe_metrics = _to_jsonable(metrics)

    def _drop_nulls(x):
        # None / 빈 dict / 빈 list 제거
        if x is None:
            return None
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                vv = _drop_nulls(v)
                if vv is None:
                    continue
                if isinstance(vv, (dict, list)) and len(vv) == 0:
                    continue
                out[str(k)] = vv
            return out
        if isinstance(x, list):
            out_list = []
            for v in x:
                vv = _drop_nulls(v)
                if vv is None:
                    continue
                if isinstance(vv, (dict, list)) and len(vv) == 0:
                    continue
                out_list.append(vv)
            return out_list
        return x

    def _topn_dict(d, n: int):
        if not isinstance(d, dict):
            return d
        return dict(list(d.items())[:n])

    # LLM에는 핵심 지표만 전달 (UI/보고서용 상세 dict는 제외)
    allow = [
        "n_total",
        "ratio_low_th","ratio_high_th","ratio_low_rate","ratio_high_rate",
        "ratio_p05","ratio_p50","ratio_p95","ratio_p99",
        "err_missing_rate","abs_p95","abs_p99","abs_tail_th","abs_tail_rate",
        "proc_p95","proc_p99","proc_max","proc_tail_th","proc_tail_rate",
        "candidate_n","candidate_tag_counts",
        "failure_definition","failure_n","failure_rate",
        "failure_type_counts_top","lane_state_counts_top",
        "failure_contours_total_zero_rate","failure_contours_kept_zero_rate",
        "failure_direction_force_abs_p95","failure_target_cx_p05","failure_target_cx_p95",
    ]
    llm_metrics = {k: safe_metrics.get(k) for k in allow if isinstance(safe_metrics, dict) and k in safe_metrics}
    if "candidate_tag_counts" in llm_metrics:
        llm_metrics["candidate_tag_counts"] = _topn_dict(llm_metrics["candidate_tag_counts"], 8)
    llm_metrics = _drop_nulls(llm_metrics) or {}

    payload = {
        "pctl": int(pctl),
        "metrics": llm_metrics,
        "metric_key_hints": [
            "ratio_p05","ratio_p50","ratio_p95","ratio_p99","ratio_low_rate","ratio_high_rate",
            "abs_p95","abs_p99","err_missing_rate",
            "proc_p95","proc_p99","proc_max",
            "candidate_tag_counts","candidate_n",
            "failure_definition","failure_n","failure_rate","failure_type_counts_top","lane_state_counts_top",
        ],
    }

    # --- structured output schema (strict)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "overview": {"type": "string", "maxLength": 700},
            "assumptions": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
            "recommendations": {
                "type": "array",
                "minItems": 4,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "priority": {"type": "integer", "minimum": 1, "maximum": 10},
                        "title": {"type": "string", "maxLength": 120},
                        "why": {"type": "string", "maxLength": 400},
                        "action": {"type": "string", "maxLength": 400},
                        "validation": {"type": "string", "maxLength": 300},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "is_speculative": {"type": "boolean"},
                    },
                    "required": ["priority", "title", "why", "action", "validation", "confidence", "is_speculative"],
                },
            },
            "notes": {"type": "string", "maxLength": 500},
        },
        "required": ["overview", "assumptions", "recommendations", "notes"],
    }

    # --- prompt (short, metric-grounded)
    system = (
        "너는 자율주행(차선 인식) 로그의 '요약 통계'만 보고 개선안을 제안한다. "
        "반드시 한국어로만 작성한다.(단위/약어: ms, p95, cx, hsv 등은 예외). "
        "요약에 없는 사실은 만들지 말고, 확신이 없으면 is_speculative=true 및 confidence=low로 표기한다. "
        "why에는 반드시 metric 키(예: abs_p95, err_missing_rate, failure_rate, failure_type_counts_top 등)를 근거로 언급한다. "
        "candidate_tag_counts가 있으면 후보(이상 구간) 분포를 근거로 활용한다. "
        "출력은 반드시 JSON 스키마를 따른다."
    )
    user = (
        "아래 JSON은 Part X에서 계산한 요약 통계다. "
        "이 정보만 근거로 개선안을 작성하라. 단위와 약어를 제외한 모든 문장은 한국어로 작성하라.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={"format": {"type": "json_schema", "name": "partx_recos", "strict": True, "schema": schema}},
        max_output_tokens=max_out,
        temperature=temperature,
        store=False,
    )

    raw = (getattr(resp, "output_text", None) or "").strip()
    if not raw:
        return {"overview": "", "assumptions": [], "recommendations": [], "notes": ""}

    # --- best-effort parse
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # fallback: wrap raw text (still editable in report)
    return {
        "overview": "LLM 출력(JSON 파싱 실패). 아래 notes를 참고하세요.",
        "assumptions": [],
        "recommendations": [],
        "notes": raw[:480],
    }

def _format_llm_recos_markdown(llm: dict | None) -> str:
    """LLM JSON 결과를 간결한 Markdown으로 렌더링 (복사/편집 용이)"""
    llm = llm or {}
    overview = str(llm.get("overview") or "").strip()
    assumptions = llm.get("assumptions") or []
    recos = llm.get("recommendations") or []
    notes = str(llm.get("notes") or "").strip()

    lines: list[str] = []
    if overview:
        lines.append(overview)

    if assumptions:
        lines.append("")
        lines.append("**가정/불확실(모델 표기)**")
        for a in assumptions[:8]:
            a = str(a).strip()
            if a:
                lines.append(f"- {a}")

    if recos:
        lines.append("")
        lines.append("**개선사항(우선순위)**")
        # sort by priority if possible
        def _pri(x):
            try:
                return int(x.get("priority", 999))
            except Exception:
                return 999

        for r in sorted(recos, key=_pri):
            try:
                pri = int(r.get("priority", 0))
            except Exception:
                pri = r.get("priority", "-")
            title = str(r.get("title") or "").strip()
            conf = str(r.get("confidence") or "").strip()
            spec = bool(r.get("is_speculative", False))
            tag = []
            if conf:
                tag.append(f"conf:{conf}")
            if spec:
                tag.append("추측")
            tag_txt = f" ({', '.join(tag)})" if tag else ""

            why = str(r.get("why") or "").strip()
            action = str(r.get("action") or "").strip()
            validation = str(r.get("validation") or "").strip()

            head = f"- **P{pri}. {title}**{tag_txt}".strip()
            lines.append(head)
            if why:
                lines.append(f"  - 근거: {why}")
            if action:
                lines.append(f"  - 조치: {action}")
            if validation:
                lines.append(f"  - 검증: {validation}")

    if notes:
        lines.append("")
        lines.append(f"> 참고: {notes}")

    return "\n".join(lines).strip()



def _build_partx_report(metrics: dict, pctl: int, llm: dict | None) -> str:
    """Part X를 위한 복사/편집 용이한 Markdown 보고서 생성"""
    llm_md = _format_llm_recos_markdown(llm)

    n_total = metrics.get("n_total")
    ratio_low_th = metrics.get("ratio_low_th")
    ratio_high_th = metrics.get("ratio_high_th")
    ratio_low_rate = metrics.get("ratio_low_rate")
    ratio_high_rate = metrics.get("ratio_high_rate")
    err_missing_rate = metrics.get("err_missing_rate")

    abs_p95 = metrics.get("abs_p95")
    abs_p99 = metrics.get("abs_p99")
    abs_tail_th = metrics.get("abs_tail_th")

    proc_p95 = metrics.get("proc_p95")
    proc_p99 = metrics.get("proc_p99")
    proc_max = metrics.get("proc_max")
    proc_tail_th = metrics.get("proc_tail_th")

    rq = metrics.get("ratio_quantiles") or {}
    cand_n = metrics.get("candidate_n")
    tag_counts = metrics.get("candidate_tag_counts") or {}
    tag_top = list(tag_counts.items())[:6]

    def fmt(v, f="{:.4f}"):
        try:
            return f.format(float(v))
        except Exception:
            return "N/A"

    def fmt2(v, f="{:.2f}"):
        try:
            return f.format(float(v))
        except Exception:
            return "N/A"

    def fmt_ms(v):
        try:
            return f"{float(v):.1f}ms"
        except Exception:
            return "N/A"

    lines: list[str] = []
    lines.append("# Part X 분석 및 개선사항 보고서")
    lines.append("")
    lines.append("## 1) 분석 요약(분포 기반)")
    lines.append(f"- 총 프레임 수: {n_total:,}" if isinstance(n_total, int) else "- 총 프레임 수: N/A")
    lines.append(f"- 분석 민감도(pctl): {pctl} (상/하위 {100 - pctl}% 꼬리 기준)")
    lines.append("")

    lines.append("### 핵심 지표")
    lines.append("| 항목 | 값 | 비고 |")
    lines.append("|---|---:|---|")
    if ratio_low_th is not None:
        lines.append(f"| Mask Ratio 하한(하위 {100 - pctl}% 분위) | {fmt(ratio_low_th)} | 낮은 가시성/미검출 후보 |")
        lines.append(f"| Mask Ratio 하한 이하 비율 | {fmt2(ratio_low_rate)}% | 꼬리 구간 비중 |")
        lines.append(f"| Mask Ratio 상한(상위 {100 - pctl}% 분위) | {fmt(ratio_high_th)} | 과검출 후보 |")
        lines.append(f"| Mask Ratio 상한 이상 비율 | {fmt2(ratio_high_rate)}% | 꼬리 구간 비중 |")
    else:
        lines.append("| Mask Ratio | N/A | 컬럼/값 부족 |")

    if err_missing_rate is not None:
        lines.append(f"| Lane Error 결측률 | {fmt2(err_missing_rate)}% | 결측이 많으면 오차 기반 분석 약화 |")
    else:
        lines.append("| Lane Error 결측률 | N/A | 컬럼/값 부족 |")

    if abs_p95 is not None:
        lines.append(f"| Abs Error p95 / p99 | {fmt2(abs_p95)}px / {fmt2(abs_p99)}px | 상위 꼬리: ≥ {fmt2(abs_tail_th)}px |")
    else:
        lines.append("| Abs Error(p95/p99) | N/A | 컬럼/값 부족 |")

    if proc_p95 is not None:
        lines.append(f"| Processing Time p95 / p99 / max | {fmt_ms(proc_p95)} / {fmt_ms(proc_p99)} / {fmt_ms(proc_max)} | 상위 꼬리: ≥ {fmt_ms(proc_tail_th)} |")
    else:
        lines.append("| Processing Time(p95/p99/max) | N/A | 컬럼/값 부족 |")

    lines.append("")
    lines.append("### 참고 분위수(요약)")
    lines.append("- Mask White Ratio: " + ", ".join([f"{k}={fmt(v)}" for k, v in rq.items() if k in ["p05","p50","p95"]]) if rq else "- Mask White Ratio: N/A")
    lines.append("")

    lines.append("## 2) 후보(이상 구간) 요약")
    if isinstance(cand_n, int):
        lines.append(f"- 전체 후보 수: {cand_n:,}")
    else:
        lines.append("- 전체 후보 수: N/A")
    if tag_top:
        lines.append("- 후보 태그 상위:")
        for k, v in tag_top:
            lines.append(f"  - {k}: {v:,}건")
    else:
        lines.append("- 후보 태그 분포: N/A")
    lines.append("")

    lines.append("## 3) 개선사항")
    if llm_md:
        # OpenAI 출력은 이미 markdown 형식. 그대로 사용
        lines.append(llm_md)
    else:
        lines.append("- (아직 생성된 개선사항이 없습니다. Part X에서 버튼을 눌러 생성한 뒤, 여기서 문구를 수정하세요.)")

    lines.append("")
    lines.append("## 4) 메모")
    lines.append("- 이 보고서는 분포 기반 자동 요약이며, 원인 확정이 아닙니다. 필요 시 원본 프레임 확인이 필요합니다.")
    lines.append("")
    return "\n".join(lines).strip() + "\n"

def _render_part_x_openai(df: pd.DataFrame, pctl: int, cand: "pd.DataFrame | None" = None) -> None:
    
    # 분포 기반 요약 지표를 한번에 계산해 metrics 딕셔너리로 반환
    metrics = _render_part_x_distribution(df, pctl, cand=cand)

    st.markdown("### 개선사항(OpenAI 자동 생성)")
    st.caption("버튼을 누르면 위 분포 요약을 근거로 개선사항을 생성")


    col_a, col_b = st.columns([1, 1])
    with col_a:
        run_btn = st.button("OpenAI로 개선사항 생성", type="primary")
    with col_b:
        clear_btn = st.button("생성 결과 지우기")

    if clear_btn:
        st.session_state.pop("partx_openai_json", None)
        st.session_state.pop("partx_openai_text", None)
        st.session_state.pop("partx_report_text", None)
    if run_btn:
        try:
            with st.spinner("OpenAI 호출 중..."):
                llm = _openai_generate_recos_from_metrics(metrics, pctl=pctl)
                md = _format_llm_recos_markdown(llm)
            st.session_state["partx_openai_json"] = llm
            st.session_state["partx_openai_text"] = md
            st.session_state["partx_report_text"] = _build_partx_report(metrics, pctl=pctl, llm=llm)
        except Exception as e:
            st.error(str(e))

    if st.session_state.get("partx_openai_text"):
        st.markdown(st.session_state["partx_openai_text"])
    else:
        st.info("아직 생성된 개선사항이 없습니다.")

    # ---- Copy/edit-friendly report
    st.markdown("### 보고서(복사/수정용)")
    st.caption("아래 텍스트는 수정 가능한 보고서 초안입니다. 필요에 맞게 문장을 편집한 뒤 복사하세요.")

    # 보고서 텍스트를 한 번 초기화 (rerun 간 사용자 편집 보존)
    if "partx_report_text" not in st.session_state:
        st.session_state["partx_report_text"] = _build_partx_report(
            metrics, pctl=pctl, llm=st.session_state.get("partx_openai_json")
        )

    st.text_area(
        "보고서 내용 (Markdown)",
        key="partx_report_text",
        height=520,
    )

    st.download_button(
        "보고서 다운로드(.md)",
        data=st.session_state.get("partx_report_text", ""),
        file_name="partx_report.md",
        mime="text/markdown",
        use_container_width=True,
    )




# =============================================================================
# Reference: Deviation regression (optional)
st.divider()
with st.expander("Reference: deviation regression", expanded=False):
    _render_reference_deviation_regression(df)

# Part X 실행
try:
    _render_part_x_openai(df, pctl, cand=cand)
except Exception as e:
    st.warning(f"Part X를 표시하는 중 문제가 발생했습니다: {e}")