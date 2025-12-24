import streamlit as st
import altair as alt
from altair.datasets import data
import polars as pl
import numpy as np
import statsmodels.api as sm


st.set_page_config(
    page_title="실패분석", page_icon=":clapper:", layout="wide"
)

# --- Mock Data Generation for Autonomous Driving ---
def generate_autonomous_data(n=1000):
    np.random.seed(42)
    weather_opts = ["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy"]
    
    # Generate base data
    weather = np.random.choice(weather_opts, n)
    white_line_present = np.random.choice([True, False], n, p=[0.9, 0.1]) # 90% frames have lines
    
    # Simulate Confidence and IoU (correlated but with noise)
    confidence = np.random.beta(5, 2, n) * 10  # Scale 0-10
    iou = (confidence * 10) + np.random.normal(0, 10, n) # Scale 0-100
    iou = np.clip(iou, 0, 100)
    
    # Inject Failures: White line exists, but low confidence/IoU (e.g., in Snowy/Rainy)
    mask_failure = (weather == "Snowy") | (weather == "Rainy")
    confidence[mask_failure] = confidence[mask_failure] * 0.6
    iou[mask_failure] = iou[mask_failure] * 0.5

    df = pl.DataFrame({
        "Frame ID": [f"frame_{i:05d}" for i in range(n)],
        "Weather": weather,
        "White Line Present": white_line_present,
        "Model Confidence": confidence,
        "IoU Score": iou,
        "Processing Time (ms)": np.random.normal(30, 5, n),
    })
    return df

df = generate_autonomous_data()
TITLE_COL = "Frame ID"
IMDB_COL = "Model Confidence"  # 0-10 scale
RT_COL = "IoU Score"           # 0-100 scale
DIRECTOR_COL = "Weather"
HAS_RATINGS = pl.col("White Line Present") == True # Only analyze frames where line exists
TEXT_WIDTH = 700
SMALLPLOT_WIDTH = 500
SMALLPLOT_HEIGHT = 500
MARK_SIZE = 70
DIRECTOR_MARK_SIZE = 150

COLUMN_CONFIG = {
    TITLE_COL: st.column_config.TextColumn(pinned=True),
    IMDB_COL: st.column_config.ProgressColumn(
        min_value=0,
        max_value=10,
        color="#f28e2b",
        format="compact",
        width=100,
    ),
    RT_COL: st.column_config.ProgressColumn(
        min_value=0,
        max_value=100,
        color="#4e79a7",
        format="compact",
        width=100,
    ),
    "Processing Time (ms)": st.column_config.NumberColumn(format="%.1f ms"),
    "Weather": st.column_config.TextColumn(),
    "White Line Present": st.column_config.CheckboxColumn(),
    "Status": st.column_config.TextColumn(
        width="medium",
    ),
}

# -----------------------------------------------------------------------------
# Helpful functions


def draw_histogram(df, metric_name):
    # Ensure we don't have nulls for the main metric
    clean_df = df.drop_nulls(subset=[metric_name])

    st.altair_chart(
        alt.Chart(clean_df, height=200, width=200)
        .mark_bar(binSpacing=0)
        .encode(
            alt.X(
                metric_name,
                type="quantitative",
            ).bin(maxbins=20),
            alt.Y("count()").axis(None),
        )
    )


def draw_director_median_chart(title, data, x_col, y_col, x_domain, color_domain):
    data = data.drop_nulls(subset=[y_col])

    medians = (
        data.group_by(y_col)
        .agg(pl.col(x_col).median().alias("median_val"))
        .sort("median_val", descending=True)
    )

    sort_order = medians.get_column(y_col).to_list()

    base = alt.Chart(data).encode(
        alt.Y(f"{y_col}:N", sort=sort_order, title=None),
    )

    points = base.mark_point(filled=True, size=DIRECTOR_MARK_SIZE).encode(
        alt.X(f"{x_col}:Q", title=f"{x_col}").scale(zero=True, domain=x_domain),
        alt.Color(DIRECTOR_COL, type="nominal").scale(domain=color_domain).legend(None),
        alt.Shape(DIRECTOR_COL, type="nominal").scale(domain=color_domain).legend(None),
        tooltip=[y_col, x_col, TITLE_COL],
    )

    ticks = (
        alt.Chart(medians.to_pandas())
        .mark_tick(
            color="red",
            thickness=2,
        )
        .encode(
            alt.Y(f"{y_col}:N", sort=sort_order),
            alt.X("median_val:Q"),
            tooltip=[alt.Tooltip("median_val:Q", title="Median Rating")],
        )
    )

    st.subheader(title)
    st.altair_chart(points + ticks, width="stretch")


def perform_linear_regression(df, x_col, y_col, sigma_threshold):
    clean_df = df.drop_nulls([x_col, y_col])

    x = clean_df[x_col].to_numpy()
    y = clean_df[y_col].to_numpy()

    # Degree 1 = Linear
    slope, intercept = np.polyfit(x, y, 1)

    predictions = (slope * x) + intercept
    residuals = y - predictions
    std_dev = np.std(residuals)

    upper_bound = predictions + (sigma_threshold * std_dev)
    lower_bound = predictions - (sigma_threshold * std_dev)

    result_df = clean_df.with_columns(
        [
            pl.Series("Predicted", predictions),
            pl.Series("Upper Bound", upper_bound),
            pl.Series("Lower Bound", lower_bound),
            # Determine Status: Outlier if outside the bounds
            pl.when(
                (pl.col(y_col) > pl.Series(upper_bound))
                | (pl.col(y_col) < pl.Series(lower_bound))
            )
            .then(pl.lit("Outlier"))
            .otherwise(pl.lit("In Range"))
            .alias("Status"),
        ]
    )

    return result_df


def perform_loess_regression(df, x_col, y_col, sigma_threshold, frac=0.66):
    """
    Calculates LOESS regression, residuals, and outlier status using Polars and Statsmodels.

    Args:
        frac (float): The fraction of the data used when estimating each y-value.
                      Between 0 and 1. Defaults to 0.66 (standard).
    """
    # Sorting by x_col is mandatory for LOESS to align predictions correctly for plotting
    clean_df = df.drop_nulls([x_col, y_col]).sort(x_col)

    x = clean_df[x_col].to_numpy()
    y = clean_df[y_col].to_numpy()

    # Returns an (n, 2) array: [sorted_x, fitted_y]
    lowess_result = sm.nonparametric.lowess(y, x, frac=frac)

    predictions = lowess_result[:, 1]

    residuals = y - predictions
    std_dev = np.std(residuals)

    upper_bound = predictions + (sigma_threshold * std_dev)
    lower_bound = predictions - (sigma_threshold * std_dev)

    result_df = clean_df.with_columns(
        [
            pl.Series("Predicted", predictions),
            pl.Series("Upper Bound", upper_bound),
            pl.Series("Lower Bound", lower_bound),
            pl.when(
                (pl.col(y_col) > pl.Series(upper_bound))
                | (pl.col(y_col) < pl.Series(lower_bound))
            )
            .then(pl.lit("Outlier"))
            .otherwise(pl.lit("In Range"))
            .alias("Status"),
        ]
    )

    return result_df


def wide_centered_layout():
    with st.container(horizontal_alignment="center"):
        return st.container(
            width=2 * SMALLPLOT_WIDTH + 16, horizontal_alignment="center"
        )


# -----------------------------------------------------------------------------
# Draw app


with wide_centered_layout():
    with st.container(width=TEXT_WIDTH):
        st.title("자율주행 차선 인식 실패 분석")

        st.space()

        """
        ## Part I: Detection Performance

        **모델의 확신(Confidence)과 실제 정확도(IoU)의 관계는?**
        아래 분석에서 **Model Confidence**는 딥러닝 모델이 차선을 인식했다고 믿는 정도이며,
        **IoU Score**는 실제 정답(Ground Truth)과 얼마나 일치하는지를 나타냅니다.

        일반적으로 두 지표는 양의 상관관계를 가져야 합니다.
        :green[**초록색 십자가(Outlier)**]는 모델이 과신했거나(Confidence 높음, IoU 낮음),
        예상보다 잘 맞춘 케이스를 의미합니다.
        """

    rating_df = (
        df.filter(HAS_RATINGS)
        .select(TITLE_COL, DIRECTOR_COL, IMDB_COL, RT_COL)
        .with_columns(
            delta=pl.col(IMDB_COL) / 10 - pl.col(RT_COL) / 100,
        )
    )

    rating_model_df = perform_linear_regression(
        rating_df, IMDB_COL, RT_COL, sigma_threshold=2
    )

    st.space()

    with st.container(width=TEXT_WIDTH):
        cols = st.columns([0.7, 0.3])

        with cols[0]:
            st.subheader("Confidence vs IoU Correlation")
            st.altair_chart(
                alt.Chart(rating_model_df)
                .mark_point(filled=True, size=MARK_SIZE, opacity=0.5)
                .encode(
                    alt.X(IMDB_COL, type="quantitative"),
                    alt.Y(RT_COL, type="quantitative"),
                    alt.Color("Status:N").legend(None),
                    alt.Shape("Status:N").scale(range=["circle", "cross"]).legend(None),
                    tooltip=[TITLE_COL, DIRECTOR_COL, IMDB_COL, RT_COL, "Status"],
                ),
                height="stretch",
            )

        with cols[1]:
            st.space("medium")
            draw_histogram(rating_df, IMDB_COL)
            draw_histogram(rating_df, RT_COL)

    diff_df = rating_df.filter(
        pl.col(IMDB_COL).is_not_null() & pl.col(RT_COL).is_not_null()
    ).sort(by="delta", descending=True)

    help_text = (
        "Confidence와 IoU의 차이를 기반으로 계산됩니다."
    )

    st.space()

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader(
            "Over-confident Failures (False Positives)",
            help=help_text,
        )

        st.dataframe(
            diff_df.select(pl.exclude("delta"))
            .head(20)
            .sort(by=IMDB_COL, descending=True),
            column_config=COLUMN_CONFIG,
            height="stretch",
        )

    with cols[1]:
        st.subheader(
            "Under-confident Successes",
            help=help_text,
        )

        st.dataframe(
            diff_df.select(pl.exclude("delta"))
            .tail(20)
            .sort(by=RT_COL, descending=True),
            column_config=COLUMN_CONFIG,
            height="stretch",
        )

    # -----------------------------------------------------------------------------
    # Part 2

    st.space("large")

    with st.container(width=TEXT_WIDTH):
        """
        ## Part II: Environmental Analysis (Weather)

        **어떤 날씨 환경에서 인식이 가장 실패할까요?**

        아래 차트는 날씨(Weather)별로 Confidence와 IoU의 중앙값(Median) 분포를 보여줍니다.
        특정 날씨(예: Snowy, Rainy)에서 성능이 저하되는지 확인할 수 있습니다.
        """

        min_movies = st.slider(
            "Minimum frames per weather condition",
            min_value=1,
            max_value=50,
            value=10,
        )

        director_df = rating_df.filter(pl.col(DIRECTOR_COL).is_not_null()).with_columns(
            first_letter=pl.col(DIRECTOR_COL).str.head(1)
        )

        director_medians_df = (
            director_df.group_by(DIRECTOR_COL)
            .agg(
                **{
                    "num_movies": pl.len(),
                    "first_letter": pl.col("first_letter").first(),
                    IMDB_COL: pl.col(IMDB_COL).median(),
                    RT_COL: pl.col(RT_COL).median(),
                }
            )
            .filter(pl.col("num_movies") >= min_movies)
        )

        all_directors_list = director_medians_df.get_column(DIRECTOR_COL).to_list()

        st.space()

        st.subheader("Performance by Weather Condition")
        st.altair_chart(
            alt.Chart(director_medians_df, height=SMALLPLOT_HEIGHT)
            .mark_point(filled=True, size=DIRECTOR_MARK_SIZE)
            .encode(
                alt.X(IMDB_COL, type="quantitative"),
                alt.Y(RT_COL, type="quantitative"),
                alt.Color(DIRECTOR_COL, type="nominal")
                .scale(domain=all_directors_list)
                .legend(None),
                alt.Shape(DIRECTOR_COL, type="nominal")
                .scale(domain=all_directors_list)
                .legend(None),
                tooltip=[DIRECTOR_COL, IMDB_COL, RT_COL, "num_movies"],
            )
        )

    st.space()

    for i in range(2):
        if i == 0:
            metric_name = IMDB_COL
            director_medians_df = director_medians_df.filter(
                pl.col(IMDB_COL).is_not_null()
            )
            x_domain = [0, 10]
        else:
            metric_name = RT_COL
            director_medians_df = director_medians_df.filter(
                pl.col(RT_COL).is_not_null()
            )
            x_domain = [0, 100]

        cols = st.columns(2, border=True)

        with cols[0]:
            top_directors_df = director_medians_df.sort(
                metric_name, descending=True
            ).head(10)

            top_directors_set = set(top_directors_df.get_column(DIRECTOR_COL).to_list())

            top_dir_df = director_df.filter(
                pl.col(DIRECTOR_COL).is_in(top_directors_set)
            )

            draw_director_median_chart(
                f"Best Conditions by {metric_name}",
                data=top_dir_df,
                x_col=metric_name,
                y_col=DIRECTOR_COL,
                x_domain=x_domain,
                color_domain=all_directors_list,
            )

        with cols[1]:
            bottom_directors_df = director_medians_df.sort(metric_name).head(10)

            bottom_directors_set = set(
                bottom_directors_df.get_column(DIRECTOR_COL).to_list()
            )

            bottom_dir_df = director_df.filter(
                pl.col(DIRECTOR_COL).is_in(bottom_directors_set)
            )

            draw_director_median_chart(
                f"Worst Conditions by {metric_name}",
                data=bottom_dir_df,
                x_col=metric_name,
                y_col=DIRECTOR_COL,
                x_domain=x_domain,
                color_domain=all_directors_list,
            )

    # -----------------------------------------------------------------------------
    # Part 3

    st.space("large")

    with st.container(width=TEXT_WIDTH):
        """
        ## Part III: Failure Prediction & Outliers

        **흰색 선을 인지하지 못하는 실패 케이스(Outlier) 탐지**
        회귀 분석을 통해 예상되는 성능 범위를 벗어나는 프레임을 찾습니다.
        특히 **흰색 선이 존재함에도 불구하고(Ground Truth)** 모델 수치가 낮은 경우를 집중적으로 확인하세요.
        """

    numeric_cols = [
        "Model Confidence",
        "IoU Score",
        "Processing Time (ms)",
    ]

    with st.container(width=TEXT_WIDTH):
        st.space()

        cols = st.columns(2)

        with cols[0]:
            x_col = st.selectbox(
                "X Axis (predictor)", options=numeric_cols, index=0
            )  # Default Confidence

        with cols[1]:
            y_col = st.selectbox(
                "Y Axis (target)", options=numeric_cols, index=1
            )  # Default IoU

        sigma_val = st.slider(
            "Confidence interval (sigma)",
            min_value=0.5,
            max_value=4.0,
            value=2.0,
            step=0.1,
            help="Determines the width of the confidence band. Points outside this band are outliers.",
        )

        regression_type = st.segmented_control(
            "Regression type",
            ["Linear regression", "LOESS regression"],
            default="Linear regression",
        )

        if not x_col or not y_col:
            st.info("Please select columns to visualize.")
            st.stop()

        if regression_type == "Linear regression":
            model_df = perform_linear_regression(df, x_col, y_col, sigma_val)
        else:
            model_df = perform_loess_regression(df, x_col, y_col, sigma_val)

        outliers = model_df.filter(pl.col("Status") == "Outlier").select(
            TITLE_COL, DIRECTOR_COL, IMDB_COL, RT_COL
        )
        num_outliers = len(model_df.filter(pl.col("Status") == "Outlier"))

        st.space()

        with st.container(height=SMALLPLOT_HEIGHT, border=False):
            base = alt.Chart(model_df).encode(
                alt.X(x_col, title=x_col, scale=alt.Scale(zero=False))
            )

            band = base.mark_area(opacity=0.1).encode(
                alt.Y("Lower Bound"),
                alt.Y2("Upper Bound"),
                tooltip=[
                    alt.Tooltip("Lower Bound", format=",.0f"),
                    alt.Tooltip("Upper Bound", format=",.0f"),
                ],
            )

            line = base.mark_line(size=3).encode(y="Predicted")

            points = base.mark_point(filled=True, size=MARK_SIZE, opacity=0.5).encode(
                alt.Y(y_col, title=y_col),
                alt.Color(
                    "Status:N",
                ).legend(None),
                alt.Shape("Status:N").scale(range=["circle", "cross"]).legend(None),
                tooltip=[TITLE_COL, x_col, y_col],
            )

            final_chart = (band + points + line).configure_legend(orient="bottom")

            st.subheader(f"{y_col} by {x_col}")
            st.altair_chart(final_chart, width="stretch", height="stretch")

    st.subheader("Detected Failures (Outliers)")

    """
    모델의 예측 경향성에서 벗어난 프레임들입니다. (흰색 선 미인식 등)
    """

    with st.container(horizontal=True):
        with st.container(width="content"):
            with st.container(border=True, width="content"):
                st.metric(
                    "Number of outliers",
                    num_outliers,
                    help="일반적인 성능 분포에서 벗어난 프레임 수",
                )

        with st.container(width="stretch"):
            st.dataframe(outliers, column_config=COLUMN_CONFIG, height=SMALLPLOT_HEIGHT)

    # -----------------------------------------------------------------------------
    # Part 4

    st.space("large")

    """
    ## Part IV: Browse Full Log Data
    """

    st.space()

    st.dataframe(
        df,
        height=SMALLPLOT_HEIGHT,
        column_config=COLUMN_CONFIG,
    )
