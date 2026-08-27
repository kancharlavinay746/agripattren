import streamlit as st
import pandas as pd
import numpy as np

from src.analysis import (
    numeric_columns,
    correlation_matrix,
    detect_anomalies,
    cluster_data,
    trend_analysis,
    unexpected_relationships,
    dataset_profile,
    make_unique_columns,
)

from src.ai_explainer import explain_patterns

from src.visualization import (
    kpi_metrics,
    histogram,
    box_plot,
    scatter_plot,
    line_chart,
    bar_chart,
    correlation_heatmap,
)

from src.prediction import (
    detect_target_columns,
    suggest_target,
    prepare_prediction_data,
    train_all_models,
    comparison_table,
    get_best_model,
    get_feature_importance,
    predict_with_model,
    actual_vs_predicted,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AgriPattern",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME
# ============================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 1rem;
    opacity: 0.65;
    margin-bottom: 1.5rem;
}

.sidebar-brand {
    font-size: 1.55rem;
    font-weight: 800;
}

.sidebar-subtitle {
    font-size: 0.82rem;
    opacity: 0.65;
}

.module-card {
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.18);
    background: rgba(128,128,128,0.05);
    min-height: 190px;
}

.section-title {
    font-size: 1.6rem;
    font-weight: 750;
}

div[data-testid="metric-container"] {
    border-radius: 16px;
    padding: 15px;
    border: 1px solid rgba(128,128,128,0.18);
    background: rgba(128,128,128,0.05);
}

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            🌾 AgriPattern
        </div>

        <div class="sidebar-subtitle">
            Agricultural Intelligence Platform
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### 🧭 Navigation")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 Dataset Explorer",
            "📊 Visualization Studio",
            "🔗 Correlation Intelligence",
            "🚨 Anomaly Detection",
            "🧩 Cluster Discovery",
            "📈 Trend Intelligence",
            "🔎 Pattern Discovery",
            "🤖 Prediction Studio",
            "🧠 AI Insights",
            "📄 Report Generator",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### 📂 Dataset")

    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
    )

    st.divider()

    st.markdown("### ⚙️ Analysis Settings")

    contamination = st.slider(
        "🚨 Anomaly Sensitivity",
        0.01,
        0.20,
        0.05,
        0.01,
    )

    k = st.slider(
        "🧩 Number of Clusters",
        2,
        10,
        3,
    )

    min_corr = st.slider(
        "🔗 Minimum Correlation",
        0.30,
        0.95,
        0.60,
        0.05,
    )

    st.divider()

    st.caption("AgriPattern v2.0")
    st.caption("Phase 4 — Prediction Intelligence")


# ============================================================
# HOME WITHOUT DATA
# ============================================================

if uploaded is None:

    st.markdown(
        """
        <div class="hero-title">
            🌾 AgriPattern
        </div>

        <div class="hero-subtitle">
            Agricultural Data Intelligence Platform
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "📂 Upload an agricultural CSV or Excel dataset "
        "from the sidebar to begin."
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="module-card">

            ### 📊 Explore & Visualize

            Analyze agricultural datasets with:

            • Histograms  
            • Box plots  
            • Scatter plots  
            • Line charts  
            • Bar charts  
            • Correlation heatmaps

            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            """
            <div class="module-card">

            ### 🧠 Machine Learning

            Discover:

            • K-Means clusters  
            • PCA  
            • Elbow analysis  
            • Silhouette analysis  
            • Anomalies

            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            """
            <div class="module-card">

            ### 🤖 Prediction Intelligence

            Predict:

            • Crop yield  
            • Agricultural production  
            • Model performance  
            • Important factors

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("🌾 Recommended Agricultural Columns")

    st.code(
        """
Year
State
District
Crop
Rainfall
Temperature
Soil_N
Soil_P
Soil_K
Fertilizer
Irrigation
Yield
Production
Area
Market_Price
        """,
        language="text",
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

except Exception as e:

    st.error(
        f"❌ Could not read dataset: {e}"
    )

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

df.columns = [
    str(c).strip()
    for c in df.columns
]

df.columns = make_unique_columns(
    df.columns
)


# ============================================================
# PROFILE
# ============================================================

profile = dataset_profile(
    df
)

nums = numeric_columns(
    df
)

metrics = kpi_metrics(
    df
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="hero-title">
        🌾 AgriPattern
    </div>

    <div class="hero-subtitle">
        Agricultural Intelligence Dashboard
        · {len(df):,} records
        · {len(df.columns):,} features
    </div>
    """,
    unsafe_allow_html=True,
)

st.success(
    f"✅ Dataset loaded successfully — "
    f"{len(df):,} rows × {len(df.columns):,} columns"
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.header("🏠 Dashboard Overview")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "📊 Records",
        f"{metrics['rows']:,}"
    )

    c2.metric(
        "📁 Features",
        metrics["columns"]
    )

    c3.metric(
        "🔢 Numeric",
        metrics["numeric_columns"]
    )

    c4.metric(
        "⚠️ Missing",
        f"{metrics['missing_cells']:,}"
    )

    c5.metric(
        "♻️ Duplicates",
        f"{metrics['duplicate_rows']:,}"
    )

    st.divider()

    st.subheader("🚀 AgriPattern Intelligence")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info(
            """
            **📊 Visualization**

            Explore agricultural trends,
            distributions and relationships.
            """
        )

    with c2:
        st.info(
            """
            **🧠 Machine Learning**

            Discover clusters, anomalies
            and hidden structures.
            """
        )

    with c3:
        st.info(
            """
            **🤖 Prediction**

            Predict crop yield and compare
            multiple ML models.
            """
        )

    st.divider()

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "📊 Dataset Explorer":

    st.header("📊 Dataset Explorer")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Numeric", len(nums))
    c4.metric(
        "Missing",
        f"{int(df.isna().sum().sum()):,}"
    )

    st.divider()

    st.subheader("Dataset")

    st.dataframe(
        df,
        use_container_width=True,
        height=500,
    )

    st.subheader("Column Profile")

    st.dataframe(
        profile["column_profile"],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "📥 Download Dataset CSV",
        df.to_csv(index=False).encode("utf-8"),
        "agripattern_dataset.csv",
        "text/csv",
    )


# ============================================================
# VISUALIZATION STUDIO
# ============================================================

elif page == "📊 Visualization Studio":

    st.header("📊 Visualization Studio")

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    chart_type = st.selectbox(
        "📈 Visualization Type",
        [
            "Histogram",
            "Box Plot",
            "Scatter Plot",
            "Line Chart",
            "Bar Chart",
            "Correlation Heatmap",
        ],
    )

    if chart_type == "Histogram":

        if not numeric_cols:
            st.warning("No numeric columns.")
        else:

            column = st.selectbox(
                "Numeric Column",
                numeric_cols
            )

            bins = st.slider(
                "Bins",
                5,
                100,
                30
            )

            fig = histogram(
                df,
                column,
                bins
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    elif chart_type == "Box Plot":

        if not numeric_cols:
            st.warning("No numeric columns.")
        else:

            column = st.selectbox(
                "Numeric Column",
                numeric_cols
            )

            fig = box_plot(
                df,
                column
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    elif chart_type == "Scatter Plot":

        if len(numeric_cols) < 2:

            st.warning(
                "At least two numeric columns are required."
            )

        else:

            c1, c2 = st.columns(2)

            with c1:
                x = st.selectbox(
                    "X Axis",
                    numeric_cols
                )

            with c2:
                y = st.selectbox(
                    "Y Axis",
                    numeric_cols,
                    index=1
                )

            color = None

            if categorical_cols:

                color_choice = st.selectbox(
                    "Color By",
                    ["None"] + categorical_cols
                )

                if color_choice != "None":
                    color = color_choice

            fig = scatter_plot(
                df,
                x,
                y,
                color
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    elif chart_type == "Line Chart":

        if not numeric_cols:
            st.warning("No numeric columns.")
        else:

            time_candidates = [
                c for c in df.columns
                if any(
                    word in str(c).lower()
                    for word in [
                        "year",
                        "date",
                        "month",
                        "time",
                        "season",
                    ]
                )
            ]

            x = st.selectbox(
                "Time / X Axis",
                time_candidates
                if time_candidates
                else df.columns.tolist()
            )

            y = st.selectbox(
                "Metric",
                numeric_cols
            )

            fig = line_chart(
                df,
                x,
                y
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    elif chart_type == "Bar Chart":

        if not categorical_cols:
            st.warning(
                "A categorical column is required."
            )

        elif not numeric_cols:
            st.warning(
                "A numeric column is required."
            )

        else:

            c1, c2 = st.columns(2)

            with c1:
                category = st.selectbox(
                    "Category",
                    categorical_cols
                )

            with c2:
                value = st.selectbox(
                    "Numeric Value",
                    numeric_cols
                )

            aggregation = st.selectbox(
                "Aggregation",
                ["mean", "sum", "count"]
            )

            fig = bar_chart(
                df,
                category,
                value,
                aggregation
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        if len(numeric_cols) < 2:
            st.warning(
                "At least two numeric columns are required."
            )
        else:

            fig = correlation_heatmap(
                df
            )

            if fig is not None:
                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


# ============================================================
# CORRELATION
# ============================================================

elif page == "🔗 Correlation Intelligence":

    st.header("🔗 Correlation Intelligence")

    if len(nums) < 2:

        st.warning(
            "At least two numeric columns are required."
        )

    else:

        corr = correlation_matrix(
            df[nums]
        )

        st.subheader("🔥 Correlation Matrix")

        st.dataframe(
            corr.round(3),
            use_container_width=True
        )

        st.subheader("🔥 Heatmap")

        fig = correlation_heatmap(df)

        if fig is not None:
            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("🔗 Strong Relationships")

        rel = unexpected_relationships(
            df,
            min_corr=min_corr
        )

        if rel.empty:
            st.info(
                "No strong relationships found."
            )
        else:
            st.dataframe(
                rel,
                use_container_width=True
            )


# ============================================================
# ANOMALIES
# ============================================================

elif page == "🚨 Anomaly Detection":

    st.header("🚨 Anomaly Detection")

    if len(nums) < 2:

        st.warning(
            "At least two numeric columns are required."
        )

    else:

        result, summary = detect_anomalies(
            df,
            nums,
            contamination
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "🚨 Anomalies",
            summary["count"]
        )

        c2.metric(
            "✅ Normal Rows",
            len(df) - summary["count"]
        )

        st.dataframe(
            result,
            use_container_width=True
        )

        st.download_button(
            "📥 Download Anomalies",
            result.to_csv(
                index=False
            ).encode("utf-8"),
            "agri_anomalies.csv",
            "text/csv"
        )


# ============================================================
# CLUSTERS
# ============================================================

elif page == "🧩 Cluster Discovery":

    st.header("🧩 Cluster Discovery")

    if len(nums) < 2:

        st.warning(
            "At least two numeric columns are required."
        )

    else:

        clustered, centers = cluster_data(
            df,
            nums,
            k
        )

        st.metric(
            "🧩 Clusters",
            k
        )

        st.subheader("Cluster Summary")

        st.dataframe(
            clustered
            .groupby("Cluster")[nums]
            .mean()
            .round(2),
            use_container_width=True
        )

        x = st.selectbox(
            "X Axis",
            nums,
            key="cluster_x"
        )

        y = st.selectbox(
            "Y Axis",
            nums,
            index=min(1, len(nums)-1),
            key="cluster_y"
        )

        st.scatter_chart(
            clustered,
            x=x,
            y=y,
            color="Cluster"
        )

        st.subheader("Cluster Centers")

        st.dataframe(
            centers.round(2),
            use_container_width=True
        )


# ============================================================
# TREND
# ============================================================

elif page == "📈 Trend Intelligence":

    st.header("📈 Trend Intelligence")

    if not nums:

        st.warning(
            "No numeric columns found."
        )

    else:

        time_candidates = [
            c for c in df.columns
            if any(
                word in str(c).lower()
                for word in [
                    "year",
                    "date",
                    "month",
                    "time",
                    "season",
                ]
            )
        ]

        time_column = st.selectbox(
            "Time Column",
            time_candidates
            if time_candidates
            else df.columns.tolist()
        )

        metric = st.selectbox(
            "Metric",
            nums
        )

        try:

            trend = trend_analysis(
                df,
                time_column,
                metric
            )

            if trend.empty:

                st.warning(
                    "No usable trend data found."
                )

            else:

                st.line_chart(
                    trend.set_index(
                        "Time"
                    )[["Value"]]
                )

                st.dataframe(
                    trend,
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Trend analysis error: {e}"
            )


# ============================================================
# PATTERN DISCOVERY
# ============================================================

elif page == "🔎 Pattern Discovery":

    st.header("🔎 Pattern Discovery")

    rel = unexpected_relationships(
        df,
        min_corr=min_corr
    )

    if rel.empty:

        st.info(
            "No strong relationships discovered."
        )

    else:

        for _, row in rel.head(10).iterrows():

            direction = (
                "positive"
                if row["Correlation"] > 0
                else "negative"
            )

            st.markdown(
                f"""
                **{row['Column A']} ↔ {row['Column B']}**

                {direction.capitalize()} relationship ·
                correlation = `{row['Correlation']:.2f}`
                """
            )

    if len(nums) >= 2:

        _, anomaly_summary = detect_anomalies(
            df,
            nums,
            contamination
        )

        st.divider()

        st.metric(
            "🚨 Anomaly Signal",
            anomaly_summary["count"]
        )


# ============================================================
# PHASE 4 — PREDICTION STUDIO
# ============================================================

elif page == "🤖 Prediction Studio":

    st.header("🤖 Crop Yield Prediction Studio")

    st.caption(
        "Train and compare machine-learning regression models "
        "to predict agricultural yield or production."
    )

    # --------------------------------------------------------
    # Target detection
    # --------------------------------------------------------

    detected_targets = detect_target_columns(
        df
    )

    suggested = suggest_target(
        df
    )

    st.subheader("🎯 Prediction Target")

    if detected_targets:

        st.success(
            f"Detected prediction targets: "
            f"{', '.join(map(str, detected_targets))}"
        )

    else:

        st.info(
            "No obvious Yield/Production column was detected. "
            "Select a numeric column manually."
        )

    numeric_targets = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_targets:

        st.error(
            "Prediction requires at least one numeric target."
        )

    else:

        default_index = 0

        if suggested in numeric_targets:

            default_index = numeric_targets.index(
                suggested
            )

        target = st.selectbox(
            "Select target variable",
            numeric_targets,
            index=default_index
        )

        st.write(
            f"**Selected target:** `{target}`"
        )

        # ----------------------------------------------------
        # Settings
        # ----------------------------------------------------

        st.subheader("⚙️ Training Settings")

        c1, c2 = st.columns(2)

        with c1:

            test_size = st.slider(
                "Test data (%)",
                10,
                40,
                20
            ) / 100

        with c2:

            random_state = st.number_input(
                "Random State",
                min_value=1,
                max_value=9999,
                value=42
            )

        st.divider()

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        if st.button(
            "🚀 Train Prediction Models",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Training Linear Regression, Random Forest "
                "and Gradient Boosting models..."
            ):

                try:

                    results, X, y = train_all_models(
                        df,
                        target,
                        test_size=test_size,
                        random_state=random_state
                    )

                    st.session_state[
                        "prediction_results"
                    ] = results

                    st.session_state[
                        "prediction_target"
                    ] = target

                    st.session_state[
                        "prediction_features"
                    ] = X.columns.tolist()

                    st.success(
                        "✅ All prediction models have been trained."
                    )

                except Exception as e:

                    st.error(
                        f"❌ Model training failed: {e}"
                    )

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        results = st.session_state.get(
            "prediction_results"
        )

        if results:

            st.divider()

            st.subheader(
                "🏆 Model Performance"
            )

            table = comparison_table(
                results
            )

            st.dataframe(
                table.style.format(
                    {
                        "R²": "{:.4f}",
                        "MAE": "{:.4f}",
                        "RMSE": "{:.4f}",
                        "MAPE (%)": "{:.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # Best model
            # ------------------------------------------------

            best = get_best_model(
                results
            )

            if best:

                best_result = results[
                    best
                ]

                st.success(
                    f"🏆 Best Model: **{best}**  "
                    f"with R² = "
                    f"**{best_result['metrics']['R²']:.4f}**"
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "R²",
                    f"{best_result['metrics']['R²']:.4f}"
                )

                c2.metric(
                    "MAE",
                    f"{best_result['metrics']['MAE']:.4f}"
                )

                c3.metric(
                    "RMSE",
                    f"{best_result['metrics']['RMSE']:.4f}"
                )

                c4.metric(
                    "MAPE",
                    f"{best_result['metrics']['MAPE (%)']:.2f}%"
                )

                # --------------------------------------------
                # Actual vs Predicted
                # --------------------------------------------

                st.divider()

                st.subheader(
                    "🎯 Actual vs Predicted"
                )

                avp = actual_vs_predicted(
                    best_result["y_test"],
                    best_result["predictions"]
                )

                st.line_chart(
                    avp[
                        ["Actual", "Predicted"]
                    ]
                )

                st.subheader(
                    "📋 Prediction Results"
                )

                st.dataframe(
                    avp,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "📥 Download Test Predictions",
                    avp.to_csv(
                        index=False
                    ).encode("utf-8"),
                    "agripattern_predictions.csv",
                    "text/csv"
                )

                # --------------------------------------------
                # Residual plot
                # --------------------------------------------

                st.subheader(
                    "📉 Prediction Error"
                )

                st.scatter_chart(
                    avp,
                    x="Actual",
                    y="Residual"
                )

                # --------------------------------------------
                # Feature importance
                # --------------------------------------------

                st.subheader(
                    "🔥 Feature Importance"
                )

                importance = get_feature_importance(
                    best_result["pipeline"]
                )

                if importance is not None:

                    st.dataframe(
                        importance.head(30),
                        use_container_width=True,
                        hide_index=True
                    )

                    chart_data = (
                        importance
                        .head(15)
                        .set_index("Feature")
                        ["Importance"]
                    )

                    st.bar_chart(
                        chart_data
                    )

                else:

                    st.info(
                        "Feature importance is available "
                        "for tree-based models such as "
                        "Random Forest and Gradient Boosting."
                    )

            # ------------------------------------------------
            # Interactive Prediction
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🔮 Interactive Prediction"
            )

            st.write(
                "Enter agricultural values below and "
                "AgriPattern will predict the selected target."
            )

            feature_columns = st.session_state.get(
                "prediction_features",
                []
            )

            input_values = {}

            input_cols = st.columns(
                min(3, max(1, len(feature_columns)))
            )

            for index, feature in enumerate(
                feature_columns
            ):

                with input_cols[
                    index % len(input_cols)
                ]:

                    original = df[feature]

                    if pd.api.types.is_numeric_dtype(
                        original
                    ):

                        median_value = pd.to_numeric(
                            original,
                            errors="coerce"
                        ).median()

                        if pd.isna(
                            median_value
                        ):
                            median_value = 0.0

                        input_values[
                            feature
                        ] = st.number_input(
                            str(feature),
                            value=float(
                                median_value
                            )
                        )

                    else:

                        values = (
                            original
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )

                        if values:

                            input_values[
                                feature
                            ] = st.selectbox(
                                str(feature),
                                values
                            )

                        else:

                            input_values[
                                feature
                            ] = ""

            if st.button(
                "🔮 Predict",
                type="primary"
            ):

                try:

                    best_pipeline = results[
                        best
                    ]["pipeline"]

                    prediction_input = pd.DataFrame(
                        [input_values]
                    )

                    prediction = predict_with_model(
                        best_pipeline,
                        prediction_input
                    )

                    value = float(
                        prediction[0]
                    )

                    st.success(
                        f"🎯 Predicted **{target}**: "
                        f"**{value:,.2f}**"
                    )

                except Exception as e:

                    st.error(
                        f"Prediction failed: {e}"
                    )


# ============================================================
# AI INSIGHTS
# ============================================================

elif page == "🧠 AI Insights":

    st.header("🧠 AI Insights")

    rel = unexpected_relationships(
        df,
        min_corr=min_corr
    )

    if len(nums) >= 2:

        _, anomaly_summary = detect_anomalies(
            df,
            nums,
            contamination
        )

    else:

        anomaly_summary = {
            "count": 0
        }

    prompt_data = {
        "rows": len(df),
        "columns": list(df.columns),
        "strong_relationships":
            rel.head(10).to_dict("records"),
        "anomalies":
            anomaly_summary["count"],
    }

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Dataset Rows",
        len(df)
    )

    c2.metric(
        "Strong Relationships",
        len(rel)
    )

    c3.metric(
        "Anomalies",
        anomaly_summary["count"]
    )

    st.divider()

    if st.button(
        "🤖 Generate AI Explanation",
        type="primary"
    ):

        with st.spinner(
            "Groq is analyzing the dataset..."
        ):

            try:

                text = explain_patterns(
                    prompt_data
                )

                st.markdown(
                    text
                )

            except Exception as e:

                st.error(
                    f"AI analysis failed: {e}"
                )


# ============================================================
# REPORT
# ============================================================

elif page == "📄 Report Generator":

    st.header(
        "📄 Report Generator"
    )

    st.info(
        "Export your agricultural dataset and analysis profile."
    )

    st.subheader(
        "📊 Dataset Profile"
    )

    st.dataframe(
        profile["column_profile"],
        use_container_width=True
    )

    st.download_button(
        "📥 Export Dataset CSV",
        df.to_csv(
            index=False
        ).encode("utf-8"),
        "agripattern_dataset.csv",
        "text/csv"
    )

    st.download_button(
        "📥 Export Profile CSV",
        profile["column_profile"]
        .to_csv(index=False)
        .encode("utf-8"),
        "agripattern_profile.csv",
        "text/csv"
    )