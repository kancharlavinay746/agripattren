# src/analysis.py

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# COLUMN UTILITIES
# ============================================================

def make_unique_columns(columns):
    """
    Make duplicate dataframe column names unique.
    Example:
        Yield, Yield -> Yield, Yield_1
    """

    result = []
    counts = {}

    for col in columns:
        col = str(col).strip()

        if col not in counts:
            counts[col] = 0
            result.append(col)
        else:
            counts[col] += 1
            result.append(f"{col}_{counts[col]}")

    return result


def numeric_columns(df):
    """Return numeric columns safely."""

    if df is None or df.empty:
        return []

    return df.select_dtypes(
        include=np.number
    ).columns.tolist()


# ============================================================
# DATASET PROFILE
# ============================================================

def dataset_profile(df):

    nums = numeric_columns(df)

    column_profile = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(df[c].dtype)
            for c in df.columns
        ],
        "Non-Null": [
            int(df[c].notna().sum())
            for c in df.columns
        ],
        "Missing": [
            int(df[c].isna().sum())
            for c in df.columns
        ],
        "Unique Values": [
            int(df[c].nunique(dropna=True))
            for c in df.columns
        ],
    })

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric": nums,
        "missing_cells": int(
            df.isna().sum().sum()
        ),
        "column_profile": column_profile,
    }


# ============================================================
# CORRELATION
# ============================================================

def correlation_matrix(df):

    if df is None or df.empty:
        return pd.DataFrame()

    return df.corr(
        numeric_only=True
    )


def unexpected_relationships(
    df,
    min_corr=0.60,
):

    nums = numeric_columns(df)

    if len(nums) < 2:
        return pd.DataFrame(
            columns=[
                "Column A",
                "Column B",
                "Correlation",
            ]
        )

    corr = df[nums].corr()

    relationships = []

    for i in range(len(nums)):

        for j in range(i + 1, len(nums)):

            value = corr.iloc[i, j]

            if pd.notna(value) and abs(value) >= min_corr:

                relationships.append({
                    "Column A": nums[i],
                    "Column B": nums[j],
                    "Correlation": round(
                        float(value),
                        4,
                    ),
                })

    result = pd.DataFrame(
        relationships
    )

    if not result.empty:
        result["Absolute Correlation"] = (
            result["Correlation"]
            .abs()
        )

        result = result.sort_values(
            "Absolute Correlation",
            ascending=False,
        ).drop(
            columns=["Absolute Correlation"]
        )

    return result.reset_index(drop=True)


# ============================================================
# ANOMALY DETECTION
# ============================================================

def detect_anomalies(
    df,
    columns=None,
    contamination=0.05,
):

    if columns is None:
        columns = numeric_columns(df)

    columns = [
        c for c in columns
        if c in df.columns
    ]

    if len(columns) < 2:
        return (
            df.copy(),
            {
                "count": 0,
                "percentage": 0,
            },
        )

    work = df[columns].copy()

    work = work.apply(
        pd.to_numeric,
        errors="coerce",
    )

    work = work.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    work = work.dropna(
        how="all"
    )

    if len(work) < 5:
        result = df.copy()
        result["Anomaly"] = 0

        return (
            result,
            {
                "count": 0,
                "percentage": 0,
            },
        )

    imputer = SimpleImputer(
        strategy="median"
    )

    X = imputer.fit_transform(work)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
    )

    predictions = model.fit_predict(
        X_scaled
    )

    result = df.copy()

    anomaly_series = pd.Series(
        0,
        index=result.index,
        dtype=int,
    )

    anomaly_indices = work.index[
        predictions == -1
    ]

    anomaly_series.loc[
        anomaly_indices
    ] = 1

    result["Anomaly"] = anomaly_series

    count = int(
        (result["Anomaly"] == 1).sum()
    )

    percentage = (
        count / len(result) * 100
        if len(result)
        else 0
    )

    return (
        result,
        {
            "count": count,
            "percentage": round(
                percentage,
                2,
            ),
        },
    )


# ============================================================
# CLUSTERING
# ============================================================

def cluster_data(
    df,
    columns,
    k=3,
):

    columns = [
        c for c in columns
        if c in df.columns
    ]

    if len(columns) < 2:
        return (
            df.copy(),
            pd.DataFrame(),
        )

    work = df[columns].copy()

    work = work.apply(
        pd.to_numeric,
        errors="coerce",
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    X = imputer.fit_transform(work)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    k = max(
        2,
        min(k, len(X_scaled)),
    )

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(
        X_scaled
    )

    result = df.copy()

    result["Cluster"] = labels

    centers_scaled = model.cluster_centers_

    centers = scaler.inverse_transform(
        centers_scaled
    )

    centers_df = pd.DataFrame(
        centers,
        columns=columns,
    )

    centers_df.index.name = "Cluster"

    return (
        result,
        centers_df,
    )


# ============================================================
# TREND ANALYSIS
# ============================================================

def trend_analysis(
    df,
    time_column,
    metric,
):

    if (
        time_column not in df.columns
        or metric not in df.columns
    ):
        return pd.DataFrame(
            columns=["Time", "Value"]
        )

    # Prevent duplicate-column problems
    time_series = df[time_column]

    metric_series = df[metric]

    # If duplicate names somehow still exist,
    # safely choose first series.
    if isinstance(time_series, pd.DataFrame):
        time_series = time_series.iloc[:, 0]

    if isinstance(metric_series, pd.DataFrame):
        metric_series = metric_series.iloc[:, 0]

    temp = pd.DataFrame({
        "Time": time_series,
        "Value": metric_series,
    })

    temp["Value"] = pd.to_numeric(
        temp["Value"],
        errors="coerce",
    )

    # Try datetime first
    parsed_time = pd.to_datetime(
        temp["Time"],
        errors="coerce",
    )

    if parsed_time.notna().sum() >= max(
        2,
        int(len(temp) * 0.5),
    ):
        temp["Time"] = parsed_time
    else:
        numeric_time = pd.to_numeric(
            temp["Time"],
            errors="coerce",
        )

        if numeric_time.notna().sum() > 0:
            temp["Time"] = numeric_time

    temp = temp.dropna(
        subset=["Time", "Value"]
    )

    if temp.empty:
        return pd.DataFrame(
            columns=["Time", "Value"]
        )

    temp = (
        temp.groupby("Time", as_index=False)
        ["Value"]
        .mean()
        .sort_values("Time")
    )

    return temp.reset_index(
        drop=True
    )


# ============================================================
# KPI METRICS
# ============================================================

def kpi_metrics(df):

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": len(
            numeric_columns(df)
        ),
        "missing_cells": int(
            df.isna().sum().sum()
        ),
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
    }


# ============================================================
# FORECASTING UTILITIES
# ============================================================

def detect_time_column(df):
    """
    Automatically identify the most likely time column.
    """

    if df is None or df.empty:
        return None

    priority_names = [
        "year",
        "date",
        "datetime",
        "time",
        "month",
        "season",
    ]

    # Exact / priority matches
    for name in priority_names:

        for col in df.columns:

            if str(col).strip().lower() == name:
                return col

    # Partial matches
    for col in df.columns:

        lower = str(col).lower()

        if any(
            word in lower
            for word in [
                "year",
                "date",
                "time",
                "month",
                "season",
            ]
        ):
            return col

    return None


def prepare_forecast_data(
    df,
    time_column,
    target_column,
):
    """
    Prepare a clean chronological series.
    """

    if (
        time_column not in df.columns
        or target_column not in df.columns
    ):
        return pd.DataFrame(
            columns=[
                "Time",
                "Target",
            ]
        )

    time_data = df[time_column]
    target_data = df[target_column]

    if isinstance(
        time_data,
        pd.DataFrame,
    ):
        time_data = time_data.iloc[:, 0]

    if isinstance(
        target_data,
        pd.DataFrame,
    ):
        target_data = target_data.iloc[:, 0]

    temp = pd.DataFrame({
        "Time": time_data,
        "Target": target_data,
    })

    temp["Target"] = pd.to_numeric(
        temp["Target"],
        errors="coerce",
    )

    # First attempt: datetime
    parsed = pd.to_datetime(
        temp["Time"],
        errors="coerce",
    )

    valid_datetime_ratio = (
        parsed.notna().mean()
    )

    if valid_datetime_ratio >= 0.5:

        temp["Time"] = parsed

    else:

        numeric_time = pd.to_numeric(
            temp["Time"],
            errors="coerce",
        )

        temp["Time"] = numeric_time

    temp = temp.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    temp = temp.dropna(
        subset=[
            "Time",
            "Target",
        ]
    )

    if temp.empty:
        return temp

    temp = (
        temp.groupby(
            "Time",
            as_index=False,
        )["Target"]
        .mean()
        .sort_values("Time")
    )

    return temp.reset_index(
        drop=True
    )


def create_time_features(
    time_values,
):
    """
    Convert dates/numeric periods into ML-friendly
    features.
    """

    values = pd.Series(
        time_values
    ).reset_index(
        drop=True
    )

    if pd.api.types.is_datetime64_any_dtype(
        values
    ):

        features = pd.DataFrame({
            "year": values.dt.year,
            "month": values.dt.month,
            "quarter": values.dt.quarter,
            "dayofyear": values.dt.dayofyear,
            "time_index": np.arange(
                len(values)
            ),
        })

    else:

        numeric = pd.to_numeric(
            values,
            errors="coerce",
        )

        features = pd.DataFrame({
            "time_value": numeric,
            "time_index": np.arange(
                len(values)
            ),
        })

    return features.astype(float)


# ============================================================
# FORECAST MODEL TRAINING
# ============================================================

def train_forecast_models(
    forecast_df,
    test_size=0.20,
):
    """
    Train:
      - Linear Regression
      - Random Forest
      - Gradient Boosting

    Returns evaluation information and fitted models.
    """

    if forecast_df is None:
        return {
            "success": False,
            "message": "No forecast data.",
        }

    if len(forecast_df) < 8:
        return {
            "success": False,
            "message": (
                "At least 8 historical observations "
                "are recommended for forecasting."
            ),
        }

    work = forecast_df.copy()

    X = create_time_features(
        work["Time"]
    )

    y = pd.to_numeric(
        work["Target"],
        errors="coerce",
    )

    valid = (
        X.notna().all(axis=1)
        & y.notna()
    )

    X = X.loc[valid].reset_index(
        drop=True
    )

    y = y.loc[valid].reset_index(
        drop=True
    )

    times = work.loc[
        valid,
        "Time"
    ].reset_index(
        drop=True
    )

    if len(X) < 8:
        return {
            "success": False,
            "message": (
                "Not enough valid observations "
                "after cleaning."
            ),
        }

    split_index = int(
        len(X) * (1 - test_size)
    )

    split_index = max(
        5,
        min(
            split_index,
            len(X) - 2,
        ),
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    models = {
        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                min_samples_leaf=2,
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            ),
    }

    evaluation = []
    fitted_models = {}

    for name, model in models.items():

        try:

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                predictions,
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions,
                )
            )

            r2 = r2_score(
                y_test,
                predictions,
            )

            evaluation.append({
                "Model": name,
                "MAE": round(
                    float(mae),
                    4,
                ),
                "RMSE": round(
                    float(rmse),
                    4,
                ),
                "R²": round(
                    float(r2),
                    4,
                ),
            })

            fitted_models[name] = model

        except Exception as exc:

            evaluation.append({
                "Model": name,
                "MAE": np.nan,
                "RMSE": np.nan,
                "R²": np.nan,
                "Error": str(exc),
            })

    evaluation_df = pd.DataFrame(
        evaluation
    )

    valid_models = evaluation_df[
        evaluation_df["RMSE"].notna()
    ]

    if valid_models.empty:
        return {
            "success": False,
            "message": (
                "All forecasting models failed."
            ),
        }

    best_row = valid_models.sort_values(
        "RMSE"
    ).iloc[0]

    best_model_name = best_row[
        "Model"
    ]

    best_model = fitted_models[
        best_model_name
    ]

    test_predictions = best_model.predict(
        X_test
    )

    prediction_df = pd.DataFrame({
        "Time": times.iloc[
            split_index:
        ],
        "Actual": y_test.values,
        "Predicted": test_predictions,
    })

    return {
        "success": True,
        "models": fitted_models,
        "evaluation": evaluation_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "prediction_df": prediction_df,
        "X": X,
        "y": y,
        "times": times,
        "feature_names": X.columns.tolist(),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


# ============================================================
# FUTURE FORECAST
# ============================================================

def generate_future_forecast(
    forecast_df,
    trained_result,
    periods=5,
):
    """
    Generate future predictions using the selected
    best model.
    """

    if not trained_result.get(
        "success",
        False,
    ):
        return pd.DataFrame()

    model = trained_result[
        "best_model"
    ]

    times = forecast_df[
        "Time"
    ]

    if len(times) == 0:
        return pd.DataFrame()

    last_time = times.iloc[-1]

    # --------------------------------------------------------
    # DATETIME FUTURE
    # --------------------------------------------------------

    if pd.api.types.is_datetime64_any_dtype(
        times
    ):

        freq = pd.infer_freq(
            times
        )

        if freq is None:

            if len(times) >= 2:

                delta = (
                    times.iloc[-1]
                    - times.iloc[-2]
                )

                if delta <= pd.Timedelta(0):
                    delta = pd.Timedelta(
                        days=365
                    )

            else:

                delta = pd.Timedelta(
                    days=365
                )

            future_times = [
                last_time + delta * i
                for i in range(
                    1,
                    periods + 1,
                )
            ]

        else:

            future_times = pd.date_range(
                start=last_time,
                periods=periods + 1,
                freq=freq,
            )[1:]

    # --------------------------------------------------------
    # NUMERIC TIME
    # --------------------------------------------------------

    else:

        numeric_times = pd.to_numeric(
            times,
            errors="coerce",
        )

        step = 1

        if len(numeric_times) >= 2:

            differences = (
                numeric_times.diff()
                .dropna()
            )

            if not differences.empty:

                median_step = differences.median()

                if pd.notna(
                    median_step
                ) and median_step != 0:

                    step = median_step

        future_times = [
            last_time + step * i
            for i in range(
                1,
                periods + 1,
            )
        ]

    future_features = create_time_features(
        future_times
    )

    predictions = model.predict(
        future_features
    )

    result = pd.DataFrame({
        "Time": future_times,
        "Forecast": predictions,
    })

    return result


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def forecast_feature_importance(
    trained_result,
):
    """
    Return model feature importance where available.
    """

    if not trained_result.get(
        "success",
        False,
    ):
        return pd.DataFrame()

    model = trained_result[
        "best_model"
    ]

    features = trained_result[
        "feature_names"
    ]

    if hasattr(
        model,
        "feature_importances_",
    ):

        importance = model.feature_importances_

    elif hasattr(
        model,
        "coef_",
    ):

        importance = np.abs(
            model.coef_
        )

    else:

        return pd.DataFrame()

    result = pd.DataFrame({
        "Feature": features,
        "Importance": importance,
    })

    return result.sort_values(
        "Importance",
        ascending=False,
    ).reset_index(
        drop=True
    )