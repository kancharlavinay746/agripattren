"""
AgriPattern - Phase 4
Crop Yield Prediction & Model Intelligence

Provides:
- Automatic target detection
- Data preprocessing
- Multiple regression models
- Model comparison
- Evaluation metrics
- Feature importance
- Predictions
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# TARGET DETECTION
# ============================================================

TARGET_KEYWORDS = [
    "yield",
    "production",
    "crop_yield",
    "crop yield",
    "output",
    "harvest",
]


def detect_target_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect likely prediction target columns.
    """

    if df is None or df.empty:
        return []

    targets = []

    for col in df.columns:

        name = str(col).strip().lower()

        if any(keyword in name for keyword in TARGET_KEYWORDS):

            if pd.api.types.is_numeric_dtype(df[col]):
                targets.append(col)

    return targets


def suggest_target(df: pd.DataFrame):
    """
    Return the most likely target column.
    """

    targets = detect_target_columns(df)

    if targets:
        # Prefer exact "Yield"
        for col in targets:
            if str(col).strip().lower() == "yield":
                return col

        return targets[0]

    # Fallback: last numeric column
    numeric = df.select_dtypes(include=np.number).columns.tolist()

    if numeric:
        return numeric[-1]

    return None


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_prediction_data(
    df: pd.DataFrame,
    target: str,
):
    """
    Prepare X and y for machine learning.
    """

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    working = df.copy()

    # Remove completely empty columns
    working = working.dropna(axis=1, how="all")

    # Remove duplicate column names safely
    working = working.loc[:, ~working.columns.duplicated()]

    # Target must be numeric
    y = pd.to_numeric(
        working[target],
        errors="coerce"
    )

    valid = y.notna()

    working = working.loc[valid].copy()
    y = y.loc[valid].copy()

    # Remove target
    X = working.drop(columns=[target])

    # Remove columns with no useful values
    X = X.dropna(axis=1, how="all")

    # Remove columns with only one unique value
    constant_cols = [
        c for c in X.columns
        if X[c].nunique(dropna=True) <= 1
    ]

    if constant_cols:
        X = X.drop(columns=constant_cols)

    if X.empty:
        raise ValueError(
            "No usable feature columns remain after preprocessing."
        )

    return X, y


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(X: pd.DataFrame):
    """
    Create preprocessing pipeline for numeric and categorical data.
    """

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    if not transformers:
        raise ValueError(
            "No usable numeric or categorical features found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ============================================================
# MODELS
# ============================================================

def get_models(random_state: int = 42):
    """
    Return supported regression models.
    """

    return {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Calculate regression evaluation metrics.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    # Avoid division by zero
    denominator = np.where(
        np.abs(y_true) < 1e-10,
        1e-10,
        np.abs(y_true)
    )

    mape = np.mean(
        np.abs(
            (y_true - y_pred) /
            denominator
        )
    ) * 100

    return {
        "R²": float(r2),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE (%)": float(mape),
    }


# ============================================================
# TRAIN SINGLE MODEL
# ============================================================

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train one regression model.
    """

    if len(X) < 10:
        raise ValueError(
            "At least 10 valid rows are recommended for prediction."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = create_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    return {
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": metrics,
    }


# ============================================================
# TRAIN ALL MODELS
# ============================================================

def train_all_models(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train all supported models.
    """

    X, y = prepare_prediction_data(
        df,
        target
    )

    models = get_models(
        random_state
    )

    results = {}

    for name, model in models.items():

        try:

            result = train_model(
                X,
                y,
                model,
                test_size=test_size,
                random_state=random_state,
            )

            results[name] = result

        except Exception as exc:

            results[name] = {
                "error": str(exc)
            }

    return results, X, y


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

def comparison_table(results):
    """
    Convert model results to a DataFrame.
    """

    rows = []

    for name, result in results.items():

        if "error" in result:
            rows.append(
                {
                    "Model": name,
                    "R²": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "MAPE (%)": np.nan,
                    "Status": "Failed",
                }
            )

        else:

            rows.append(
                {
                    "Model": name,
                    "R²": result["metrics"]["R²"],
                    "MAE": result["metrics"]["MAE"],
                    "RMSE": result["metrics"]["RMSE"],
                    "MAPE (%)": result["metrics"]["MAPE (%)"],
                    "Status": "Success",
                }
            )

    table = pd.DataFrame(rows)

    if not table.empty:
        table = table.sort_values(
            "R²",
            ascending=False,
            na_position="last"
        )

    return table.reset_index(
        drop=True
    )


# ============================================================
# BEST MODEL
# ============================================================

def get_best_model(results):
    """
    Select the model with highest R².
    """

    valid = []

    for name, result in results.items():

        if "error" not in result:

            r2 = result["metrics"]["R²"]

            if np.isfinite(r2):
                valid.append(
                    (name, r2)
                )

    if not valid:
        return None

    valid.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return valid[0][0]


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    pipeline,
):
    """
    Extract feature importance from tree models.
    """

    if pipeline is None:
        return None

    model = pipeline.named_steps.get(
        "model"
    )

    preprocessor = pipeline.named_steps.get(
        "preprocessor"
    )

    if not hasattr(
        model,
        "feature_importances_"
    ):
        return None

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        importances = (
            model
            .feature_importances_
        )

        result = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances,
            }
        )

        # Clean sklearn prefixes
        result["Feature"] = (
            result["Feature"]
            .str.replace(
                "numeric__",
                "",
                regex=False
            )
            .str.replace(
                "categorical__",
                "",
                regex=False
            )
        )

        return (
            result
            .sort_values(
                "Importance",
                ascending=False
            )
            .reset_index(drop=True)
        )

    except Exception:
        return None


# ============================================================
# PREDICTION
# ============================================================

def predict_with_model(
    pipeline,
    input_data: pd.DataFrame,
):
    """
    Generate predictions for new observations.
    """

    if pipeline is None:
        raise ValueError(
            "A trained model is required."
        )

    if input_data is None or input_data.empty:
        raise ValueError(
            "Prediction input is empty."
        )

    return pipeline.predict(
        input_data
    )


# ============================================================
# ACTUAL VS PREDICTED DATA
# ============================================================

def actual_vs_predicted(
    y_true,
    predictions,
):
    """
    Return dataframe for visualization.
    """

    return pd.DataFrame(
        {
            "Actual": np.asarray(y_true),
            "Predicted": np.asarray(predictions),
            "Residual": (
                np.asarray(y_true)
                -
                np.asarray(predictions)
            ),
        }
    )


# ============================================================
# FEATURE DESCRIPTIONS
# ============================================================

def get_feature_columns(
    df: pd.DataFrame,
    target: str,
):
    """
    Return usable prediction features.
    """

    X, _ = prepare_prediction_data(
        df,
        target
    )

    return X.columns.tolist()