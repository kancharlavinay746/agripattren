import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.ensemble import IsolationForest


# ============================================================
# PREPARE NUMERIC DATA
# ============================================================

def prepare_numeric_data(df, columns):

    valid_columns = [
        c for c in columns
        if c in df.columns
    ]

    if len(valid_columns) < 2:
        return pd.DataFrame(), pd.Index([])

    temp = df[valid_columns].copy()

    # Protect against duplicate column names
    cleaned = {}

    for column in valid_columns:

        series = temp[column]

        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        cleaned[column] = pd.to_numeric(
            series,
            errors="coerce"
        )

    temp = pd.DataFrame(cleaned)

    # Replace infinite values
    temp = temp.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid_mask = temp.notna().all(axis=1)

    clean = temp.loc[valid_mask]

    return clean, clean.index


# ============================================================
# STANDARDIZATION
# ============================================================

def standardize_data(df, columns):

    clean, indices = prepare_numeric_data(
        df,
        columns
    )

    if clean.empty:
        return pd.DataFrame(), indices

    scaler = StandardScaler()

    X = scaler.fit_transform(clean)

    X_scaled = pd.DataFrame(
        X,
        columns=clean.columns,
        index=clean.index
    )

    return X_scaled, indices


# ============================================================
# PCA
# ============================================================

def pca_analysis(
    df,
    columns,
    n_components=2,
    clusters=3
):

    X_scaled, indices = standardize_data(
        df,
        columns
    )

    if X_scaled.empty:
        return None

    if len(X_scaled) < clusters:
        return None

    # PCA
    pca = PCA(
        n_components=n_components,
        random_state=42
    )

    components = pca.fit_transform(
        X_scaled
    )

    # KMeans
    kmeans = KMeans(
        n_clusters=clusters,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        X_scaled
    )

    result = pd.DataFrame(
        {
            "PC1": components[:, 0],
            "PC2": components[:, 1],
            "Cluster": labels.astype(str),
        },
        index=indices
    )

    variance = pca.explained_variance_ratio_

    explained_variance = {
        "PC1": float(variance[0]),
        "PC2": float(variance[1]),
        "Total": float(variance.sum()),
    }

    return {
        "data": result,
        "pca": pca,
        "explained_variance": explained_variance,
        "components": pca.components_,
        "features": list(X_scaled.columns),
    }


# ============================================================
# ELBOW METHOD
# ============================================================

def elbow_analysis(
    df,
    columns,
    min_k=2,
    max_k=10
):

    X_scaled, _ = standardize_data(
        df,
        columns
    )

    if X_scaled.empty:
        return pd.DataFrame()

    max_k = min(
        max_k,
        len(X_scaled) - 1
    )

    if max_k < min_k:
        return pd.DataFrame()

    values = []

    for k in range(
        min_k,
        max_k + 1
    ):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X_scaled)

        values.append(
            {
                "K": k,
                "Inertia": float(
                    model.inertia_
                )
            }
        )

    return pd.DataFrame(values)


# ============================================================
# SILHOUETTE ANALYSIS
# ============================================================

def silhouette_analysis(
    df,
    columns,
    min_k=2,
    max_k=10
):

    X_scaled, _ = standardize_data(
        df,
        columns
    )

    if X_scaled.empty:
        return pd.DataFrame()

    max_k = min(
        max_k,
        len(X_scaled) - 1
    )

    values = []

    for k in range(
        min_k,
        max_k + 1
    ):

        if k >= len(X_scaled):
            continue

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(
            X_scaled
        )

        score = silhouette_score(
            X_scaled,
            labels
        )

        values.append(
            {
                "K": k,
                "Silhouette Score": float(
                    score
                )
            }
        )

    return pd.DataFrame(values)


# ============================================================
# SILHOUETTE DETAILS
# ============================================================

def silhouette_detail(
    df,
    columns,
    k
):

    X_scaled, indices = standardize_data(
        df,
        columns
    )

    if X_scaled.empty:
        return pd.DataFrame()

    if k >= len(X_scaled):
        return pd.DataFrame()

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        X_scaled
    )

    scores = silhouette_samples(
        X_scaled,
        labels
    )

    result = pd.DataFrame(
        {
            "Cluster": labels,
            "Silhouette": scores,
        },
        index=indices
    )

    return result


# ============================================================
# ANOMALY VISUALIZATION DATA
# ============================================================

def anomaly_visualization(
    df,
    columns,
    contamination=0.05
):

    X_scaled, indices = standardize_data(
        df,
        columns
    )

    if X_scaled.empty:
        return pd.DataFrame()

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    predictions = model.fit_predict(
        X_scaled
    )

    scores = model.decision_function(
        X_scaled
    )

    result = df.loc[
        indices
    ].copy()

    result["Anomaly"] = np.where(
        predictions == -1,
        "Anomaly",
        "Normal"
    )

    result["Anomaly Score"] = scores

    return result


# ============================================================
# ANOMALY SUMMARY
# ============================================================

def anomaly_summary(
    anomaly_df
):

    if anomaly_df.empty:
        return {
            "total": 0,
            "anomalies": 0,
            "normal": 0,
            "percentage": 0.0
        }

    total = len(anomaly_df)

    anomalies = int(
        (
            anomaly_df["Anomaly"]
            == "Anomaly"
        ).sum()
    )

    normal = total - anomalies

    percentage = (
        anomalies / total * 100
    )

    return {
        "total": total,
        "anomalies": anomalies,
        "normal": normal,
        "percentage": percentage
    }