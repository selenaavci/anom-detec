import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score


MODELS = {
    "Isolation Forest": "isolation_forest",
    "Local Outlier Factor": "lof",
    "One-Class SVM": "ocsvm",
}

MODEL_DESCRIPTIONS = {
    "isolation_forest": "Veriyi rastgele bölerek sıra dışı kayıtları hızlıca ayırır.",
    "lof": "Her kaydı komşularıyla karşılaştırarak yoğunluk farkı arar.",
    "ocsvm": "Normal verilerin sınırını çizerek dışında kalanları bulur.",
}


def run_isolation_forest(data, contamination=0.05, random_state=42):
    model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=100)
    model.fit(data)
    return -model.decision_function(data)


def run_lof(data, contamination=0.05, n_neighbors=20):
    n_neighbors = min(n_neighbors, len(data) - 1)
    model = LocalOutlierFactor(n_neighbors=max(n_neighbors, 2), contamination=contamination, novelty=False)
    model.fit_predict(data)
    return -model.negative_outlier_factor_


def run_ocsvm(data, nu=0.05):
    model = OneClassSVM(kernel="rbf", nu=nu)
    model.fit(data)
    return -model.decision_function(data)


def _label_from_scores(scores, contamination):
    threshold = np.quantile(scores, 1 - contamination)
    return (scores >= threshold).astype(int)


def benchmark_models(data, contamination=0.05):
    """Run all models and return benchmark results with auto-selected best model.

    Returns:
        results: list of dicts with model name, scores, and quality metric
        best_model_key: key of the best performing model
    """
    runners = {
        "isolation_forest": lambda: run_isolation_forest(data, contamination),
        "lof": lambda: run_lof(data, contamination),
        "ocsvm": lambda: run_ocsvm(data, nu=contamination),
    }

    results = []
    for key, runner in runners.items():
        try:
            scores = runner()
            labels = _label_from_scores(scores, contamination)

            n_unique = len(np.unique(labels))
            if n_unique > 1 and len(data) > 10:
                sample_size = min(len(data), 5000)
                if sample_size < len(data):
                    idx = np.random.RandomState(42).choice(len(data), sample_size, replace=False)
                    sil = silhouette_score(data.iloc[idx], labels[idx])
                else:
                    sil = silhouette_score(data, labels)
            else:
                sil = -1.0

            results.append({
                "model_key": key,
                "scores": scores,
                "labels": labels,
                "silhouette": round(sil, 4),
                "n_anomalies": int(labels.sum()),
                "success": True,
            })
        except Exception as e:
            results.append({
                "model_key": key,
                "scores": None,
                "labels": None,
                "silhouette": -1.0,
                "n_anomalies": 0,
                "success": False,
                "error": str(e),
            })

    successful = [r for r in results if r["success"]]
    if successful:
        best = max(successful, key=lambda r: r["silhouette"])
        best_key = best["model_key"]
    else:
        best_key = "isolation_forest"

    return results, best_key


def get_top_deviating_features(row_original, df_original, selected_columns, top_k=5):
    """For a single row, find which original features deviate most from the median.

    Returns a DataFrame with feature name, row value, typical (median) value,
    and a 'deviation' percentage showing how far the value is from typical.
    """
    records = []
    for col in selected_columns:
        if col not in df_original.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df_original[col]):
            continue

        val = row_original[col] if col in row_original.index else None
        if val is None or pd.isna(val):
            continue

        median = df_original[col].median()
        std = df_original[col].std()
        if std == 0 or pd.isna(std):
            continue

        deviation = abs(val - median) / std
        records.append({
            "feature": col,
            "value": val,
            "typical": round(median, 2),
            "deviation": round(deviation, 2),
        })

    if not records:
        return pd.DataFrame(columns=["feature", "value", "typical", "deviation"])

    result = pd.DataFrame(records).sort_values("deviation", ascending=False).head(top_k)
    return result


def detect_temporal_anomalies(df, date_col, value_col, window=14):
    """Detect anomalies in a time series using rolling statistics.

    Flags points that fall outside rolling_mean ± 2.5 * rolling_std.
    Returns the original df augmented with rolling stats and anomaly flags.
    """
    ts = df[[date_col, value_col]].copy()
    ts = ts.sort_values(date_col).reset_index(drop=True)
    ts[date_col] = pd.to_datetime(ts[date_col])

    ts["rolling_mean"] = ts[value_col].rolling(window=window, center=True, min_periods=3).mean()
    ts["rolling_std"] = ts[value_col].rolling(window=window, center=True, min_periods=3).std()
    ts["rolling_std"] = ts["rolling_std"].fillna(ts[value_col].std())

    ts["upper"] = ts["rolling_mean"] + 2.5 * ts["rolling_std"]
    ts["lower"] = ts["rolling_mean"] - 2.5 * ts["rolling_std"]
    ts["is_anomaly"] = (ts[value_col] > ts["upper"]) | (ts[value_col] < ts["lower"])

    return ts


def run_semi_supervised(data, labels, contamination=0.05, random_state=42):
    """Semi-supervised: train only on rows not marked as anomaly."""
    train_mask = labels != 1
    train_data = data.loc[train_mask]

    model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=150)
    model.fit(train_data)
    scores = -model.decision_function(data)
    return pd.Series(scores, index=data.index, name="anomaly_score")
