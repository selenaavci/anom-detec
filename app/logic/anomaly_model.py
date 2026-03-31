import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


MODELS = {
    "Isolation Forest": "isolation_forest",
    "Local Outlier Factor": "lof",
    "One-Class SVM": "ocsvm",
}


def run_isolation_forest(
    data: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[np.ndarray, IsolationForest]:
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    model.fit(data)
    scores = -model.decision_function(data) 
    return scores, model


def run_lof(
    data: pd.DataFrame,
    contamination: float = 0.05,
    n_neighbors: int = 20,
) -> tuple[np.ndarray, LocalOutlierFactor]:
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
    )
    model.fit_predict(data)
    scores = -model.negative_outlier_factor_
    return scores, model


def run_ocsvm(
    data: pd.DataFrame,
    nu: float = 0.05,
) -> tuple[np.ndarray, OneClassSVM]:
    model = OneClassSVM(kernel="rbf", nu=nu)
    model.fit(data)
    scores = -model.decision_function(data)
    return scores, model


def detect_anomalies(
    data: pd.DataFrame,
    model_name: str = "isolation_forest",
    contamination: float = 0.05,
) -> tuple[pd.Series, object]:
    """Run selected anomaly detection model. Returns (scores, fitted_model)."""
    if model_name == "isolation_forest":
        scores, model = run_isolation_forest(data, contamination=contamination)
    elif model_name == "lof":
        scores, model = run_lof(data, contamination=contamination)
    elif model_name == "ocsvm":
        scores, model = run_ocsvm(data, nu=contamination)
    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")

    return pd.Series(scores, index=data.index, name="anomaly_score"), model


def get_top_contributing_features(
    processed_row: pd.Series,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return top-k features with highest absolute z-score for a single row."""
    abs_z = processed_row.abs().sort_values(ascending=False).head(top_k)
    contrib = pd.DataFrame({
        "feature": abs_z.index,
        "z_score": abs_z.values,
        "direction": ["+" if processed_row[f] > 0 else "-" for f in abs_z.index],
    })
    return contrib


def run_semi_supervised(
    data: pd.DataFrame,
    labels: pd.Series,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.Series:
    """Semi-supervised anomaly detection using feedback.

    Strategy:
    - Rows marked as normal (label=0) are used to fit the model.
    - Rows marked as anomaly (label=1) are excluded from training.
    - Unlabeled rows (label=-1) are included in training.
    - Scores are computed for ALL rows.
    """

    train_mask = labels != 1
    train_data = data.loc[train_mask]

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=150,
    )
    model.fit(train_data)

    # Score all rows
    scores = -model.decision_function(data)
    return pd.Series(scores, index=data.index, name="anomaly_score")
