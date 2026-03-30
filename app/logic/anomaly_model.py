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
) -> np.ndarray:
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    model.fit(data)
    return -model.decision_function(data)  # higher = more anomalous


def run_lof(
    data: pd.DataFrame,
    contamination: float = 0.05,
    n_neighbors: int = 20,
) -> np.ndarray:
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
    )
    model.fit_predict(data)
    return -model.negative_outlier_factor_


def run_ocsvm(
    data: pd.DataFrame,
    nu: float = 0.05,
) -> np.ndarray:
    model = OneClassSVM(kernel="rbf", nu=nu)
    model.fit(data)
    return -model.decision_function(data)


def detect_anomalies(
    data: pd.DataFrame,
    model_name: str = "isolation_forest",
    contamination: float = 0.05,
) -> pd.Series:
    """Run selected anomaly detection model and return anomaly scores."""
    if model_name == "isolation_forest":
        scores = run_isolation_forest(data, contamination=contamination)
    elif model_name == "lof":
        scores = run_lof(data, contamination=contamination)
    elif model_name == "ocsvm":
        scores = run_ocsvm(data, nu=contamination)
    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")

    return pd.Series(scores, index=data.index, name="anomaly_score")
