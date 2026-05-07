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


def benchmark_models(data, contamination=0.05, model_weights: dict | None = None):
    """Run all models and return benchmark results with auto-selected best model.

    The "best" model is chosen by silhouette score multiplied by an optional
    preset weight (model_weights[model_key] defaults to 1.0). This lets
    presets bias selection toward a model better suited to that data type
    without removing the silhouette signal entirely.
    """
    runners = {
        "isolation_forest": lambda: run_isolation_forest(data, contamination),
        "lof": lambda: run_lof(data, contamination),
        "ocsvm": lambda: run_ocsvm(data, nu=contamination),
    }
    weights = model_weights or {}

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
        def weighted_score(r):
            w = float(weights.get(r["model_key"], 1.0))
            # Map silhouette from [-1, 1] to [0, 1] before scaling
            normed = (r["silhouette"] + 1) / 2 if r["silhouette"] is not None else 0
            return normed * w
        best = max(successful, key=weighted_score)
        best_key = best["model_key"]
    else:
        best_key = "isolation_forest"

    return results, best_key


# ── Ensemble ────────────────────────────────────────────────────────────────
def _percentile_rank(scores: np.ndarray) -> np.ndarray:
    """Return per-element percentile rank in [0, 1] (higher = more anomalous)."""
    s = pd.Series(scores)
    return s.rank(pct=True, method="average").to_numpy()


def compute_ensemble(
    bench_results: list[dict],
    contamination: float = 0.05,
    model_weights: dict | None = None,
) -> dict:
    """Combine per-model scores into one ensemble signal.

    Returns dict with:
        ensemble_score: np.array — weighted average of percentile ranks (0..1)
        any_top_mask:   bool array — True if record is in top contamination of ANY model
        per_model_pct:  dict[model_key] -> np.array of percentile ranks
        per_model_top:  dict[model_key] -> bool array of "in this model's top-q"
    """
    weights = model_weights or {}
    successful = [r for r in bench_results if r.get("success") and r.get("scores") is not None]
    if not successful:
        return {
            "ensemble_score": None,
            "any_top_mask": None,
            "per_model_pct": {},
            "per_model_top": {},
        }

    per_model_pct = {}
    per_model_top = {}
    weighted_sum = None
    weight_total = 0.0
    any_top = None

    for r in successful:
        key = r["model_key"]
        scores = np.asarray(r["scores"], dtype=float)
        pct = _percentile_rank(scores)
        per_model_pct[key] = pct
        threshold = np.quantile(scores, 1 - contamination)
        top_mask = scores >= threshold
        per_model_top[key] = top_mask
        any_top = top_mask if any_top is None else (any_top | top_mask)

        w = float(weights.get(key, 1.0))
        weighted_sum = pct * w if weighted_sum is None else weighted_sum + pct * w
        weight_total += w

    ensemble = weighted_sum / weight_total if weight_total > 0 else weighted_sum
    return {
        "ensemble_score": ensemble,
        "any_top_mask": any_top,
        "per_model_pct": per_model_pct,
        "per_model_top": per_model_top,
    }


# ── Explanations ────────────────────────────────────────────────────────────
def get_top_deviating_features(row_original, df_original, selected_columns, top_k=5):
    """For a single row, find which numeric features deviate most from the median.

    Returns a DataFrame with feature name, row value, typical (median) value,
    and a 'deviation' percentage showing how far the value is from typical.
    Backward-compatible API.
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


def explain_record(
    row_original: pd.Series,
    df_original: pd.DataFrame,
    selected_columns: list[str],
    rare_threshold: float = 0.01,
    combo_columns: list[str] | None = None,
    dq_flags_row: pd.Series | None = None,
    top_k: int = 8,
) -> list[dict]:
    """Generate human-readable reasons for a record being flagged.

    Returns a list of dicts: {kind, column, message, score} where kind is one of
    {"numeric", "kategorik", "kombinasyon", "kural"}. Sorted by descending score.
    """
    reasons: list[dict] = []

    for col in selected_columns:
        if col not in df_original.columns:
            continue
        if col not in row_original.index:
            continue
        val = row_original[col]
        if pd.isna(val):
            continue
        ser = df_original[col]

        if pd.api.types.is_numeric_dtype(ser):
            median = ser.median()
            std = ser.std()
            if std and not pd.isna(std) and std > 0:
                z = abs(val - median) / std
                direction = "yüksek" if val > median else "düşük"
                if z >= 2:
                    reasons.append({
                        "kind": "numeric",
                        "column": col,
                        "message": (
                            f"**{col}** değeri **{val:.2f}** — tipik **{median:.2f}**'den "
                            f"{z:.1f}σ daha {direction}"
                        ),
                        "score": float(z),
                    })
        else:
            # Categorical: rare-value check
            freq = ser.value_counts(normalize=True, dropna=False)
            f = float(freq.get(val, 0.0))
            if f > 0 and f < rare_threshold:
                reasons.append({
                    "kind": "kategorik",
                    "column": col,
                    "message": (
                        f"**{col}** = `{val}` veri setinde **%{f * 100:.2f}** sıklıkta — "
                        f"nadir görülen bir değer"
                    ),
                    "score": float((rare_threshold - f) / rare_threshold * 5),
                })
            elif f == 0:
                reasons.append({
                    "kind": "kategorik",
                    "column": col,
                    "message": f"**{col}** = `{val}` — veri setinde başka örneği yok",
                    "score": 5.0,
                })

    if combo_columns and len(combo_columns) >= 2:
        if all(c in df_original.columns for c in combo_columns):
            joined = df_original[combo_columns].astype(str).agg("|".join, axis=1)
            row_combo = "|".join(str(row_original.get(c, "")) for c in combo_columns)
            joint_freq = joined.value_counts(normalize=True)
            f = float(joint_freq.get(row_combo, 0.0))
            if 0 < f < rare_threshold:
                pretty = " + ".join(f"{c}=`{row_original.get(c, '')}`" for c in combo_columns)
                reasons.append({
                    "kind": "kombinasyon",
                    "column": " + ".join(combo_columns),
                    "message": (
                        f"Kombinasyon ({pretty}) veri setinde **%{f * 100:.2f}** "
                        f"sıklıkta — olağan dışı"
                    ),
                    "score": float((rare_threshold - f) / rare_threshold * 4),
                })

    if dq_flags_row is not None:
        for flag_name, val in dq_flags_row.items():
            if not val:
                continue
            try:
                rule, _, column = flag_name.partition("__")
            except Exception:
                rule, column = flag_name, ""
            reasons.append({
                "kind": "kural",
                "column": column,
                "message": f"Kural ihlali: **{rule.replace('_', ' ')}**" + (f" ({column})" if column else ""),
                "score": 6.0,
            })

    reasons.sort(key=lambda r: r["score"], reverse=True)
    return reasons[:top_k]


# ── Time series & semi-supervised (unchanged) ──────────────────────────────
def detect_temporal_anomalies(df, date_col, value_col, window=14):
    """Detect anomalies in a time series using rolling statistics."""
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
