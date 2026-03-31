import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def detect_column_types(df: pd.DataFrame) -> dict:
    """Classify columns as numeric or categorical."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return {"numeric": numeric_cols, "categorical": categorical_cols}


def analyze_column_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze each column and flag quality issues.

    Flags:
    - id_like: unique ratio > 0.9 for non-numeric or > 0.95 for numeric
    - constant: only 1 unique value (excluding NaN)
    - high_cardinality: categorical with > 50 unique values
    - free_text: object column with high avg word count (>3 words)
    """
    rows = []
    n = len(df)
    for col in df.columns:
        nunique = df[col].nunique()
        unique_ratio = nunique / n if n > 0 else 0
        dtype = str(df[col].dtype)
        is_numeric = pd.api.types.is_numeric_dtype(df[col])

        flags = []

        # Constant column
        if nunique <= 1:
            flags.append("constant")

        # ID-like detection
        if is_numeric and unique_ratio > 0.95:
            flags.append("id_like")
        elif not is_numeric and unique_ratio > 0.9:
            flags.append("id_like")

        # Free text detection (object columns with avg >3 words per value)
        if not is_numeric and nunique > 1:
            sample = df[col].dropna().head(200).astype(str)
            avg_words = sample.str.split().str.len().mean() if len(sample) > 0 else 0
            if avg_words > 3:
                flags.append("free_text")

        # High cardinality
        if not is_numeric and nunique > 50 and "id_like" not in flags:
            flags.append("high_cardinality")

        rows.append({
            "column": col,
            "dtype": dtype,
            "unique": nunique,
            "unique_ratio": round(unique_ratio, 3),
            "missing": int(df[col].isna().sum()),
            "missing_pct": round(df[col].isna().sum() / n * 100, 1) if n > 0 else 0,
            "flags": ", ".join(flags) if flags else "-",
            "recommended": "id_like" not in flags and "constant" not in flags and "free_text" not in flags,
        })

    return pd.DataFrame(rows)


def get_safe_columns(quality_df: pd.DataFrame) -> list[str]:
    """Return columns that passed quality checks (no id_like, constant, free_text)."""
    return quality_df.loc[quality_df["recommended"], "column"].tolist()


def preprocess(
    df: pd.DataFrame,
    selected_columns: list[str],
    max_onehot_categories: int = 15,
) -> tuple[pd.DataFrame, StandardScaler, list[str]]:
    """Preprocess selected columns for anomaly detection.

    Returns:
        processed: scaled DataFrame ready for modelling
        scaler: fitted StandardScaler
        feature_names: final column names after encoding
    """
    data = df[selected_columns].copy()

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

    # ── Fill missing values ──────────────────────────────────────────────────
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())
    for col in categorical_cols:
        mode = data[col].mode()
        data[col] = data[col].fillna(mode.iloc[0] if not mode.empty else "UNKNOWN")

    # ── Encode categoricals ──────────────────────────────────────────────────
    encoded_parts = [data[numeric_cols]]
    encoded_cat_names = []

    for col in categorical_cols:
        nunique = data[col].nunique()
        if nunique <= max_onehot_categories:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="if_binary")
            transformed = ohe.fit_transform(data[[col]])
            cat_names = [f"{col}_{cat}" for cat in ohe.categories_[0]]
            if transformed.shape[1] < len(cat_names):
                # drop="if_binary" removed one column
                cat_names = [f"{col}_{cat}" for cat in ohe.get_feature_names_out([col])]
                cat_names = [c.replace(f"{col}_{col}_", f"{col}_") for c in cat_names]
            ohe_df = pd.DataFrame(transformed, columns=cat_names[:transformed.shape[1]], index=data.index)
            encoded_parts.append(ohe_df)
            encoded_cat_names.extend(cat_names[:transformed.shape[1]])
        else:
            # High cardinality: use frequency encoding
            freq_map = data[col].value_counts(normalize=True).to_dict()
            freq_col = data[col].map(freq_map).fillna(0)
            freq_df = pd.DataFrame({f"{col}_freq": freq_col}, index=data.index)
            encoded_parts.append(freq_df)
            encoded_cat_names.append(f"{col}_freq")

    data_encoded = pd.concat(encoded_parts, axis=1)
    feature_names = list(data_encoded.columns)

    # ── Scale ────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(data_encoded)
    processed = pd.DataFrame(scaled_array, columns=feature_names, index=data.index)

    return processed, scaler, feature_names


def compute_feature_contributions(
    processed_df: pd.DataFrame,
    original_df: pd.DataFrame,
    selected_columns: list[str],
) -> pd.DataFrame:
    """Compute per-row z-score based feature contributions.

    For each row, returns the absolute z-score of each feature.
    Higher values indicate the feature deviates more from the norm.
    """
    # Use the processed (already z-scored) data — absolute values show deviation
    abs_z = processed_df.abs()
    return abs_z
