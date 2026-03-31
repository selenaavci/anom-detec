import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def detect_column_types(df: pd.DataFrame) -> dict:
    """Classify columns as numeric, categorical, or datetime."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    for col in list(categorical_cols):
        sample = df[col].dropna().head(50)
        if len(sample) == 0:
            continue
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            if parsed.notna().sum() / len(sample) > 0.8:
                datetime_cols.append(col)
                categorical_cols.remove(col)
        except Exception:
            pass

    return {"numeric": numeric_cols, "categorical": categorical_cols, "datetime": datetime_cols}


def analyze_column_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze each column and flag quality issues."""
    rows = []
    n = len(df)
    for col in df.columns:
        nunique = df[col].nunique()
        unique_ratio = nunique / n if n > 0 else 0
        is_numeric = pd.api.types.is_numeric_dtype(df[col])

        flags = []
        flag_reasons = []

        if nunique <= 1:
            flags.append("constant")
            flag_reasons.append("Tek bir değer içeriyor, analiz için bilgi taşımaz")

        if is_numeric and unique_ratio > 0.95:
            flags.append("id_like")
            flag_reasons.append("Her satırda neredeyse farklı bir değer var — muhtemelen ID sütunu")
        elif not is_numeric and unique_ratio > 0.9:
            flags.append("id_like")
            flag_reasons.append("Her satırda neredeyse farklı bir değer var — muhtemelen ID sütunu")

        if not is_numeric and nunique > 1:
            sample = df[col].dropna().head(200).astype(str)
            avg_words = sample.str.split().str.len().mean() if len(sample) > 0 else 0
            if avg_words > 3:
                flags.append("free_text")
                flag_reasons.append("Serbest metin alanı — anomali tespitine uygun değil")

        if not is_numeric and nunique > 50 and "id_like" not in flags:
            flags.append("high_cardinality")
            flag_reasons.append("Çok fazla kategori var, otomatik olarak frekans bazlı dönüşüm uygulanacak")

        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "unique": nunique,
            "unique_ratio": round(unique_ratio, 3),
            "missing": int(df[col].isna().sum()),
            "missing_pct": round(df[col].isna().sum() / n * 100, 1) if n > 0 else 0,
            "flags": ", ".join(flags) if flags else "-",
            "flag_reasons": " | ".join(flag_reasons) if flag_reasons else "",
            "recommended": "id_like" not in flags and "constant" not in flags and "free_text" not in flags,
        })

    return pd.DataFrame(rows)


def detect_id_columns(df: pd.DataFrame, quality_df: pd.DataFrame) -> list[str]:
    """Detect columns that are likely identifiers.

    Two signals:
    1. Column name contains 'id' (case-insensitive, whole word-like patterns)
    2. Column was flagged as 'id_like' by quality analysis
    """
    id_cols = set()

    # Name-based: "id", "ID", "_id", "Id", columns ending/starting with id
    import re
    id_pattern = re.compile(r'(?:^|[_\- ])id(?:[_\- ]|$)|^id$|_id$|Id$|ID$|^id_|^ID_', re.IGNORECASE)
    for col in df.columns:
        if id_pattern.search(col):
            id_cols.add(col)

    # Quality-based: flagged as id_like
    flagged_ids = quality_df.loc[quality_df["flags"].str.contains("id_like", na=False), "column"].tolist()
    id_cols.update(flagged_ids)

    return list(id_cols)


def get_safe_columns(quality_df: pd.DataFrame) -> list[str]:
    """Return columns that passed quality checks."""
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

    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())
    for col in categorical_cols:
        mode = data[col].mode()
        data[col] = data[col].fillna(mode.iloc[0] if not mode.empty else "UNKNOWN")

    encoded_parts = [data[numeric_cols]]

    for col in categorical_cols:
        nunique = data[col].nunique()
        if nunique <= max_onehot_categories:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="if_binary")
            transformed = ohe.fit_transform(data[[col]])
            cat_names = list(ohe.get_feature_names_out([col]))
            ohe_df = pd.DataFrame(transformed, columns=cat_names, index=data.index)
            encoded_parts.append(ohe_df)
        else:
            freq_map = data[col].value_counts(normalize=True).to_dict()
            freq_col = data[col].map(freq_map).fillna(0)
            encoded_parts.append(pd.DataFrame({f"{col}_freq": freq_col}, index=data.index))

    data_encoded = pd.concat(encoded_parts, axis=1)
    feature_names = list(data_encoded.columns)

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(data_encoded)
    processed = pd.DataFrame(scaled_array, columns=feature_names, index=data.index)

    return processed, scaler, feature_names
