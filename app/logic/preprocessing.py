import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def detect_column_types(df: pd.DataFrame) -> dict:
    """Classify columns as numeric or categorical."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return {"numeric": numeric_cols, "categorical": categorical_cols}


def preprocess(df: pd.DataFrame, selected_columns: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    """Preprocess selected columns for anomaly detection.

    Steps:
    1. Subset to selected columns
    2. Handle missing values
    3. Encode categorical variables
    4. Scale numeric features

    Returns the processed DataFrame and the fitted scaler.
    """
    data = df[selected_columns].copy()

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

    # Fill missing values
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode().iloc[0] if not data[col].mode().empty else "UNKNOWN")

    # Encode categoricals
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Scale all features
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(data)
    processed = pd.DataFrame(scaled_array, columns=data.columns, index=data.index)

    return processed, scaler
