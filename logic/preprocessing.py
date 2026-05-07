import re
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def detect_column_types(df: pd.DataFrame) -> dict:
    """Classify columns as numeric, categorical, or datetime."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    import warnings
    for col in list(categorical_cols):
        sample = df[col].dropna().head(50)
        if len(sample) == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce")
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
    """Detect columns that are likely identifiers."""
    id_cols = set()

    id_pattern = re.compile(
        r'(?:^|[_\- ])id(?:[_\- ]|$)|^id$|_id$|Id$|ID$|^id_|^ID_',
        re.IGNORECASE,
    )
    for col in df.columns:
        if id_pattern.search(col):
            id_cols.add(col)

    flagged_ids = quality_df.loc[quality_df["flags"].str.contains("id_like", na=False), "column"].tolist()
    id_cols.update(flagged_ids)

    return list(id_cols)


def get_safe_columns(quality_df: pd.DataFrame) -> list[str]:
    """Return columns that passed quality checks (kept for backward compat)."""
    return quality_df.loc[quality_df["recommended"], "column"].tolist()


# ── Column grouping (prefix-based) ──────────────────────────────────────────
_SPLIT_RE = re.compile(r"[_\.\-]")


def detect_column_groups(columns: list[str], min_group: int = 2) -> dict[str, list[str]]:
    """Detect prefix-based logical groups in a list of column names.

    Splits names on `_`, `.` or `-` and groups by the first token if at least
    `min_group` columns share that prefix. Singletons fall under "Diğer".
    """
    by_prefix: dict[str, list[str]] = defaultdict(list)
    for col in columns:
        parts = _SPLIT_RE.split(col, maxsplit=1)
        prefix = parts[0] if parts else col
        by_prefix[prefix].append(col)

    groups: dict[str, list[str]] = {}
    others: list[str] = []
    for prefix, cols in by_prefix.items():
        if len(cols) >= min_group:
            groups[prefix] = cols
        else:
            others.extend(cols)
    if others:
        groups["Diğer"] = others
    return groups


# ── Balanced default selection ──────────────────────────────────────────────
def propose_balanced_selection(
    df: pd.DataFrame,
    quality_df: pd.DataFrame,
    id_cols: list[str],
    max_columns: int = 30,
    max_high_card_cat: int = 6,
) -> tuple[list[str], list[dict]]:
    """Build a balanced default analysis set: numerics + balanced categoricals.

    Returns:
        selected: column list
        coverage_report: list of dicts per column with `included` and `reason`.
    """
    flag_lookup = quality_df.set_index("column").to_dict(orient="index")

    selected: list[str] = []
    report: list[dict] = []

    def add(col: str, reason_in: str) -> None:
        selected.append(col)
        report.append({"column": col, "included": True, "reason": reason_in})

    def skip(col: str, reason_out: str) -> None:
        report.append({"column": col, "included": False, "reason": reason_out})

    high_card_added = 0

    for col in df.columns:
        info = flag_lookup.get(col, {})
        flags = info.get("flags", "-")
        if col in id_cols:
            skip(col, "ID/Referans olarak işaretlendi")
            continue
        if "constant" in flags:
            skip(col, "Tek değer içeriyor (bilgi taşımaz)")
            continue
        if "id_like" in flags:
            skip(col, "ID benzeri (her satırda neredeyse farklı)")
            continue
        if "free_text" in flags:
            skip(col, "Serbest metin alanı")
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            add(col, "Sayısal alan")
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            skip(col, "Tarih alanı (zaman serisi panelinde kullanılır)")
            continue

        nunique = info.get("unique", df[col].nunique())
        if nunique <= 30:
            add(col, "Düşük kardinaliteli kategorik")
        elif nunique <= 200:
            if high_card_added < max_high_card_cat:
                add(col, "Orta kardinaliteli kategorik (frekans-encode)")
                high_card_added += 1
            else:
                skip(col, "Orta kardinaliteli — kapsam dengesi için dışarıda")
        else:
            skip(col, f"Yüksek kardinaliteli ({nunique} kategori) — modele eklenmedi")

        if len(selected) >= max_columns:
            # remaining: mark skipped with capacity reason
            break

    if len(selected) >= max_columns:
        for col in df.columns:
            if col in selected:
                continue
            already = any(r["column"] == col for r in report)
            if not already:
                report.append({
                    "column": col,
                    "included": False,
                    "reason": f"Default kapsam üst sınırı ({max_columns}) doldu",
                })
    return selected, report


# ── Rare category & rule-based features ─────────────────────────────────────
def compute_rare_category_features(
    df: pd.DataFrame,
    columns: list[str],
    rare_threshold: float = 0.005,
) -> pd.DataFrame:
    """For each column, generate two features:
        {col}_is_rare  — 1 if the value's frequency < rare_threshold
        {col}_freq     — relative frequency of the value in the column
    """
    out = pd.DataFrame(index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        freq = df[col].value_counts(normalize=True, dropna=False)
        mapped = df[col].map(freq).fillna(0).astype(float)
        out[f"{col}_freq"] = mapped
        out[f"{col}_is_rare"] = (mapped < rare_threshold).astype(int)
    return out


def compute_combo_frequency(
    df: pd.DataFrame,
    combo_cols: list[str],
    rare_threshold: float = 0.005,
) -> pd.DataFrame:
    """Produce a combo-frequency + rare-combo feature for a tuple of columns.

    Example: combo_cols = ['cur', 'chnl', 'ctg'] → joint frequency of the
    triple (currency, channel, type) per row. Rare combos get is_rare=1.
    """
    if not combo_cols or any(c not in df.columns for c in combo_cols):
        return pd.DataFrame(index=df.index)
    key = "__|__".join(combo_cols)
    joined = df[combo_cols].astype(str).agg("|".join, axis=1)
    freq = joined.value_counts(normalize=True, dropna=False)
    mapped = joined.map(freq).fillna(0).astype(float)
    out = pd.DataFrame({
        f"combo_{key}_freq": mapped,
        f"combo_{key}_is_rare": (mapped < rare_threshold).astype(int),
    }, index=df.index)
    return out


_NAME_HINTS_CAT = ("name", "isim", "ad", "soyad", "unvan")
_BRANCH_HINTS = ("brchn", "branch", "sube", "şube", "office", "location")
_COUNTRY_TOKENS = (
    "İNGİLTERE", "AMERİKA", "ALMANYA", "FRANSA", "İTALYA", "İSPANYA",
    "ABD", "ENGLAND", "USA", "GERMANY", "FRANCE", "ITALY",
    "TURKİYE", "CHINA", "ÇİN", "RUSSIA", "RUSYA",
)
_CONTROL_RE = re.compile(r"[\r\n\t\x00-\x08\x0b\x0c\x0e-\x1f]")
_DIGITS_ONLY_RE = re.compile(r"^\s*\d+\s*$")


def compute_rule_features(
    df: pd.DataFrame,
    selected_columns: list[str],
) -> pd.DataFrame:
    """Generate rule-based binary features that flag obvious data anomalies.

    Per numeric column:
        {col}_is_negative, {col}_is_zero, {col}_is_extreme (z>4)
    Per name-like categorical:
        {col}_digits_only       (e.g. "ODEA BANK A.Ş." → "01460146")
        {col}_has_control_char  (e.g. "0\\n\\n\\n.90")
    Per branch-like categorical:
        {col}_country_in_branch (e.g. branch = "İNGİLTERE")
    """
    out = pd.DataFrame(index=df.index)
    for col in selected_columns:
        if col not in df.columns:
            continue
        ser = df[col]
        if pd.api.types.is_numeric_dtype(ser):
            out[f"{col}_is_negative"] = (ser < 0).fillna(False).astype(int)
            out[f"{col}_is_zero"] = (ser == 0).fillna(False).astype(int)
            mu, sigma = ser.mean(), ser.std()
            if sigma and not pd.isna(sigma) and sigma > 0:
                z = (ser - mu).abs() / sigma
                out[f"{col}_is_extreme"] = (z > 4).fillna(False).astype(int)
            else:
                out[f"{col}_is_extreme"] = 0
            continue

        # Categorical: name-like / branch-like rules
        col_l = col.lower()
        s_str = ser.astype(str)
        if any(h in col_l for h in _NAME_HINTS_CAT):
            out[f"{col}_digits_only"] = s_str.str.match(_DIGITS_ONLY_RE, na=False).astype(int)
            out[f"{col}_has_control_char"] = s_str.str.contains(_CONTROL_RE, regex=True, na=False).astype(int)
        if any(h in col_l for h in _BRANCH_HINTS):
            up = s_str.str.upper()
            country_mask = up.apply(lambda x: any(tok in x for tok in _COUNTRY_TOKENS))
            out[f"{col}_country_in_branch"] = country_mask.astype(int)
    return out


# ── Preprocessing pipeline ──────────────────────────────────────────────────
def preprocess(
    df: pd.DataFrame,
    selected_columns: list[str],
    max_onehot_categories: int = 15,
    rare_category_columns: list[str] | None = None,
    combo_columns: list[str] | None = None,
    add_rule_features: bool = False,
    extra_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, StandardScaler, list[str]]:
    """Preprocess selected columns for anomaly detection.

    Args:
        rare_category_columns: list of cat cols to attach rare/freq features for.
        combo_columns: tuple of cat cols whose joint frequency becomes a feature.
        add_rule_features: include rule-based binary features.
        extra_features: caller-supplied dataframe (e.g. data-quality flags) merged in.

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

    if rare_category_columns:
        rare_part = compute_rare_category_features(df, rare_category_columns)
        if not rare_part.empty:
            encoded_parts.append(rare_part)

    if combo_columns and len(combo_columns) >= 2:
        combo_part = compute_combo_frequency(df, combo_columns)
        if not combo_part.empty:
            encoded_parts.append(combo_part)

    if add_rule_features:
        rule_part = compute_rule_features(df, selected_columns)
        if not rule_part.empty:
            encoded_parts.append(rule_part)

    if extra_features is not None and not extra_features.empty:
        extra_aligned = extra_features.reindex(data.index).fillna(0)
        encoded_parts.append(extra_aligned)

    data_encoded = pd.concat(encoded_parts, axis=1)
    # Deduplicate identical column names that may arise from extra features
    data_encoded = data_encoded.loc[:, ~data_encoded.columns.duplicated()]
    feature_names = list(data_encoded.columns)

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(data_encoded)
    processed = pd.DataFrame(scaled_array, columns=feature_names, index=data.index)

    return processed, scaler, feature_names
