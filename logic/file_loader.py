import zipfile
import io
import re
import pandas as pd
import numpy as np
import streamlit as st


SUPPORTED_DATA_EXTENSIONS = (".csv", ".xls", ".xlsx", ".xml")


# ── Column kind taxonomy ────────────────────────────────────────────────────
# Kullanıcıya UI'da gösterilen "Kolon Tipi" seçenekleri.
KIND_NUMERIC = "Sayısal/Tutar"
KIND_CATEGORICAL = "Kategorik/Metin"
KIND_DATETIME = "Tarih"
KIND_ID = "ID/Referans"
KIND_EXCLUDED = "Analiz dışı"

ALL_KINDS = (KIND_NUMERIC, KIND_CATEGORICAL, KIND_DATETIME, KIND_ID, KIND_EXCLUDED)


# ── Name-based detection signals (multi-language) ───────────────────────────
NUMERIC_NAME_HINTS = (
    "amount", "amt", "tutar", "balance", "bakiye", "blnc",
    "price", "fiyat", "rate", "oran", "score", "skor", "puan",
    "count", "adet", "sayisi", "sayı", "miktar", "qty", "quantity",
    "fee", "ucret", "ücret", "tax", "vergi", "interest", "faiz",
    "cost", "maliyet", "value", "deger", "değer",
    "duration", "sure", "süre", "age", "yas", "yaş",
    "amnttl", "amntfc", "salary", "maas", "maaş",
    "revenue", "gelir", "expense", "gider",
)

DATETIME_NAME_HINTS = (
    "date", "tarih", "time", "saat", "datetime", "timestamp",
    "created", "updated", "olusturma", "oluşturma", "guncelleme",
    "guncellendi", "year", "yil", "yıl", "month", "ay",
)

ID_NAME_HINTS = (
    "id", "uuid", "ref", "kod", "code", "no", "number",
    "tcno", "tckn", "iban", "accno", "acct", "msisdn",
    "kimlik", "guid",
)


def _name_has_hint(name: str, hints: tuple[str, ...]) -> bool:
    n = str(name).lower()
    for h in hints:
        if h in n:
            return True
    return False


def _is_id_name(name: str) -> bool:
    """Stricter ID name pattern (avoids matching e.g. 'video' for 'id')."""
    n = str(name).lower()
    pat = re.compile(r"(?:^|[_\-\s])id(?:[_\-\s]|$)|^id$|_id$|^id_|"
                     r"(?:^|[_\-\s])(uuid|ref|kod|code|no|number|tcno|tckn|"
                     r"iban|accno|acct|msisdn|guid|kimlik)(?:[_\-\s]|$)")
    return bool(pat.search(n))


# ── Numeric coercion ────────────────────────────────────────────────────────
_NEWLINE_RE = re.compile(r"[\r\n\t]+")
_WS_RE = re.compile(r"\s+")
_TR_NUM_RE = re.compile(r"^-?\d{1,3}(\.\d{3})*(,\d+)?$")
_NUMERIC_LIKE_RE = re.compile(r"^-?\d+([.,]\d+)?$")


def _clean_numeric_string(series: pd.Series) -> pd.Series:
    """Best-effort cleanup before pd.to_numeric.

    Handles:
    - Stray newlines/tabs (e.g. "0\n\n\n.90" → "0.90")
    - Whitespace
    - Turkish decimal "1.234,56" → "1234.56"
    - Currency suffixes/prefixes ("123 TL", "$45.00", "₺120")
    - Thousand separators (commas in en, dots in tr)
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(_NEWLINE_RE, "", regex=True)
    s = s.str.replace(_WS_RE, "", regex=True)
    # Strip common currency symbols/codes
    s = s.str.replace(r"^(₺|\$|€|£)", "", regex=True)
    s = s.str.replace(r"(TL|TRY|USD|EUR|GBP)$", "", regex=True, case=False)

    # Turkish thousand-separator pattern: "1.234,56"
    tr_mask = s.str.match(_TR_NUM_RE, na=False)
    if tr_mask.any():
        tmp = s[tr_mask].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        s.loc[tr_mask] = tmp

    return s


def coerce_numeric_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 0.85,
) -> tuple[pd.DataFrame, dict]:
    """Coerce columns to numeric with a per-column report.

    If ``columns`` is None, auto-detects: name-hint OR ≥threshold values
    parseable as numeric.

    Returns:
        new_df: DataFrame with target columns converted
        report: dict[col] -> {converted, failed, non_null_original, failed_samples}
    """
    out = df.copy()
    report: dict[str, dict] = {}

    if columns is None:
        columns = []
        for col in out.columns:
            if pd.api.types.is_numeric_dtype(out[col]):
                continue
            if _is_id_name(col):
                continue
            if _name_has_hint(col, NUMERIC_NAME_HINTS):
                columns.append(col)
                continue
            sample = out[col].dropna().astype(str).head(500)
            if len(sample) == 0:
                continue
            cleaned = _clean_numeric_string(sample)
            parsed = pd.to_numeric(cleaned, errors="coerce")
            if len(sample) > 0 and parsed.notna().sum() / len(sample) >= threshold:
                columns.append(col)

    for col in columns:
        if col not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            report[col] = {
                "converted": int(out[col].notna().sum()),
                "failed": 0,
                "non_null_original": int(out[col].notna().sum()),
                "failed_samples": [],
                "already_numeric": True,
            }
            continue
        original = out[col]
        cleaned = _clean_numeric_string(original)
        new_values = pd.to_numeric(cleaned, errors="coerce")
        non_null_orig = int(original.notna().sum())
        converted = int(new_values.notna().sum())
        failed_mask = original.notna() & new_values.isna()
        failed = int(failed_mask.sum())
        failed_samples = (
            original[failed_mask].astype(str).unique()[:5].tolist()
            if failed > 0 else []
        )
        report[col] = {
            "converted": converted,
            "failed": failed,
            "non_null_original": non_null_orig,
            "failed_samples": failed_samples,
            "already_numeric": False,
        }
        out[col] = new_values
    return out, report


def coerce_datetime_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Coerce columns to datetime with a per-column report."""
    out = df.copy()
    report: dict[str, dict] = {}
    for col in columns:
        if col not in out.columns:
            continue
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            report[col] = {"converted": int(out[col].notna().sum()), "failed": 0,
                           "failed_samples": [], "already_datetime": True}
            continue
        original = out[col]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_values = pd.to_datetime(original, errors="coerce")
        non_null_orig = int(original.notna().sum())
        converted = int(new_values.notna().sum())
        failed_mask = original.notna() & new_values.isna()
        failed = int(failed_mask.sum())
        failed_samples = (
            original[failed_mask].astype(str).unique()[:5].tolist()
            if failed > 0 else []
        )
        report[col] = {
            "converted": converted,
            "failed": failed,
            "non_null_original": non_null_orig,
            "failed_samples": failed_samples,
            "already_datetime": False,
        }
        out[col] = new_values
    return out, report


# ── Kind inference & application ────────────────────────────────────────────
def _content_looks_numeric(series: pd.Series, threshold: float = 0.85) -> bool:
    """True only when content looks like a measure/amount (not a code).

    A column counts as numeric-by-content if (a) ≥threshold of values parse,
    AND (b) values look like measures rather than identifiers — i.e. some
    decimals, or notable variance, or no fixed-width digit-string pattern.
    """
    sample = series.dropna().astype(str).head(500)
    if len(sample) == 0:
        return False
    cleaned = _clean_numeric_string(sample)
    parsed = pd.to_numeric(cleaned, errors="coerce")
    parse_share = parsed.notna().sum() / len(sample)
    if parse_share < threshold:
        return False

    raw = sample.astype(str).str.strip()
    has_decimal = raw.str.contains(r"[.,]\d", regex=True).mean() > 0.05
    has_negative = (raw.str.startswith("-")).any()
    if has_decimal or has_negative:
        return True

    # All-integer content. Could still be numeric (counts) — but reject
    # patterns that look like codes / IDs:
    leading_zero_share = raw.str.match(r"^0\d", na=False).mean()
    lengths = raw.str.len()
    fixed_width = lengths.nunique() <= 2 and lengths.iloc[0] >= 4
    if leading_zero_share > 0.05 or fixed_width:
        return False
    # Otherwise (varying-length integer column without leading zeros): treat
    # as a count/quantity-style numeric.
    return True


def _content_looks_datetime(series: pd.Series, threshold: float = 0.8) -> bool:
    sample = series.dropna().astype(str).head(200)
    if len(sample) == 0:
        return False
    try:
        # Quick gate: must contain digits (avoids costly fallback parsing on plain text)
        digit_share = sample.str.contains(r"\d", regex=True).mean()
        if digit_share < 0.5:
            return False
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().sum() / len(sample) >= threshold
    except Exception:
        return False


def infer_column_kinds(df: pd.DataFrame) -> dict[str, str]:
    """Suggest a Kind label (KIND_*) for each column.

    Order of precedence:
        1. Constant column → Excluded
        2. Already numeric / datetime dtype  → corresponding Kind
        3. Name-based numeric / datetime / ID hint
        4. Content-based numeric / datetime check
        5. Near-unique strings  → ID  (only after name + content failed)
        6. Categorical
    """
    kinds: dict[str, str] = {}
    n_rows = len(df)
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1:
            kinds[col] = KIND_EXCLUDED
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            kinds[col] = KIND_NUMERIC
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            kinds[col] = KIND_DATETIME
            continue

        # Name-based hints take precedence over unique-ratio heuristics so a
        # column called "amnttl" with 100% unique values isn't mistakenly
        # classified as an ID.
        if _name_has_hint(col, NUMERIC_NAME_HINTS):
            kinds[col] = KIND_NUMERIC
            continue
        if _name_has_hint(col, DATETIME_NAME_HINTS):
            kinds[col] = KIND_DATETIME
            continue
        if _is_id_name(col):
            kinds[col] = KIND_ID
            continue

        # Content-based detection for columns whose name gave no hint.
        if _content_looks_numeric(df[col]):
            kinds[col] = KIND_NUMERIC
            continue
        if _content_looks_datetime(df[col]):
            kinds[col] = KIND_DATETIME
            continue

        unique_ratio = nunique / n_rows if n_rows else 0
        if unique_ratio > 0.95:
            kinds[col] = KIND_ID
            continue

        kinds[col] = KIND_CATEGORICAL
    return kinds


def apply_column_kinds(
    df: pd.DataFrame,
    kinds: dict[str, str],
) -> tuple[pd.DataFrame, dict]:
    """Apply user-confirmed kinds to df and return conversion report.

    - KIND_NUMERIC: pd.to_numeric with cleanup
    - KIND_DATETIME: pd.to_datetime
    - KIND_CATEGORICAL: cast to object/string
    - KIND_ID / KIND_EXCLUDED: leave dtype alone (selection layer handles them)
    """
    out = df.copy()
    report: dict[str, dict] = {}

    numeric_cols = [c for c, k in kinds.items() if k == KIND_NUMERIC]
    datetime_cols = [c for c, k in kinds.items() if k == KIND_DATETIME]

    if numeric_cols:
        out, num_report = coerce_numeric_columns(out, columns=numeric_cols)
        report.update(num_report)

    if datetime_cols:
        out, dt_report = coerce_datetime_columns(out, columns=datetime_cols)
        for c, r in dt_report.items():
            report[c] = {**r, "kind": KIND_DATETIME}

    for col, kind in kinds.items():
        if kind == KIND_CATEGORICAL and col in out.columns:
            if not pd.api.types.is_object_dtype(out[col]) and \
               not pd.api.types.is_string_dtype(out[col]):
                out[col] = out[col].astype("object")
    return out, report


# ── XML / file readers (preserved verbatim) ─────────────────────────────────
def _read_xml(buffer) -> pd.DataFrame:
    """Read an XML file into a DataFrame.

    Supports:
    - nested bank/report XML files with <records><record>...</record></records>
      where each <record> may contain further nested elements (e.g. <originator>)
    - single-record XML files
    - flat XML structures readable by pandas directly
    """
    raw = buffer.read() if hasattr(buffer, "read") else buffer
    if isinstance(raw, str):
        raw = raw.encode("utf-8")

    import xml.etree.ElementTree as ET

    def strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    def flatten_element(elem, parent_key=""):
        row = {}
        children = list(elem)

        for attr_key, attr_val in elem.attrib.items():
            key = f"{parent_key}_@{attr_key}" if parent_key else f"@{attr_key}"
            row[key] = attr_val

        if not children:
            text = elem.text.strip() if elem.text and elem.text.strip() else None
            if parent_key:
                row[parent_key] = text
            return row

        for child in children:
            child_tag = strip_ns(child.tag)
            new_key = f"{parent_key}_{child_tag}" if parent_key else child_tag
            row.update(flatten_element(child, new_key))

        return row

    try:
        root = ET.fromstring(raw)
        records = root.findall(".//record")
        if records:
            rows = [flatten_element(rec) for rec in records]
            df = pd.DataFrame(rows)
            if not df.empty:
                return df
    except ET.ParseError:
        pass

    last_err: Exception | None = None
    for parser in ("lxml", "etree"):
        try:
            df = pd.read_xml(io.BytesIO(raw), parser=parser)
            if df is not None and not df.empty:
                return df
        except ImportError:
            continue
        except Exception as e:
            last_err = e

    if last_err:
        raise last_err
    raise ValueError("XML dosyası okunamadı.")


def _read_single_file(name: str, buffer) -> pd.DataFrame | None:
    """Read a single CSV, Excel or XML file from a name + file-like buffer."""
    name = name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(buffer)
    elif name.endswith((".xls", ".xlsx")):
        return pd.read_excel(buffer, engine="openpyxl")
    elif name.endswith(".xml"):
        return _read_xml(buffer)
    return None


def load_file(uploaded_file) -> pd.DataFrame | None:
    """Read an uploaded CSV, Excel, XML or ZIP file into a DataFrame.

    ZIP files: the archive is scanned for CSV/Excel/XML files.
    - If one data file is found, it is loaded directly.
    - If multiple are found, the user picks which one to use.
    """
    name = uploaded_file.name.lower()

    try:
        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as zf:
                data_files = [
                    f for f in zf.namelist()
                    if not f.startswith("__MACOSX")
                    and not f.startswith(".")
                    and f.lower().endswith(SUPPORTED_DATA_EXTENSIONS)
                ]

                if not data_files:
                    st.error("ZIP dosyasında CSV, Excel veya XML dosyası bulunamadı.")
                    return None

                if len(data_files) == 1:
                    target = data_files[0]
                else:
                    target = st.selectbox(
                        "ZIP içinde birden fazla dosya bulundu. Hangisini yüklemek istersiniz?",
                        options=data_files,
                    )

                with zf.open(target) as f:
                    df = _read_single_file(target, io.BytesIO(f.read()))

                if df is None:
                    st.error(f"'{target}' dosyası okunamadı.")
                    return None

        elif name.endswith(SUPPORTED_DATA_EXTENSIONS):
            df = _read_single_file(name, uploaded_file)
            if df is None:
                st.error("Dosya okunamadı.")
                return None

        else:
            st.error("Desteklenmeyen dosya formatı. CSV, Excel, XML veya ZIP yükleyin.")
            return None

        if df.empty:
            st.error("Yüklenen dosya boş.")
            return None

        return df

    except zipfile.BadZipFile:
        st.error("Geçersiz ZIP dosyası. Lütfen dosyayı kontrol edin.")
        return None
    except Exception as e:
        st.error(f"Dosya okunurken hata oluştu: {e}")
        return None
