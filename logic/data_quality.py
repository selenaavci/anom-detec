"""Rule-based data quality checks.

Run these before ML so that obvious format/content anomalies are reported as
their own signal — independent of statistical anomaly detection. The same
row-level flags can also be injected into the model as binary features.
"""
from __future__ import annotations

import re
import pandas as pd
import numpy as np

# Local kind constants (mirrored from file_loader to avoid circular imports)
KIND_NUMERIC = "Sayısal/Tutar"
KIND_CATEGORICAL = "Kategorik/Metin"
KIND_DATETIME = "Tarih"
KIND_ID = "ID/Referans"

_CONTROL_RE = re.compile(r"[\r\n\t\x00-\x08\x0b\x0c\x0e-\x1f]")
_DIGITS_ONLY_RE = re.compile(r"^\s*\d+\s*$")
_NAME_TOKENS = ("name", "isim", "ad", "soyad", "unvan")
_CURRENCY_TOKENS = ("cur", "currency", "para", "doviz", "döviz")


def _name_matches(col: str, tokens) -> bool:
    cl = col.lower()
    return any(t in cl for t in tokens)


def _make_summary_row(
    rule: str,
    column: str,
    severity: str,
    mask: pd.Series,
    df: pd.DataFrame,
    n: int,
    description: str,
) -> dict:
    count = int(mask.sum())
    samples = []
    if count > 0 and column in df.columns:
        samples = df.loc[mask, column].astype(str).head(3).tolist()
    return {
        "rule": rule,
        "column": column,
        "severity": severity,
        "count": count,
        "share": round(count / n * 100, 2) if n else 0,
        "samples": samples,
        "description": description,
    }


def run_quality_checks(
    df: pd.DataFrame,
    kinds: dict[str, str] | None = None,
    allowed_currencies: list[str] | None = None,
    required_columns: list[str] | None = None,
    long_text_threshold: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run rule-based checks and return (summary, row_flags).

    summary: long-form DataFrame with one row per (rule, column) hit.
    row_flags: wide DataFrame with boolean columns named ``{rule}__{column}``,
        aligned to df.index — can be fed straight into preprocess(extra_features=).

    Args:
        kinds: column kind hints from infer_column_kinds(). When None, dtypes
            are used.
        allowed_currencies: e.g. ["TRY", "USD", "EUR"]. Values outside the list
            are flagged for any currency-like column.
        required_columns: columns whose missing values are flagged as
            "boş_zorunlu_alan".
    """
    n = len(df)
    summary: list[dict] = []
    flags = pd.DataFrame(index=df.index)

    kinds = kinds or {}
    required_columns = required_columns or []
    allowed_currencies = [c.upper() for c in (allowed_currencies or [])]

    for col in df.columns:
        kind = kinds.get(col)
        ser = df[col]

        is_num = kind == KIND_NUMERIC or (kind is None and pd.api.types.is_numeric_dtype(ser))
        is_dt = kind == KIND_DATETIME or (kind is None and pd.api.types.is_datetime64_any_dtype(ser))

        # ── Required-but-empty ──
        if col in required_columns:
            mask = ser.isna()
            if is_num is False and pd.api.types.is_object_dtype(ser):
                mask = mask | (ser.astype(str).str.strip() == "")
            if mask.any():
                summary.append(_make_summary_row(
                    "boş_zorunlu_alan", col, "high", mask, df, n,
                    "Bu kolonda zorunlu olduğu halde değer eksik",
                ))
                flags[f"boş_zorunlu_alan__{col}"] = mask.astype(int)

        if is_num:
            num_ser = pd.to_numeric(ser, errors="coerce")
            # Negative, zero, extreme, NaN-after-coerce
            neg_mask = num_ser < 0
            if neg_mask.any():
                summary.append(_make_summary_row(
                    "negatif_tutar", col, "medium", neg_mask, df, n,
                    "Sayısal alanda negatif değer (tutar/bakiye için sıra dışı)",
                ))
                flags[f"negatif_tutar__{col}"] = neg_mask.fillna(False).astype(int)

            zero_mask = num_ser == 0
            if zero_mask.any():
                summary.append(_make_summary_row(
                    "sıfır_tutar", col, "low", zero_mask, df, n,
                    "Tutar tam sıfır",
                ))
                flags[f"sıfır_tutar__{col}"] = zero_mask.fillna(False).astype(int)

            mu, sigma = num_ser.mean(), num_ser.std()
            if sigma and not pd.isna(sigma) and sigma > 0:
                z = (num_ser - mu).abs() / sigma
                ext_mask = z > 4
                if ext_mask.any():
                    summary.append(_make_summary_row(
                        "aşırı_uç_değer", col, "medium", ext_mask, df, n,
                        "Standart sapmanın 4 katından fazla sapma",
                    ))
                    flags[f"aşırı_uç_değer__{col}"] = ext_mask.fillna(False).astype(int)

            parse_fail = ser.notna() & num_ser.isna()
            if parse_fail.any():
                summary.append(_make_summary_row(
                    "parse_edilemeyen_sayı", col, "high", parse_fail, df, n,
                    "Sayısal olması beklenen alanda parse edilemeyen değer",
                ))
                flags[f"parse_edilemeyen_sayı__{col}"] = parse_fail.astype(int)

        elif is_dt:
            dt_ser = pd.to_datetime(ser, errors="coerce")
            parse_fail = ser.notna() & dt_ser.isna()
            if parse_fail.any():
                summary.append(_make_summary_row(
                    "parse_edilemeyen_tarih", col, "high", parse_fail, df, n,
                    "Tarih olması beklenen alanda parse edilemeyen değer",
                ))
                flags[f"parse_edilemeyen_tarih__{col}"] = parse_fail.astype(int)

            now = pd.Timestamp.now()
            future_mask = dt_ser > now + pd.Timedelta(days=365 * 10)
            past_mask = dt_ser < pd.Timestamp("1900-01-01")
            if future_mask.any():
                summary.append(_make_summary_row(
                    "uzak_gelecek_tarih", col, "medium", future_mask, df, n,
                    "10 yıldan uzak gelecek tarih",
                ))
                flags[f"uzak_gelecek_tarih__{col}"] = future_mask.fillna(False).astype(int)
            if past_mask.any():
                summary.append(_make_summary_row(
                    "uzak_geçmiş_tarih", col, "medium", past_mask, df, n,
                    "1900 öncesi tarih",
                ))
                flags[f"uzak_geçmiş_tarih__{col}"] = past_mask.fillna(False).astype(int)

        elif pd.api.types.is_object_dtype(ser) or pd.api.types.is_string_dtype(ser):
            s = ser.fillna("").astype(str)
            # Control chars
            ctrl_mask = s.str.contains(_CONTROL_RE, regex=True, na=False)
            if ctrl_mask.any():
                summary.append(_make_summary_row(
                    "kontrol_karakteri", col, "high", ctrl_mask, df, n,
                    "Metinde \\n, \\t, \\r gibi kontrol karakterleri",
                ))
                flags[f"kontrol_karakteri__{col}"] = ctrl_mask.astype(int)

            # Too long
            too_long_mask = s.str.len() > long_text_threshold
            if too_long_mask.any():
                summary.append(_make_summary_row(
                    "aşırı_uzun_metin", col, "low", too_long_mask, df, n,
                    f"{long_text_threshold} karakterden uzun değer",
                ))
                flags[f"aşırı_uzun_metin__{col}"] = too_long_mask.astype(int)

            # Name-like with digits-only content
            if _name_matches(col, _NAME_TOKENS):
                digits_mask = s.str.match(_DIGITS_ONLY_RE, na=False) & (s.str.len() > 0)
                if digits_mask.any():
                    summary.append(_make_summary_row(
                        "isim_alanında_sadece_rakam", col, "high", digits_mask, df, n,
                        "İsim alanında sadece rakam — bozuk veya kasıtlı manipülasyon olabilir",
                    ))
                    flags[f"isim_alanında_sadece_rakam__{col}"] = digits_mask.astype(int)

            # Currency outside whitelist
            if allowed_currencies and _name_matches(col, _CURRENCY_TOKENS):
                cur_up = s.str.upper().str.strip()
                bad_cur = (~cur_up.isin(allowed_currencies)) & (cur_up != "")
                if bad_cur.any():
                    summary.append(_make_summary_row(
                        "beklenmeyen_para_birimi", col, "medium", bad_cur, df, n,
                        f"{', '.join(allowed_currencies)} dışında para birimi",
                    ))
                    flags[f"beklenmeyen_para_birimi__{col}"] = bad_cur.astype(int)

        # ── Duplicate-id check (any column with KIND_ID or 'proc'/'id' in name) ──
        if kind == KIND_ID or col.lower() in ("proc", "id", "ref"):
            dup_mask = ser.duplicated(keep=False) & ser.notna()
            if dup_mask.any():
                summary.append(_make_summary_row(
                    "tekrarlayan_id", col, "high", dup_mask, df, n,
                    "Aynı ID/Referans birden fazla kez görülüyor",
                ))
                flags[f"tekrarlayan_id__{col}"] = dup_mask.astype(int)

    summary_df = pd.DataFrame(summary)
    if summary_df.empty:
        summary_df = pd.DataFrame(
            columns=["rule", "column", "severity", "count", "share", "samples", "description"]
        )
    return summary_df, flags


def severity_label(sev: str) -> str:
    return {"high": "Yüksek", "medium": "Orta", "low": "Düşük"}.get(sev, "Bilgi")
