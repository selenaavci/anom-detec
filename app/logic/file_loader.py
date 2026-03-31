import zipfile
import io
import pandas as pd
import streamlit as st


def _read_single_file(name: str, buffer) -> pd.DataFrame | None:
    """Read a single CSV or Excel file from a name + file-like buffer."""
    name = name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(buffer)
    elif name.endswith((".xls", ".xlsx")):
        return pd.read_excel(buffer, engine="openpyxl")
    return None


def load_file(uploaded_file) -> pd.DataFrame | None:
    """Read an uploaded CSV, Excel, or ZIP file into a DataFrame.

    ZIP files: the archive is scanned for CSV/Excel files.
    - If one data file is found, it is loaded directly.
    - If multiple are found, the user picks which one to use.
    """
    name = uploaded_file.name.lower()

    try:
        # ── ZIP handling ─────────────────────────────────────────────────
        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as zf:
                data_files = [
                    f for f in zf.namelist()
                    if not f.startswith("__MACOSX")
                    and not f.startswith(".")
                    and (f.lower().endswith(".csv") or f.lower().endswith((".xls", ".xlsx")))
                ]

                if not data_files:
                    st.error("ZIP dosyasında CSV veya Excel dosyası bulunamadı.")
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

        # ── Direct CSV / Excel ───────────────────────────────────────────
        elif name.endswith(".csv") or name.endswith((".xls", ".xlsx")):
            df = _read_single_file(name, uploaded_file)
            if df is None:
                st.error("Dosya okunamadı.")
                return None

        else:
            st.error("Desteklenmeyen dosya formatı. CSV, Excel veya ZIP yükleyin.")
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
