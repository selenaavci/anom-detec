import zipfile
import io
import pandas as pd
import streamlit as st


SUPPORTED_DATA_EXTENSIONS = (".csv", ".xls", ".xlsx", ".xml")


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

    # Manual flattening handles deeply nested <record> structures that
    # pd.read_xml cannot reach.
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
