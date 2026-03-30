import pandas as pd
import streamlit as st


def load_file(uploaded_file) -> pd.DataFrame | None:
    """Read an uploaded CSV or Excel file into a DataFrame."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            st.error("Desteklenmeyen dosya formatı. Lütfen CSV veya Excel yükleyin.")
            return None

        if df.empty:
            st.error("Yüklenen dosya boş.")
            return None

        return df
    except Exception as e:
        st.error(f"Dosya okunurken hata oluştu: {e}")
        return None
