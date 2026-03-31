import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from app.logic.file_loader import load_file
from app.logic.preprocessing import detect_column_types, preprocess
from app.logic.anomaly_model import detect_anomalies, MODELS

st.set_page_config(page_title="Anomaly Detection", page_icon="🔍", layout="wide")
st.title("Anomaly Detection Agent")
st.markdown("Veri setinizi yükleyin, sütunları seçin ve anomali tespiti çalıştırın.")

st.sidebar.header("Ayarlar")

model_display = st.sidebar.selectbox("Model", list(MODELS.keys()))
model_key = MODELS[model_display]

contamination = st.sidebar.slider(
    "Kontaminasyon Oranı",
    min_value=0.01,
    max_value=0.50,
    value=0.05,
    step=0.01,
    help="Veri setindeki tahmini anomali yüzdesi",
)

top_n = st.sidebar.slider(
    "Gösterilecek Üst Anomali Sayısı",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
)

st.header("1. Veri Yükleme")
uploaded_file = st.file_uploader(
    "CSV veya Excel dosyası yükleyin",
    type=["csv", "xls", "xlsx"],
)

if uploaded_file is None:
    st.info("Başlamak için bir dosya yükleyin.")
    st.stop()

df = load_file(uploaded_file)
if df is None:
    st.stop()


st.header("2. Veri Önizleme")

col1, col2, col3 = st.columns(3)
col1.metric("Satır Sayısı", df.shape[0])
col2.metric("Sütun Sayısı", df.shape[1])
col3.metric("Eksik Değer", int(df.isna().sum().sum()))

with st.expander("İlk 100 satırı göster", expanded=True):
    st.dataframe(df.head(100), use_container_width=True)

with st.expander("Veri tipleri ve istatistikler"):
    tab1, tab2 = st.tabs(["Veri Tipleri", "İstatistikler"])
    with tab1:
        dtype_df = pd.DataFrame({
            "Sütun": df.columns,
            "Tip": df.dtypes.astype(str).values,
            "Eksik": df.isna().sum().values,
            "Benzersiz": df.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(df.describe(), use_container_width=True)


st.header("3. Sütun Seçimi")

col_types = detect_column_types(df)
all_usable = col_types["numeric"] + col_types["categorical"]

if not all_usable:
    st.error("Veri setinde kullanılabilir sayısal veya kategorik sütun bulunamadı.")
    st.stop()

st.markdown(
    f"**Sayısal sütunlar:** {len(col_types['numeric'])} &nbsp;|&nbsp; "
    f"**Kategorik sütunlar:** {len(col_types['categorical'])}"
)

st.warning("Yalnızca ID içeren sütunları ve serbest metin alanlarını seçime dahil etmemeye dikkat edin. Bu tür sütunlar analiz sonuçlarını olumsuz etkileyebilir.")

selected_columns = st.multiselect(
    "Analiz edilecek sütunları seçin",
    options=all_usable,
    default=col_types["numeric"][:10] if col_types["numeric"] else all_usable[:5],
)

if not selected_columns:
    st.warning("En az bir sütun seçmelisiniz.")
    st.stop()


st.header("4. Anomali Tespiti")

if st.button("Anomali Tespiti Çalıştır", type="primary", use_container_width=True):
    with st.spinner("Veri işleniyor ve model çalıştırılıyor..."):
        processed, scaler = preprocess(df, selected_columns)
        scores = detect_anomalies(processed, model_name=model_key, contamination=contamination)

    result_df = df.copy()
    result_df["anomaly_score"] = scores
    result_df = result_df.sort_values("anomaly_score", ascending=False)
    result_df["rank"] = range(1, len(result_df) + 1)

    st.session_state["result_df"] = result_df
    st.session_state["scores"] = scores


if "result_df" in st.session_state:
    result_df = st.session_state["result_df"]
    scores = st.session_state["scores"]

    st.header("5. Sonuçlar")

    threshold = scores.quantile(1 - contamination)
    n_anomalies = int((scores >= threshold).sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Toplam Kayıt", len(result_df))
    m2.metric("Tespit Edilen Anomali", n_anomalies)
    m3.metric("Anomali Oranı", f"%{n_anomalies / len(result_df) * 100:.1f}")

    st.subheader("Anomali Skor Dağılımı")
    import plotly.express as px

    fig = px.histogram(
        result_df,
        x="anomaly_score",
        nbins=50,
        title="Anomali Skorları Dağılımı",
        labels={"anomaly_score": "Anomali Skoru", "count": "Kayıt Sayısı"},
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text="Eşik Değer")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"En Şüpheli {top_n} Kayıt")
    top_anomalies = result_df.head(top_n)
    st.dataframe(
        top_anomalies.style.background_gradient(subset=["anomaly_score"], cmap="Reds"),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Kayıt Detay İnceleme")
    selected_rank = st.selectbox(
        "İncelemek istediğiniz kaydın sırasını seçin",
        options=top_anomalies["rank"].tolist(),
        format_func=lambda x: f"Sıra {x} — Skor: {top_anomalies.loc[top_anomalies['rank'] == x, 'anomaly_score'].values[0]:.4f}",
    )
    row = top_anomalies[top_anomalies["rank"] == selected_rank].iloc[0]
    st.json(row.to_dict())

    st.subheader("Sonuçları İndir")

    csv_data = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Tüm Sonuçları CSV Olarak İndir",
        data=csv_data,
        file_name="anomaly_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    csv_anomalies = top_anomalies.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Üst {top_n} Anomaliyi CSV Olarak İndir",
        data=csv_anomalies,
        file_name="top_anomalies.csv",
        mime="text/csv",
        use_container_width=True,
    )
