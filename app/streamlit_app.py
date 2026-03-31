import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from app.logic.file_loader import load_file
from app.logic.preprocessing import (
    detect_column_types,
    analyze_column_quality,
    get_safe_columns,
    preprocess,
    compute_feature_contributions,
)
from app.logic.anomaly_model import (
    detect_anomalies,
    get_top_contributing_features,
    run_semi_supervised,
    MODELS,
)
from app.logic.presets import PRESETS

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Anomaly Detection", page_icon="🔍", layout="wide")
st.title("Anomaly Detection Agent")
st.markdown("Veri setinizi yükleyin, sütunları seçin ve anomali tespiti çalıştırın.")

# ── Sidebar — Preset + Settings ──────────────────────────────────────────────
st.sidebar.header("Senaryo Seçimi")

preset_name = st.sidebar.selectbox("Örnek Senaryolar", list(PRESETS.keys()))
preset = PRESETS[preset_name]

st.sidebar.info(preset["description"])
for tip in preset["tips"]:
    st.sidebar.markdown(f"- {tip}")

st.sidebar.divider()
st.sidebar.header("Model Ayarları")

default_model_idx = list(MODELS.values()).index(preset["model"]) if preset["model"] in MODELS.values() else 0
model_display = st.sidebar.selectbox("Model", list(MODELS.keys()), index=default_model_idx)
model_key = MODELS[model_display]

contamination = st.sidebar.slider(
    "Kontaminasyon Orani",
    min_value=0.01,
    max_value=0.50,
    value=preset["contamination"],
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

# ── 1. File upload ───────────────────────────────────────────────────────────
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

# ── 2. Data preview ──────────────────────────────────────────────────────────
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
            "Sutun": df.columns,
            "Tip": df.dtypes.astype(str).values,
            "Eksik": df.isna().sum().values,
            "Benzersiz": df.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(df.describe(), use_container_width=True)

# ── 3. Column quality analysis ───────────────────────────────────────────────
st.header("3. Kolon Kalite Kontrolü")

quality_df = analyze_column_quality(df)

flagged = quality_df[quality_df["flags"] != "-"]
if not flagged.empty:
    st.warning(f"{len(flagged)} sütunda kalite uyarısı tespit edildi.")
    with st.expander("Kalite uyarıları (detay)", expanded=True):
        flag_display = flagged[["column", "dtype", "unique", "unique_ratio", "missing_pct", "flags"]].copy()
        flag_display.columns = ["Sütun", "Tip", "Benzersiz", "Benzersiz Oranı", "Eksik %", "Uyarı"]
        st.dataframe(flag_display, use_container_width=True, hide_index=True)

        st.markdown("""
        **Uyarı türleri:**
        - **id_like**: Benzersiz değer oranı cok yüksek — muhtemelen ID sütunu
        - **constant**: Tek bir değer içeriyor — analiz için bilgi taşımaz
        - **free_text**: Serbest metin alanı — anomali tespitine uygun değil
        - **high_cardinality**: Çok fazla kategori — frequency encoding uygulanacak
        """)
else:
    st.success("Tüm sütunlar kalite kontrolünden geçti.")

# ── 4. Column selection ──────────────────────────────────────────────────────
st.header("4. Sütun Seçimi")

col_types = detect_column_types(df)
all_usable = col_types["numeric"] + col_types["categorical"]

if not all_usable:
    st.error("Veri setinde kullanılabilir sayısal veya kategorik sütun bulunamadı.")
    st.stop()

st.markdown(
    f"**Sayısal sütunlar:** {len(col_types['numeric'])} &nbsp;|&nbsp; "
    f"**Kategorik sütunlar:** {len(col_types['categorical'])}"
)

st.warning(
    "ID içeren sütunlar, serbest metin alanlarını ve sabit değerli sütunları "
    "seçime dahil etmemeye dikkat edin. Bu tür sütunlar analiz sonuçlarını olumsuz etkiler."
)


safe_columns = get_safe_columns(quality_df)
safe_defaults = [c for c in safe_columns if c in all_usable]

selected_columns = st.multiselect(
    "Analiz edilecek sütunları seçin",
    options=all_usable,
    default=safe_defaults[:15],
)

if not selected_columns:
    st.warning("En az bir sütun seçmelisiniz.")
    st.stop()

selected_flagged = flagged[flagged["column"].isin(selected_columns)]
if not selected_flagged.empty:
    cols_str = ", ".join(selected_flagged["column"].tolist())
    st.error(f"Uyarı: Seçtiğiniz şu sütunlar kalite kontrolünden geçemedi: **{cols_str}**. Çıkarmak isteyebilirsiniz.")

# ── 5. Run detection ─────────────────────────────────────────────────────────
st.header("5. Anomali Tespiti")

if st.button("Anomali Tespiti Çalıştır", type="primary", use_container_width=True):
    with st.spinner("Veri işleniyor ve model çalıştırılıyor..."):
        processed, scaler, feature_names = preprocess(df, selected_columns)
        scores, fitted_model = detect_anomalies(processed, model_name=model_key, contamination=contamination)
        contributions = compute_feature_contributions(processed, df, selected_columns)
        
    result_df = df.copy()
    result_df["anomaly_score"] = scores
    result_df = result_df.sort_values("anomaly_score", ascending=False)
    result_df["rank"] = range(1, len(result_df) + 1)

    st.session_state["result_df"] = result_df
    st.session_state["scores"] = scores
    st.session_state["processed"] = processed
    st.session_state["feature_names"] = feature_names
    st.session_state["contributions"] = contributions

    if "feedback" not in st.session_state:
        st.session_state["feedback"] = {}

# ── 6. Results ───────────────────────────────────────────────────────────────
if "result_df" not in st.session_state:
    st.stop()

result_df = st.session_state["result_df"]
scores = st.session_state["scores"]
processed = st.session_state["processed"]
feature_names = st.session_state["feature_names"]
contributions = st.session_state["contributions"]

st.header("6. Sonuçlar")

threshold = scores.quantile(1 - contamination)
n_anomalies = int((scores >= threshold).sum())

m1, m2, m3 = st.columns(3)
m1.metric("Toplam Kayıt", len(result_df))
m2.metric("Tespit Edilen Anomali", n_anomalies)
m3.metric("Anomali Oranı", f"%{n_anomalies / len(result_df) * 100:.1f}")

st.subheader("Anomali Skor Dağılımı")
fig = px.histogram(
    result_df,
    x="anomaly_score",
    nbins=50,
    title="Anomali Skorları Dağılımı",
    labels={"anomaly_score": "Anomali Skoru", "count": "Kayıt Sayısı"},
)
fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Eşik Değer")
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"En Şüpheli {top_n} Kayıt")
top_anomalies = result_df.head(top_n)
st.dataframe(
    top_anomalies.style.background_gradient(subset=["anomaly_score"], cmap="Reds"),
    use_container_width=True,
    hide_index=True,
)

# ── 7. Detail inspection + Feature contributions ────────────────────────────
st.header("7. Kayıt Detay İnceleme ve Feature Analizi")

selected_rank = st.selectbox(
    "İncelemek istediğiniz kaydın sırasını seçin",
    options=top_anomalies["rank"].tolist(),
    format_func=lambda x: f"Sira {x} — Skor: {top_anomalies.loc[top_anomalies['rank'] == x, 'anomaly_score'].values[0]:.4f}",
)

row_mask = top_anomalies["rank"] == selected_rank
original_idx = top_anomalies.loc[row_mask].index[0]
row = top_anomalies.loc[row_mask].iloc[0]

detail_col, contrib_col = st.columns(2)

with detail_col:
    st.subheader("Kayıt Detayı")
    row_dict = row.drop(["anomaly_score", "rank"]).to_dict()
    st.json(row_dict)

with contrib_col:
    st.subheader("En Etkili Özellikler")
    st.caption("Yüksek z-skoru, o özelliğin normalden ne kadar saptığını gösterir.")
    if original_idx in processed.index:
        processed_row = processed.loc[original_idx]
        top_features = get_top_contributing_features(processed_row, top_k=8)

        fig_bar = px.bar(
            top_features,
            x="z_score",
            y="feature",
            color="direction",
            orientation="h",
            title="Z-Score ile Özellik Katkıları",
            labels={"z_score": "|Z-Score|", "feature": "Özellik", "direction": "Yön"},
            color_discrete_map={"+": "#EF553B", "-": "#636EFA"},
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"}, height=350)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Bu kayıt için feature analizi mevcut değil.")

# ── 8. Feedback loop ─────────────────────────────────────────────────────────
st.header("8. Geri Bildirim")
st.markdown(
    "Tespit edilen anomalileri **gerçek anomali** veya **yanlış alarm** olarak işaretleyin. "
    "Yeterli geri bildirim toplandıktan sonra yarı gözetimli modeli çalıştırabilirsiniz."
)

if "feedback" not in st.session_state:
    st.session_state["feedback"] = {}

feedback_ranks = top_anomalies["rank"].tolist()
for rank_val in feedback_ranks[:top_n]:
    idx = top_anomalies.loc[top_anomalies["rank"] == rank_val].index[0]
    score_val = top_anomalies.loc[top_anomalies["rank"] == rank_val, "anomaly_score"].values[0]

    col_label, col_btn1, col_btn2, col_status = st.columns([3, 2, 2, 2])
    col_label.markdown(f"**Sira {rank_val}** — Skor: {score_val:.4f}")

    current_fb = st.session_state["feedback"].get(idx, None)

    if col_btn1.button("Gerçek Anomali", key=f"anom_{idx}"):
        st.session_state["feedback"][idx] = 1
        current_fb = 1
    if col_btn2.button("Yanlş Alarm", key=f"normal_{idx}"):
        st.session_state["feedback"][idx] = 0
        current_fb = 0

    if current_fb == 1:
        col_status.success("Anomali")
    elif current_fb == 0:
        col_status.info("Normal")
    else:
        col_status.markdown("_Bekliyor_")

fb = st.session_state["feedback"]
n_feedback = len(fb)
n_marked_anomaly = sum(1 for v in fb.values() if v == 1)
n_marked_normal = sum(1 for v in fb.values() if v == 0)

st.divider()
fb1, fb2, fb3 = st.columns(3)
fb1.metric("Toplam Geri Bildirim", n_feedback)
fb2.metric("Gerçek Anomali", n_marked_anomaly)
fb3.metric("Yanlış Alarm", n_marked_normal)

# ── 9. Semi-supervised re-run ────────────────────────────────────────────────
st.header("9. Yarı Gözetimli Yeniden Analiz")
st.markdown(
    "Geri bildirimleriniz kullanılarak model yeniden eğitilir. "
    "**Normal** olarak işaretlenen kayıtlar eğitim setine dahil edilir, "
    "**Anomali** olarak işaretlenenler eğitimden çıkarılır."
)

min_feedback = 3
if n_feedback < min_feedback:
    st.info(f"Yarı gözetimli analiz için en az {min_feedback} geri bildirim gereklidir. Şimdilik {n_feedback} adet var.")
else:
    if st.button("Yarı Gözetimli Modeli Çalıştır", type="primary", use_container_width=True):
        with st.spinner("Geri bildirimlerle model yeniden eğitiliyor..."):
            labels = pd.Series(-1, index=processed.index)
            for idx_key, label_val in fb.items():
                if idx_key in labels.index:
                    labels.loc[idx_key] = label_val

            new_scores = run_semi_supervised(
                processed, labels, contamination=contamination,
            )

        result_df_v2 = df.copy()
        result_df_v2["anomaly_score"] = new_scores
        result_df_v2 = result_df_v2.sort_values("anomaly_score", ascending=False)
        result_df_v2["rank"] = range(1, len(result_df_v2) + 1)

        st.success("Yarı gözetimli analiz tamamlandı!")

        new_threshold = new_scores.quantile(1 - contamination)
        new_n_anomalies = int((new_scores >= new_threshold).sum())

        r1, r2 = st.columns(2)
        r1.metric("Yeni Anomali Sayısı", new_n_anomalies, delta=new_n_anomalies - n_anomalies)
        r2.metric("Yeni Eşik Değer", f"{new_threshold:.4f}")

        st.subheader(f"Güncellenmis En Şüpheli {top_n} Kayıt")
        st.dataframe(
            result_df_v2.head(top_n).style.background_gradient(subset=["anomaly_score"], cmap="Reds"),
            use_container_width=True,
            hide_index=True,
        )

        csv_v2 = result_df_v2.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Güncellenmis Sonuçları CSV Olarak İndir",
            data=csv_v2,
            file_name="anomaly_results_v2.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── 10. Download ─────────────────────────────────────────────────────────────
st.header("10. Sonuçları İndir")

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
