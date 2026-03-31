import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from app.logic.file_loader import load_file
from app.logic.preprocessing import (
    detect_column_types,
    analyze_column_quality,
    get_safe_columns,
    preprocess,
)
from app.logic.anomaly_model import (
    benchmark_models,
    get_top_deviating_features,
    detect_temporal_anomalies,
    run_semi_supervised,
    MODEL_DESCRIPTIONS,
    MODELS,
)
from app.logic.presets import PRESETS

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Anomaly Detection", page_icon="🔍", layout="wide")
st.title("Anomaly Detection Agent")
st.markdown("Veri setinizi yükleyin, sütunları seçin ve olağandışı kayıtları tespit edin.")

# ── Sidebar — Preset + Settings ──────────────────────────────────────────────
st.sidebar.header("Senaryo Seçimi")

preset_name = st.sidebar.selectbox("Örnek Senaryolar", list(PRESETS.keys()))
preset = PRESETS[preset_name]

st.sidebar.info(preset["description"])
for tip in preset["tips"]:
    st.sidebar.markdown(f"- {tip}")

st.sidebar.divider()
st.sidebar.header("Ayarlar")

contamination = st.sidebar.slider(
    "Beklenen Anomali Oranı",
    min_value=0.01,
    max_value=0.50,
    value=preset["contamination"],
    step=0.01,
    help="Veri setinde yaklaşık ne kadar olağandışı kayıt bekliyorsunuz?",
)

top_n = st.sidebar.slider(
    "Gösterilecek Şüpheli Kayıt Sayısı",
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
            "Sütun": df.columns,
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
    st.info(
        f"{len(flagged)} sütunda öneri tespit edildi. "
        f"Bu sütunları yine de kullanabilirsiniz, ancak sonuçları etkileyebilir."
    )
    with st.expander("Kalite önerileri (detay)", expanded=True):
        flag_display = flagged[["column", "dtype", "unique", "unique_ratio", "missing_pct", "flags"]].copy()
        flag_display.columns = ["Sütun", "Tip", "Benzersiz", "Benzersiz Oranı", "Eksik %", "Öneri"]
        st.dataframe(flag_display, use_container_width=True, hide_index=True)

        st.markdown("""
        **Öneri türleri:**
        - **id_like**: Benzersiz değer oranı çok yüksek — muhtemelen ID sütunu
        - **constant**: Tek bir değer içeriyor — analiz için bilgi taşımaz
        - **free_text**: Serbest metin alanı — anomali tespitine uygun değil
        - **high_cardinality**: Çok fazla kategori — otomatik olarak frekans bazlı dönüşüm uygulanacak
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

if col_types["datetime"]:
    st.markdown(f"**Tarih sütunları:** {', '.join(col_types['datetime'])}")

st.info(
    "Analiz için anlamlı sütunları seçin. ID numaraları, serbest metin gibi sütunlar "
    "sonuçların doğruluğunu azaltabilir. Önerilen sütunlar otomatik seçilmiştir."
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

# Soft warning for flagged columns
selected_flagged = flagged[flagged["column"].isin(selected_columns)]
if not selected_flagged.empty:
    cols_list = selected_flagged["column"].tolist()
    st.info(
        f"Seçtiğiniz **{', '.join(cols_list)}** sütun(lar)ı için önerilerimiz var. "
        f"Kullanmaya devam edebilirsiniz, sonuçları inceledikten sonra çıkarmayı deneyebilirsiniz."
    )

# ── 5. Run detection (benchmark all models) ─────────────────────────────────
st.header("5. Anomali Tespiti")

st.markdown(
    "Sistem üç farklı yöntemi otomatik olarak çalıştırır, karşılaştırır ve "
    "verilerinize en uygun olanını seçer."
)

if st.button("Analizi Başlat", type="primary", use_container_width=True):
    with st.spinner("Veri hazırlanıyor ve modeller çalıştırılıyor..."):
        processed, scaler, feature_names = preprocess(df, selected_columns)
        bench_results, best_key = benchmark_models(processed, contamination=contamination)

    st.session_state["processed"] = processed
    st.session_state["feature_names"] = feature_names
    st.session_state["bench_results"] = bench_results
    st.session_state["best_key"] = best_key

    # Use best model's scores
    best_result = next(r for r in bench_results if r["model_key"] == best_key)
    scores = pd.Series(best_result["scores"], index=processed.index, name="anomaly_score")

    result_df = df.copy()
    result_df["anomaly_score"] = scores
    result_df = result_df.sort_values("anomaly_score", ascending=False)
    result_df["rank"] = range(1, len(result_df) + 1)

    st.session_state["result_df"] = result_df
    st.session_state["scores"] = scores
    st.session_state["selected_columns"] = selected_columns

    if "feedback" not in st.session_state:
        st.session_state["feedback"] = {}

# ── 6. Benchmark results ────────────────────────────────────────────────────
if "bench_results" not in st.session_state:
    st.stop()

bench_results = st.session_state["bench_results"]
best_key = st.session_state["best_key"]
result_df = st.session_state["result_df"]
scores = st.session_state["scores"]
processed = st.session_state["processed"]
sel_columns = st.session_state["selected_columns"]

st.header("6. Model Karşılaştırması")

best_display = [k for k, v in MODELS.items() if v == best_key][0]
st.success(f"En uygun model otomatik seçildi: **{best_display}**")

bench_rows = []
for r in bench_results:
    display_name = [k for k, v in MODELS.items() if v == r["model_key"]][0]
    bench_rows.append({
        "Model": display_name,
        "Açıklama": MODEL_DESCRIPTIONS[r["model_key"]],
        "Bulunan Anomali": r["n_anomalies"] if r["success"] else "-",
        "Uyum Skoru": r["silhouette"] if r["success"] else "-",
        "Seçildi": "✓" if r["model_key"] == best_key else "",
    })
st.dataframe(pd.DataFrame(bench_rows), use_container_width=True, hide_index=True)

# ── 7. Results ───────────────────────────────────────────────────────────────
st.header("7. Sonuçlar")

threshold = scores.quantile(1 - contamination)
n_anomalies = int((scores >= threshold).sum())

m1, m2, m3 = st.columns(3)
m1.metric("Toplam Kayıt", len(result_df))
m2.metric("Şüpheli Kayıt", n_anomalies)
m3.metric("Şüpheli Oranı", f"%{n_anomalies / len(result_df) * 100:.1f}")

# Score distribution — simplified labels
st.subheader("Şüphelilik Dağılımı")
fig = px.histogram(
    result_df,
    x="anomaly_score",
    nbins=50,
    labels={"anomaly_score": "Şüphelilik Puanı", "count": "Kayıt Sayısı"},
)
fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Eşik")
fig.update_layout(title=None)
st.plotly_chart(fig, use_container_width=True)

# Top anomalies table
st.subheader(f"En Şüpheli {top_n} Kayıt")
top_anomalies = result_df.head(top_n)
st.dataframe(
    top_anomalies.style.background_gradient(subset=["anomaly_score"], cmap="Reds"),
    use_container_width=True,
    hide_index=True,
)

# ── 8. Detail inspection — non-technical ─────────────────────────────────────
st.header("8. Kayıt İnceleme")
st.markdown("Bir kaydı seçerek neden şüpheli bulunduğunu inceleyin.")

selected_rank = st.selectbox(
    "İncelemek istediğiniz kaydı seçin",
    options=top_anomalies["rank"].tolist(),
    format_func=lambda x: f"Sıra {x} — Şüphelilik: {top_anomalies.loc[top_anomalies['rank'] == x, 'anomaly_score'].values[0]:.2f}",
)

row_mask = top_anomalies["rank"] == selected_rank
original_idx = top_anomalies.loc[row_mask].index[0]
row = top_anomalies.loc[row_mask].iloc[0]

detail_col, reason_col = st.columns(2)

with detail_col:
    st.subheader("Kayıt Bilgileri")
    row_dict = row.drop(["anomaly_score", "rank"]).to_dict()
    display_items = []
    for k, v in row_dict.items():
        display_items.append({"Alan": k, "Değer": v})
    st.dataframe(pd.DataFrame(display_items), use_container_width=True, hide_index=True)

with reason_col:
    st.subheader("Neden Şüpheli?")
    st.markdown("Bu kaydın diğer kayıtlardan en çok farklılaştığı alanlar:")

    deviations = get_top_deviating_features(row, df, sel_columns, top_k=5)
    if not deviations.empty:
        for _, d in deviations.iterrows():
            diff_pct = d["deviation"]
            direction = "yüksek" if d["value"] > d["typical"] else "düşük"
            st.markdown(
                f"- **{d['feature']}**: Değeri **{d['value']:.2f}**, "
                f"tipik değer **{d['typical']:.2f}** — normalden **{diff_pct:.1f}x** daha {direction}"
            )
    else:
        st.info("Bu kayıt için sayısal karşılaştırma yapılamadı.")

# ── 9. Temporal anomaly ──────────────────────────────────────────────────────
datetime_cols = col_types["datetime"]
numeric_cols_for_ts = [c for c in sel_columns if c in col_types["numeric"]]

if datetime_cols and numeric_cols_for_ts:
    st.header("9. Zaman Serisi Analizi")
    st.markdown(
        "Verilerinizde tarih sütunu tespit edildi. Zaman içindeki olağandışı değişimleri "
        "aşağıda inceleyebilirsiniz."
    )

    ts_col1, ts_col2 = st.columns(2)
    with ts_col1:
        ts_date_col = st.selectbox("Tarih sütunu", datetime_cols)
    with ts_col2:
        ts_value_col = st.selectbox("İncelenecek değer sütunu", numeric_cols_for_ts)

    ts_window = st.slider("Hareketli ortalama penceresi (gün)", 3, 60, 14)

    if st.button("Zaman Serisi Analizi Çalıştır", use_container_width=True):
        with st.spinner("Zaman serisi analiz ediliyor..."):
            ts_result = detect_temporal_anomalies(df, ts_date_col, ts_value_col, window=ts_window)

        n_ts_anomalies = int(ts_result["is_anomaly"].sum())
        st.metric("Zaman Serisinde Şüpheli Nokta", n_ts_anomalies)

        fig_ts = go.Figure()
        # Normal points
        normal_ts = ts_result[~ts_result["is_anomaly"]]
        fig_ts.add_trace(go.Scatter(
            x=normal_ts[ts_date_col], y=normal_ts[ts_value_col],
            mode="markers", name="Normal", marker=dict(color="#636EFA", size=4),
        ))
        # Anomaly points
        anom_ts = ts_result[ts_result["is_anomaly"]]
        fig_ts.add_trace(go.Scatter(
            x=anom_ts[ts_date_col], y=anom_ts[ts_value_col],
            mode="markers", name="Şüpheli", marker=dict(color="#EF553B", size=10, symbol="x"),
        ))
        # Rolling mean
        fig_ts.add_trace(go.Scatter(
            x=ts_result[ts_date_col], y=ts_result["rolling_mean"],
            mode="lines", name="Ortalama Trend", line=dict(color="gray", dash="dash"),
        ))
        # Band
        fig_ts.add_trace(go.Scatter(
            x=ts_result[ts_date_col], y=ts_result["upper"],
            mode="lines", name="Üst Sınır", line=dict(color="rgba(200,200,200,0.5)"),
        ))
        fig_ts.add_trace(go.Scatter(
            x=ts_result[ts_date_col], y=ts_result["lower"],
            mode="lines", name="Alt Sınır", line=dict(color="rgba(200,200,200,0.5)"),
            fill="tonexty", fillcolor="rgba(200,200,200,0.15)",
        ))
        fig_ts.update_layout(
            xaxis_title="Tarih",
            yaxis_title=ts_value_col,
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        if n_ts_anomalies > 0:
            st.subheader("Şüpheli Zaman Noktaları")
            st.dataframe(
                anom_ts[[ts_date_col, ts_value_col, "rolling_mean"]].rename(columns={
                    ts_date_col: "Tarih",
                    ts_value_col: "Değer",
                    "rolling_mean": "Beklenen Değer",
                }),
                use_container_width=True,
                hide_index=True,
            )

    section_offset = 10
else:
    section_offset = 9

# ── Feedback loop ────────────────────────────────────────────────────────────
st.header(f"{section_offset}. Geri Bildirim")
st.markdown(
    "Tespit edilen kayıtları **gerçek anomali** veya **yanlış alarm** olarak işaretleyin. "
    "Geri bildirimleriniz modeli iyileştirmek için kullanılabilir."
)

if "feedback" not in st.session_state:
    st.session_state["feedback"] = {}

feedback_ranks = top_anomalies["rank"].tolist()
for rank_val in feedback_ranks[:top_n]:
    idx = top_anomalies.loc[top_anomalies["rank"] == rank_val].index[0]
    score_val = top_anomalies.loc[top_anomalies["rank"] == rank_val, "anomaly_score"].values[0]

    col_label, col_btn1, col_btn2, col_status = st.columns([3, 2, 2, 2])
    col_label.markdown(f"**Sıra {rank_val}** — Puan: {score_val:.2f}")

    current_fb = st.session_state["feedback"].get(idx, None)

    if col_btn1.button("Gerçek Anomali", key=f"anom_{idx}"):
        st.session_state["feedback"][idx] = 1
        current_fb = 1
    if col_btn2.button("Yanlış Alarm", key=f"normal_{idx}"):
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

# ── Semi-supervised re-run ───────────────────────────────────────────────────
st.header(f"{section_offset + 1}. Geri Bildirimle Yeniden Analiz")
st.markdown(
    "Geri bildirimleriniz kullanılarak model güncellenir. "
    "Normal olarak işaretlediğiniz kayıtlar modele öğretilir, "
    "anomali dedikleriniz eğitimden çıkarılır."
)

min_feedback = 3
if n_feedback < min_feedback:
    st.info(f"Yeniden analiz için en az {min_feedback} geri bildirim gerekli. Şu an {n_feedback} adet var.")
else:
    if st.button("Geri Bildirimle Yeniden Analiz Et", type="primary", use_container_width=True):
        with st.spinner("Model geri bildirimlerinizle güncelleniyor..."):
            labels = pd.Series(-1, index=processed.index)
            for idx_key, label_val in fb.items():
                if idx_key in labels.index:
                    labels.loc[idx_key] = label_val

            new_scores = run_semi_supervised(processed, labels, contamination=contamination)

        result_df_v2 = df.copy()
        result_df_v2["anomaly_score"] = new_scores
        result_df_v2 = result_df_v2.sort_values("anomaly_score", ascending=False)
        result_df_v2["rank"] = range(1, len(result_df_v2) + 1)

        st.success("Model güncellendi!")

        new_threshold = new_scores.quantile(1 - contamination)
        new_n_anomalies = int((new_scores >= new_threshold).sum())

        r1, r2 = st.columns(2)
        r1.metric("Güncel Şüpheli Sayısı", new_n_anomalies, delta=new_n_anomalies - n_anomalies)
        r2.metric("Değişim", f"{abs(new_n_anomalies - n_anomalies)} kayıt")

        st.subheader(f"Güncellenmiş En Şüpheli {top_n} Kayıt")
        st.dataframe(
            result_df_v2.head(top_n).style.background_gradient(subset=["anomaly_score"], cmap="Reds"),
            use_container_width=True,
            hide_index=True,
        )

        csv_v2 = result_df_v2.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Güncellenmiş Sonuçları İndir",
            data=csv_v2,
            file_name="anomaly_results_v2.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── Download ─────────────────────────────────────────────────────────────────
st.header(f"{section_offset + 2}. Sonuçları İndir")

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
    label=f"En Şüpheli {top_n} Kaydı İndir",
    data=csv_anomalies,
    file_name="top_anomalies.csv",
    mime="text/csv",
    use_container_width=True,
)
