# 🔍 Anomaly Detection Agent

## Project Overview
Anomaly Detection Agent, kullanıcı tarafından yüklenen veri setleri üzerinde **otomatik anomali tespiti** yaparak, olağandışı kayıtları **iş birimi tarafından incelenebilir bir formatta** sunmak üzere tasarlanmış bir analiz modülüdür.

Klasik makine öğrenmesi (Isolation Forest, Local Outlier Factor, One-Class SVM) ile üç model paralel çalıştırılır; sonuçları **ensemble** edilerek tek bir modelin zayıf olduğu anomali türlerinin gözden kaçması engellenir. Her kayıt için anomali skoru üretilir, sapma açıklamaları çıkarılır (sayısal + kategorik + kombinasyon + kural ihlali) ve sonuçlar Excel raporu olarak indirilebilir. Kullanıcı geri bildirimleri **oturumlar arası kalıcı** olarak saklanır ve preset model ağırlıklarının zaman içinde iyileşmesinde kullanılır.

> ⚠️ **LLM entegrasyonu şu an aktif değildir.** Mevcut versiyon tamamen klasik ML üzerine kuruludur. İleride doğal dil açıklama katmanı (Phase 2) eklenmek üzere planlanmıştır.

---

## 🎯 Project Purpose
Ham veri setlerinden olağandışı kayıtları çıkararak, bu kayıtları iş diliyle anlaşılabilir hale getirmek ve kurum içinde **denetim, risk yönetimi ve operasyonel kalite süreçlerinin hızlanmasını sağlamak**.

Skor üretmenin ötesine geçerek, her şüpheli kayıt için **hangi alanların ne kadar saptığını**, **hangi değerlerin nadir olduğunu** ve **hangi kuralların ihlal edildiğini** göstermek; kullanıcı geri bildirimini modele geri besleyerek **iyileştirme döngüsü** kurmak hedeflenir.

---

## 👥 Target Use Cases

### 1. Financial Transaction Monitoring
- Ödeme, fatura, banka işlemleri üzerinde olağandışı tutar/davranış tespiti
- Sigorta tazminat taleplerinde sahte/aşırı kayıt analizi
- Mutabakat raporlarında tutarsız satır tespiti

### 2. HR & Workforce Analytics
- Maaş, performans ve devam kayıtlarında uç değer tespiti
- Çalışan davranışında ani değişim analizi
- Eğitim katılım anomalilerinin tespiti

### 3. Operational Quality Control
- Çağrı merkezi süre/bekleme/memnuniyet metriklerinde sapma analizi
- Üretim/sensör loglarında olağandışı davranış tespiti
- Veri kalitesi denetiminde kuraldışı kayıt tespiti

---

## ⚙️ End-to-End Workflow

1. **Data Upload**
   Kullanıcı CSV, Excel (XLS/XLSX), XML veya ZIP dosyasını yükler. Maksimum upload boyutu 1 GB.

2. **Automatic Data Analysis**
   Sistem veri tiplerini otomatik analiz eder:
   - Sayısal kolonlar (ad ve içerik bazlı sezgi)
   - Kategorik kolonlar
   - Tarih alanları
   - ID benzeri ve anlamsız kolonlar

3. **Column Type Control (Yeni)**
   Sistem her kolonu otomatik tipler; kullanıcı yanlış algılananları açılır listeden değiştirir:
   - `Sayısal/Tutar`, `Kategorik/Metin`, `Tarih`, `ID/Referans`, `Analiz dışı`
   - Sayısal'a çevrilen kolonlar için **dönüşüm raporu**: kaç değer çevrildi, kaç değer çevrilemedi, çevrilemeyen örnekler
   - Türkçe ondalık (`1.234,56`), kontrol karakteri (`\n\t\r`), para birimi sembolleri otomatik temizlenir

4. **Column Quality Analysis**
   Her kolon için kalite işaretleri çıkarılır:
   - `id_like`: yüksek benzersizlik oranı
   - `constant`: tek değerli kolon
   - `free_text`: serbest metin alanı
   - `high_cardinality`: çok kategorili alan

5. **Identifier Selection**
   Tanımlayıcı (ID) kolonları otomatik tespit edilir. Kullanıcı seçtiği ID kolonları sonuçlarda referans olarak gösterilir; analize dahil edilmez.

6. **Column Coverage Control (Yeni)**
   Sabit `safe_defaults[:15]` kesimi yerine **dengeli kapsam** önerilir:
   - Sayısal alanlar + düşük/orta kardinaliteli kategorikler birlikte
   - Hangi kolonun **dahil**, hangisinin **hariç** olduğunu ve nedenini gösteren rapor
   - Prefix bazlı **kolon grupları** (ör. `originator_*`, `beneficiary_*`, `customer_*`) otomatik tespit edilir
   - Beneficiary tarafı artık otomatik kapsamda — kör nokta yok

7. **Data Quality Checks (Yeni)**
   ML'den önce bağımsız bir kural katmanı çalışır. 11 kural sınıfı: negatif/sıfır tutar, parse-edilemeyen sayı veya tarih, kontrol karakteri, isim alanında sadece rakam, şubede ülke adı gibi beklenmeyen değer, beklenmeyen para birimi, tekrarlayan referans, aşırı uzun metin, uzak gelecek/geçmiş tarih, boş zorunlu alan. Sonuç hem **kullanıcıya kalite uyarısı** olarak hem de istenirse modele **binary feature** olarak girer.

8. **Preprocessing Pipeline**
   - Missing value handling (numeric → median, categorical → mode)
   - OneHotEncoding (preset eşiği — 15-25 kategori)
   - Frequency encoding (yüksek kardinaliteli kategorikler için)
   - **Rare category flag + freq** (her kategorik için ek özellik)
   - **Combo frequency** (örn. para birimi + kanal + işlem tipi nadir kombinasyonlar)
   - **Rule-based binary features** (DQ kontrol bayraklarından)
   - StandardScaler normalizasyonu

9. **Model Benchmarking + Ensemble**
   Üç algoritma paralel çalıştırılır: **Isolation Forest**, **Local Outlier Factor**, **One-Class SVM**. Tek bir "en iyi model" seçimi yerine üç sonuç birden tutulur ve **ensemble skor** üretilir:
   - Her modelin percentile rank'ı (preset ağırlıklı) ortalaması
   - Herhangi bir modelin top-%5'ine giren kayıtlar (`any-of-top` mask)
   - Otomatik seçimde silhouette skoru **preset model ağırlığı** ile çarpılır → kategorik anomali için LOF'un, tutar uçları için IF/OCSVM'in görünmemesi engellenir
   - UI'da sekmeler: `Önerilen (Ensemble)`, `Isolation Forest`, `Local Outlier Factor`, `One-Class SVM`

10. **Anomaly Scoring & Ranking**
    - Her kayda anomali skoru atanır
    - Kayıtlar şüphelilikten en az şüpheliye sıralanır
    - Top-N en şüpheli kayıtlar tabloda gösterilir

11. **Feature-Level + Categorical Explanation (Genişletildi)**
    Her şüpheli kayıt için kategorize edilmiş açıklamalar:
    - **Sayısal sapmalar** — değer, tipik medyan, z-skor yönü
    - **Nadir kategorik değerler** — değerin global frekansı yüzde olarak
    - **Olağan dışı kombinasyon** — seçilen kolonların birlikte ne sıklıkta görüldüğü
    - **Kural ihlalleri** — DQ katmanından gelen tetiklenen kurallar

12. **Temporal Anomaly Detection (opsiyonel)**
    - Tarih kolonu varsa rolling mean/std analizi yapılır
    - `rolling_mean ± 2.5 * rolling_std` bandı dışındaki noktalar işaretlenir
    - Trend, güven bandı ve şüpheli noktalar görselleştirilir

13. **Persistent Human Feedback Loop (Genişletildi)**
    - Kullanıcı her şüpheli kaydı **gerçek anomali** veya **yanlış alarm** olarak işaretler
    - Geri bildirimler `~/.anomaly_detection_feedback.json` dosyasında **oturumlar arası kalıcı** saklanır
    - Dataset adı + kolon seçimi hash'i ile tekrar açılınca aynı feedback'ler hatırlanır
    - Birikim (≥10 gözlem) preset bazında model ağırlıklarını otomatik ayarlar

14. **Semi-Supervised Retraining**
    - Yanlış alarm olarak işaretlenen kayıtlar eğitime dahil edilir
    - Gerçek anomali olarak işaretlenen kayıtlar eğitimden çıkarılır
    - Güncellenmiş skorlar ve sıralama yeniden üretilir

15. **Output & Reporting**
    - Excel export (.xlsx) — anomali satırları **neon sarı** ile vurgulanır
    - Tüm sonuçlar veya yalnızca top-N şüpheli kayıt ayrı ayrı indirilebilir
    - Geri bildirim sonrası güncellenmiş rapor da indirilebilir

---

## 🧩 Architecture Overview

**Core Layers:**

- **Data Loading & Type Inference Layer**
  CSV / Excel / XML / ZIP okuma; ad ve içerik bazlı tip tespiti; numeric coercion (Türkçe ondalık + kontrol karakteri temizleme); kullanıcı override'ı için `apply_column_kinds` API.

- **Data Quality Layer (Yeni)**
  Kural tabanlı 11 kontrol; severity bazında raporlama; row-level binary feature üretimi.

- **Feature Engineering Layer**
  Dengeli default seçim; OneHot + frequency + rare-flag + combo-frequency + rule-feature kombinasyonu; StandardScaler.

- **ML Layer (Anomaly Engine)**
  Üç klasik algoritma paralel; **rank-percentile ensemble**; preset-ağırlıklı silhouette ile model seçimi; yarı denetimli yeniden eğitim için Isolation Forest.

- **Explanation Layer (Genişletildi)**
  Sayısal z-skor + kategorik nadir değer + kombinasyon nadirliği + kural ihlali sinyallerini birleştiren `explain_record` API'si.

- **Temporal Layer**
  Tarih kolonu olan veri setleri için rolling istatistik tabanlı zaman serisi anomali tespiti.

- **UI Layer (Streamlit)**
  Veri yükleme, kolon tipi kontrolü, kapsam kontrolü, DQ paneli, ensemble sekmeli sonuçlar, kayıt inceleme, geri bildirim toplama.

- **Feedback Persistence Layer (Yeni)**
  Dataset signature bazlı JSON store; preset bazında precision-aware model ağırlık türetimi.

- **Export & Reporting Layer**
  Anomali satırları sarı ile vurgulanmış Excel çıktıları.

---

## 🤖 Model & Technology Stack

### Machine Learning
- **Isolation Forest** — random partitioning ile anomali izolasyonu (genel amaçlı, tutar uçları için güçlü)
- **Local Outlier Factor (LOF)** — yoğunluk tabanlı, küme içi outlier tespiti (kategorik manipülasyon için güçlü)
- **One-Class SVM** — RBF kernel, normal sınırını çizen model
- **Ensemble** — rank-percentile ortalaması + any-of-top mask
- **Preset-Weighted Silhouette** — model seçim metriği (preset ağırlığı ile çarpılır)
- **Rolling Statistics** — zaman serisi anomali tespiti

### Backend & UI
- Python 3.10+
- Pandas / NumPy
- Scikit-learn (IsolationForest, LocalOutlierFactor, OneClassSVM, StandardScaler, OneHotEncoder)
- Plotly (görselleştirmeler)
- Streamlit (Cloud için ≥1.30, Windows terminal için 1.26.0 desteği)
- OpenPyXL (Excel export & vurgulama)
- lxml (XML parsing için)

---

## 🧠 ML Strategy

Anomali tespiti, erken aşamalarda LLM tabanlı mimarilerden çok klasik ML için uygundur. Sebepleri:

- Daha düşük karmaşıklık
- Daha hızlı geliştirme
- İç ortamlara daha kolay deployment
- Daha düşük operasyonel maliyet
- Daha öngörülebilir davranış
- Dış AI servislerine bağımlılık yok

LLM, mevcut versiyonda **kullanılmaz**. Phase 2'de yorumlama katmanı olarak eklenmesi planlanır:
- Şüpheli kayıt için doğal dil açıklaması
- Konuşmalı arayüz
- Otomatik root-cause yorumu

Bu yaklaşım, hem **açıklanabilirlik** hem de **kontrol edilebilirlik** sağlar.

---

## 📊 Example Output

Her şüpheli kayıt için sistem aşağıdaki çıktıları üretir:

- Anomaly Score (ensemble veya tek model)
- Rank (en şüpheliden başlayarak)
- ID Reference (kullanıcı seçtiği tanımlayıcı kolonlardan)
- **Sapma açıklamaları** — sayısal, kategorik, kombinasyon, kural ihlali kategorilerinde
- Excel Export (anomali satırları **neon sarı** ile)
- Time Series Visualization (uygulanabilir veri setleri için)
- Feedback Summary (gerçek anomali / yanlış alarm sayıları, kalıcı kayıt durumu)

### Preset Senaryolar

Presetler artık sadece ipucu değil; **contamination aralığı**, **model ağırlıkları**, **rule check toggles**, **allowed currencies**, **encoding stratejisi** ve **numeric hint listesi** preset bazında değişir.

| Preset | Açıklama | Önerilen Contamination | Model Ağırlığı (IF / LOF / OCSVM) |
|--------|----------|------------------------|------------------------------------|
| Genel (Varsayılan) | Genel amaçlı veri setleri | 5% | 1.0 / 1.0 / 1.0 |
| İşlem / Finans Verisi | Ödeme, fatura, sigorta talepleri | 2% | 1.1 / 1.3 / 1.0 |
| İnsan Kaynakları Verisi | Maaş, performans, devam | 5% | 1.0 / 1.2 / 0.9 |
| Çağrı Merkezi Verisi | Süre, bekleme, memnuniyet | 5% | 1.1 / 1.0 / 1.0 |
| Operasyon / Üretim Verisi | Sensör, lojistik, üretim hattı | 4% | 1.2 / 1.0 / 1.0 |

---

## 🔐 Banking & Compliance Considerations

- Kişisel veri ve hassas veri kullanımına dikkat edilmelidir
- Model çıktıları **karar destek aracı** olarak konumlandırılmalıdır
- ID kolonları analiz dışı tutulur; yalnızca referans olarak gösterilir
- Feature seçim süreci kullanıcı kontrolünde tutulur
- Kalite uyarıları kullanıcıya açıkça iletilir (id_like / constant / free_text / high_cardinality + 11 DQ kuralı)
- Açıklanabilirlik (sayısal + kategorik + kombinasyon + kural ihlali) ön planda tutulur
- Geri bildirim döngüsü insan denetimini koruyacak şekilde tasarlanmıştır
- Feedback dosyası kullanıcının lokal makinesinde saklanır (`~/.anomaly_detection_feedback.json`); merkezi sunucuya gönderilmez

---

## 🚀 Business Impact

- Manuel denetim süresini ciddi ölçüde azaltır
- Teknik olmayan ekiplerin olağandışı kayıtları tespit etmesini sağlar
- Risk, denetim ve operasyon ekiplerinin önceliklendirme süreçlerini hızlandırır
- Geri bildirim yoluyla zamanla daha hassas hale gelir (kalıcı feedback + preset ağırlık ayarı)
- Excel raporlarıyla mevcut kurumsal iş akışına kolayca entegre olur
- Açıklanabilir çıktılar sayesinde karar gerekçeleri loglanabilir hale gelir

---

## 🔮 Future Enhancements

- LLM tabanlı doğal dil açıklama katmanı (Phase 2)
- Konuşmalı kayıt sorgulama arayüzü
- Otomatik feature öneri motoru
- Drift detection entegrasyonu
- API endpoint desteği
- Çok dosyalı / akış bazlı analiz
- Domain-spesifik konfigürasyon profillerinin paylaşılabilmesi (preset import/export)
