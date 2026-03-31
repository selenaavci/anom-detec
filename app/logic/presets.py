"""Preset scenario configurations for common dataset types."""

PRESETS = {
    "Genel (Varsayılan)": {
        "description": "Herhangi bir veri seti tipi için genel ayarlar.",
        "contamination": 0.05,
        "model": "isolation_forest",
        "tips": [
            "Sayısal sütunları tercih edin",
            "ID ve tarih sütunlarını hariç tutun",
        ],
    },
    "İşlem / Finans Verisi": {
        "description": "Ödeme, Fatura, Sigorta Talepleri, Banka İşlem Kayıtları.",
        "contamination": 0.02,
        "model": "isolation_forest",
        "tips": [
            "Tutar, adet, oran gibi sayısal sütunları seçin",
            "Müşteri ID sütunlarını hariç tutun",
            "Düşük kontaminasyon oranı sahtecilik tespitinde daha etkilidir",
        ],
    },
    "İnsan Kaynakları Verisi": {
        "description": "Çalışan Performans, Maaş, Devamsızlık, İşe Alım Verileri.",
        "contamination": 0.05,
        "model": "lof",
        "tips": [
            "Maaş, deneyim yılı, performans puanı gibi sütunları seçin",
            "İsim ve TC kimlik no sütunlarını hariç tutun",
        ],
    },
    "Çağrı Merkezi Verisi": {
        "description": "Çağrı Süresi, Bekleme Süresi, Memnuniyet Puanı, Çağrı Hacmi Verileri.",
        "contamination": 0.05,
        "model": "isolation_forest",
        "tips": [
            "Çağrı süresi, bekleme süresi, memnuniyet skoru seçin",
            "Agent ID ve müşteri ID sütunlarını hariç tutun",
            "Aşırı uzun veya kısa çağrılarını tespit etmeye odaklanın",
        ],
    },
}
