"""Preset scenario configurations for common dataset types.

Each preset can change UI defaults and pipeline behavior:
    contamination_default / contamination_range
    model_weights         — bias auto-selection toward a model
    numeric_hints         — extra column-name fragments treated as numeric
    rule_checks           — toggles for data-quality checks
    allowed_currencies    — for the beklenmeyen_para_birimi check
    encoding_strategy     — onehot threshold, rare cutoff
"""

PRESETS = {
    "Genel (Varsayılan)": {
        "description": "Herhangi bir veri seti tipi için dengeli ayarlar.",
        "contamination_default": 0.05,
        "contamination_range": (0.01, 0.30),
        "model": "isolation_forest",
        "model_weights": {"isolation_forest": 1.0, "lof": 1.0, "ocsvm": 1.0},
        "numeric_hints": (),
        "rule_checks": {
            "isim_alanında_sadece_rakam": True,
            "kontrol_karakteri": True,
            "negatif_tutar": False,
            "sıfır_tutar": False,
            "aşırı_uç_değer": True,
            "tekrarlayan_id": True,
            "beklenmeyen_para_birimi": False,
        },
        "allowed_currencies": [],
        "encoding": {"onehot_max_unique": 15, "rare_threshold": 0.005},
        "tips": [
            "Sayısal sütunları tercih edin",
            "ID ve tarih sütunlarını hariç tutun",
        ],
    },
    "İşlem / Finans Verisi": {
        "description": "Ödeme, fatura, sigorta talepleri, banka işlem kayıtları.",
        "contamination_default": 0.02,
        "contamination_range": (0.005, 0.15),
        "model": "isolation_forest",
        # Finansta tutar uçları için IF/OCSVM güçlü, kategorik manipülasyon için LOF
        # tek başına seçim metriğine yenilmesin diye ağırlığı yukarı çekiyoruz.
        "model_weights": {"isolation_forest": 1.1, "lof": 1.3, "ocsvm": 1.0},
        "numeric_hints": ("amnttl", "amntfc", "blnc", "tutar", "bakiye", "fee", "ucret"),
        "rule_checks": {
            "isim_alanında_sadece_rakam": True,
            "kontrol_karakteri": True,
            "negatif_tutar": True,
            "sıfır_tutar": True,
            "aşırı_uç_değer": True,
            "tekrarlayan_id": True,
            "beklenmeyen_para_birimi": True,
        },
        "allowed_currencies": ["TRY", "USD", "EUR", "GBP", "CHF", "JPY", "CAD", "AUD",
                                "AED", "RUB", "CNY", "XAU", "XAG"],
        "encoding": {"onehot_max_unique": 20, "rare_threshold": 0.003},
        "tips": [
            "Tutar, adet, oran gibi sayısal sütunları seçin",
            "Müşteri ID sütunlarını hariç tutun",
            "Sahtecilik testlerinde düşük kontaminasyon (≤%2) önerilir",
            "Ensemble sekmesi tutar + kategorik manipülasyonu birlikte yakalar",
        ],
    },
    "İnsan Kaynakları Verisi": {
        "description": "Çalışan performans, maaş, devamsızlık, işe alım.",
        "contamination_default": 0.05,
        "contamination_range": (0.02, 0.20),
        "model": "lof",
        "model_weights": {"isolation_forest": 1.0, "lof": 1.2, "ocsvm": 0.9},
        "numeric_hints": ("salary", "maas", "maaş", "yas", "yaş", "deneyim",
                           "performance", "performans", "puan", "skor"),
        "rule_checks": {
            "isim_alanında_sadece_rakam": True,
            "kontrol_karakteri": True,
            "negatif_tutar": True,
            "sıfır_tutar": False,
            "aşırı_uç_değer": True,
            "tekrarlayan_id": True,
            "beklenmeyen_para_birimi": False,
        },
        "allowed_currencies": [],
        "encoding": {"onehot_max_unique": 25, "rare_threshold": 0.01},
        "tips": [
            "Maaş, deneyim yılı, performans puanı gibi sütunları seçin",
            "İsim ve TC kimlik no sütunlarını hariç tutun",
            "LOF, departman/grup içi sapmaları daha iyi yakalar",
        ],
    },
    "Çağrı Merkezi Verisi": {
        "description": "Çağrı süresi, bekleme, memnuniyet skoru, hacim.",
        "contamination_default": 0.05,
        "contamination_range": (0.02, 0.20),
        "model": "isolation_forest",
        "model_weights": {"isolation_forest": 1.1, "lof": 1.0, "ocsvm": 1.0},
        "numeric_hints": ("duration", "sure", "süre", "wait", "bekleme",
                           "satisfaction", "memnuniyet", "puan", "skor", "count"),
        "rule_checks": {
            "isim_alanında_sadece_rakam": False,
            "kontrol_karakteri": True,
            "negatif_tutar": True,
            "sıfır_tutar": True,
            "aşırı_uç_değer": True,
            "tekrarlayan_id": False,
            "beklenmeyen_para_birimi": False,
        },
        "allowed_currencies": [],
        "encoding": {"onehot_max_unique": 20, "rare_threshold": 0.01},
        "tips": [
            "Çağrı süresi, bekleme süresi, memnuniyet skoru seçin",
            "Agent ID ve müşteri ID sütunlarını hariç tutun",
            "Aşırı uzun/kısa çağrılarına odaklanın",
        ],
    },
    "Operasyon / Üretim Verisi": {
        "description": "Üretim hatları, sensör ölçümleri, lojistik kayıtları.",
        "contamination_default": 0.04,
        "contamination_range": (0.01, 0.20),
        "model": "isolation_forest",
        "model_weights": {"isolation_forest": 1.2, "lof": 1.0, "ocsvm": 1.0},
        "numeric_hints": ("temp", "sicaklik", "sıcaklık", "pressure", "basinc", "basınç",
                           "speed", "hiz", "hız", "rpm", "qty", "adet", "weight",
                           "agirlik", "ağırlık"),
        "rule_checks": {
            "isim_alanında_sadece_rakam": False,
            "kontrol_karakteri": True,
            "negatif_tutar": True,
            "sıfır_tutar": True,
            "aşırı_uç_değer": True,
            "tekrarlayan_id": False,
            "beklenmeyen_para_birimi": False,
        },
        "allowed_currencies": [],
        "encoding": {"onehot_max_unique": 20, "rare_threshold": 0.005},
        "tips": [
            "Sensör değerleri ve süre/adet sütunlarını seçin",
            "Hat/makine ID sütunlarını ID olarak işaretleyin",
            "Aşırı uç değerleri yakalamak için kontaminasyonu düşük tutun",
        ],
    },
}
