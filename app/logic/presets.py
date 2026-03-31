"""Preset scenario configurations for common dataset types."""

PRESETS = {
    "Genel (Varsayılan)": {
        "description": "Herhangi bir veri seti tipi icin genel ayarlar.",
        "contamination": 0.05,
        "model": "isolation_forest",
        "tips": [
            "Sayisal sutunlari tercih edin",
            "ID ve tarih sutunlarini haric tutun",
        ],
    },
    "Operasyon Verisi": {
        "description": "Uretim hatlari, makine sensoru verileri, lojistik operasyon verileri.",
        "contamination": 0.03,
        "model": "isolation_forest",
        "tips": [
            "Sensor olcumleri, sicaklik, basinc gibi sutunlari secin",
            "Zaman damgasi sutunlarini haric tutun",
            "Dusuk kontaminasyon orani nadir arizalari yakalar",
        ],
    },
    "Islem / Finans Verisi": {
        "description": "Odeme, fatura, sigorta talepleri, banka islem kayitlari.",
        "contamination": 0.02,
        "model": "isolation_forest",
        "tips": [
            "Tutar, adet, oran gibi sayisal sutunlari secin",
            "Musteri ID sutunlarini haric tutun",
            "Dusuk kontaminasyon orani sahtecilik tespitinde daha etkilidir",
        ],
    },
    "HR / Insan Kaynaklari Verisi": {
        "description": "Calisan performans, maas, devamsizlik, ise alim verileri.",
        "contamination": 0.05,
        "model": "lof",
        "tips": [
            "Maas, deneyim yili, performans puani gibi sutunlari secin",
            "Isim ve TC kimlik no sutunlarini haric tutun",
            "LOF modeli kume ici aykiri degerleri iyi yakalar",
        ],
    },
    "Cagri Merkezi Verisi": {
        "description": "Cagri suresi, bekleme suresi, memnuniyet puani, cagri hacmi verileri.",
        "contamination": 0.05,
        "model": "isolation_forest",
        "tips": [
            "Cagri suresi, bekleme suresi, memnuniyet skoru secin",
            "Agent ID ve musteri ID sutunlarini haric tutun",
            "Asiri uzun veya kisa cagrilari tespit etmeye odaklanin",
        ],
    },
}
