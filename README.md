# PM2.5 Surakarta Forecasting

Analisis deret waktu mingguan PM2.5 Kota Surakarta menggunakan data model CAMS global
melalui Open-Meteo Air Quality API. Proyek ini membandingkan MA, Exponential Smoothing,
AR, ARMA, ARIMA, SARIMA, dan Fourier-ARIMA, lalu meramalkan 12 minggu ke depan.

## Data

- **Sumber:** [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api)
  (keluaran [CAMS global](https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4)).
- **Lokasi:** Surakarta, `-7,5545; 110,8060`, zona waktu `Asia/Jakarta`.
- **Satuan:** µg/m³.
- **Periode model:** 8 Agustus 2022 – 21 Oktober 2024.
- **Jumlah data:** 116 minggu, tanpa nilai hilang; 104 minggu latih dan 12 minggu uji.
- **Catatan:** periode setelah Oktober 2024 dikeluarkan karena patahan rasio PM10/PM2.5
  pada model CAMS (sekitar 1,46 menjadi 1,04).

Data CAMS adalah **grid model**, bukan stasiun pengukuran darat Surakarta.

## Hasil Utama

| Metrik | Nilai |
|---|---:|
| Model terpilih | SARIMA(2,1,0)(1,0,0)[52] |
| RMSE uji | 4,81 µg/m³ |
| MAE uji | 2,95 µg/m³ |
| MAPE uji | 15,95% |
| MASE uji | 0,38 |
| Naive RMSE | 4,64 µg/m³ |
| Uji Diebold-Mariano vs naive | p = 0,197 (tidak signifikan) |

Model menang menurut AICc, tetapi **tidak berbeda signifikan dari naive random walk**.
Peramalan 12 minggu relatif datar sekitar 14–15 µg/m³ dengan interval 80% dan 95%
yang melebar. Keterbatasan utama: seri hanya 2,2 siklus tahunan, suku musiman tidak
signifikan, dan residual heteroskedastik.

## Struktur Repo

```text
.
├── src/
│   ├── fetch_openmeteo.py      # unduh data mentah
│   ├── preprocess.py           # agregasi harian/mingguan/bulanan
│   ├── check_homogeneity.py    # cek patahan PM10/PM2.5
│   ├── analysis.py             # pipeline analisis dan peramalan
│   └── build_report.py         # generator laporan DOCX
├── notebooks/
│   └── 01_analisis_pm25_surakarta.ipynb
├── data/
│   ├── raw/                    # data mentah CSV/JSON
│   └── processed/              # data siap analisis
├── outputs/
│   ├── figures/                # 10 gambar
│   ├── tables/                 # tabel CSV
│   ├── results.json
│   ├── Hasil_Analisis_PM25_Surakarta.xlsx
│   ├── Laporan_PM25_Surakarta.docx
│   └── Laporan_PM25_Surakarta.pdf
├── requirements.txt
└── README.md
```

## Cara Menjalankan

```bash
pip install -r requirements.txt

# Pipeline lengkap
python src/fetch_openmeteo.py
python src/preprocess.py
python src/check_homogeneity.py
python src/analysis.py
python src/build_report.py
```

`fetch_openmeteo.py` memerlukan internet dan akan memperbarui data mentah. Untuk
reproduksi hasil yang sudah ada, cukup jalankan `analysis.py` karena data mentah dan
data olahan sudah tersedia di repo.

### Notebook

```bash
jupyter notebook notebooks/01_analisis_pm25_surakarta.ipynb
```

Notebook menjalankan `src/analysis.py`, lalu menampilkan tabel dan gambar utama.
Pipeline `.py` tetap menjadi sumber utama; notebook tidak menduplikasi logika model.

## Output

- `outputs/Laporan_PM25_Surakarta.pdf` — laporan lengkap.
- `outputs/Hasil_Analisis_PM25_Surakarta.xlsx` — seluruh tabel dalam satu workbook.
- `outputs/figures/` — gambar untuk laporan.
- `outputs/results.json` — ringkasan angka analisis.

## Lisensi dan Sitasi

Kode proyek ini dibuat untuk keperluan mata kuliah Analisis Data Time Series.
Data mengikuti ketentuan lisensi Open-Meteo/CAMS; sitasi sumber data tercantum di
laporan. Sitasi data:

> Copernicus Atmosphere Monitoring Service. (2024). *CAMS global reanalysis (EAC4)*.
> ECMWF. https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4

> Open-Meteo. (2026). *Air quality API documentation*.
> https://open-meteo.com/en/docs/air-quality-api
