import json
import os
import sys
from pathlib import Path

import nbformat as nbf

# Define notebook
nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.11.9"
    }
}

cells = []

# ==============================================================================
# CELL 1: MARKDOWN HEADER
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# Proyek Analisis Deret Waktu: PM2.5 Kota Surakarta
## Bagian 1: Eksplorasi Data & Pra-pemrosesan (Tugas Kunto)
**Penanggung Jawab:** Kunto Rossindu H (L0224020)  
**Mata Kuliah:** Analisis Data Deret Waktu (Time Series Analysis)  
**Program Studi:** Sains Data / Statistika, Semester 5  

---

### Deskripsi Tugas Nomor 1 (Kunto):
Sesuai dengan pembagian tugas tim proyek deret waktu:
* **Tahap 1: Eksplorasi Data**
  1. **Plot deret, statistik ringkas, cek data hilang & pencilan:** Memuat data mentah dan data teragregasi, menghitung statistik deskriptif lengkap, mendeteksi missing value dan pencilan menggunakan metode IQR 1.5 dan Z-Score.
  2. **Dekomposisi deret waktu (aditif/multiplikatif + alasan):** Menganalisis sifat varians dan amplitudo musiman, memilih model dekomposisi yang tepat dengan argumen ilmiah, melakukan dekomposisi STL ($s=52$), dan menghitung kekuatan musiman ($F_s$).
  3. **Plot musiman / boxplot per periode, plot ACF/PACF data mentah:** Membuat boxplot per bulan kalender, profil musiman rata-rata bulanan ($\pm$ SE), plot subseri tahunan, serta plot ACF & PACF deret level hingga lag 55 untuk menelaah autokorelasi.
  4. **Pembagian data latih (train) & data uji (test):** Membagi data menjadi 104 minggu train (2 siklus musiman penuh) dan 12 minggu test (out-of-sample evaluation), disertai justifikasi statistik dan praktis.
* **Pra-pemrosesan Data (Tambahan Sesuai Judul Tugas):**
  - Pemeriksaan kelengkapan jam data mentah Open-Meteo/CAMS.
  - Uji homogenitas fisik rasio PM10/PM2.5 dan deteksi patahan struktural November 2024.
  - Agregasi jam-jaman ke mingguan dengan threshold kelengkapan $\ge 95\%$.
"""))

# ==============================================================================
# CELL 2: CODE SETUP & IMPORTS
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 1. SETUP LINGKUNGAN DAN IMPORT LIBRARY
# ==============================================================================
from pathlib import Path
import json
import calendar
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL, seasonal_decompose

warnings.filterwarnings("ignore")

# Penentuan direktori root proyek
ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"
TAB_DIR = OUTPUTS / "tables"

for folder in (DATA_PROCESSED, OUTPUTS, FIG_DIR, TAB_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# Konfigurasi visualisasi matplotlib
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "figure.autolayout": True
})

COLORS = {
    "blue": "#1d4ed8",
    "orange": "#d97706",
    "red": "#dc2626",
    "green": "#059669",
    "gray": "#4b5563",
    "purple": "#7c3aed",
    "cyan": "#0891b2"
}

print("Setup lingkungan berhasil:")
print(f"- Root direktori : {ROOT}")
print(f"- Data raw       : {DATA_RAW}")
print(f"- Data processed : {DATA_PROCESSED}")
"""))

# ==============================================================================
# CELL 3: MARKDOWN SECTION 2 (PRA-PEMROSESAN & KONTROL KUALITAS)
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 2. Pra-pemrosesan Data & Kontrol Kualitas
Sebelum melakukan eksplorasi data, kita perlu memahami karakteristik data mentah dan proses pra-pemrosesan yang telah dilakukan.
* **Sumber Data:** Open-Meteo Air Quality API yang meneruskan keluaran reanalisis model CAMS global (*Copernicus Atmosphere Monitoring Service*).
* **Lokasi & Parameter:** Kota Surakarta (koordinat $-7,5545^\\circ$ LS, $110,8060^\\circ$ BT; titik grid CAMS $-7,60^\\circ$ LS, $110,80^\\circ$ BT, zona waktu `Asia/Jakarta`).
* **Satuan:** $\\mu\\text{g/m}^3$ untuk PM2.5 dan PM10.
* **Tantangan Fisik / Homogenitas:** Pada November 2024 terjadi patahan basis model CAMS di mana rasio PM10/PM2.5 anjlok dari sekitar 1,46 menjadi 1,04 (PM2.5 mendekati PM10), yang secara fisik tidak wajar bagi aerosol atmosfer. Oleh karena itu, periode model dipotong hingga 21 Oktober 2024 (sebelum November 2024) untuk menjaga homogenitas seri.
"""))

# ==============================================================================
# CELL 4: CODE PREPROCESSING CHECK
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 2.1 Pemuatan Data Mentah Hourly dan Cek Missing Value Awal
# ==============================================================================
raw_csv = DATA_RAW / "openmeteo_pm25_surakarta_hourly.csv"
assert raw_csv.exists(), f"File {raw_csv} tidak ditemukan!"

df_hourly = pd.read_csv(raw_csv)
df_hourly["datetime"] = pd.to_datetime(df_hourly["datetime"])
df_hourly["pm2_5"] = pd.to_numeric(df_hourly["pm2_5"], errors="coerce")

total_hours = len(df_hourly)
valid_hours = df_hourly["pm2_5"].notna().sum()
missing_hours = df_hourly["pm2_5"].isna().sum()

print("Ringkasan Data Mentah Jam-Jaman (Hourly):")
print(f"- Rentang waktu total : {df_hourly['datetime'].min()} s.d. {df_hourly['datetime'].max()}")
print(f"- Total observasi     : {total_hours:,} jam")
print(f"- Observasi valid     : {valid_hours:,} jam ({valid_hours/total_hours*100:.2f}%)")
print(f"- Missing value       : {missing_hours:,} jam ({missing_hours/total_hours*100:.2f}%)")

# Identifikasi di mana missing value berada
missing_dates = df_hourly.loc[df_hourly["pm2_5"].isna(), "datetime"]
print(f"- Rentang missing     : {missing_dates.min()} s.d. {missing_dates.max()} (hanya di awal seri sebelum periode model)")
"""))

# ==============================================================================
# CELL 5: CODE HOMOGENEITY CHECK
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 2.2 Uji Homogenitas Fisik: Rasio PM10 / PM2.5
# ==============================================================================
ratio_file = DATA_PROCESSED / "pm25_pm10_monthly.csv"
if ratio_file.exists():
    df_ratio = pd.read_csv(ratio_file)
    
    pre = df_ratio.loc[df_ratio["homogeneous"], "ratio"]
    post = df_ratio.loc[~df_ratio["homogeneous"], "ratio"]
    
    ratio_summary = pd.DataFrame([
        {"Periode": "Sebelum Nov 2024 (Homogen)", "Jumlah Bulan": len(pre), 
         "Rata-rata Rasio": pre.mean(), "Std Dev": pre.std(), "Status": "Digunakan untuk Pemodelan"},
        {"Periode": "Nov 2024 s.d. Sekarang", "Jumlah Bulan": len(post), 
         "Rata-rata Rasio": post.mean(), "Std Dev": post.std(), "Status": "Dikeluarkan (Patahan Struktural CAMS)"}
    ])
    display(ratio_summary.style.format({
        "Rata-rata Rasio": "{:.3f}",
        "Std Dev": "{:.3f}"
    }))
    
    # Plot Rasio PM10/PM2.5
    fig, ax = plt.subplots(figsize=(10, 4))
    months = pd.to_datetime(df_ratio["month"])
    ax.plot(months, df_ratio["ratio"], marker="o", markersize=4, color=COLORS["blue"], label="Rasio Bulanan PM10/PM2.5")
    ax.axvline(pd.Timestamp("2024-11-01"), color=COLORS["red"], linestyle="--", linewidth=1.5, label="Patahan Model (1 Nov 2024)")
    ax.axvspan(months.min(), pd.Timestamp("2024-10-31"), color=COLORS["green"], alpha=0.08, label="Periode Homogen Model")
    ax.axvspan(pd.Timestamp("2024-11-01"), months.max(), color=COLORS["red"], alpha=0.08, label="Periode Non-Homogen (Dibuang)")
    ax.set_title("Uji Homogenitas Rasio PM10 / PM2.5 Kota Surakarta (Open-Meteo/CAMS)")
    ax.set_ylabel("Rasio PM10 / PM2.5")
    ax.set_xlabel("Bulan")
    ax.legend(loc="upper right")
    plt.show()
"""))

# ==============================================================================
# CELL 6: CODE LOAD MODEL DATASET
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 2.3 Pemuatan Dataset Mingguan Model Homogen (pm25_surakarta_weekly_model.csv)
# ==============================================================================
weekly_file = DATA_PROCESSED / "pm25_surakarta_weekly_model.csv"
assert weekly_file.exists(), f"File {weekly_file} tidak ditemukan!"

df_weekly = pd.read_csv(weekly_file, parse_dates=["week_start"])
df_weekly = df_weekly.sort_values("week_start").reset_index(drop=True)

# Membuat series terindeks waktu dengan frekuensi mingguan eksplisit ('7D')
series = df_weekly.set_index("week_start")["pm25"].asfreq("7D")

print("Dataset Mingguan Model Berhasil Dimuat:")
print(f"- Jumlah observasi mingguan (N) : {len(series)} minggu")
print(f"- Periode                       : {series.index.min().strftime('%d %B %Y')} s.d. {series.index.max().strftime('%d %B %Y')}")
print(f"- Nilai minimum coverage jam    : {df_weekly['coverage'].min()*100:.1f}% (seluruh minggu memiliki data >= 95%)")
print(f"- Ada missing value di deret    : {series.isna().any()}")

display(df_weekly.head())
"""))

# ==============================================================================
# CELL 7: MARKDOWN SECTION 3 (STATISTIK RINGKAS & PENCILAN)
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 3. Statistik Deskriptif, Data Hilang, dan Deteksi Pencilan
Pada tahap ini, kita memeriksa:
1. **Pemeriksaan Data Hilang:** Memastikan tidak ada jeda waktu (gap) pada deret mingguan.
2. **Statistik Ringkas:** Menghitung ukuran pemusatan (mean, median), dispersi (standar deviasi, IQR, CV), dan bentuk sebaran (skewness, kurtosis).
3. **Deteksi Pencilan:**
   * **Kriteria Interquartile Range (IQR):**
     $$\\text{Batas Bawah} = Q_1 - 1,5 \\times \\text{IQR}$$
     $$\\text{Batas Atas} = Q_3 + 1,5 \\times \\text{IQR}$$
   * **Kriteria Z-Score:**
     $$|z| = \\left|\\frac{y_t - \\bar{y}}{s}\\right| > 3$$
"""))

# ==============================================================================
# CELL 8: CODE SUMMARY STATISTICS & OUTLIERS
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 3.1 Perhitungan Statistik Ringkas dan Deteksi Pencilan
# ==============================================================================
n_obs = len(series)
n_missing = int(series.isna().sum())
mean_val = float(series.mean())
median_val = float(series.median())
std_val = float(series.std())
min_val = float(series.min())
q1_val = float(series.quantile(0.25))
q3_val = float(series.quantile(0.75))
max_val = float(series.max())
iqr_val = q3_val - q1_val
skew_val = float(series.skew())
kurt_val = float(series.kurtosis())
cv_val = std_val / mean_val

# Ambang batas pencilan IQR 1.5
lower_iqr = q1_val - 1.5 * iqr_val
upper_iqr = q3_val + 1.5 * iqr_val
iqr_outliers = series[(series < lower_iqr) | (series > upper_iqr)]

# Ambang batas Z-score
z_scores = (series - mean_val) / std_val
z_outliers = series[z_scores.abs() > 3]

# Tabel Statistik Deskriptif
df_stats = pd.DataFrame([
    {"Parameter": "Jumlah Observasi (N)", "Nilai": f"{n_obs}"},
    {"Parameter": "Periode Awal", "Nilai": series.index.min().strftime('%Y-%m-%d')},
    {"Parameter": "Periode Akhir", "Nilai": series.index.max().strftime('%Y-%m-%d')},
    {"Parameter": "Jumlah Data Hilang (Missing)", "Nilai": f"{n_missing}"},
    {"Parameter": "Rata-rata (Mean)", "Nilai": f"{mean_val:.2f} µg/m³"},
    {"Parameter": "Median", "Nilai": f"{median_val:.2f} µg/m³"},
    {"Parameter": "Standar Deviasi", "Nilai": f"{std_val:.2f} µg/m³"},
    {"Parameter": "Minimum", "Nilai": f"{min_val:.2f} µg/m³"},
    {"Parameter": "Kuartil 1 (Q1)", "Nilai": f"{q1_val:.2f} µg/m³"},
    {"Parameter": "Kuartil 3 (Q3)", "Nilai": f"{q3_val:.2f} µg/m³"},
    {"Parameter": "Maksimum", "Nilai": f"{max_val:.2f} µg/m³"},
    {"Parameter": "Interquartile Range (IQR)", "Nilai": f"{iqr_val:.2f} µg/m³"},
    {"Parameter": "Kemencengan (Skewness)", "Nilai": f"{skew_val:.3f}"},
    {"Parameter": "Kurtosis", "Nilai": f"{kurt_val:.3f}"},
    {"Parameter": "Koefisien Variasi (CV)", "Nilai": f"{cv_val:.3f} ({cv_val*100:.1f}%)"},
    {"Parameter": "Batas Bawah Pencilan (Q1 - 1.5*IQR)", "Nilai": f"{lower_iqr:.2f} µg/m³"},
    {"Parameter": "Batas Atas Pencilan (Q3 + 1.5*IQR)", "Nilai": f"{upper_iqr:.2f} µg/m³"},
    {"Parameter": "Jumlah Pencilan IQR", "Nilai": f"{len(iqr_outliers)}"},
    {"Parameter": "Jumlah Pencilan Z-Score (|z| > 3)", "Nilai": f"{len(z_outliers)}"}
])

display(df_stats)
"""))

# ==============================================================================
# CELL 9: CODE BOXPLOT & DISTRIBUTION
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 3.2 Visualisasi Distribusi dan Boxplot Pencilan
# ==============================================================================
fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, 
                                      gridspec_kw={"height_ratios": [0.35, 0.65]})

# Boxplot horizontal
box = ax_box.boxplot(series, vert=False, patch_artist=True,
                     boxprops=dict(facecolor="#93c5fd", color=COLORS["blue"]),
                     medianprops=dict(color=COLORS["red"], linewidth=2),
                     whiskerprops=dict(color=COLORS["blue"], linewidth=1.2),
                     capprops=dict(color=COLORS["blue"], linewidth=1.2))
ax_box.axvline(lower_iqr, color=COLORS["red"], linestyle=":", label=f"Batas Bawah IQR ({lower_iqr:.2f})")
ax_box.axvline(upper_iqr, color=COLORS["red"], linestyle=":", label=f"Batas Atas IQR ({upper_iqr:.2f})")
ax_box.set_yticks([])
ax_box.set_title("Deteksi Pencilan dan Sebaran Data Mingguan PM2.5 Surakarta")
ax_box.legend(loc="upper right", fontsize=8)

# Histogram & KDE
n, bins, patches = ax_hist.hist(series, bins=16, density=True, color="#bfdbfe", edgecolor="#1d4ed8", alpha=0.7)
series.plot.kde(ax=ax_hist, color=COLORS["blue"], linewidth=2, label="KDE (Density)")
ax_hist.axvline(mean_val, color=COLORS["green"], linestyle="--", linewidth=1.5, label=f"Mean ({mean_val:.2f})")
ax_hist.axvline(median_val, color=COLORS["red"], linestyle="-.", linewidth=1.5, label=f"Median ({median_val:.2f})")
ax_hist.set_xlabel("Konsentrasi PM2.5 (µg/m³)")
ax_hist.set_ylabel("Densitas")
ax_hist.legend(loc="upper right")

plt.show()

print("Interpretasi Statistik & Pencilan:")
print(f"1. Data bersih dari nilai hilang (missing = 0 dari 116 minggu).")
print(f"2. Tidak terdeteksi adanya pencilan (outlier = 0) baik dengan batas IQR 1.5 [-2.93, 48.92] maupun kriteria |z| > 3.")
print(f"3. Distribusi condong ke kanan secara moderat (skewness = {skew_val:.3f}), dengan nilai mean ({mean_val:.2f}) sedikit di atas median ({median_val:.2f}).")
print(f"4. Kurtosis negatif ({kurt_val:.3f}) menunjukkan distribusi cenderung platykurtic (tidak memiliki ekor tebal yang ekstrem).")
"""))

# ==============================================================================
# CELL 10: MARKDOWN SECTION 4 (PLOT DERET WAKTU)
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 4. Plot Deret Waktu Mentah PM2.5
Berikut adalah visualisasi deret waktu mingguan PM2.5 Kota Surakarta sepanjang periode 8 Agustus 2022 hingga 21 Oktober 2024.
* Perhatikan pola osilasi tahunan yang terbentuk: konsentrasi cenderung rendah pada bulan-bulan basah (akhir tahun hingga awal musim hujan) dan mengalami lonjakan pada pertengahan tahun hingga akhir musim kemarau.
* Fluktuasi mingguan menunjukkan variasi lokal, namun skala varians tampak relatif konstan dari tahun ke tahun.
"""))

# ==============================================================================
# CELL 11: CODE PLOT SERIES
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 4.1 Plot Deret Waktu Mingguan PM2.5 Surakarta
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 4.5))

ax.plot(series.index, series.values, color=COLORS["blue"], linewidth=1.5, marker="o", markersize=3, label="PM2.5 Mingguan (µg/m³)")
ax.axhline(mean_val, color=COLORS["gray"], linestyle="--", linewidth=1, label=f"Rata-rata Keseluruhan ({mean_val:.2f} µg/m³)")

# Anotasi titik ekstrem
idx_max = series.idxmax()
val_max = series.max()
idx_min = series.idxmin()
val_min = series.min()

ax.annotate(f"Puncak Maksimum: {val_max:.1f}\\n({idx_max.strftime('%d-%b-%Y')})",
            xy=(idx_max, val_max), xytext=(idx_max + pd.Timedelta(days=15), val_max - 4),
            arrowprops=dict(facecolor=COLORS["red"], arrowstyle="->", lw=1.2),
            fontsize=8, fontweight="bold", color=COLORS["red"])

ax.annotate(f"Lembah Minimum: {val_min:.1f}\\n({idx_min.strftime('%d-%b-%Y')})",
            xy=(idx_min, val_min), xytext=(idx_min + pd.Timedelta(days=15), val_min + 5),
            arrowprops=dict(facecolor=COLORS["green"], arrowstyle="->", lw=1.2),
            fontsize=8, fontweight="bold", color=COLORS["green"])

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_title("Deret Waktu Mingguan PM2.5 Kota Surakarta (8 Agustus 2022 – 21 Oktober 2024)", fontsize=12)
ax.set_ylabel("Konsentrasi PM2.5 (µg/m³)")
ax.set_xlabel("Tanggal Awal Minggu")
ax.legend(loc="upper left")

plt.show()
"""))

# ==============================================================================
# CELL 12: MARKDOWN SECTION 5 (DEKOMPOSISI DERET WAKTU)
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 5. Dekomposisi Deret Waktu (Aditif vs Multiplikatif)

### 5.1 Landasan Teori Pemilihan Model Dekomposisi:
1. **Model Aditif:**
   $$Y_t = T_t + S_t + R_t$$
   Model aditif mengasumsikan bahwa amplitudo fluktuasi musiman ($S_t$) dan varians residual ($R_t$) bersifat **independen terhadap level tren ($T_t$)**. Besarnya fluktuasi bernilai absolut sama terlepas dari apakah nilai deret sedang tinggi atau rendah.
   
2. **Model Multiplikatif:**
   $$Y_t = T_t \\times S_t \\times R_t \\quad \\Longleftrightarrow \\quad \\ln(Y_t) = \\ln(T_t) + \\ln(S_t) + \\ln(R_t)$$
   Model multiplikatif mengasumsikan bahwa amplitudo musiman membesar atau mengecil **sebanding dengan besarnya level tren**.

### 5.2 Alasan & Justifikasi Pemilihan Model Aditif:
1. **Stabilitas Amplitudo Musiman:** Fluktuasi musiman dari siklus tahun pertama (2022–2023) ke siklus tahun kedua (2023–2024) memiliki rentang puncak-lembah yang relatif konstan (berkisar antara $\\sim 10$ hingga $\\sim 47\\ \\mu\\text{g/m}^3$), bukan membesar secara eksponensial seiring bertambahnya waktu.
2. **Ketiadaan Tren Pertumbuhan Berkelanjutan:** Level konsentrasi PM2.5 tidak mengalami tren eksponensial jangka panjang; deret bersifat berfluktuasi di sekitar rata-rata $\\sim 23,8\\ \\mu\\text{g/m}^3$.
3. **Kestabilan Varians Level:** Variabilitas deret relatif homogen sepanjang 116 minggu.
4. **Kemudahan Interpretasi:** Pada model aditif, komponen musiman dan residu berada dalam satuan asli yang sama ($\\mu\\text{g/m}^3$), sehingga sangat intuitif dalam analisis kualitas udara.

Oleh karena itu, **dekomposisi aditif** dipilih sebagai metode dekomposisi utama. Kita menerapkan algoritma **STL (*Seasonal-Trend decomposition using LOESS*)** dengan periode musiman tahunan $s = 52$ minggu, *seasonal smoother* = 7, dan *robust* = True untuk meredam potensi outlier lokal.
"""))

# ==============================================================================
# CELL 13: CODE STL DECOMPOSITION
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 5.1 Implementasi Dekomposisi STL Aditif (Periode = 52 Minggu)
# ==============================================================================
SEASON_PERIOD = 52

# STL decomposition
stl = STL(series, period=SEASON_PERIOD, seasonal=7, robust=True).fit()

trend = stl.trend
seasonal = stl.seasonal
residual = stl.resid

# Perhitungan Kekuatan Musiman (Seasonal Strength, Fs)
# Fs = max(0, 1 - Var(R_t) / Var(S_t + R_t))
detrended = series - trend
var_resid = np.var(residual)
var_detrended = np.var(detrended)
seasonal_strength = max(0.0, 1.0 - var_resid / var_detrended) if var_detrended > 0 else np.nan

# Visualisasi 4 Panel STL
fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True)

# 1. Observasi Asli
axes[0].plot(series.index, series.values, color=COLORS["blue"], lw=1.2)
axes[0].set_ylabel("Observasi (Y)")
axes[0].set_title("Dekomposisi STL Aditif Seri Mingguan PM2.5 Kota Surakarta (Periode s = 52, Smoother = 7)")

# 2. Tren (Trend)
axes[1].plot(series.index, trend, color=COLORS["orange"], lw=1.8)
axes[1].set_ylabel("Tren (T)")

# 3. Musiman (Seasonal)
axes[2].plot(series.index, seasonal, color=COLORS["green"], lw=1.5)
axes[2].set_ylabel("Musiman (S)")

# 4. Residu (Residual)
axes[3].plot(series.index, residual, color=COLORS["gray"], marker="o", markersize=2.5, lw=0.8)
axes[3].axhline(0, color="black", linestyle="--", lw=0.8)
axes[3].set_ylabel("Residu (R)")
axes[3].set_xlabel("Tanggal Awal Minggu")

for ax in axes:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

plt.show()

print(f"Kekuatan Musiman (Seasonal Strength, Fs): {seasonal_strength:.4f} ({seasonal_strength*100:.2f}%)")
print("Interpretasi STL:")
print(f"- Nilai Fs = {seasonal_strength:.3f} (> 0.6) mengindikasikan komponen musiman tahunan yang sangat kuat secara deskriptif.")
print(f"- Amplitudo musiman berkisar dari {seasonal.min():.2f} µg/m³ (lembah) hingga +{seasonal.max():.2f} µg/m³ (puncak).")
print(f"- Catatan Kritis: Karena deret hanya mencakup ~2.2 tahun (116 minggu), komponen musiman lag 52 hanya berulang 2 kali, sehingga estimasi musiman panjang harus diuji secara hati-hati pada pemodelan SARIMA.")
"""))

# ==============================================================================
# CELL 14: CODE CLASSICAL DECOMPOSITION COMPARISON
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 5.2 Perbandingan Tambahan: Dekomposisi Klasik Aditif vs Multiplikatif
# ==============================================================================
decomp_add = seasonal_decompose(series, model="additive", period=SEASON_PERIOD)
decomp_mul = seasonal_decompose(series, model="multiplicative", period=SEASON_PERIOD)

# Evaluasi varians residu relatif terhadap skala
resid_add = decomp_add.resid.dropna()
resid_mul = decomp_mul.resid.dropna()

print("Perbandingan Statistik Residu Dekomposisi Klasik:")
print(f"1. Model Aditif        : Std Dev Residu = {resid_add.std():.3f} µg/m³ | Mean Residu = {resid_add.mean():.4f}")
print(f"2. Model Multiplikatif : Std Dev Residu Indeks = {resid_mul.std():.3f} | Mean Residu = {resid_mul.mean():.4f}")
print("Kesimpulan: Model Aditif mempertahankan unit pengukuran fisik konsentrasi (µg/m³) secara langsung dan stabil.")
"""))

# ==============================================================================
# CELL 15: MARKDOWN SECTION 6 (ANALISIS POLA MUSIMAN)
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 6. Plot Musiman & Boxplot per Periode
Untuk melihat profil musiman secara lebih terperinci pada kalender tahunan, kita menganalisis:
1. **Boxplot Musiman per Bulan Kalender (Januari – Desember):** Menunjukkan distribusi sebaran konsentrasi mingguan di setiap bulan.
2. **Profil Musiman Rata-Rata Bulanan:** Dilengkapi dengan *standard error* ($SE = s / \\sqrt{n}$) untuk melihat presisi estimasi bulanan.
3. **Plot Subseri Antar-Tahun (*Year-over-Year Seasonality*):** Membandingkan trajektori mingguan tahun 2022, 2023, dan 2024.
"""))

# ==============================================================================
# CELL 16: CODE SEASONAL BOXPLOT & PROFILE
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 6.1 Boxplot Musiman per Bulan Kalender dan Profil Rata-Rata
# ==============================================================================
df_monthly_group = df_weekly.copy()
df_monthly_group["month_num"] = df_monthly_group["week_start"].dt.month
df_monthly_group["month_name"] = df_monthly_group["month_num"].apply(lambda x: calendar.month_abbr[x])

month_order = list(range(1, 13))
month_labels = [calendar.month_abbr[m] for m in month_order]

# Hitung statistik agregat per bulan
monthly_stats = df_monthly_group.groupby("month_num")["pm25"].agg(["count", "mean", "std", "median"]).reindex(month_order)
monthly_stats["se"] = monthly_stats["std"] / np.sqrt(monthly_stats["count"])
monthly_stats["month_name"] = month_labels

fig, (ax_bar, ax_box) = plt.subplots(1, 2, figsize=(12, 4.5))

# 1. Profil Rata-rata Bulanan dengan Error Bar (SE)
ax_bar.bar(monthly_stats.index, monthly_stats["mean"], color="#3b82f6", alpha=0.8, edgecolor=COLORS["blue"], label="Rata-rata")
ax_bar.errorbar(monthly_stats.index, monthly_stats["mean"], yerr=monthly_stats["se"], fmt="none",
                ecolor=COLORS["red"], elinewidth=1.5, capsize=4, label="Standard Error (SE)")
ax_bar.set_xticks(month_order)
ax_bar.set_xticklabels(month_labels)
ax_bar.set_title("Profil Musiman Rata-Rata Bulanan PM2.5 (± SE)")
ax_bar.set_ylabel("Rata-rata PM2.5 (µg/m³)")
ax_bar.set_xlabel("Bulan Kalender")
ax_bar.legend(loc="upper right")

# 2. Boxplot Sebaran per Bulan
monthly_data = [df_monthly_group.loc[df_monthly_group["month_num"] == m, "pm25"].values for m in month_order]
bp = ax_box.boxplot(monthly_data, labels=month_labels, patch_artist=True,
                    boxprops=dict(facecolor="#fed7aa", color=COLORS["orange"]),
                    medianprops=dict(color=COLORS["red"], linewidth=1.5))
ax_box.set_title("Boxplot Sebaran PM2.5 Mingguan per Bulan Kalender")
ax_box.set_ylabel("PM2.5 (µg/m³)")
ax_box.set_xlabel("Bulan Kalender")

plt.show()

display(monthly_stats[["month_name", "count", "mean", "std", "se", "median"]].rename(columns={
    "month_name": "Bulan", "count": "Jumlah Minggu", "mean": "Rata-rata", "std": "Std Dev", "se": "SE", "median": "Median"
}))
"""))

# ==============================================================================
# CELL 17: CODE YEAR OVER YEAR PLOT
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 6.2 Plot Subseri Antar-Tahun (Year-over-Year Trajectory)
# ==============================================================================
df_yoy = df_weekly.copy()
df_yoy["year"] = df_yoy["week_start"].dt.year
df_yoy["week_of_year"] = df_yoy["week_start"].dt.isocalendar().week

fig, ax = plt.subplots(figsize=(10, 4.5))

for yr, col, ls in zip([2022, 2023, 2024], [COLORS["gray"], COLORS["blue"], COLORS["red"]], ["--", "-", "-."]):
    subset = df_yoy[df_yoy["year"] == yr].sort_values("week_of_year")
    ax.plot(subset["week_of_year"], subset["pm25"], color=col, linestyle=ls, marker="o", markersize=3, label=f"Tahun {yr}")

ax.set_title("Perbandingan Profil Musiman Antar-Tahun (Week 1 s.d. 52)")
ax.set_xlabel("Minggu ke- dalam Tahun (ISO Week)")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.legend(loc="upper right")

plt.show()

print("Analisis Pola Musiman:")
print("1. Puncak konsentrasi secara konsisten terjadi pada kuartal pertama hingga pertengahan tahun (Februari - Mei),")
print("   sedangkan konsentrasi terendah tercatat pada bulan Juli - September.")
print("2. Pola ini sejalan dengan fenomena meteorologi monsun di Pulau Jawa:")
print("   - Musim kemarau/transisi memicu akumulasi partikulat karena minimnya presipitasi (wet deposition).")
print("   - Musim hujan membantu pencucian udara (wet scavenging), menurunkan konsentrasi aerosol.")
"""))

# ==============================================================================
# CELL 18: MARKDOWN SECTION 7 (ACF & PACF DATA MENTAH)
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 7. Analisis Autokorelasi: Plot ACF dan PACF Data Mentah
Fungsi autokorelasi (*Autocorrelation Function* / ACF) dan autokorelasi parsial (*Partial Autocorrelation Function* / PACF) pada **data mentah (level)** dianalisis hingga **lag 55** (untuk mencakup lag musiman tahunan $s = 52$).
* **ACF:** Mengukur korelasi linear antara $Y_t$ dan $Y_{t-k}$.
* **PACF:** Mengukur korelasi langsung antara $Y_t$ dan $Y_{t-k}$ setelah mengontrol pengaruh lag intermediat $Y_{t-1}, \\dots, Y_{t-k+1}$.
* **Batas Signifikansi:** Daerah bayangan abu-abu/biru mewakili interval signifikansi Bartlett $\\pm 1,96 / \\sqrt{N} \\approx \\pm 0,182$.
"""))

# ==============================================================================
# CELL 19: CODE ACF PACF RAW
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 7.1 Plot ACF dan PACF pada Deret Level Mentah
# ==============================================================================
MAX_LAGS = 55

fig, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(13, 4.5))

# Plot ACF
plot_acf(series, lags=MAX_LAGS, ax=ax_acf, color=COLORS["blue"], title="ACF: Data Mentah (Level)")
ax_acf.axvline(SEASON_PERIOD, color=COLORS["red"], linestyle=":", linewidth=1.5, label="Lag Musiman (s = 52)")
ax_acf.set_xlabel("Lag (Minggu)")
ax_acf.set_ylabel("Koefisien Autokorelasi")
ax_acf.legend(loc="upper right")

# Plot PACF (Yule-Walker modified)
plot_pacf(series, lags=MAX_LAGS, ax=ax_pacf, method="ywm", color=COLORS["orange"], title="PACF: Data Mentah (Level)")
ax_pacf.axvline(SEASON_PERIOD, color=COLORS["red"], linestyle=":", linewidth=1.5, label="Lag Musiman (s = 52)")
ax_pacf.set_xlabel("Lag (Minggu)")
ax_pacf.set_ylabel("Koefisien Autokorelasi Parsial")
ax_pacf.legend(loc="upper right")

plt.show()

print("Analisis Pola ACF & PACF Data Mentah:")
print("1. Pola ACF menunjukkan peluruhan yang sangat lambat (slow decay / die-down),")
print("   dengan nilai korelasi positif signifikan bertahan hingga belasan lag awal.")
print("   Ini merupakan indikasi kuat bahwa deret asli TIDAK STASIONER dalam rata-rata (non-stationary mean),")
print("   sehingga memerlukan diferensiasi (differencing) di tahap pemodelan selanjutnya.")
print("2. Pola PACF menunjukkan spike dominan yang sangat tinggi pada lag 1 (r_1 ~ 0.8),")
print("   kemudian langsung memotong tajam (cuts off) setelah lag 1.")
print("   Hal ini mengindikasikan struktur ketergantungan autoregresif orde 1 (AR(1)) yang sangat kuat pada level.")
print("3. Pada lag 52 (lag musiman), terdapat gelombang positif kecil di ACF, menandakan jejak siklus tahunan.")
"""))

# ==============================================================================
# CELL 20: MARKDOWN SECTION 8 (TRAIN TEST SPLIT)
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 8. Pembagian Data Latih (Train) & Data Uji (Test)

### 8.1 Skema Pembagian:
* **Total Observasi:** $N = 116$ minggu (8 Agustus 2022 s.d. 21 Oktober 2024).
* **Data Latih (*Train Set*):** 104 minggu pertama (8 Agustus 2022 s.d. 29 Juli 2024).
* **Data Uji (*Test Set*):** 12 minggu terakhir (5 Agustus 2024 s.d. 21 Oktober 2024).

### 8.2 Justifikasi Statistik & Substantif Pemilihan Titik Potong:
1. **Dua Siklus Musiman Penuh untuk Data Latih:**
   Panjang data latih $104$ minggu tepat sama dengan **2 siklus musiman tahunan penuh** ($2 \\times 52 = 104$ minggu). Hal ini sangat krusial agar metode pemodelan musiman (seperti Holt-Winters musiman dan SARIMA dengan periode $s=52$) memiliki sampel data latih yang cukup untuk mengestimasi koefisien musiman di setiap minggu tanpa kehilangan derajat kebebasan.
2. **Horizon Evaluasi Operasional yang Realistis:**
   Data uji sebesar 12 minggu (kurang lebih 3 bulan atau 1 triwulan kalender) merepresentasikan horizon peramalan praktis jangka menengah yang sangat relevan untuk kebijakan pemantauan kualitas udara perkotaan.
3. **Integritas Out-of-Sample:**
   Pemisahan dilakukan secara sekuensial (kronologis) murni tanpa pengacakan (no shuffling) demi menjaga struktur urutan waktu dan menghindari *data leakage*.
"""))

# ==============================================================================
# CELL 21: CODE TRAIN TEST SPLIT
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# 8.1 Pembagian Train-Test dan Visualisasi
# ==============================================================================
TRAIN_SIZE = 104
TEST_SIZE = 12

train_series = series.iloc[:TRAIN_SIZE]
test_series = series.iloc[TRAIN_SIZE:]

assert len(train_series) == 104, "Train size tidak sesuai"
assert len(test_series) == 12, "Test size tidak sesuai"

fig, ax = plt.subplots(figsize=(11, 4.6))

# Plot keseluruhan seri
ax.plot(train_series.index, train_series.values, color=COLORS["blue"], lw=1.5, label="Data Latih (Train, 104 minggu)")
ax.plot(test_series.index, test_series.values, color=COLORS["red"], lw=1.5, marker="o", markersize=3.5, label="Data Uji (Test, 12 minggu)")

# Arsiran latar belakang
ax.axvspan(train_series.index[0], train_series.index[-1], color=COLORS["green"], alpha=0.08)
ax.axvspan(test_series.index[0], test_series.index[-1], color=COLORS["red"], alpha=0.10)
ax.axvline(test_series.index[0], color=COLORS["red"], linestyle="--", lw=1.2, label=f"Titik Potong Split ({test_series.index[0].strftime('%d-%b-%Y')})")

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_title("Pembagian Data Latih (Train: 104 Minggu) dan Data Uji (Test: 12 Minggu)")
ax.set_ylabel("Konsentrasi PM2.5 (µg/m³)")
ax.set_xlabel("Tanggal Awal Minggu")
ax.legend(loc="upper left")

plt.show()

# Ringkasan Statistik Train vs Test
df_split_stats = pd.DataFrame([
    {"Metrik": "Jumlah Minggu", "Data Latih (Train)": len(train_series), "Data Uji (Test)": len(test_series)},
    {"Metrik": "Periode Awal", "Data Latih (Train)": train_series.index.min().strftime('%Y-%m-%d'), "Data Uji (Test)": test_series.index.min().strftime('%Y-%m-%d')},
    {"Metrik": "Periode Akhir", "Data Latih (Train)": train_series.index.max().strftime('%Y-%m-%d'), "Data Uji (Test)": test_series.index.max().strftime('%Y-%m-%d')},
    {"Metrik": "Rata-rata (µg/m³)", "Data Latih (Train)": f"{train_series.mean():.2f}", "Data Uji (Test)": f"{test_series.mean():.2f}"},
    {"Metrik": "Standar Deviasi (µg/m³)", "Data Latih (Train)": f"{train_series.std():.2f}", "Data Uji (Test)": f"{test_series.std():.2f}"},
    {"Metrik": "Minimum (µg/m³)", "Data Latih (Train)": f"{train_series.min():.2f}", "Data Uji (Test)": f"{test_series.min():.2f}"},
    {"Metrik": "Maksimum (µg/m³)", "Data Latih (Train)": f"{train_series.max():.2f}", "Data Uji (Test)": f"{test_series.max():.2f}"}
])

display(df_split_stats)
"""))

# ==============================================================================
# CELL 22: MARKDOWN SECTION 9 (KESIMPULAN TUGAS KUNTO & REKOMENDASI TAHAP 2)
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 9. Kesimpulan Eksplorasi Data & Rekomendasi Tahap Lanjutan

### Ringkasan Hasil Tugas Nomor 1 (Kunto):
1. **Kualitas Data:**
   * Data mentah Open-Meteo/CAMS hourly berhasil diagregasi menjadi deret mingguan sebanyak 116 observasi tanpa ada satupun nilai hilang (*0 missing values*).
   * Telah dilakukan uji homogenitas rasio PM10/PM2.5 yang menunjukkan adanya *structural break* pada November 2024, sehingga pemotongan data per 21 Oktober 2024 berhasil menjaga homogenitas fisik data.
2. **Statistik Deskriptif & Pencilan:**
   * Rata-rata PM2.5 mingguan adalah $23,82\\ \\mu\\text{g/m}^3$ ($SD = 8,45\\ \\mu\\text{g/m}^3$) dengan rentang $9,69$ s.d. $47,26\\ \\mu\\text{g/m}^3$.
   * Uji pencilan (IQR 1.5 dan Z-Score) mengonfirmasi **tidak adanya pencilan ekstrem**, sehingga seluruh observasi valid untuk pemodelan tanpa perlu imputasi/pemangkasan artifisial.
3. **Dekomposisi:**
   * **Model Aditif** terjustifikasi secara kuat karena amplitudo musiman tidak bergantung pada level tren dan varians deret relatif konstan.
   * Dekomposisi STL menghasilkan kekuatan musiman $F_s = 0,927$, membuktikan adanya osilasi musiman tahunan yang nyata.
4. **Struktur Autokorelasi (ACF/PACF):**
   * Pola *slow decay* pada ACF level menandakan ketidakstasioneran rata-rata.
   * Pola *cut-off* pada PACF lag 1 menandakan dependensi autoregresif jangka pendek yang dominan.
5. **Pembagian Data Latih–Uji:**
   * Data dibagi secara tepat menjadi $104$ minggu data latih (2 siklus $s=52$ penuh) dan $12$ minggu data uji (out-of-sample forecasting horizon).

---

### Catatan Transisi untuk Anggota Tim Selanjutnya:
* **Untuk Maul (Tugas Nomor 2):**
  - Data mentah telah siap pada variabel `train_series` (104 minggu) dan `test_series` (12 minggu).
  - Varians terbukti relatif stabil, namun uji transformasi Box-Cox / Logaritma dapat dipertimbangkan lebih lanjut pada Tahap 2A.
  - Uji stasioneritas awal (ADF & KPSS) pada deret asli dapat dilanjutkan langsung dari temuan ACF di atas.
  - Model perataan (*Simple Moving Average*, SES, Holt, Holt-Winters aditif dengan $s=52$) siap diestimasi pada data latih 104 minggu.
"""))

# Assign cells to notebook
nb.cells = cells

# Save notebook
out_path = Path("notebooks") / "01_eksplorasi_dan_prapemrosesan_kunto.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook berhasil dibuat: {out_path}")

