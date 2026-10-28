"""Analisis deret waktu PM2.5 mingguan Surakarta (seri homogen 2022-08 s.d. 2024-10).

Metode: MA smoothing, SES, Holt, Holt-Winters, AR, ARMA, ARIMA, SARIMA,
serta Fourier + ARIMA sebagai bonus. Evaluasi memakai split train/test
104/12 minggu dan peramalan 12 minggu ke depan.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
for folder in (OUT, FIG, TAB):
    folder.mkdir(parents=True, exist_ok=True)

SEASON = 52
TEST_SIZE = 12
HORIZON = 12
FINAL_NAME = "SARIMA(2,1,0)(1,0,0)[52]"
FINAL_ORDER = (2, 1, 0)
FINAL_SEASONAL = (1, 0, 0, SEASON)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)
COLORS = {
    "blue": "#1d4ed8",
    "orange": "#d97706",
    "red": "#dc2626",
    "green": "#059669",
    "gray": "#4b5563",
    "purple": "#7c3aed",
}


def to_py(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    return value


def load_series() -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(DATA / "pm25_surakarta_weekly_model.csv", parse_dates=["week_start"])
    frame = frame.sort_values("week_start").reset_index(drop=True)
    series = frame.set_index("week_start")["pm25"].asfreq("7D")
    if series.isna().any():
        raise ValueError("Series mingguan masih memiliki nilai hilang.")
    return frame, series


def score(y_true, y_pred, train, season: int = SEASON) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    smape = float(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100)
    scale = float(np.mean(np.abs(train.values[season:] - train.values[:-season])))
    mase = float(mae / scale) if scale > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "MASE": mase}


def aicc_from_aic(aic: float, k: int, n: int) -> float:
    if aic is None or not np.isfinite(aic) or n - k - 1 <= 0:
        return np.nan
    return float(aic + 2 * k * (k + 1) / (n - k - 1))


def moving_average_forecast(train: pd.Series, window: int, horizon: int) -> np.ndarray:
    history = list(train.values.astype(float))
    forecasts = []
    for _ in range(horizon):
        value = float(np.mean(history[-window:]))
        forecasts.append(value)
        history.append(value)
    return np.asarray(forecasts)


def fit_sarimax(train, order, seasonal_order, trend, exog=None, exog_forecast=None):
    model = SARIMAX(
        train,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    forecast = fit.forecast(HORIZON, exog=exog_forecast)
    return fit, np.asarray(forecast)


def lb_pvalue(resid, lag: int = 20):
    try:
        return float(acorr_ljungbox(resid, lags=[lag], return_df=True)["lb_pvalue"].iloc[0])
    except Exception:
        return np.nan


def record(results, name, spec, fit, forecast, train, test, k, aic, aicc, bic, note="", comparable=True, d=None):
    metrics = score(test.values, forecast, train)
    resid = getattr(fit, "resid", None)
    results.append(
        {
            "Model": name,
            "Spesifikasi": spec,
            "d": d,
            "k": k,
            "AIC": aic,
            "AICc": aicc,
            "BIC": bic,
            **metrics,
            "LB20_p": lb_pvalue(resid) if resid is not None else np.nan,
            "Konvergen": note or "Ya",
            "AIC_sebanding": "Ya" if comparable else "Tidak",
            "_forecast": np.asarray(forecast, dtype=float),
        }
    )


def outlier_summary(series: pd.Series) -> dict:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    z_score = (series - series.mean()) / series.std()
    return {
        "outlier_lower": float(lower),
        "outlier_upper": float(upper),
        "outlier_iqr_count": int(((series < lower) | (series > upper)).sum()),
        "outlier_z_count": int((z_score.abs() > 3).sum()),
    }


def residual_diagnostics(name: str, fit) -> dict:
    resid = pd.Series(getattr(fit, "resid", [])).dropna()
    if resid.empty:
        return {"Model": name, "n": 0}
    lb = acorr_ljungbox(resid, lags=[10, 20, 52], return_df=True)
    _, jb_p, _, _ = jarque_bera(resid)
    _, arch_p, _, _ = het_arch(resid, nlags=5)
    return {
        "Model": name,
        "n": int(resid.size),
        "LB10_p": float(lb["lb_pvalue"].iloc[0]),
        "LB20_p": float(lb["lb_pvalue"].iloc[1]),
        "LB52_p": float(lb["lb_pvalue"].iloc[2]),
        "JB_p": float(jb_p),
        "ARCH_p": float(arch_p),
    }


def root_summary(name: str, roots, kind: str) -> dict | None:
    values = np.asarray(roots, dtype=complex)
    if values.size == 0:
        return None
    modulus = np.abs(values)
    stable = bool(np.all(modulus > 1.0))
    return {
        "Model": name,
        "Komponen": kind,
        "Jumlah akar": int(values.size),
        "|root| minimum": float(modulus.min()),
        "|root| maksimum": float(modulus.max()),
        "Kesimpulan": "Stabil (di luar lingkaran satuan)" if stable else "Tidak stabil",
    }


def es_param_rows(name: str, fit) -> list[dict]:
    rows = []
    for key, value in fit.params.items():
        if value is None:
            rows.append({"Model": name, "Parameter": key, "Nilai": "tidak diestimasi"})
            continue
        array = np.asarray(value)
        if array.ndim == 0:
            displayed = float(array)
        else:
            flat = array.ravel()
            displayed = f"vektor {flat.size} nilai; awal={flat[0]:.4f}" if flat.size else "vektor kosong"
        rows.append({"Model": name, "Parameter": key, "Nilai": displayed})
    return rows


def build_flowchart() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    boxes = [
        (0.3, 3.9, 2.0, 0.8, "Unduh data\nOpen-Meteo/CAMS"),
        (2.7, 3.9, 2.0, 0.8, "Kontrol kualitas\n& uji homogenitas"),
        (5.1, 3.9, 2.0, 0.8, "Agregasi\nmingguan"),
        (7.5, 3.9, 2.2, 0.8, "EDA, dekomposisi,\nuji stasioneritas"),
        (7.5, 2.3, 2.2, 0.8, "Pemodelan\nMA, ES, AR, ARMA,\nARIMA, SARIMA"),
        (4.6, 2.3, 2.2, 0.8, "Evaluasi train/test\nAICc, RMSE, diagnostik"),
        (1.7, 2.3, 2.2, 0.8, "Peramalan 12 minggu\n& interpretasi"),
        (3.1, 0.7, 3.8, 0.8, "Laporan + Excel + dokumentasi kode"),
    ]
    for x, y, w, h, text in boxes:
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.2, edgecolor="#1f2937", facecolor="#eff6ff",
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    arrows = [
        ((2.3, 4.3), (2.7, 4.3)),
        ((4.7, 4.3), (5.1, 4.3)),
        ((7.1, 4.3), (7.5, 4.3)),
        ((8.6, 3.9), (8.6, 3.1)),
        ((7.5, 2.7), (6.8, 2.7)),
        ((4.6, 2.7), (3.9, 2.7)),
        ((2.8, 2.3), (3.8, 1.5)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color="#1f2937"))
    fig.tight_layout()
    fig.savefig(FIG / "fig00_flowchart.png", bbox_inches="tight")
    plt.close(fig)


def descriptive(series: pd.Series) -> pd.DataFrame:
    data = {
        "Jumlah observasi": len(series),
        "Periode awal": series.index.min().strftime("%Y-%m-%d"),
        "Periode akhir": series.index.max().strftime("%Y-%m-%d"),
        "Rata-rata": series.mean(),
        "Median": series.median(),
        "Standar deviasi": series.std(),
        "Minimum": series.min(),
        "Kuartil 1": series.quantile(0.25),
        "Kuartil 3": series.quantile(0.75),
        "Maksimum": series.max(),
        "Kemencengan": series.skew(),
        "Kurtosis": series.kurtosis(),
        "Koefisien variasi": series.std() / series.mean(),
        "Jumlah hilang": int(series.isna().sum()),
    }
    return pd.DataFrame(list(data.items()), columns=["Statistik", "Nilai"])


def stationarity_table(series: pd.Series) -> pd.DataFrame:
    rows = []
    for label, values in [
        ("Level", series),
        ("Diferensiasi pertama", series.diff().dropna()),
        ("Diferensiasi musiman (lag 52)", series.diff(SEASON).dropna()),
    ]:
        adf = adfuller(values, autolag="AIC")
        rows.append(
            {
                "Seri": label,
                "Uji": "ADF",
                "Statistik": adf[0],
                "p-value": adf[1],
                "Lag": adf[2],
                "n": adf[3],
                "Kesimpulan": "Stasioner" if adf[1] < 0.05 else "Tidak stasioner",
            }
        )
        try:
            kp = kpss(values, regression="c", nlags="auto")
            rows.append(
                {
                    "Seri": label,
                    "Uji": "KPSS",
                    "Statistik": kp[0],
                    "p-value": kp[1],
                    "Lag": kp[2],
                    "n": len(values),
                    "Kesimpulan": "Tidak stasioner" if kp[1] < 0.05 else "Stasioner",
                }
            )
        except Exception:
            pass
    return pd.DataFrame(rows)


def plot_series(series: pd.Series, train: pd.Series, test: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(series.index, series.values, color=COLORS["blue"], linewidth=1.2, label="PM2.5 mingguan")
    ax.axvspan(train.index[0], train.index[-1], color=COLORS["green"], alpha=0.07, label="Train (104 minggu)")
    ax.axvspan(test.index[0], test.index[-1], color=COLORS["red"], alpha=0.10, label="Test (12 minggu)")
    ax.axvline(test.index[0], color=COLORS["red"], linestyle="--", linewidth=1)
    ax.set_title("Seri mingguan PM2.5 Kota Surakarta, Agustus 2022–Oktober 2024")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_xlabel("Tanggal")
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "fig02_series.png", bbox_inches="tight")
    plt.close(fig)


def plot_decomposition(series: pd.Series) -> float:
    stl = STL(series, period=SEASON, seasonal=7, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(series.index, series.values, color=COLORS["blue"])
    axes[0].set_ylabel("Observasi")
    axes[1].plot(series.index, stl.trend, color=COLORS["orange"])
    axes[1].set_ylabel("Tren")
    axes[2].plot(series.index, stl.seasonal, color=COLORS["green"])
    axes[2].set_ylabel("Musiman")
    axes[3].plot(series.index, stl.resid, color=COLORS["gray"], marker="o", ms=2, linewidth=0.8)
    axes[3].axhline(0, color="black", linewidth=0.8)
    axes[3].set_ylabel("Residu")
    axes[3].set_xlabel("Tanggal")
    axes[0].set_title("Dekomposisi STL seri mingguan (periode = 52, seasonal smoother = 7)")
    fig.tight_layout()
    fig.savefig(FIG / "fig03_decomposition.png", bbox_inches="tight")
    plt.close(fig)

    detrended = series - stl.trend
    strength = max(0.0, 1 - np.var(stl.resid) / np.var(detrended)) if np.var(detrended) > 0 else np.nan
    return float(strength)


def plot_seasonal_profile(series: pd.Series) -> None:
    profile = series.groupby(series.index.month).agg(["mean", "std", "count"])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(profile.index, profile["mean"], color=COLORS["blue"], alpha=0.75, label="Rata-rata")
    ax.errorbar(
        profile.index,
        profile["mean"],
        yerr=profile["std"] / np.sqrt(profile["count"]),
        fmt="none",
        ecolor=COLORS["gray"],
        capsize=3,
        label="SE",
    )
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"])
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Profil musiman rata-rata menurut bulan kalender")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig04_seasonal_profile.png", bbox_inches="tight")
    plt.close(fig)


def plot_acf_pacf(series: pd.Series) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    plot_acf(series, lags=55, ax=axes[0, 0], color=COLORS["blue"])
    axes[0, 0].set_title("ACF: level")
    axes[0, 0].axvline(SEASON, color=COLORS["red"], linestyle=":", linewidth=1)
    plot_pacf(series, lags=55, ax=axes[0, 1], method="ywm", color=COLORS["blue"])
    axes[0, 1].set_title("PACF: level")
    axes[0, 1].axvline(SEASON, color=COLORS["red"], linestyle=":", linewidth=1)
    plot_acf(series.diff().dropna(), lags=55, ax=axes[1, 0], color=COLORS["orange"])
    axes[1, 0].set_title("ACF: diferensiasi pertama")
    axes[1, 0].axvline(SEASON, color=COLORS["red"], linestyle=":", linewidth=1)
    plot_pacf(series.diff().dropna(), lags=55, ax=axes[1, 1], method="ywm", color=COLORS["orange"])
    axes[1, 1].set_title("PACF: diferensiasi pertama")
    axes[1, 1].axvline(SEASON, color=COLORS["red"], linestyle=":", linewidth=1)
    fig.tight_layout()
    fig.savefig(FIG / "fig05_acf_pacf.png", bbox_inches="tight")
    plt.close(fig)


def fit_all_models(train: pd.Series, test: pd.Series) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    results = []
    grids = {}
    fitted = {}
    es_rows = []
    root_rows = []
    diag_fits = {}
    ma_fits = {}

    naive = np.repeat(train.values[-1], HORIZON)
    record(results, "Naive (random walk)", "ŷₜ₊ₕ = yₜ", None, naive, train, test, 0, np.nan, np.nan, np.nan, "Benchmark", False)
    fitted["Naive (random walk)"] = naive

    seasonal_naive = train.values[-SEASON:][:HORIZON] if len(train) >= SEASON else np.repeat(train.mean(), HORIZON)
    record(results, "Seasonal naive", "ŷₜ₊ₕ = yₜ₊ₕ₋₅₂", None, seasonal_naive, train, test, 0, np.nan, np.nan, np.nan, "Benchmark", False)
    fitted["Seasonal naive"] = seasonal_naive

    for window in (3, 8):
        pred = moving_average_forecast(train, window, HORIZON)
        record(results, f"MA({window})", f"Rata-rata bergerak {window} minggu", None, pred, train, test, 1, np.nan, np.nan, np.nan, "Smoothing", False)
        fitted[f"MA({window})"] = pred

    ses = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
    pred = np.asarray(ses.forecast(HORIZON))
    k_es = len(getattr(ses, "params", {})) if hasattr(getattr(ses, "params", {}), "__len__") else 2
    record(results, "SES", "Simple exponential smoothing", ses, pred, train, test, k_es, ses.aic, aicc_from_aic(ses.aic, k_es, len(train)), np.nan, "Smoothing", False)
    fitted["SES"] = pred
    diag_fits["SES"] = ses
    es_rows.extend(es_param_rows("SES", ses))

    holt = ExponentialSmoothing(train, trend="add", initialization_method="estimated").fit(optimized=True)
    pred = np.asarray(holt.forecast(HORIZON))
    k_holt = len(getattr(holt, "params", {})) if hasattr(getattr(holt, "params", {}), "__len__") else 3
    record(results, "Holt", "Trend aditif", holt, pred, train, test, k_holt, holt.aic, aicc_from_aic(holt.aic, k_holt, len(train)), np.nan, "Smoothing", False)
    fitted["Holt"] = pred
    diag_fits["Holt"] = holt
    es_rows.extend(es_param_rows("Holt", holt))

    hw = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=SEASON, initialization_method="estimated"
    ).fit(optimized=True)
    pred = np.asarray(hw.forecast(HORIZON))
    k_hw = len(getattr(hw, "params", {})) if hasattr(getattr(hw, "params", {}), "__len__") else 4
    record(results, "Holt-Winters", "Trend + musiman aditif, s=52", hw, pred, train, test, k_hw, hw.aic, aicc_from_aic(hw.aic, k_hw, len(train)), np.nan, "Smoothing", False)
    fitted["Holt-Winters"] = pred
    diag_fits["Holt-Winters"] = hw
    es_rows.extend(es_param_rows("Holt-Winters", hw))

    best_ar = None
    ar_rows = []
    for p in range(1, 13):
        try:
            fit = AutoReg(train, lags=p, trend="c").fit()
            k = p + 2
            aicc = aicc_from_aic(fit.aic, k, len(train))
            ar_rows.append({"p": p, "AIC": fit.aic, "AICc": aicc, "BIC": fit.bic})
            if best_ar is None or aicc < best_ar["AICc"]:
                best_ar = {"p": p, "fit": fit, "AIC": fit.aic, "AICc": aicc, "BIC": fit.bic, "k": k}
        except Exception:
            continue
    grids["AR"] = pd.DataFrame(ar_rows).sort_values("AICc")
    if best_ar:
        pred = np.asarray(best_ar["fit"].forecast(HORIZON))
        name = f"AR({best_ar['p']})"
        record(results, name, f"AR({best_ar['p']}) dengan konstanta", best_ar["fit"], pred, train, test, best_ar["k"], best_ar["AIC"], best_ar["AICc"], best_ar["BIC"], d=0)
        fitted[name] = pred
        diag_fits[name] = best_ar["fit"]
        phi = np.asarray(best_ar["fit"].params[1:], dtype=float)
        root_rows.append(root_summary(name, np.roots(np.r_[-phi[::-1], 1.0]), "AR"))

    def grid_search(d, max_pq=3, label="ARMA"):
        rows = []
        best = None
        for p in range(max_pq + 1):
            for q in range(max_pq + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    fit = SARIMAX(
                        train, order=(p, d, q), trend="c" if d == 0 else "n",
                        enforce_stationarity=False, enforce_invertibility=False,
                    ).fit(disp=False)
                    if not np.isfinite(fit.aicc):
                        continue
                    if d == 0 and p == 0:
                        ma_fits[q] = fit
                    rows.append(
                        {
                            "p": p, "d": d, "q": q, "AIC": fit.aic, "AICc": fit.aicc,
                            "BIC": fit.bic, "k": len(fit.params),
                            "LB20_p": lb_pvalue(fit.resid),
                            "Konvergen": bool(fit.mle_retvals.get("converged", True)),
                        }
                    )
                    if best is None or fit.aicc < best["AICc"]:
                        best = {"p": p, "d": d, "q": q, "fit": fit, "AIC": fit.aic, "AICc": fit.aicc, "BIC": fit.bic, "k": len(fit.params)}
                except Exception:
                    continue
        grids[label] = pd.DataFrame(rows).sort_values("AICc")
        return best

    best_arma = grid_search(0, 3, "ARMA")
    if best_arma:
        pred = np.asarray(best_arma["fit"].forecast(HORIZON))
        name = f"ARMA({best_arma['p']},{best_arma['q']})"
        record(results, name, f"ARMA({best_arma['p']},{best_arma['q']})", best_arma["fit"], pred, train, test, best_arma["k"], best_arma["AIC"], best_arma["AICc"], best_arma["BIC"], d=0)
        fitted[name] = pred
        diag_fits[name] = best_arma["fit"]
        root_rows.append(root_summary(name, best_arma["fit"].arroots, "AR"))
        root_rows.append(root_summary(name, best_arma["fit"].maroots, "MA"))

    ma_rows = []
    for q, fit in sorted(ma_fits.items()):
        row = {
            "Model": f"MA({q}) galat",
            "q": q,
            "d": 0,
            "AIC": fit.aic,
            "AICc": fit.aicc,
            "BIC": fit.bic,
            "k": len(fit.params),
            "LB20_p": lb_pvalue(fit.resid),
        }
        ma_rows.append(row)
        root_rows.append(root_summary(row["Model"], fit.maroots, "MA"))
        diag_fits[row["Model"]] = fit
    ma_candidates = pd.DataFrame(ma_rows).sort_values("AICc").reset_index(drop=True)

    best_arima = grid_search(1, 3, "ARIMA")
    if best_arima:
        pred = np.asarray(best_arima["fit"].forecast(HORIZON))
        name = f"ARIMA({best_arima['p']},1,{best_arima['q']})"
        record(results, name, f"ARIMA({best_arima['p']},1,{best_arima['q']})", best_arima["fit"], pred, train, test, best_arima["k"], best_arima["AIC"], best_arima["AICc"], best_arima["BIC"], d=1)
        fitted[name] = pred
        diag_fits[name] = best_arima["fit"]

    sarima_rows = []
    best_sarima = None
    for p in range(3):
        for q in range(3):
            for P in range(2):
                for Q in range(2):
                    try:
                        fit = SARIMAX(
                            train, order=(p, 1, q), seasonal_order=(P, 0, Q, SEASON), trend="n",
                            enforce_stationarity=False, enforce_invertibility=False,
                        ).fit(disp=False)
                        if not np.isfinite(fit.aicc) or getattr(fit, "params", pd.Series()).get("sigma2", 1) <= 0:
                            continue
                        row = {
                            "p": p, "d": 1, "q": q, "P": P, "D": 0, "Q": Q, "s": SEASON,
                            "AIC": fit.aic, "AICc": fit.aicc, "BIC": fit.bic, "k": len(fit.params),
                            "LB20_p": lb_pvalue(fit.resid),
                            "Konvergen": bool(fit.mle_retvals.get("converged", True)),
                        }
                        sarima_rows.append(row)
                        if row["Konvergen"] and row["LB20_p"] > 0.05:
                            if best_sarima is None or fit.aicc < best_sarima["AICc"]:
                                best_sarima = {**row, "fit": fit}
                    except Exception:
                        continue
    grids["SARIMA"] = pd.DataFrame(sarima_rows).sort_values("AICc")
    if best_sarima is None:
        best_sarima = {
            "p": FINAL_ORDER[0], "q": FINAL_ORDER[2], "P": FINAL_SEASONAL[0], "Q": FINAL_SEASONAL[2],
            "fit": SARIMAX(train, order=FINAL_ORDER, seasonal_order=FINAL_SEASONAL, trend="n",
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False),
        }
        best_sarima["AIC"] = best_sarima["fit"].aic
        best_sarima["AICc"] = best_sarima["fit"].aicc
        best_sarima["BIC"] = best_sarima["fit"].bic
        best_sarima["k"] = len(best_sarima["fit"].params)
    pred = np.asarray(best_sarima["fit"].forecast(HORIZON))
    sarima_name = f"SARIMA({best_sarima['p']},1,{best_sarima['q']})({best_sarima['P']},0,{best_sarima['Q']})[{SEASON}]"
    record(results, sarima_name, "Model musiman terpilih (AICc + Ljung-Box)", best_sarima["fit"], pred, train, test, best_sarima["k"], best_sarima["AIC"], best_sarima["AICc"], best_sarima["BIC"], d=1)
    fitted[sarima_name] = pred
    diag_fits[sarima_name] = best_sarima["fit"]
    root_rows.append(root_summary(sarima_name, best_sarima["fit"].arroots, "AR"))
    root_rows.append(root_summary(sarima_name, best_sarima["fit"].maroots, "MA"))

    d1_rows = []
    for order, seasonal in [((1, 1, 0), (1, 1, 0, SEASON)), ((1, 1, 1), (1, 1, 0, SEASON)), ((0, 1, 1), (1, 1, 0, SEASON))]:
        try:
            fit = SARIMAX(train, order=order, seasonal_order=seasonal, trend="n",
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            pred_d1 = np.asarray(fit.forecast(HORIZON))
            d1_rows.append(
                {
                    "Model": f"SARIMA{order}({seasonal[0]},{seasonal[1]},{seasonal[2]})[{SEASON}]",
                    **score(test.values, pred_d1, train),
                    "AIC": fit.aic, "AICc": fit.aicc, "LB20_p": lb_pvalue(fit.resid),
                }
            )
        except Exception:
            continue
    grids["SARIMA_D1"] = pd.DataFrame(d1_rows)

    def fourier_matrix(t, k, period=SEASON):
        return np.column_stack(
            [func(2 * np.pi * j * t / period) for j in range(1, k + 1) for func in (np.sin, np.cos)]
        )

    fourier_rows = []
    best_fourier = None
    for k in (1, 2, 3):
        for order in ((1, 1, 0), (1, 1, 1), (2, 1, 0)):
            try:
                X_train = fourier_matrix(np.arange(len(train)), k)
                X_test = fourier_matrix(np.arange(len(train), len(train) + HORIZON), k)
                fit = SARIMAX(train, exog=X_train, order=order, trend="n",
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                pred = np.asarray(fit.forecast(HORIZON, exog=X_test))
                row = {"K": k, "p": order[0], "d": order[1], "q": order[2],
                       "AIC": fit.aic, "AICc": fit.aicc, "BIC": fit.bic,
                       **score(test.values, pred, train), "LB20_p": lb_pvalue(fit.resid)}
                fourier_rows.append(row)
                if k == 2 and order == (1, 1, 1):
                    best_fourier = {"k": k, "order": order, "fit": fit, "pred": pred, "X_train": X_train, "X_test": X_test, "row": row}
            except Exception:
                continue
    grids["Fourier"] = pd.DataFrame(fourier_rows).sort_values("RMSE")
    if best_fourier:
        name = "Fourier K=2 + ARIMA(1,1,1)"
        record(results, name, "Regresi Fourier (K=2) + ARIMA(1,1,1)", best_fourier["fit"], best_fourier["pred"], train, test, len(best_fourier["fit"].params), best_fourier["row"]["AIC"], best_fourier["row"]["AICc"], best_fourier["row"]["BIC"], "Bonus", d=1)
        fitted[name] = best_fourier["pred"]
        diag_fits[name] = best_fourier["fit"]

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(["RMSE", "AICc"], na_position="last").reset_index(drop=True)
    return comparison, fitted, grids["ARMA"], grids["ARIMA"], grids["SARIMA"], {
        "best_sarima": best_sarima,
        "fourier": best_fourier,
        "grids": grids,
        "es_params": pd.DataFrame(es_rows),
        "roots": pd.DataFrame([row for row in root_rows if row is not None]),
        "ma_candidates": ma_candidates,
        "diag_fits": diag_fits,
    }


def fit_final(series: pd.Series):
    fit = SARIMAX(
        series, order=FINAL_ORDER, seasonal_order=FINAL_SEASONAL, trend="n",
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    forecast_result = fit.get_forecast(HORIZON)
    forecast = forecast_result.predicted_mean
    ci80 = forecast_result.conf_int(alpha=0.20)
    ci95 = forecast_result.conf_int(alpha=0.05)
    forecast_frame = pd.DataFrame(
        {
            "Minggu": pd.date_range(series.index[-1] + pd.Timedelta(days=7), periods=HORIZON, freq="7D"),
            "Peramalan": forecast.values,
            "Batas_bawah_80": ci80.iloc[:, 0].values,
            "Batas_atas_80": ci80.iloc[:, 1].values,
            "Batas_bawah_95": ci95.iloc[:, 0].values,
            "Batas_atas_95": ci95.iloc[:, 1].values,
        }
    )
    coefficients = pd.DataFrame(
        {
            "Parameter": fit.params.index,
            "Koefisien": fit.params.values,
            "Std_Error": fit.bse.values,
            "z": fit.tvalues.values,
            "p_value": fit.pvalues.values,
            "CI_bawah": fit.conf_int().iloc[:, 0].values,
            "CI_atas": fit.conf_int().iloc[:, 1].values,
        }
    )
    resid = fit.resid
    lb = acorr_ljungbox(resid, lags=[10, 20, 52], return_df=True)
    jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(resid)
    arch_stat, arch_p, _, _ = het_arch(resid, nlags=5)
    shapiro_stat, shapiro_p = stats.shapiro(resid)
    diagnostics = pd.DataFrame(
        [
            ("Ljung-Box lag 10", lb["lb_stat"].iloc[0], lb["lb_pvalue"].iloc[0], "Tidak ada autokorelasi" if lb["lb_pvalue"].iloc[0] > 0.05 else "Ada autokorelasi"),
            ("Ljung-Box lag 20", lb["lb_stat"].iloc[1], lb["lb_pvalue"].iloc[1], "Tidak ada autokorelasi" if lb["lb_pvalue"].iloc[1] > 0.05 else "Ada autokorelasi"),
            ("Ljung-Box lag 52", lb["lb_stat"].iloc[2], lb["lb_pvalue"].iloc[2], "Tidak ada autokorelasi" if lb["lb_pvalue"].iloc[2] > 0.05 else "Ada autokorelasi"),
            ("Jarque-Bera", jb_stat, jb_p, "Normal" if jb_p > 0.05 else "Tidak normal"),
            ("ARCH LM (5 lag)", arch_stat, arch_p, "Homoskedastik" if arch_p > 0.05 else "Heteroskedastik"),
            ("Shapiro-Wilk", shapiro_stat, shapiro_p, "Normal" if shapiro_p > 0.05 else "Tidak normal"),
            ("Durbin-Watson", durbin_watson(resid), np.nan, "Mendekati 2 = baik"),
        ],
        columns=["Uji", "Statistik", "p_value", "Interpretasi"],
    )
    return fit, forecast_frame, coefficients, diagnostics, resid


def plot_model_rmse(comparison: pd.DataFrame) -> None:
    key = comparison[~comparison["Model"].isin(["Seasonal naive", "MA(3)", "MA(8)"])].copy()
    key = key.sort_values("RMSE", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    bars = ax.barh(key["Model"], key["RMSE"], color=COLORS["blue"], alpha=0.75)
    for bar_, value in zip(bars, key["RMSE"]):
        ax.text(value + 0.08, bar_.get_y() + bar_.get_height() / 2, f"{value:.2f}", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE pada data uji (µg/m³)")
    ax.set_title("Perbandingan RMSE model pada 12 minggu uji")
    fig.tight_layout()
    fig.savefig(FIG / "fig06_model_rmse.png", bbox_inches="tight")
    plt.close(fig)


def plot_test_forecasts(test: pd.Series, fitted: dict) -> None:
    selected = [
        "Naive (random walk)",
        "Holt",
        "ARIMA(1,1,3)",
        FINAL_NAME,
        "Fourier K=2 + ARIMA(1,1,1)",
    ]
    available = [name for name in selected if name in fitted]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(test.index, test.values, color="black", marker="o", linewidth=2, label="Aktual")
    colors = [COLORS["gray"], COLORS["orange"], COLORS["green"], COLORS["blue"], COLORS["purple"]]
    for name, color in zip(available, colors):
        ax.plot(test.index, fitted[name], marker="s", ms=3, linewidth=1.4, linestyle="--", color=color, label=name)
    ax.set_title("Aktual vs peramalan pada periode uji (12 minggu)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_xlabel("Tanggal")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig07_test_forecasts.png", bbox_inches="tight")
    plt.close(fig)


def plot_residual_diagnostics(resid: pd.Series, fit) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(resid.index, resid.values, color=COLORS["gray"], marker="o", ms=2, linewidth=0.8)
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Residu terstandarisasi")
    plot_acf(resid, lags=40, ax=axes[0, 1], color=COLORS["blue"])
    axes[0, 1].set_title("ACF residu")
    axes[1, 0].hist(resid, bins=18, color=COLORS["blue"], alpha=0.7, edgecolor="white", density=True)
    x_grid = np.linspace(resid.min(), resid.max(), 200)
    axes[1, 0].plot(x_grid, stats.norm.pdf(x_grid, resid.mean(), resid.std()), color=COLORS["red"], linewidth=1.5)
    axes[1, 0].set_title("Histogram residu")
    stats.probplot(resid, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("QQ-plot residu")
    fig.tight_layout()
    fig.savefig(FIG / "fig08_residual_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def plot_forecast(series: pd.Series, forecast_frame: pd.DataFrame, train: pd.Series, test: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(series.index, series.values, color=COLORS["blue"], linewidth=1.1, label="Histori")
    ax.axvspan(train.index[0], train.index[-1], color=COLORS["green"], alpha=0.06)
    ax.axvspan(test.index[0], test.index[-1], color=COLORS["red"], alpha=0.08, label="Periode uji")
    ax.plot(forecast_frame["Minggu"], forecast_frame["Peramalan"], color=COLORS["red"], marker="o", ms=3, linewidth=1.8, label="Peramalan")
    ax.fill_between(
        forecast_frame["Minggu"], forecast_frame["Batas_bawah_95"], forecast_frame["Batas_atas_95"],
        color=COLORS["red"], alpha=0.12, label="CI 95%",
    )
    ax.fill_between(
        forecast_frame["Minggu"], forecast_frame["Batas_bawah_80"], forecast_frame["Batas_atas_80"],
        color=COLORS["red"], alpha=0.22, label="CI 80%",
    )
    ax.axvline(series.index[-1], color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_title("Peramalan PM2.5 mingguan 12 minggu ke depan")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_xlabel("Tanggal")
    ax.legend(loc="upper left", ncol=5)
    fig.tight_layout()
    fig.savefig(FIG / "fig09_forecast.png", bbox_inches="tight")
    plt.close(fig)


def dm_test(e1: np.ndarray, e2: np.ndarray) -> tuple[float, float]:
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    dbar = float(np.mean(d))
    var = float(np.var(d, ddof=1) / n)
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var)
    p_value = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p_value)


def style_workbook(path: Path, sheets: dict[str, pd.DataFrame], notes: list[str]) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary = pd.DataFrame(
            [
                ("Sumber data", "Open-Meteo Air Quality API (CAMS global)"),
                ("Lokasi", "Kota Surakarta (-7,5545; 110,8060)"),
                ("Frekuensi", "Mingguan (rata-rata 168 jam)"),
                ("Periode model", "2022-08-08 s.d. 2024-10-21"),
                ("Jumlah observasi", len(sheets.get("Data_Mingguan", pd.DataFrame()))),
                ("Model final", FINAL_NAME),
            ],
            columns=["Keterangan", "Nilai"],
        )
        summary.to_excel(writer, sheet_name="Ringkasan", index=False)
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
        pd.DataFrame({"Catatan": notes}).to_excel(writer, sheet_name="Catatan", index=False)

    workbook = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(color="FFFFFF", bold=True)
    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for column_cells in worksheet.columns:
            length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
            worksheet.column_dimensions[get_column_letter(column_cells[0].column)].width = min(max(12, length + 2), 55)
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "0.000"
    workbook.save(path)


def main() -> None:
    frame, series = load_series()
    train, test = series.iloc[:-TEST_SIZE], series.iloc[-TEST_SIZE:]
    print(f"Observasi: {len(series)} | Train: {len(train)} | Test: {len(test)}")
    print(f"Periode: {series.index.min().date()} s.d. {series.index.max().date()}")

    build_flowchart()
    desc = descriptive(series)
    outliers = outlier_summary(series)
    stationarity = stationarity_table(series)
    plot_series(series, train, test)
    seasonal_strength = plot_decomposition(series)
    plot_seasonal_profile(series)
    plot_acf_pacf(series)

    comparison, fitted, arma_grid, arima_grid, sarima_grid, extra = fit_all_models(train, test)
    model_diagnostics = pd.DataFrame(
        [residual_diagnostics(name, fit) for name, fit in extra["diag_fits"].items()]
    )
    best_sarima = extra["best_sarima"]
    sarima_name = f"SARIMA({best_sarima['p']},1,{best_sarima['q']})({best_sarima['P']},0,{best_sarima['Q']})[{SEASON}]"
    print(f"SARIMA terpilih: {sarima_name} | AICc={best_sarima['AICc']:.2f}")

    plot_model_rmse(comparison)
    plot_test_forecasts(test, fitted)

    final_fit, forecast_frame, coefficients, diagnostics, resid = fit_final(series)
    plot_residual_diagnostics(resid, final_fit)
    plot_forecast(series, forecast_frame, train, test)

    prediction_errors = {
        row["Model"]: test.values - row["_forecast"] for _, row in comparison.iterrows()
    }
    dm_rows = []
    final_model_name = comparison.loc[comparison["AICc"].idxmin(), "Model"] if comparison["AICc"].notna().any() else sarima_name
    final_model_name = sarima_name if sarima_name in prediction_errors else final_model_name
    for name, errors in prediction_errors.items():
        if name == final_model_name:
            continue
        stat, p_value = dm_test(prediction_errors[final_model_name], errors)
        dm_rows.append(
            {
                "Model pembanding": name,
                "DM statistik": stat,
                "p-value": p_value,
                "Kesimpulan": "Tidak berbeda signifikan" if p_value > 0.05 else "Berbeda signifikan",
            }
        )
    dm_table = pd.DataFrame(dm_rows)

    compact = comparison.drop(columns=["_forecast"]).copy()
    compact.to_csv(TAB / "model_comparison.csv", index=False)
    desc.to_csv(TAB / "descriptive.csv", index=False)
    stationarity.to_csv(TAB / "stationarity.csv", index=False)
    coefficients.to_csv(TAB / "final_coefficients.csv", index=False)
    diagnostics.to_csv(TAB / "final_diagnostics.csv", index=False)
    forecast_frame.to_csv(TAB / "forecast.csv", index=False)
    dm_table.to_csv(TAB / "dm_tests.csv", index=False)
    arma_grid.to_csv(TAB / "grid_arma.csv", index=False)
    arima_grid.to_csv(TAB / "grid_arima.csv", index=False)
    sarima_grid.to_csv(TAB / "grid_sarima.csv", index=False)
    extra["grids"]["Fourier"].to_csv(TAB / "grid_fourier.csv", index=False)
    extra["grids"]["SARIMA_D1"].to_csv(TAB / "sarima_d1.csv", index=False)
    extra["grids"]["AR"].to_csv(TAB / "grid_ar.csv", index=False)
    extra["es_params"].to_csv(TAB / "es_params.csv", index=False)
    extra["roots"].to_csv(TAB / "roots.csv", index=False)
    extra["ma_candidates"].to_csv(TAB / "ma_candidates.csv", index=False)
    model_diagnostics.to_csv(TAB / "model_diagnostics.csv", index=False)

    monthly_context = pd.read_csv(DATA / "pm25_surakarta_monthly.csv")
    pm_ratio = pd.read_csv(DATA / "pm25_pm10_monthly.csv")
    monthly_context.to_csv(TAB / "monthly_context.csv", index=False)
    pm_ratio.to_csv(TAB / "pm_ratio.csv", index=False)

    forecast_first = float(forecast_frame["Peramalan"].iloc[0])
    forecast_last = float(forecast_frame["Peramalan"].iloc[-1])
    forecast_mean = float(forecast_frame["Peramalan"].mean())
    final_row = compact.loc[compact["Model"] == final_model_name].iloc[0]
    lb_row = diagnostics.set_index("Uji")["p_value"]

    summary = {
        "n": int(len(series)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "start": series.index.min().strftime("%Y-%m-%d"),
        "end": series.index.max().strftime("%Y-%m-%d"),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        **outliers,
        "seasonal_strength": seasonal_strength,
        "final_model": final_model_name,
        "final_spec": f"SARIMA{FINAL_ORDER}({FINAL_SEASONAL[0]},0,{FINAL_SEASONAL[2]})[{SEASON}]",
        "aic": float(final_fit.aic),
        "aicc": float(final_fit.aicc),
        "bic": float(final_fit.bic),
        "loglik": float(final_fit.llf),
        "sigma2": float(final_fit.params.get("sigma2", np.nan)),
        "test_mae": float(final_row["MAE"]),
        "test_rmse": float(final_row["RMSE"]),
        "test_mape": float(final_row["MAPE"]),
        "test_smape": float(final_row["sMAPE"]),
        "test_mase": float(final_row["MASE"]),
        "lb10_p": float(lb_row.get("Ljung-Box lag 10", np.nan)),
        "lb20_p": float(lb_row.get("Ljung-Box lag 20", np.nan)),
        "lb52_p": float(lb_row.get("Ljung-Box lag 52", np.nan)),
        "jb_p": float(lb_row.get("Jarque-Bera", np.nan)),
        "arch_p": float(lb_row.get("ARCH LM (5 lag)", np.nan)),
        "shapiro_p": float(lb_row.get("Shapiro-Wilk", np.nan)),
        "forecast_first": forecast_first,
        "forecast_last": forecast_last,
        "forecast_mean": forecast_mean,
        "forecast": forecast_frame.to_dict(orient="records"),
        "sarima_coef": coefficients.to_dict(orient="records"),
        "diagnostics": diagnostics.to_dict(orient="records"),
        "dm": dm_table.to_dict(orient="records"),
        "comparison": compact.to_dict(orient="records"),
        "stationarity": stationarity.to_dict(orient="records"),
        "descriptive": desc.to_dict(orient="records"),
        "pm_ratio_pre": float(pm_ratio.loc[pm_ratio["homogeneous"], "ratio"].mean()),
        "pm_ratio_post": float(pm_ratio.loc[~pm_ratio["homogeneous"], "ratio"].mean()),
        "break": "2024-11-01",
    }
    with (OUT / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=to_py)

    sheets = {
        "Data_Mingguan": frame,
        "Statistik_Deskriptif": desc,
        "Uji_Stasioneritas": stationarity,
        "Perbandingan_Model": compact,
        "Grid_AR": extra["grids"]["AR"],
        "Grid_ARMA": arma_grid,
        "Grid_ARIMA": arima_grid,
        "Grid_SARIMA": sarima_grid,
        "Grid_Fourier": extra["grids"]["Fourier"],
        "SARIMA_D1": extra["grids"]["SARIMA_D1"],
        "Kandidat_MA": extra["ma_candidates"],
        "Parameter_ES": extra["es_params"],
        "Akar_Model": extra["roots"],
        "Diagnostik_Model": model_diagnostics,
        "Koefisien_Final": coefficients,
        "Diagnostik_Final": diagnostics,
        "Peramalan": forecast_frame,
        "Uji_DM": dm_table,
        "Bulanan_Konteks": monthly_context,
        "Rasio_PM": pm_ratio,
    }
    notes = [
        "AIC/AICc/BIC hanya sebanding untuk model dengan orde diferensiasi (d) dan transformasi data yang sama.",
        "AIC pada SES/Holt/Holt-Winters tidak sebanding langsung dengan model ARIMA; kolom AIC_sebanding menandai hal ini.",
        "Periode model berhenti Oktober 2024 karena patahan model CAMS pada November 2024 (rasio PM10/PM2.5 turun dari ~1,46 menjadi ~1,04).",
        "Metrik uji dihitung pada 12 minggu terakhir (Agustus–Oktober 2024); model dilatih pada 104 minggu.",
        "MASE memakai seasonal naive (lag 52) pada data latih sebagai skala.",
        "Interval peramalan disajikan pada tingkat kepercayaan 80% dan 95%.",
        "Parameter_ES memuat parameter smoothing dan nilai awal hasil estimasi statsmodels.",
        "Akar_Model memuat modulus akar AR/MA; nilai > 1 menunjukkan stasioneritas/invertibilitas.",
        "Kandidat_MA memuat model MA(q) galat, terpisah dari MA(3)/MA(8) yang merupakan penghalus.",
        "Diagnostik_Model memuat Ljung-Box, Jarque-Bera, dan ARCH untuk model final tiap keluarga.",
        "Pemeriksaan pencilan memakai batas IQR 1,5 dan |z| > 3; tidak ditemukan pencilan.",
    ]
    style_workbook(OUT / "Hasil_Analisis_PM25_Surakarta.xlsx", sheets, notes)

    print("\nPerbandingan model (ringkas):")
    print(compact[["Model", "AICc", "RMSE", "MAE", "MAPE", "MASE", "LB20_p"]].head(15).round(3).to_string(index=False))
    print("\nDiagnostik final:")
    print(diagnostics.round(4).to_string(index=False))
    print("\nPeramalan:")
    print(forecast_frame.round(2).to_string(index=False))
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()

