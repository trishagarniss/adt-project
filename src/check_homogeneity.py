"""Periksa homogenitas seri PM2.5 Open-Meteo/CAMS memakai rasio PM2.5/PM10.

Perubahan basis model CAMS pada November 2024 membuat PM2.5 hampir sama
dengan PM10. Seri setelah patahan tersebut tidak dipakai untuk pemodelan.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"

LATITUDE = -7.5545
LONGITUDE = 110.8060
TIMEZONE = "Asia/Jakarta"
START_DATE = "2022-08-01"
END_DATE = "2026-10-28"


def build_url(variables: str) -> str:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": variables,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": TIMEZONE,
    }
    return "https://air-quality-api.open-meteo.com/v1/air-quality?" + urllib.parse.urlencode(params)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    url = build_url("pm10,pm2_5")
    print(f"Mengunduh: {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.load(response)

    raw_path = RAW_DIR / "openmeteo_pm10_pm25_surakarta_hourly.json"
    with raw_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(payload["hourly"]["time"]),
            "pm10": payload["hourly"]["pm10"],
            "pm25": payload["hourly"]["pm2_5"],
        }
    )
    frame["month"] = frame["datetime"].dt.to_period("M").astype(str)
    monthly = (
        frame.groupby("month")[["pm10", "pm25"]]
        .mean()
        .reset_index()
        .rename(columns={"pm10": "pm10_mean", "pm25": "pm25_mean"})
    )
    monthly["ratio"] = monthly["pm10_mean"] / monthly["pm25_mean"]
    monthly["homogeneous"] = monthly["month"] < "2024-11"
    monthly.to_csv(OUT_DIR / "pm25_pm10_monthly.csv", index=False)

    pre = monthly.loc[monthly["homogeneous"], "ratio"]
    print("\nRasio PM10/PM2.5 sebelum Nov 2024:")
    print(pre.describe().round(3).to_string())
    print("\nRasio PM10/PM2.5 sesudah Nov 2024:")
    print(monthly.loc[~monthly["homogeneous"], "ratio"].describe().round(3).to_string())

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[3, 2])
    x = pd.to_datetime(monthly["month"])

    axes[0].plot(x, monthly["pm10_mean"], color="#d97706", marker="o", ms=3, label="PM10")
    axes[0].plot(x, monthly["pm25_mean"], color="#1d4ed8", marker="o", ms=3, label="PM2.5")
    axes[0].axvline(pd.Timestamp("2024-11-01"), color="#dc2626", linestyle="--", linewidth=1.5)
    axes[0].text(
        pd.Timestamp("2024-11-10"),
        axes[0].get_ylim()[1] * 0.9,
        "Patahan model CAMS",
        color="#dc2626",
        fontsize=10,
    )
    axes[0].set_ylabel("Konsentrasi (µg/m³)")
    axes[0].set_title("PM2.5 dan PM10 bulanan, Surakarta (CAMS via Open-Meteo)")
    axes[0].legend(loc="upper left", ncol=2)

    axes[1].plot(x, monthly["ratio"], color="#374151", marker="o", ms=3)
    axes[1].axhline(pre.mean(), color="#059669", linestyle=":", linewidth=1.5, label=f"Rata-rata pra-patahan = {pre.mean():.2f}")
    axes[1].axvline(pd.Timestamp("2024-11-01"), color="#dc2626", linestyle="--", linewidth=1.5)
    axes[1].set_ylabel("Rasio PM10/PM2.5")
    axes[1].set_xlabel("Bulan")
    axes[1].set_ylim(0.8, 1.7)
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_data_break.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nTabel : {OUT_DIR / 'pm25_pm10_monthly.csv'}")
    print(f"Gambar: {FIG_DIR / 'fig01_data_break.png'}")


if __name__ == "__main__":
    main()
