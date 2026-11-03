"""Bersihkan data hourly Open-Meteo menjadi series harian, mingguan, dan bulanan.

Output:
- data/processed/pm25_surakarta_daily.csv
- data/processed/pm25_surakarta_weekly.csv
- data/processed/pm25_surakarta_monthly.csv
- data/processed/pm25_surakarta_monthly_model.csv
"""

from __future__ import annotations

import calendar
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "openmeteo_pm25_surakarta_hourly.csv"
OUT_DIR = ROOT / "data" / "processed"


def load_hourly() -> pd.DataFrame:
    frame = pd.read_csv(RAW_CSV)
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["pm2_5"] = pd.to_numeric(frame["pm2_5"], errors="coerce")
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime")
    return frame


def aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = (
        frame.groupby(keys, dropna=False)["pm2_5"]
        .agg(pm25_mean="mean", pm25_median="median", pm25_min="min", pm25_max="max", n_hours="count")
        .reset_index()
    )
    return grouped


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly = load_hourly()

    valid = hourly.dropna(subset=["pm2_5"])
    print(f"Hourly valid: {len(valid)} / {len(hourly)}")
    print(f"Periode     : {valid['datetime'].min()} s.d. {valid['datetime'].max()}")

    hourly["date"] = hourly["datetime"].dt.date
    daily = aggregate(hourly, ["date"])
    daily["date"] = pd.to_datetime(daily["date"])
    daily["expected_hours"] = 24
    daily["coverage"] = daily["n_hours"] / daily["expected_hours"]
    daily = daily.sort_values("date").reset_index(drop=True)
    daily.to_csv(OUT_DIR / "pm25_surakarta_daily.csv", index=False)

    hourly["month"] = hourly["datetime"].dt.to_period("M").astype(str)
    monthly = aggregate(hourly, ["month"])
    monthly["expected_hours"] = monthly["month"].map(
        lambda value: calendar.monthrange(int(value[:4]), int(value[5:7]))[1] * 24
    )
    monthly["coverage"] = monthly["n_hours"] / monthly["expected_hours"]
    monthly["complete"] = monthly["coverage"] >= 0.95
    monthly = monthly.sort_values("month").reset_index(drop=True)
    monthly.to_csv(OUT_DIR / "pm25_surakarta_monthly.csv", index=False)

    hourly["week_start"] = (
        hourly["datetime"] - pd.to_timedelta(hourly["datetime"].dt.weekday, unit="D")
    ).dt.normalize()
    weekly = aggregate(hourly, ["week_start"])
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    weekly["expected_hours"] = 7 * 24
    weekly["coverage"] = weekly["n_hours"] / weekly["expected_hours"]
    weekly["complete"] = weekly["coverage"] >= 0.95
    weekly = weekly.sort_values("week_start").reset_index(drop=True)
    weekly.to_csv(OUT_DIR / "pm25_surakarta_weekly.csv", index=False)

    break_start = pd.Timestamp("2024-11-01")
    monthly["homogeneous"] = monthly["month"] < break_start.strftime("%Y-%m")
    weekly["homogeneous"] = (weekly["week_start"] + pd.Timedelta(days=6)) < break_start
    monthly.to_csv(OUT_DIR / "pm25_surakarta_monthly.csv", index=False)
    weekly.to_csv(OUT_DIR / "pm25_surakarta_weekly.csv", index=False)

    monthly_model = monthly.loc[
        monthly["complete"] & monthly["homogeneous"], ["month", "pm25_mean", "coverage"]
    ].rename(columns={"pm25_mean": "pm25"})
    monthly_model.to_csv(OUT_DIR / "pm25_surakarta_monthly_model.csv", index=False)

    weekly_model = weekly.loc[
        weekly["complete"] & weekly["homogeneous"],
        ["week_start", "pm25_mean", "n_hours", "coverage"],
    ].rename(columns={"pm25_mean": "pm25"})
    weekly_model = weekly_model.sort_values("week_start").reset_index(drop=True)

    expected_weeks = pd.date_range(
        weekly_model["week_start"].min(), weekly_model["week_start"].max(), freq="7D"
    )
    missing_weeks = expected_weeks.difference(weekly_model["week_start"])
    if len(missing_weeks):
        print(f"Minggu hilang di dalam rentang: {list(missing_weeks)}")
        weekly_model = (
            weekly_model.set_index("week_start")
            .reindex(expected_weeks)
            .rename_axis("week_start")
            .reset_index()
        )
        weekly_model["pm25"] = weekly_model["pm25"].interpolate(method="linear")
    weekly_model.to_csv(OUT_DIR / "pm25_surakarta_weekly_model.csv", index=False)

    print("\nRingkasan bulanan:")
    print(monthly[["month", "pm25_mean", "coverage", "complete"]].to_string(index=False))
    print("\nRingkasan series homogen (sebelum patahan Nov 2024):")
    print(f"Bulan lengkap : {len(monthly_model)} (tidak cukup untuk model bulanan)")
    print(f"Minggu lengkap: {len(weekly_model)}")
    print(
        f"Rentang minggu: {weekly_model['week_start'].min().date()} s.d. "
        f"{weekly_model['week_start'].max().date()}"
    )
    print(f"Minggu hilang : {len(missing_weeks)}")
    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
