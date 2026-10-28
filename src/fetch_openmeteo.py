"""Unduh PM2.5 historis Surakarta dari Open-Meteo Air Quality API.

Sumber data : Copernicus Atmosphere Monitoring Service (CAMS) melalui Open-Meteo.
Tanpa API key. Dokumentasi: https://open-meteo.com/en/docs/air-quality-api
"""

from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

LATITUDE = -7.5545
LONGITUDE = 110.8060
TIMEZONE = "Asia/Jakarta"
START_DATE = "2022-08-01"
END_DATE = date.today().isoformat()
HOURLY_VARIABLE = "pm2_5"


def build_url() -> str:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": HOURLY_VARIABLE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": TIMEZONE,
    }
    return "https://air-quality-api.open-meteo.com/v1/air-quality?" + urllib.parse.urlencode(params)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = build_url()
    print(f"Mengunduh: {url}")

    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.load(response)

    json_path = RAW_DIR / "openmeteo_pm25_surakarta_hourly.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    times = payload["hourly"]["time"]
    values = payload["hourly"][HOURLY_VARIABLE]

    csv_path = RAW_DIR / "openmeteo_pm25_surakarta_hourly.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["datetime", "pm2_5"])
        for timestamp, value in zip(times, values):
            writer.writerow([timestamp, "" if value is None else value])

    valid = [(t, v) for t, v in zip(times, values) if v is not None]
    missing = len(times) - len(valid)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = OUT_DIR / "pm25_surakarta_hourly_clean.csv"
    with clean_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["datetime", "pm2_5"])
        writer.writerows(valid)

    print(f"Lokasi        : {payload.get('latitude')}, {payload.get('longitude')}")
    print(f"Zona waktu    : {payload.get('timezone')}")
    print(f"Satuan        : {payload.get('hourly_units', {}).get(HOURLY_VARIABLE)}")
    print(f"Total jam     : {len(times)}")
    print(f"Jam valid     : {len(valid)}")
    print(f"Jam hilang    : {missing} ({missing / len(times):.2%})")
    if valid:
        print(f"Periode valid : {valid[0][0]} s.d. {valid[-1][0]}")
        print(f"Nilai pertama : {valid[0][0]} = {valid[0][1]} ug/m3")
    print(f"Raw JSON      : {json_path}")
    print(f"Raw CSV       : {csv_path}")
    print(f"CSV bersih    : {clean_path}")


if __name__ == "__main__":
    main()
