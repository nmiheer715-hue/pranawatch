"""CPCB real-time air quality from the Government of India open data portal (data.gov.in).

Dataset: "Real time Air Quality Index from various locations" (CPCB), resource
3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69. Free API key: register at https://data.gov.in,
then My Account -> API key. Set DATA_GOV_IN_KEY.

Used to (1) plot official ground monitors on the India map and (2) validate the CAMS
model values PranaWatch screens with, city by city.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests

from .hotspots import haversine_km

RESOURCE = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
URL = f"https://api.data.gov.in/resource/{RESOURCE}"


def available() -> bool:
    return bool(os.environ.get("DATA_GOV_IN_KEY"))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def parse_records(records: list[dict]) -> pd.DataFrame:
    """Normalise raw records (field names differ slightly between API versions)."""
    rows = []
    for r in records:
        pol = str(r.get("pollutant_id", "")).upper().replace(" ", "")
        if pol not in ("PM2.5", "PM25"):
            continue
        avg = _num(r.get("avg_value", r.get("pollutant_avg")))
        lat, lon = _num(r.get("latitude")), _num(r.get("longitude"))
        if np.isnan(avg) or np.isnan(lat) or np.isnan(lon):
            continue
        rows.append({"station": r.get("station", ""), "city": r.get("city", ""), "state": r.get("state", ""),
                     "lat": lat, "lon": lon, "pm25": avg,
                     "pm25_max": _num(r.get("max_value", r.get("pollutant_max"))),
                     "last_update": r.get("last_update", "")})
    return pd.DataFrame(rows, columns=["station", "city", "state", "lat", "lon", "pm25", "pm25_max", "last_update"])


def fetch(max_pages: int = 5, timeout: int = 25) -> pd.DataFrame:
    key = os.environ.get("DATA_GOV_IN_KEY")
    if not key:
        raise RuntimeError("DATA_GOV_IN_KEY not set")
    records, limit = [], 1000
    for page in range(max_pages):
        r = requests.get(URL, params={"api-key": key, "format": "json", "limit": limit,
                                      "offset": page * limit, "filters[pollutant_id]": "PM2.5"}, timeout=timeout)
        r.raise_for_status()
        batch = r.json().get("records", [])
        records += batch
        if len(batch) < limit:
            break
    df = parse_records(records)
    if df.empty:
        raise RuntimeError("data.gov.in returned no PM2.5 records")
    return df


def match_cities(summary: pd.DataFrame, stations: pd.DataFrame, radius_km: float = 25.0) -> pd.DataFrame:
    """Add CPCB station mean PM2.5 (within radius) to each city row, plus CAMS - CPCB bias."""
    out = summary.copy()
    vals, counts = [], []
    for _, c in out.iterrows():
        if stations is None or stations.empty:
            vals.append(np.nan); counts.append(0); continue
        d = stations.apply(lambda s: haversine_km(c["lat"], c["lon"], s["lat"], s["lon"]), axis=1)
        near = stations.loc[d <= radius_km, "pm25"]
        vals.append(float(near.mean()) if len(near) else np.nan)
        counts.append(int(len(near)))
    out["cpcb_pm25"] = vals
    out["cpcb_stations"] = counts
    out["cams_minus_cpcb"] = out["pm25_now"] - out["cpcb_pm25"]
    return out


def validation(summary: pd.DataFrame) -> dict:
    m = summary.dropna(subset=["cpcb_pm25"])
    if m.empty:
        return {"n": 0}
    err = m["pm25_now"] - m["cpcb_pm25"]
    return {"n": int(len(m)), "stations": int(m["cpcb_stations"].sum()), "mae": float(err.abs().mean()),
            "bias": float(err.mean()), "corr": float(m["pm25_now"].corr(m["cpcb_pm25"])) if len(m) > 2 else np.nan}


def load() -> dict:
    try:
        st_df = fetch()
        return {"stations": st_df, "mode": "live", "error": None}
    except Exception as e:
        return {"stations": pd.DataFrame(), "mode": "none" if not available() else "error", "error": str(e)[:200]}
