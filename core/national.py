"""National screening layer: all-India city air quality, 48 h forecast, and
trans-boundary smoke detection from crop-residue fires.

Data (live, free):
  * Open-Meteo Air Quality API: CAMS model PM2.5 / PM10 / NO2, past 2 days + 3-day forecast,
    ~40 cities in ONE multi-location request.
  * Open-Meteo Forecast API: current wind at each city.
  * Fires: NASA FIRMS VIIRS (set FIRMS_MAP_KEY, free at firms.modaps.eosdis.nasa.gov/api/map_key)
    or Google Earth Engine FIRMS; otherwise a clearly-labelled simulated early-season pattern.

AQI follows India's NAQI method: category from the 24-hour mean PM2.5.
"""
from __future__ import annotations

import io
import math
import os

import numpy as np
import pandas as pd
import requests

from . import config
from .hotspots import angle_diff, bearing_deg, haversine_km

TZ = "Asia/Kolkata"

# name, state, lat, lon, NCR?
CITIES = [
    ("Delhi", "Delhi", 28.61, 77.21, True), ("Gurugram", "Haryana", 28.46, 77.03, True),
    ("Noida", "Uttar Pradesh", 28.54, 77.39, True), ("Ghaziabad", "Uttar Pradesh", 28.67, 77.45, True),
    ("Faridabad", "Haryana", 28.41, 77.32, True), ("Chandigarh", "Chandigarh", 30.73, 76.78, False),
    ("Ludhiana", "Punjab", 30.90, 75.85, False), ("Amritsar", "Punjab", 31.63, 74.87, False),
    ("Jaipur", "Rajasthan", 26.91, 75.79, False), ("Lucknow", "Uttar Pradesh", 26.85, 80.95, False),
    ("Kanpur", "Uttar Pradesh", 26.45, 80.33, False), ("Agra", "Uttar Pradesh", 27.18, 78.01, False),
    ("Varanasi", "Uttar Pradesh", 25.32, 82.97, False), ("Patna", "Bihar", 25.59, 85.14, False),
    ("Kolkata", "West Bengal", 22.57, 88.36, False), ("Guwahati", "Assam", 26.14, 91.74, False),
    ("Bhubaneswar", "Odisha", 20.30, 85.82, False), ("Ranchi", "Jharkhand", 23.34, 85.31, False),
    ("Raipur", "Chhattisgarh", 21.25, 81.63, False), ("Bhopal", "Madhya Pradesh", 23.26, 77.41, False),
    ("Indore", "Madhya Pradesh", 22.72, 75.86, False), ("Ahmedabad", "Gujarat", 23.02, 72.57, False),
    ("Surat", "Gujarat", 21.17, 72.83, False), ("Mumbai", "Maharashtra", 19.08, 72.88, False),
    ("Pune", "Maharashtra", 18.52, 73.86, False), ("Nagpur", "Maharashtra", 21.15, 79.09, False),
    ("Hyderabad", "Telangana", 17.39, 78.49, False), ("Visakhapatnam", "Andhra Pradesh", 17.69, 83.22, False),
    ("Vijayawada", "Andhra Pradesh", 16.51, 80.65, False), ("Chennai", "Tamil Nadu", 13.08, 80.27, False),
    ("Coimbatore", "Tamil Nadu", 11.02, 76.96, False), ("Madurai", "Tamil Nadu", 9.93, 78.12, False),
    ("Bengaluru", "Karnataka", 12.97, 77.59, False), ("Mysuru", "Karnataka", 12.30, 76.64, False),
    ("Kochi", "Kerala", 9.93, 76.27, False), ("Thiruvananthapuram", "Kerala", 8.52, 76.94, False),
    ("Dehradun", "Uttarakhand", 30.32, 78.03, False), ("Srinagar", "Jammu & Kashmir", 34.08, 74.80, False),
]

INDIA_BBOX = (68.0, 6.0, 98.0, 37.5)  # lon_min, lat_min, lon_max, lat_max
STUBBLE_STATES_BOX = (73.8, 28.8, 77.6, 32.3)  # Punjab + Haryana (approx)
CAQM = "Commission for Air Quality Management (CAQM, NCR & adjoining areas)"


def state_board(state: str) -> str:
    if state == "Delhi":
        return "Delhi Pollution Control Committee (DPCC)"
    if state == "Chandigarh":
        return "Chandigarh Pollution Control Committee"
    if state == "Jammu & Kashmir":
        return "J&K Pollution Control Committee"
    return f"{state} State Pollution Control Board"


def cities_df() -> pd.DataFrame:
    return pd.DataFrame(CITIES, columns=["city", "state", "lat", "lon", "ncr"])


# ---------------------------------------------------------------------------
# Live fetch
# ---------------------------------------------------------------------------

def fetch_live_aq(timeout: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (hourly long df: city,time,pm25,pm10,no2 ; wind df: city,wind_speed,wind_dir)."""
    c = cities_df()
    lat = ",".join(f"{v:.2f}" for v in c["lat"])
    lon = ",".join(f"{v:.2f}" for v in c["lon"])
    r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
        "latitude": lat, "longitude": lon, "timezone": TZ,
        "hourly": "pm2_5,pm10,nitrogen_dioxide", "past_days": 2, "forecast_days": 3,
    }, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    js = js if isinstance(js, list) else [js]
    rows = []
    for city, j in zip(c["city"], js):
        h = j["hourly"]
        rows.append(pd.DataFrame({"city": city, "time": pd.to_datetime(h["time"]),
                                  "pm25": h["pm2_5"], "pm10": h["pm10"], "no2": h["nitrogen_dioxide"]}))
    aq = pd.concat(rows, ignore_index=True)
    w = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon, "timezone": TZ, "wind_speed_unit": "ms",
        "current": "wind_speed_10m,wind_direction_10m",
    }, timeout=timeout)
    w.raise_for_status()
    wj = w.json()
    wj = wj if isinstance(wj, list) else [wj]
    wind = pd.DataFrame({"city": c["city"],
                         "wind_speed": [x["current"]["wind_speed_10m"] for x in wj],
                         "wind_dir": [x["current"]["wind_direction_10m"] for x in wj]})
    return aq, wind


def fetch_firms(days: int = 2, timeout: int = 25) -> list[dict]:
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise RuntimeError("FIRMS_MAP_KEY not set")
    b = ",".join(str(x) for x in INDIA_BBOX)
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_NRT/{b}/{days}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "latitude" not in df:
        raise RuntimeError(r.text[:120])
    if "confidence" in df:
        df = df[df["confidence"].astype(str).str.lower().isin(["n", "h", "nominal", "high"])]
    return [{"lat": float(a), "lon": float(o), "frp": float(f), "source": "NASA FIRMS VIIRS"}
            for a, o, f in zip(df["latitude"], df["longitude"], df.get("frp", pd.Series([0] * len(df))))]


def fetch_gee_fires(days: int = 2) -> list[dict]:
    from .satellite import _ee
    ee = _ee()
    end = ee.Date(pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    region = ee.Geometry.Rectangle(list(INDIA_BBOX))
    img = ee.ImageCollection("FIRMS").filterDate(end.advance(-days, "day"), end).select("T21").max()
    pts = img.sample(region=region, scale=2000, geometries=True).limit(1500).getInfo()["features"]
    return [{"lat": p["geometry"]["coordinates"][1], "lon": p["geometry"]["coordinates"][0],
             "frp": 0.0, "source": "NASA FIRMS (Earth Engine)"} for p in pts]


# ---------------------------------------------------------------------------
# Simulated fallback
# ---------------------------------------------------------------------------

_REGION_BASE = {  # typical late-Sep / Oct PM2.5 levels (ug/m3)
    "Delhi": 95, "Haryana": 85, "Uttar Pradesh": 80, "Punjab": 75, "Chandigarh": 55, "Bihar": 70,
    "Rajasthan": 55, "Madhya Pradesh": 45, "West Bengal": 45, "Jharkhand": 45, "Uttarakhand": 40,
    "Gujarat": 45, "Maharashtra": 38, "Chhattisgarh": 45, "Odisha": 35, "Assam": 35, "Telangana": 35,
    "Andhra Pradesh": 30, "Tamil Nadu": 25, "Karnataka": 25, "Kerala": 18, "Jammu & Kashmir": 30,
}


def simulated_aq(now: pd.Timestamp, seed: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(now - pd.Timedelta(hours=48), now + pd.Timedelta(hours=72), freq="h")
    h = idx.hour.values
    k = np.arange(len(idx))
    rows, wind = [], []
    for city, state, lat, lon, ncr in CITIES:
        base = _REGION_BASE.get(state, 35) * rng.uniform(0.85, 1.15)
        diurnal = 1 + 0.35 * np.exp(-((h - 8) ** 2) / 6) + 0.45 * np.exp(-((h - 22) ** 2) / 8) - 0.25 * np.exp(-((h - 15) ** 2) / 10)
        # north-west India: smoke build-up over the next 2 days (early stubble-burning season)
        trend = 1 + (0.45 * np.clip((k - 48) / 48, 0, 1) if (ncr or state in ("Punjab", "Haryana", "Delhi")) else 0.05 * np.sin(k / 20))
        pm25 = np.clip(base * diurnal * trend + rng.normal(0, base * 0.06, len(idx)), 5, None)
        pm10 = pm25 * rng.uniform(1.6, 2.2)
        no2 = np.clip(15 + 0.25 * pm25 + rng.normal(0, 3, len(idx)), 3, None)
        rows.append(pd.DataFrame({"city": city, "time": idx, "pm25": pm25, "pm10": pm10, "no2": no2}))
        north = lat > 26
        wind.append({"city": city, "wind_speed": float(rng.uniform(1.5, 4.5)),
                     "wind_dir": float((305 if north else 250) + rng.normal(0, 12)) % 360})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(wind)


def simulated_fires(seed: int = 9) -> list[dict]:
    rng = np.random.default_rng(seed)
    clusters = [(30.45, 75.55, 40), (30.95, 75.05, 30), (30.15, 74.90, 25), (29.95, 76.35, 25),
                (29.60, 76.10, 15), (31.40, 75.40, 20)]
    out = []
    for la, lo, n in clusters:
        for _ in range(n):
            out.append({"lat": float(la + rng.normal(0, 0.18)), "lon": float(lo + rng.normal(0, 0.22)),
                        "frp": float(rng.gamma(2, 6)), "source": "simulated"})
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def smoke_transport(lat: float, lon: float, wind_from: float, wind_speed: float, fires: list[dict],
                    min_km: float = 30, max_km: float = 600, half_angle: float = 35) -> dict:
    """Count fires lying upwind of a city (within a wind-aligned sector)."""
    if not fires or wind_speed < 0.5:
        return {"n": 0, "weighted": 0.0, "mean_km": None, "centroid": None}
    sel = []
    for f in fires:
        d = haversine_km(lat, lon, f["lat"], f["lon"])
        if min_km <= d <= max_km and angle_diff(bearing_deg(lat, lon, f["lat"], f["lon"]), wind_from) <= half_angle:
            sel.append((f, d))
    if not sel:
        return {"n": 0, "weighted": 0.0, "mean_km": None, "centroid": None}
    w = sum(100.0 / d for _, d in sel)
    return {"n": len(sel), "weighted": w, "mean_km": float(np.mean([d for _, d in sel])),
            "centroid": (float(np.mean([f["lat"] for f, _ in sel])), float(np.mean([f["lon"] for f, _ in sel]))),
            "hours_to_arrive": float(np.mean([d for _, d in sel]) / max(wind_speed * 3.6, 1))}


def summarize(aq: pd.DataFrame, wind: pd.DataFrame, fires: list[dict], now: pd.Timestamp) -> pd.DataFrame:
    meta = cities_df()
    out = []
    for _, m in meta.iterrows():
        g = aq[aq["city"] == m["city"]].set_index("time").sort_index()
        past = g[g.index <= now]
        fut = g[(g.index > now) & (g.index <= now + pd.Timedelta(hours=48))]
        if past.empty:
            continue
        mean24 = float(past["pm25"].tail(24).mean())
        pm10_24 = float(past["pm10"].tail(24).mean())
        roll_fut = g["pm25"].rolling(24, min_periods=12).mean()
        roll_fut = roll_fut[(roll_fut.index > now) & (roll_fut.index <= now + pd.Timedelta(hours=48))]
        fc24_peak = float(roll_fut.max()) if len(roll_fut) else mean24
        fc_peak_t = roll_fut.idxmax() if len(roll_fut) else now
        hourly_peak = float(fut["pm25"].max()) if len(fut) else float(past["pm25"].iloc[-1])
        aqi, cat = config.pm25_to_aqi(mean24)
        aqi_f, cat_f = config.pm25_to_aqi(fc24_peak)
        wrow = wind[wind["city"] == m["city"]].iloc[0]
        sm = smoke_transport(m["lat"], m["lon"], float(wrow["wind_dir"]), float(wrow["wind_speed"]), fires)
        out.append({
            "city": m["city"], "state": m["state"], "lat": m["lat"], "lon": m["lon"], "ncr": m["ncr"],
            "pm25_now": float(past["pm25"].iloc[-1]), "pm25_24h": mean24, "pm10_24h": pm10_24,
            "ratio": mean24 / max(pm10_24, 1), "aqi": aqi, "category": cat,
            "fc_24h_peak": fc24_peak, "fc_peak_time": fc_peak_t, "fc_aqi": aqi_f, "fc_category": cat_f,
            "hourly_peak_48h": hourly_peak,
            "rising": fc24_peak > mean24 * 1.2,
            "alert": fc24_peak >= config.ALERT_THRESHOLD_PM25 or mean24 >= config.ALERT_THRESHOLD_PM25,
            "wind_speed": float(wrow["wind_speed"]), "wind_dir": float(wrow["wind_dir"]),
            "upwind_fires": sm["n"], "smoke_score": sm["weighted"], "fire_km": sm["mean_km"],
            "fire_centroid": sm["centroid"], "smoke_eta_h": sm.get("hours_to_arrive"),
            "smoke_flag": sm["n"] >= 10 and sm["weighted"] >= 8.0 and (sm.get("hours_to_arrive") or 999) <= 60,
        })
    return pd.DataFrame(out)


def load(prefer_live: bool = True, use_gee: bool = False) -> dict:
    now = pd.Timestamp.now(tz=TZ).tz_localize(None).floor("h")
    mode, fire_mode, err = "demo", "simulated", None
    try:
        if not prefer_live:
            raise RuntimeError("live disabled")
        aq, wind = fetch_live_aq()
        mode = "live"
    except Exception as e:
        aq, wind = simulated_aq(now)
        err = str(e)[:150]
    fires = None
    for name, fn in (("firms", fetch_firms), ("gee", fetch_gee_fires if use_gee else None)):
        if fn is None or not prefer_live:
            continue
        try:
            fires = fn()
            fire_mode = name
            break
        except Exception:
            continue
    if fires is None:
        if mode == "live":
            # never mix real air-quality data with simulated fires
            fires, fire_mode = [], "none"
        else:
            fires = simulated_fires()
    summ = summarize(aq, wind, fires, now)
    return {"aq": aq, "wind": wind, "fires": fires, "summary": summ, "now": now,
            "mode": mode, "fire_mode": fire_mode, "error": err}


def alert_context(row: pd.Series, summary: pd.DataFrame) -> dict:
    """Build the context dict consumed by gemini.write_alert() for a city."""
    nat_median = float(summary["pm25_24h"].median())
    evidence = [f"24-h mean PM2.5 {row['pm25_24h']:.0f} ug/m3 (NAQI {row['aqi']}, {row['category']})",
                f"CAMS 48-h forecast: 24-h mean peaks at {row['fc_24h_peak']:.0f} ug/m3 "
                f"({row['fc_category']}) around {row['fc_peak_time']:%d %b %H:%M}"]
    authority = state_board(row["state"]) + (f" + {CAQM}" if row["ncr"] else "")
    if row["smoke_flag"]:
        eta = f", smoke arrival in roughly {row['smoke_eta_h']:.0f} h" if row["smoke_eta_h"] else ""
        evidence.append(f"{int(row['upwind_fires'])} active crop-residue fires {row['fire_km']:.0f} km UPWIND "
                        f"(wind from {row['wind_dir']:.0f} deg at {row['wind_speed']:.1f} m/s{eta})")
        if row["ratio"] >= 0.55:
            evidence.append(f"PM2.5/PM10 ratio {row['ratio']:.2f} indicates combustion smoke")
        src = {"name": "Crop-residue burning upwind (Punjab / Haryana)", "kind": "crop_burning",
               "authority": f"{authority} + Punjab & Haryana State Pollution Control Boards + {CAQM}"
               if not row["ncr"] else f"{authority} + Punjab & Haryana State Pollution Control Boards",
               "confidence": min(0.9, 0.4 + row["smoke_score"] / 40), "evidence": evidence}
    else:
        evidence.append("no significant upwind fire activity; local urban sources dominate")
        src = {"name": "Local urban emissions (traffic, industry, construction, dust)", "kind": "unknown",
               "authority": authority, "confidence": 0.6, "evidence": evidence}
    return {"node_name": row["city"], "lat": row["lat"], "lon": row["lon"], "pm25": row["pm25_24h"],
            "pm10": row["pm10_24h"], "excess": row["pm25_24h"] - nat_median, "ratio": row["ratio"],
            "wind": {"from_deg": round(row["wind_dir"]), "speed_ms": round(row["wind_speed"], 1)},
            "top_source": src,
            "forecast_peak": {"time": f"{row['fc_peak_time']:%d %b %H:%M}", "value": round(row["fc_24h_peak"], 1)},
            "reports": [], "baseline_label": "national median of 38 cities"}
