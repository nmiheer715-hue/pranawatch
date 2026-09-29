"""Data layer.

Two modes:
  * LIVE  - regional background air quality (CAMS model via Open-Meteo) and weather
            (Open-Meteo forecast API) for the corridor. Free, no key needed.
  * DEMO  - a physically-motivated synthetic dataset, used automatically when the
            internet/API is unavailable, so the demo never breaks on stage.

Node-level (street-level) readings are simulated on top of the regional background
using a simple plume model from known source categories, until real PranaWatch
nodes are deployed. Low-cost sensor error (humidity over-reading + noise) is
simulated too, so the calibration step has something real to fix.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from . import config

OPEN_METEO_AQ = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_WX = "https://api.open-meteo.com/v1/forecast"

TZ = "Asia/Kolkata"


@dataclass
class Dataset:
    weather: pd.DataFrame      # index: time (hourly, past + 48h future); temp, rh, wind_speed, wind_dir, blh
    background: pd.DataFrame   # index: time (past only); pm25_bg, pm10_bg, no2_bg
    obs: pd.DataFrame          # long: time, node_id, pm25_true, pm10_true, pm25_raw, pm10_raw
    now: pd.Timestamp          # last observed hour
    mode: str                  # "live" or "demo"


# ---------------------------------------------------------------------------
# Live fetch
# ---------------------------------------------------------------------------

def fetch_live(past_days: int = 45, timeout: int = 15):
    """Fetch regional AQ + weather from Open-Meteo. Returns (weather, background) or raises."""
    lat, lon = config.CENTER
    aq = requests.get(OPEN_METEO_AQ, params={
        "latitude": lat, "longitude": lon, "timezone": TZ,
        "hourly": "pm2_5,pm10,nitrogen_dioxide",
        "past_days": min(past_days, 92), "forecast_days": 1,
    }, timeout=timeout)
    aq.raise_for_status()
    wx = requests.get(OPEN_METEO_WX, params={
        "latitude": lat, "longitude": lon, "timezone": TZ,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,boundary_layer_height",
        "past_days": min(past_days, 92), "forecast_days": 3, "wind_speed_unit": "ms",
    }, timeout=timeout)
    wx.raise_for_status()

    a = aq.json()["hourly"]
    w = wx.json()["hourly"]
    bg = pd.DataFrame({
        "time": pd.to_datetime(a["time"]),
        "pm25_bg": a["pm2_5"], "pm10_bg": a["pm10"], "no2_bg": a["nitrogen_dioxide"],
    }).set_index("time").astype(float)
    weather = pd.DataFrame({
        "time": pd.to_datetime(w["time"]),
        "temp": w["temperature_2m"], "rh": w["relative_humidity_2m"],
        "wind_speed": w["wind_speed_10m"], "wind_dir": w["wind_direction_10m"],
        "blh": w["boundary_layer_height"],
    }).set_index("time").astype(float)
    weather = weather.interpolate(limit_direction="both")

    now = pd.Timestamp.now(tz=TZ).tz_localize(None).floor("h")
    bg = bg[bg.index <= now].interpolate(limit_direction="both").dropna()
    if len(bg) < 24 * 7:
        raise ValueError("Not enough live history returned")
    return weather, bg, now


# ---------------------------------------------------------------------------
# Synthetic demo data
# ---------------------------------------------------------------------------

def _synthetic_weather(idx: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    h = idx.hour.values
    doy = idx.dayofyear.values
    n = len(idx)
    # Bengaluru late monsoon: 20-29 C, humid nights, W/SW winds
    temp = 24 + 4.5 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 0.6, n)
    rh = np.clip(72 - 18 * np.sin(2 * np.pi * (h - 9) / 24) + 6 * np.sin(2 * np.pi * doy / 9)
                 + rng.normal(0, 4, n), 35, 98)
    # wind: slowly-varying direction around 250 deg (WSW), with synoptic swings
    walk = np.cumsum(rng.normal(0, 6, n))
    walk = walk - pd.Series(walk).rolling(72, min_periods=1).mean().values
    wind_dir = (250 + 40 * np.sin(2 * np.pi * np.arange(n) / (24 * 6)) + walk) % 360
    wind_speed = np.clip(2.2 + 1.4 * np.sin(2 * np.pi * (h - 11) / 24)
                         + 0.8 * np.sin(2 * np.pi * np.arange(n) / (24 * 4.5)) + rng.normal(0, 0.4, n), 0.3, 7)
    # boundary layer: shallow at night (~200 m), deep afternoon (~1500 m)
    blh = np.clip(250 + 1200 * np.clip(np.sin(np.pi * (h - 7) / 12), 0, None) + rng.normal(0, 60, n), 120, 2200)
    return pd.DataFrame({"temp": temp, "rh": rh, "wind_speed": wind_speed, "wind_dir": wind_dir, "blh": blh},
                        index=idx)


def _synthetic_background(idx: pd.DatetimeIndex, wx: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    h = idx.hour.values
    n = len(idx)
    ventilation = (wx["wind_speed"].values * wx["blh"].values) / 1500.0
    base = 28 + 10 * np.sin(2 * np.pi * np.arange(n) / (24 * 11))
    diurnal = 10 * np.exp(-((h - 9) ** 2) / 6) + 12 * np.exp(-((h - 22) ** 2) / 8)
    pm25 = np.clip(base + diurnal + 14 / np.clip(ventilation, 0.3, None) + rng.normal(0, 3, n), 8, None)
    pm10 = pm25 * rng.uniform(1.7, 2.1, n)
    no2 = np.clip(18 + 16 * np.exp(-((h - 9) ** 2) / 5) + 14 * np.exp(-((h - 20) ** 2) / 6)
                  + 10 / np.clip(ventilation, 0.3, None) + rng.normal(0, 3, n), 4, None)
    return pd.DataFrame({"pm25_bg": pm25, "pm10_bg": pm10, "no2_bg": no2}, index=idx)


def _source_activity(kind: str, idx: pd.DatetimeIndex) -> np.ndarray:
    h = idx.hour.values
    wd = idx.dayofweek.values
    if kind == "industrial":   # shifts 6am-10pm, lighter on Sundays
        return np.where((h >= 6) & (h <= 22), 1.0, 0.35) * np.where(wd == 6, 0.5, 1.0)
    if kind == "traffic":      # rush hours
        return 0.3 + np.exp(-((h - 9) ** 2) / 3) + 0.9 * np.exp(-((h - 19) ** 2) / 4)
    if kind == "construction":  # daytime, not Sunday
        return np.where((h >= 8) & (h <= 18) & (wd != 6), 1.0, 0.05)
    if kind == "waste_burning":  # evening episodes
        return np.where((h >= 19) | (h <= 1), 1.0, 0.02)
    return np.ones(len(idx))


# emission strength and PM2.5 share of PM10 per source kind
_SOURCE_PARAMS = {
    "industrial": (55.0, 0.62),
    "traffic": (35.0, 0.55),
    "construction": (60.0, 0.22),
    "waste_burning": (70.0, 0.80),
}


def _plume(src, node, wind_dir_deg, wind_speed) -> np.ndarray:
    """Very simplified Gaussian-plume footprint: impact at node from source."""
    dy = (node["lat"] - src["lat"]) * 111.0
    dx = (node["lon"] - src["lon"]) * 111.0 * math.cos(math.radians(src["lat"]))
    # wind_dir is where wind comes FROM; plume travels toward wind_dir+180
    travel = np.radians((np.asarray(wind_dir_deg) + 180.0) % 360.0)
    ux, uy = np.sin(travel), np.cos(travel)
    along = dx * ux + dy * uy
    cross = -dx * uy + dy * ux
    dist = math.hypot(dx, dy)
    downwind = np.clip(along, 0.05, None)
    sigma = 0.25 + 0.35 * downwind
    f = np.exp(-(cross ** 2) / (2 * sigma ** 2)) / (1 + 2.5 * downwind) * (along > -0.1)
    near = math.exp(-dist / 0.35)  # a node sitting on a source always sees some of it
    return (f + 0.4 * near) / np.clip(np.asarray(wind_speed), 0.5, None)


def humidity_growth(rh) -> np.ndarray:
    """Hygroscopic growth factor for optical PM sensors (kappa-Koehler, kappa~0.3)."""
    rh = np.clip(np.asarray(rh, dtype=float), 0, 95) / 100.0
    return 1 + 0.3 * rh / (1 - rh) * 0.55


def simulate_nodes(weather: pd.DataFrame, background: pd.DataFrame, seed: int = 7,
                   inject_episode: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = background.index
    wx = weather.reindex(idx).interpolate(limit_direction="both")
    rows = []
    episode_start = idx[-1] - pd.Timedelta(hours=5)
    for node in config.NODES:
        pm25 = background["pm25_bg"].values * rng.uniform(0.92, 1.05)
        pm10 = background["pm10_bg"].values * rng.uniform(0.92, 1.05)
        for src in config.SOURCES:
            q, share = _SOURCE_PARAMS[src["kind"]]
            act = _source_activity(src["kind"], idx)
            if inject_episode and src["kind"] == "industrial":
                # demo episode: last ~5 hours, the industrial estate emits heavily
                act = act * np.where(idx >= episode_start, 5.0, 1.0)
            fp = _plume(src, node, wx["wind_dir"].values, wx["wind_speed"].values)
            contrib10 = q * act * fp
            pm10 = pm10 + contrib10
            pm25 = pm25 + contrib10 * share
        pm25 = np.clip(pm25 + rng.normal(0, 2.0, len(idx)), 3, None)
        pm10 = np.maximum(np.clip(pm10 + rng.normal(0, 4.0, len(idx)), 5, None), pm25 * 1.05)
        if node["type"] == "reference":
            raw25, raw10 = pm25.copy(), pm10.copy()
        else:
            g = humidity_growth(wx["rh"].values)
            gain = rng.uniform(1.05, 1.25)   # unit-to-unit sensor gain error
            drift = 1 + 0.06 * np.linspace(0, 1, len(idx))          # slow optical drift
            mult = rng.lognormal(0, 0.07, len(idx))                  # particle-mix variability
            raw25 = pm25 * gain * g * drift * mult + rng.normal(0, 3, len(idx))
            raw10 = pm10 * gain * (1 + (g - 1) * 0.6) * drift * rng.lognormal(0, 0.09, len(idx)) \
                + rng.normal(0, 5, len(idx))
        rows.append(pd.DataFrame({
            "time": idx, "node_id": node["id"],
            "pm25_true": pm25, "pm10_true": pm10,
            "pm25_raw": np.clip(raw25, 1, None), "pm10_raw": np.clip(raw10, 2, None),
        }))
    return pd.concat(rows, ignore_index=True)


def demo_dataset(days: int = 45, seed: int = 42, now: pd.Timestamp | None = None) -> Dataset:
    rng = np.random.default_rng(seed)
    now = (now or pd.Timestamp.now(tz=TZ).tz_localize(None)).floor("h")
    idx_all = pd.date_range(now - pd.Timedelta(days=days), now + pd.Timedelta(hours=48), freq="h")
    weather = _synthetic_weather(idx_all, rng)
    # demo episode conditions: calm, stable air in the last hours + next morning
    last = weather.index >= now - pd.Timedelta(hours=6)
    nxt = (weather.index > now) & (weather.index <= now + pd.Timedelta(hours=14))
    weather.loc[last | nxt, "wind_speed"] = np.clip(weather.loc[last | nxt, "wind_speed"] * 0.5, 0.4, None)
    weather.loc[last | nxt, "wind_dir"] = 325 + rng.normal(0, 6, (last | nxt).sum())  # NW: estate -> Peenya Ph2
    weather.loc[last | nxt, "blh"] = np.clip(weather.loc[last | nxt, "blh"] * 0.6, 120, None)
    idx_past = idx_all[idx_all <= now]
    background = _synthetic_background(idx_past, weather.loc[idx_past], rng)
    obs = simulate_nodes(weather, background, seed=seed)
    return Dataset(weather=weather, background=background, obs=obs, now=now, mode="demo")


def load_dataset(prefer_live: bool = True, days: int = 45) -> Dataset:
    if prefer_live:
        try:
            weather, bg, now = fetch_live(past_days=days)
            obs = simulate_nodes(weather, bg)
            return Dataset(weather=weather, background=bg, obs=obs, now=now, mode="live")
        except Exception:
            pass
    return demo_dataset(days=days)


def latest_snapshot(ds: Dataset, pm_col: str = "pm25_cal") -> pd.DataFrame:
    """Latest reading per node joined with node metadata."""
    last = ds.obs[ds.obs["time"] == ds.obs["time"].max()]
    meta = pd.DataFrame(config.NODES)
    snap = meta.merge(last, left_on="id", right_on="node_id", how="left")
    return snap
