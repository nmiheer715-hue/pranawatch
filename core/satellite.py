"""Satellite layers via Google Earth Engine.

  * Sentinel-5P TROPOMI tropospheric NO2 (COPERNICUS/S5P/OFFL/L3_NO2)
  * NASA FIRMS active fires (FIRMS)

Needs `earthengine-api` and an authenticated Earth Engine account with a Cloud
project (set EE_PROJECT). Without it, a clearly-labelled simulated layer is used
so the dashboard still demonstrates the fusion logic.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from . import config

BBOX = (77.47, 12.99, 77.58, 13.08)  # lon_min, lat_min, lon_max, lat_max


def _ee():
    """Initialise Earth Engine. Never opens a login flow inside the web app:
    run `earthengine authenticate` once in a terminal instead."""
    import ee  # noqa: WPS433
    project = os.environ.get("EE_PROJECT")
    if not project:
        raise RuntimeError("EE_PROJECT not set: add EE_PROJECT = \"your-cloud-project-id\" to .streamlit/secrets.toml")
    try:
        ee.Initialize(project=project)
    except Exception as e:
        raise RuntimeError(f"Earth Engine not authenticated for project '{project}'. Run `earthengine authenticate` "
                           f"in a terminal, then restart the app. ({type(e).__name__}: {str(e)[:120]})") from e
    return ee


def gee_no2(days: int = 30, step: float = 0.01) -> tuple[pd.DataFrame, str | None]:
    """Mean tropospheric NO2 on a grid (umol/m2) + a map tile URL for overlays."""
    ee = _ee()
    end = ee.Date(pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    start = end.advance(-days, "day")
    region = ee.Geometry.Rectangle(list(BBOX))
    img = (ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
           .select("tropospheric_NO2_column_number_density")
           .filterDate(start, end).filterBounds(region).mean()
           .multiply(1e6).rename("no2"))  # mol/m2 -> umol/m2
    lons = np.arange(BBOX[0], BBOX[2] + 1e-9, step)
    lats = np.arange(BBOX[1], BBOX[3] + 1e-9, step)
    feats = [ee.Feature(ee.Geometry.Point([float(x), float(y)]), {"lat": float(y), "lon": float(x)})
             for y in lats for x in lons]
    fc = img.reduceRegions(ee.FeatureCollection(feats), ee.Reducer.mean(), scale=1113)
    rows = [f["properties"] for f in fc.getInfo()["features"]]
    df = pd.DataFrame(rows).rename(columns={"mean": "no2"}).dropna()
    vis = {"min": 40, "max": 160, "palette": ["0000ff", "00ffff", "00ff00", "ffff00", "ff0000"]}
    tile = img.clip(region.buffer(5000)).getMapId(vis)["tile_fetcher"].url_format
    return df, tile


def gee_fires(days: int = 10) -> list[dict]:
    ee = _ee()
    end = ee.Date(pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    region = ee.Geometry.Rectangle([BBOX[0] - 0.3, BBOX[1] - 0.3, BBOX[2] + 0.3, BBOX[3] + 0.3])
    img = ee.ImageCollection("FIRMS").filterDate(end.advance(-days, "day"), end).select("T21").max()
    pts = img.sample(region=region, scale=1000, geometries=True).limit(200).getInfo()["features"]
    return [{"lat": p["geometry"]["coordinates"][1], "lon": p["geometry"]["coordinates"][0],
             "brightness_k": p["properties"].get("T21"), "source": "NASA FIRMS"} for p in pts]


# ---------------------------------------------------------------------------
# Simulated fallback (clearly labelled in the UI)
# ---------------------------------------------------------------------------

def simulated_no2(step: float = 0.005, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for y in np.arange(BBOX[1], BBOX[3] + 1e-9, step):
        for x in np.arange(BBOX[0], BBOX[2] + 1e-9, step):
            v = 65 + 8 * (y - BBOX[1]) / (BBOX[3] - BBOX[1]) + rng.normal(0, 3)
            for s in config.SOURCES:
                if s["kind"] in ("industrial", "traffic"):
                    d = math.hypot((y - s["lat"]) * 111, (x - s["lon"]) * 109)
                    amp = 70 if s["kind"] == "industrial" else 35
                    v += amp * math.exp(-(d ** 2) / (2 * 1.2 ** 2))
            rows.append({"lat": float(y), "lon": float(x), "no2": float(v)})
    return pd.DataFrame(rows)


def simulated_fires() -> list[dict]:
    s = next(s for s in config.SOURCES if s["kind"] == "waste_burning")
    return [{"lat": s["lat"] + 0.002, "lon": s["lon"] - 0.001, "brightness_k": 318.0, "source": "simulated"}]


def load_layers(use_gee: bool = True) -> dict:
    """Returns dict(no2=df, fires=list, tile=url|None, mode='gee'|'simulated', error=str|None)."""
    if use_gee:
        try:
            no2, tile = gee_no2()
            fires = gee_fires()
            if len(no2) > 0:
                return {"no2": no2, "fires": fires, "tile": tile, "mode": "gee", "error": None}
        except Exception as e:  # not installed / not authenticated / offline
            err = str(e)[:300]
            return {"no2": simulated_no2(), "fires": simulated_fires(), "tile": None, "mode": "simulated", "error": err}
    return {"no2": simulated_no2(), "fires": simulated_fires(), "tile": None, "mode": "simulated", "error": None}


def no2_enhancement_at(no2: pd.DataFrame, lat: float, lon: float, radius_km: float = 1.5) -> float:
    """Fractional NO2 enhancement near a point vs the corridor median."""
    if no2 is None or no2.empty:
        return 0.0
    d = np.hypot((no2["lat"] - lat) * 111, (no2["lon"] - lon) * 109)
    near = no2.loc[d <= radius_km, "no2"]
    if near.empty:
        near = no2.loc[[d.idxmin()], "no2"]
    med = no2["no2"].median()
    return float((near.mean() - med) / med) if med else 0.0
