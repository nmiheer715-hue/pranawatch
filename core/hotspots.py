"""Hidden hotspot detection and source attribution.

1. Detection: a node is a hotspot when its calibrated PM2.5 is far above what the rest
   of the network (and the regional/satellite background) says it should be:
   robust z-score vs network median, sustained over the last few hours.
2. Attribution: combine independent evidence into a score for each candidate source:
     - upwind geometry: is the source upwind of the node given current wind?
     - particle fingerprint: PM2.5/PM10 ratio (combustion ~0.6-0.85, dust ~0.2-0.35)
     - satellite: NO2 enhancement (industry/traffic), VIIRS fire points (burning)
     - citizen reports: Gemini-classified photos near the node
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config

EXPECTED_RATIO = {"industrial": 0.62, "traffic": 0.55, "construction": 0.25, "waste_burning": 0.80,
                  "crop_burning": 0.82}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def angle_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def classify_ratio(ratio: float) -> str:
    if ratio >= 0.72:
        return "fresh combustion (burning)"
    if ratio >= 0.48:
        return "combustion (industry / vehicles)"
    if ratio <= 0.35:
        return "coarse dust (construction / road dust)"
    return "mixed"


def detect_hotspots(obs: pd.DataFrame, now: pd.Timestamp, window_h: int = 3, z_thresh: float = 2.0,
                    min_excess: float = 15.0) -> pd.DataFrame:
    """Return per-node stats over the last `window_h` hours with hotspot flag."""
    recent = obs[(obs["time"] > now - pd.Timedelta(hours=window_h)) & (obs["time"] <= now)]
    agg = recent.groupby("node_id").agg(pm25=("pm25_cal", "mean"), pm10=("pm10_cal", "mean")).reset_index()
    med = agg["pm25"].median()
    mad = (agg["pm25"] - med).abs().median() * 1.4826
    mad = max(mad, 6.0)
    agg["network_median"] = med
    agg["excess"] = agg["pm25"] - med
    agg["z"] = agg["excess"] / mad
    agg["ratio"] = agg["pm25"] / agg["pm10"].clip(lower=1)
    agg["fingerprint"] = agg["ratio"].apply(classify_ratio)
    agg["hotspot"] = (agg["z"] >= z_thresh) & (agg["excess"] >= min_excess)
    meta = pd.DataFrame(config.NODES)[["id", "name", "lat", "lon", "type"]]
    return meta.merge(agg, left_on="id", right_on="node_id").drop(columns="node_id")


def attribute(node: dict, wind_dir_from: float, wind_speed: float, ratio: float,
              no2_enhancement: float = 0.0, fires: list | None = None, reports: list | None = None,
              sources: list | None = None) -> list[dict]:
    """Score each candidate source for a hotspot node. Returns sorted list with evidence."""
    sources = sources or config.SOURCES
    fires = fires or []
    reports = reports or []
    results = []
    for s in sources:
        dist = haversine_km(node["lat"], node["lon"], s["lat"], s["lon"])
        brg = bearing_deg(node["lat"], node["lon"], s["lat"], s["lon"])
        evidence = []
        # upwind geometry: source should lie in the direction the wind is coming FROM
        if dist < 0.3:
            geo = 0.8
            evidence.append(f"source is {dist*1000:.0f} m from the node")
        else:
            d_ang = angle_diff(brg, wind_dir_from)
            geo = max(0.0, math.cos(math.radians(min(d_ang, 90)))) ** 2 * math.exp(-dist / 3.0)
            if d_ang <= 35:
                evidence.append(f"source is {dist:.1f} km UPWIND (bearing {brg:.0f}°, wind from {wind_dir_from:.0f}°)")
        # particle fingerprint
        exp_r = EXPECTED_RATIO.get(s["kind"], 0.5)
        fp = math.exp(-((ratio - exp_r) ** 2) / (2 * 0.12 ** 2))
        if fp > 0.6:
            evidence.append(f"PM2.5/PM10 ratio {ratio:.2f} matches {s['kind'].replace('_', ' ')} signature (~{exp_r:.2f})")
        # satellite
        sat = 0.0
        if s["kind"] in ("industrial", "traffic") and no2_enhancement > 0.15:
            sat = min(no2_enhancement, 1.0)
            evidence.append(f"Sentinel-5P NO2 elevated {no2_enhancement*100:.0f}% above city background")
        if s["kind"] in ("waste_burning", "crop_burning"):
            near = [f for f in fires if haversine_km(f["lat"], f["lon"], s["lat"], s["lon"]) < 2.0]
            if near:
                sat = 1.0
                evidence.append(f"{len(near)} NASA FIRMS active-fire detection(s) within 2 km")
        # citizen reports (Gemini-classified photos)
        cit = 0.0
        for r in reports:
            if r.get("category") == s["kind"] and haversine_km(r["lat"], r["lon"], s["lat"], s["lon"]) < 1.5:
                cit = max(cit, float(r.get("confidence", 0.7)))
        if cit:
            evidence.append(f"citizen photo report classified by Gemini as {s['kind'].replace('_', ' ')}")
        wind_factor = 1.0 if wind_speed < 3.5 else 0.8
        score = wind_factor * (0.45 * geo + 0.30 * fp + 0.15 * sat + 0.10 * cit)
        results.append({**s, "distance_km": dist, "bearing": brg, "score": score, "evidence": evidence,
                        "geo": geo, "fingerprint_match": fp})
    total = sum(r["score"] for r in results) or 1.0
    for r in results:
        r["confidence"] = r["score"] / total
    return sorted(results, key=lambda r: r["score"], reverse=True)


def back_trajectory(lat: float, lon: float, weather: pd.DataFrame, now: pd.Timestamp, hours: int = 3) -> list[tuple]:
    """Trace air parcel backwards in time using hourly wind (simple Lagrangian step)."""
    pts = [(lat, lon)]
    t = now
    for _ in range(hours):
        if t not in weather.index:
            break
        ws, wd = weather.loc[t, "wind_speed"], weather.loc[t, "wind_dir"]
        # parcel came FROM wind_dir: step backwards toward wind_dir
        step_km = ws * 3.6 * 0.5   # half-hour effective step keeps it within the corridor
        la, lo = pts[-1]
        dlat = step_km * math.cos(math.radians(wd)) / 111.0
        dlon = step_km * math.sin(math.radians(wd)) / (111.0 * math.cos(math.radians(la)))
        pts.append((la + dlat, lo + dlon))
        t -= pd.Timedelta(hours=1)
    return pts
