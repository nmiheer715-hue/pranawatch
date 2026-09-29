"""Low-cost sensor calibration.

Optical PM sensors (PMS5003 etc.) over-read in humid air because particles absorb
water and grow. Each node is first co-located with a CPCB reference monitor for a
week; we fit a per-node model  PM_ref ~ f(PM_raw, humidity term, temperature)  and
apply it to all later readings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def _features(raw: np.ndarray, rh: np.ndarray, temp: np.ndarray) -> np.ndarray:
    rh = np.clip(rh, 0, 95)
    hum = (rh / 100.0) / (1 - rh / 100.0)
    return np.column_stack([raw, raw / (1 + 0.165 * hum), hum, temp])


def calibrate(obs: pd.DataFrame, weather: pd.DataFrame, colocation_days: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (obs with pm25_cal/pm10_cal columns, per-node metrics table)."""
    wx = weather[["rh", "temp"]]
    df = obs.merge(wx, left_on="time", right_index=True, how="left")
    df[["rh", "temp"]] = df[["rh", "temp"]].interpolate(limit_direction="both")
    t0 = df["time"].min()
    coloc_end = t0 + pd.Timedelta(days=colocation_days)
    df["pm25_cal"] = df["pm25_raw"]
    df["pm10_cal"] = df["pm10_raw"]
    metrics = []
    for node_id, g in df.groupby("node_id"):
        train = g[g["time"] < coloc_end]
        test = g[g["time"] >= coloc_end]
        if (g["pm25_raw"] == g["pm25_true"]).all():  # reference station, no calibration needed
            continue
        for pol in ("pm25", "pm10"):
            X_tr = _features(train[f"{pol}_raw"].values, train["rh"].values, train["temp"].values)
            m = LinearRegression().fit(X_tr, train[f"{pol}_true"].values)
            X_all = _features(g[f"{pol}_raw"].values, g["rh"].values, g["temp"].values)
            df.loc[g.index, f"{pol}_cal"] = np.clip(m.predict(X_all), 1, None)
            if pol == "pm25":
                cal_test = df.loc[test.index, "pm25_cal"].values
                metrics.append({
                    "node_id": node_id,
                    "mae_raw": mean_absolute_error(test["pm25_true"], test["pm25_raw"]),
                    "mae_cal": mean_absolute_error(test["pm25_true"], cal_test),
                    "r2_raw": r2_score(test["pm25_true"], test["pm25_raw"]),
                    "r2_cal": r2_score(test["pm25_true"], cal_test),
                })
    return df.drop(columns=["rh", "temp"]), pd.DataFrame(metrics)
