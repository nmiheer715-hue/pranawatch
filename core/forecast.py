"""24-48 h PM2.5 forecast per node.

Gradient-boosted trees on lagged PM2.5 + meteorology (humidity, wind, boundary-layer
height, a "ventilation" index) + time-of-day. The model is rolled forward hour by hour
using the weather forecast for future hours. Backtested against persistence and
same-hour-yesterday baselines so the gain is measurable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

LAGS = (1, 2, 3, 6, 12, 24)
FEATURES = [f"lag{l}" for l in LAGS] + [
    "roll6", "roll24", "hour_sin", "hour_cos", "dow", "rh", "wind_speed",
    "wind_u", "wind_v", "blh", "ventilation",
]


def _weather_feats(wx: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame(index=wx.index)
    rad = np.radians(wx["wind_dir"])
    w["rh"] = wx["rh"]
    w["wind_speed"] = wx["wind_speed"]
    w["wind_u"] = wx["wind_speed"] * np.sin(rad)
    w["wind_v"] = wx["wind_speed"] * np.cos(rad)
    w["blh"] = wx["blh"]
    w["ventilation"] = wx["wind_speed"] * wx["blh"]
    h = wx.index.hour
    w["hour_sin"] = np.sin(2 * np.pi * h / 24)
    w["hour_cos"] = np.cos(2 * np.pi * h / 24)
    w["dow"] = wx.index.dayofweek
    return w


def build_training(series: pd.Series, weather: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    for l in LAGS:
        df[f"lag{l}"] = series.shift(l)
    df["roll6"] = series.shift(1).rolling(6).mean()
    df["roll24"] = series.shift(1).rolling(24).mean()
    df = df.join(_weather_feats(weather), how="left")
    return df.dropna()


class PM25Forecaster:
    def __init__(self, **kw):
        params = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
        params.update(kw)
        self.model = HistGradientBoostingRegressor(**params)
        self.resid_std = None

    def fit(self, series: pd.Series, weather: pd.DataFrame):
        tr = build_training(series, weather)
        self.model.fit(tr[FEATURES], tr["y"])
        return self

    def predict(self, history: pd.Series, weather: pd.DataFrame, horizon: int = 24) -> pd.Series:
        hist = history.copy().astype(float)
        wf = _weather_feats(weather)
        t = hist.index[-1]
        out = {}
        for _ in range(horizon):
            t = t + pd.Timedelta(hours=1)
            if t not in wf.index:
                break
            row = {f"lag{l}": hist.iloc[-l] for l in LAGS}
            row["roll6"] = hist.iloc[-6:].mean()
            row["roll24"] = hist.iloc[-24:].mean()
            row.update(wf.loc[t].to_dict())
            yhat = float(max(self.model.predict(pd.DataFrame([row])[FEATURES])[0], 1.0))
            out[t] = yhat
            hist.loc[t] = yhat
        return pd.Series(out, name="forecast")


def backtest(series: pd.Series, weather: pd.DataFrame, test_days: int = 5, horizon: int = 24,
             step_hours: int = 12) -> dict:
    """Rolling-origin backtest vs persistence and same-hour-yesterday baselines."""
    end = series.index[-1]
    split = end - pd.Timedelta(days=test_days)
    model = PM25Forecaster().fit(series[series.index <= split], weather)
    errs, pers, seas = [], [], []
    by_h = {h: [] for h in range(1, horizon + 1)}
    origin = split
    while origin + pd.Timedelta(hours=horizon) <= end:
        hist = series[series.index <= origin]
        fc = model.predict(hist, weather, horizon)
        actual = series.reindex(fc.index)
        e = (fc - actual).abs()
        errs.append(e.mean())
        pers.append((actual - hist.iloc[-1]).abs().mean())
        yday = series.reindex(fc.index - pd.Timedelta(hours=24)).values
        seas.append(np.nanmean(np.abs(actual.values - yday)))
        for i, v in enumerate((fc - actual).values, start=1):
            by_h[i].append(v)
        origin += pd.Timedelta(hours=step_hours)
    resid_std = np.array([np.std(by_h[h]) if by_h[h] else np.nan for h in range(1, horizon + 1)])
    mae, mae_p, mae_s = float(np.mean(errs)), float(np.mean(pers)), float(np.mean(seas))
    return {
        "mae_model": mae, "mae_persistence": mae_p, "mae_yesterday": mae_s,
        "skill_vs_persistence": 1 - mae / mae_p if mae_p else np.nan,
        "n_origins": len(errs), "resid_std": resid_std,
    }


def forecast_node(series: pd.Series, weather: pd.DataFrame, horizon: int = 24, with_backtest: bool = True):
    """Fit on all history and forecast `horizon` hours ahead with an uncertainty band."""
    bt = backtest(series, weather, horizon=horizon) if with_backtest else None
    model = PM25Forecaster().fit(series, weather)
    fc = model.predict(series, weather, horizon)
    std = bt["resid_std"][: len(fc)] if bt is not None else np.linspace(3, 12, len(fc))
    std = np.nan_to_num(std, nan=np.nanmean(std) if np.isfinite(std).any() else 8.0)
    band = pd.DataFrame({"forecast": fc, "lo": np.clip(fc - 1.28 * std, 0, None), "hi": fc + 1.28 * std})
    return band, bt
