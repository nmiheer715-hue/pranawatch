"""Federated learning simulation (FedAvg) for BRICS interoperability.

Each city trains a short-term PM2.5 model on its OWN data and only shares model
weights with the coordinator; raw sensor data never leaves the country. The
coordinator averages the weights (FedAvg, weighted by sample count) and sends the
global model back.

Demo: three BRICS cities with different pollution regimes. A newly joined city with
only a week of data gets a much better model from the federation than it could
train alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as data_mod
from .calibration import calibrate
from .forecast import build_training

# Shared feature schema = the interoperability contract between countries
SCHEMA = [
    ("lag1", 100), ("lag2", 100), ("lag3", 100), ("lag24", 100), ("roll6", 100), ("roll24", 100),
    ("hour_sin", 1), ("hour_cos", 1), ("rh", 100), ("wind_u", 5), ("wind_v", 5), ("blh", 1000),
    ("ventilation", 5000),
]

CITIES = [
    {"city": "Bengaluru", "country": "India", "seed": 42, "scale": 1.0, "days": 45},
    {"city": "Sao Paulo", "country": "Brazil", "seed": 11, "scale": 0.8, "days": 45},
    {"city": "Johannesburg (new member)", "country": "South Africa", "seed": 23, "scale": 1.2, "days": 10},
]


def _xy(df: pd.DataFrame):
    X = np.column_stack([df[c].values / s for c, s in SCHEMA] + [np.ones(len(df))])
    return X, df["y"].values


def city_data(spec: dict, now: pd.Timestamp):
    ds = data_mod.demo_dataset(days=spec["days"], seed=spec["seed"], now=now)
    ds.background[["pm25_bg", "pm10_bg"]] *= spec["scale"]
    ds.obs = data_mod.simulate_nodes(ds.weather, ds.background, seed=spec["seed"], inject_episode=False)
    obs, _ = calibrate(ds.obs, ds.weather)
    ser = obs[obs["node_id"] == "PW-01"].set_index("time")["pm25_cal"].asfreq("h").interpolate()
    tr = build_training(ser, ds.weather)
    cut = int(len(tr) * 0.75)
    return tr.iloc[:cut], tr.iloc[cut:]


def _local_sgd(w, X, y, epochs=40, lr=0.05):
    n = len(y)
    for _ in range(epochs):
        grad = X.T @ (X @ w - y) / n
        w = w - lr * grad
    return w


def mae(w, X, y):
    return float(np.mean(np.abs(X @ w - y)))


def run(rounds: int = 30, now: pd.Timestamp | None = None) -> dict:
    now = (now or pd.Timestamp.now()).floor("h")
    clients = []
    for spec in CITIES:
        tr, te = city_data(spec, now)
        Xtr, ytr = _xy(tr)
        Xte, yte = _xy(te)
        clients.append({**spec, "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte, "n": len(ytr)})

    d = clients[0]["Xtr"].shape[1]
    # local-only baselines (same optimiser & budget)
    for c in clients:
        c["w_local"] = _local_sgd(np.zeros(d), c["Xtr"], c["ytr"], epochs=40 * rounds)

    # FedAvg
    w = np.zeros(d)
    history = []
    total = sum(c["n"] for c in clients)
    for r in range(rounds):
        updates = [(_local_sgd(w.copy(), c["Xtr"], c["ytr"]), c["n"]) for c in clients]
        w = sum(u * n for u, n in updates) / total
        history.append({"round": r + 1, **{c["city"]: mae(w, c["Xte"], c["yte"]) for c in clients}})

    # personalised: 1 short local fine-tune of the global model
    rows = []
    for c in clients:
        w_pers = _local_sgd(w.copy(), c["Xtr"], c["ytr"], epochs=10)
        rows.append({
            "city": c["city"], "country": c["country"], "train_samples": c["n"],
            "mae_local_only": mae(c["w_local"], c["Xte"], c["yte"]),
            "mae_federated": mae(w, c["Xte"], c["yte"]),
            "mae_federated_personalised": mae(w_pers, c["Xte"], c["yte"]),
            "raw_data_shared": "No (weights only)",
        })
    return {"summary": pd.DataFrame(rows), "history": pd.DataFrame(history), "n_params": d}
