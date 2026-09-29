"""OGC SensorThings API (v1.1) export: the open, vendor-neutral data contract that lets
any BRICS city's sensors join the network without custom integration."""
from __future__ import annotations

import pandas as pd

from . import config


def thing(node: dict, city: str = config.CITY, country: str = "IN") -> dict:
    return {
        "@iot.id": node["id"],
        "name": node["name"],
        "description": f"PranaWatch air-quality node ({node['type']})",
        "properties": {"city": city, "country": country, "corridor": config.CORRIDOR_NAME,
                       "hardware": "ESP32 + PMS5003 + BME280" if node["type"] == "pranawatch" else "CPCB reference"},
        "Locations": [{"name": node["name"], "encodingType": "application/geo+json",
                       "location": {"type": "Point", "coordinates": [node["lon"], node["lat"]]}}],
        "Datastreams": [
            {"name": f"{node['id']}-PM2.5", "unitOfMeasurement": {"name": "microgram per cubic meter",
             "symbol": "ug/m3", "definition": "http://qudt.org/vocab/unit/MicroGM-PER-M3"},
             "ObservedProperty": {"name": "PM2.5", "definition": "http://vocabs.lter-europe.net/EnvThes/22035"},
             "Sensor": {"name": "Plantower PMS5003", "encodingType": "application/pdf"}},
        ],
    }


def observations(obs: pd.DataFrame, node_id: str, col: str = "pm25_cal", last_n: int = 24) -> list[dict]:
    g = obs[obs["node_id"] == node_id].sort_values("time").tail(last_n)
    return [{"phenomenonTime": t.isoformat() + "+05:30", "result": round(float(v), 1),
             "parameters": {"calibrated": col.endswith("cal")}} for t, v in zip(g["time"], g[col])]
