"""Google BigQuery export: store national snapshots and alerts for trend analysis.

Works in the free BigQuery sandbox (no billing) using load jobs. Credentials, in order:
Application Default Credentials (`gcloud auth application-default login`), else the
Earth Engine login saved by `earthengine authenticate` (it includes the cloud-platform
scope). Project = BQ_PROJECT or EE_PROJECT. Dataset = BQ_DATASET (default "pranawatch").
"""
from __future__ import annotations

import json
import os

import pandas as pd


def project() -> str | None:
    return os.environ.get("BQ_PROJECT") or os.environ.get("EE_PROJECT")


def available() -> bool:
    if not project():
        return False
    try:
        import google.cloud.bigquery  # noqa: F401
        return True
    except ImportError:
        return False


def _client():
    from google.cloud import bigquery
    proj = project()
    try:
        import google.auth
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    except Exception:
        import ee
        creds = ee.data.get_persistent_credentials()
    return bigquery.Client(project=proj, credentials=creds)


def export(summary: pd.DataFrame, alerts: list[dict], now) -> dict:
    """Append the city snapshot (and any alerts) to BigQuery. Returns table ids and row counts."""
    from google.cloud import bigquery
    client = _client()
    ds_id = f"{project()}.{os.environ.get('BQ_DATASET', 'pranawatch')}"
    client.create_dataset(bigquery.Dataset(ds_id), exists_ok=True)
    cols = ["city", "state", "lat", "lon", "pm25_now", "pm25_24h", "aqi", "category", "fc_24h_peak",
            "fc_category", "upwind_fires", "smoke_flag", "cpcb_pm25", "cpcb_stations"]
    snap = summary[[c for c in cols if c in summary.columns]].copy()
    snap.insert(0, "snapshot_time", pd.Timestamp(now))
    cfg = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND", autodetect=True)
    client.load_table_from_dataframe(snap, f"{ds_id}.city_snapshots", job_config=cfg).result()
    out = {"city_snapshots": len(snap)}
    if alerts:
        al = pd.DataFrame([{"created": pd.Timestamp(now), "node": a.get("node"), "authority": a.get("authority"),
                            "subject": a.get("subject"), "priority": a.get("priority"),
                            "message": a.get("authority_message"), "model": a.get("_model"),
                            "advisory_json": json.dumps(a.get("public_advisory", {}), ensure_ascii=False)}
                           for a in alerts])
        client.load_table_from_dataframe(al, f"{ds_id}.alerts", job_config=cfg).result()
        out["alerts"] = len(al)
    out["dataset"] = ds_id
    return out
