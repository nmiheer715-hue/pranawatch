# 🌬️ PranaWatch: Hyperlocal Air-Quality Intelligence

**Code for Communities 2.0 · Track 2: Clean Air & Climate Resilience (BRICS theme: Sustainability)**

City monitors report macro-level AQI. A foundry, a burning waste pile or a construction site 500 m away
gets averaged out, and satellites see only km-scale pixels. **PranaWatch** fuses low-cost street sensors,
citizen photos, Sentinel-5P satellite data and weather to **find hidden pollution hotspots, attribute them to
a likely source, forecast spikes 24–48 h ahead, and alert the responsible authority**. It runs as a
federated platform that BRICS cities can share models on without sharing raw data.

Pilot corridor: **Peenya Industrial Area, Bengaluru**.

---

## Two levels: national screening, then street-level action

- **🇮🇳 India overview (live):** 38 cities. CAMS PM2.5/PM10/NO₂ analysis and a 48 h forecast from Open-Meteo, AQI by
  India's NAQI method (24-h mean), and **trans-boundary smoke detection**: active fires (NASA FIRMS) lying upwind of a
  city, with an estimated smoke arrival time. Alerts are routed to the right state PCB / DPCC, plus **CAQM** for NCR.
- **Peenya pilot (hyperlocal):** once sensor nodes are deployed in a city, PranaWatch finds hidden street-level
  hotspots and attributes them to a specific source (tabs 2–6).

## What it does (maps to the challenge)

| Challenge asks for | PranaWatch |
|---|---|
| Citizen-sourced data: photos | Photo upload → **Gemini multimodal** classifies source (industrial / traffic / construction / burning) + severity |
| Citizen-sourced data: local sensor readings | ESP32 + PMS5003 + BME280 nodes ([firmware](firmware/pranawatch_node/pranawatch_node.ino)); humidity-aware **calibration** vs CPCB reference (error −94%) |
| National satellite imagery | **Google Earth Engine**: Sentinel-5P tropospheric NO₂ + NASA FIRMS active fires |
| Meteorological data | **Open-Meteo**: wind, humidity, boundary-layer height + regional CAMS air quality (live, free) |
| Detect hidden hotspots | Robust network-anomaly detection (z-score vs corridor median) |
| (and find their source) | Evidence fusion: upwind geometry + PM2.5/PM10 fingerprint + NO₂/fire satellite evidence + citizen photos; 3-h back-trajectory |
| Forecast AQ spikes | Gradient-boosted 24/48 h PM2.5 forecast with 80% interval; **77% better than persistence** in backtest |
| Alert authorities | **Gemini** writes an evidence-backed alert to KSPCB / BBMP / Traffic Police + public advisory in **English, ಕನ್ನಡ, हिन्दी** |
| Interoperable, share predictive models | **Federated learning (FedAvg)**: weights only, no raw data; **OGC SensorThings API** data standard |

## Architecture

```
 ESP32 nodes ─┐                                   ┌─► Hotspot detection ─► Source attribution ─┐
 Citizen photos ─► Gemini vision ─┐               │                                            ├─► Gemini alert writer
 Earth Engine (S5P NO2, FIRMS) ───┼─► Calibration ┼─► 24–48 h PM2.5 forecast ──────────────────┘   (EN / KN / HI)
 Open-Meteo weather + CAMS ───────┘   & fusion    │
                                                  └─► FedAvg across BRICS cities (OGC SensorThings)
```

## Run it locally (5 minutes)

```bash
git clone <your-repo-url> && cd pranawatch
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app works immediately with no keys. Each missing service falls back automatically:
- **Gemini key.** Get a free one at https://aistudio.google.com/apikey. Paste it in the sidebar, or copy
  `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill it in. Without a key, alerts use
  built-in templates and photos are labelled manually.
- **Earth Engine (optional).** Register at https://code.earthengine.google.com/register (free for noncommercial
  use), then run `earthengine authenticate`, set `EE_PROJECT=<your-cloud-project-id>`, and turn on the sidebar
  toggle. Without it, a clearly labelled simulated NO₂ layer is used.
- **NASA FIRMS key (optional, free, instant):** get one at https://firms.modaps.eosdis.nasa.gov/api/map_key and add
  `FIRMS_MAP_KEY = "..."` to secrets for live fire detections across India. Without it, a clearly labelled simulated
  early-season stubble-fire pattern is used.
- **Offline.** If Open-Meteo is unreachable, a built-in demo dataset is used, so the demo never breaks.

Run the tests with `pip install pytest && pytest -q` (11 tests, including a headless UI test of every tab).

## Deploy (free, about 5 minutes)

1. Push this folder to a **public GitHub repo** (make sure `secrets.toml` is NOT committed; `.gitignore` handles it).
2. Go to https://share.streamlit.io, choose **New app**, pick the repo, and set main file `app.py`.
3. Under **Advanced settings → Secrets**, paste `GEMINI_API_KEY = "..."`, then deploy.

(For Google Cloud: the app runs unchanged on Cloud Run with `streamlit run app.py --server.port $PORT`.)

## Project layout

```
app.py                     Streamlit dashboard (8 tabs)
core/config.py             Corridor nodes, sources, authorities, India NAQI
core/data.py               Open-Meteo live fetch + physics-based demo/node simulation
core/calibration.py        Humidity-aware low-cost sensor calibration
core/hotspots.py           Hotspot detection, source attribution, back-trajectory
core/forecast.py           24–48 h PM2.5 forecaster + backtest vs baselines
core/satellite.py          Earth Engine Sentinel-5P NO2 + FIRMS (with fallback)
core/gemini.py             Gemini photo classification + multilingual alerts
core/federated.py          FedAvg simulation across 3 BRICS cities
core/sensorthings.py       OGC SensorThings API export
core/national.py           All-India screening: 38 cities, CAMS forecast, fire-smoke transport
firmware/                  ESP32 node firmware (PMS5003 + BME280)
tests/                     Unit + headless UI tests
```

## Honest status

- **Live:** Open-Meteo weather and regional (CAMS) air quality; Gemini; Earth Engine when authenticated.
- **Simulated:** street-level node readings, generated on top of the live regional background with a Gaussian-plume
  model from known source categories (hardware nodes are designed but not yet deployed); the other two
  federation cities.
- The detection, attribution, calibration, forecasting and alerting logic is the same for real data.

## Roadmap

1. Build 5 nodes on the custom KiCad PCB (solar + Li-ion) and co-locate them at the CPCB Peenya station for calibration.
2. Pilot with one BBMP ward and KSPCB regional office; WhatsApp alert channel for residents.
3. BigQuery + Cloud Run backend; onboard a second BRICS city through the SensorThings API.

---
Built by **Miheer Nikam**, EEE, BMS College of Engineering, Bengaluru.
