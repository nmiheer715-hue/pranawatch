import pandas as pd
from core import cpcb, gemini, national


def test_cpcb_parse_and_match():
    recs = [
        {"station": "Anand Vihar, Delhi - DPCC", "city": "Delhi", "state": "Delhi", "latitude": "28.647",
         "longitude": "77.316", "pollutant_id": "PM2.5", "avg_value": "120", "max_value": "180"},
        {"station": "ITO, Delhi", "city": "Delhi", "state": "Delhi", "latitude": "28.628", "longitude": "77.241",
         "pollutant_id": "PM2.5", "pollutant_avg": "100", "pollutant_max": "150"},
        {"station": "x", "latitude": "28.6", "longitude": "77.2", "pollutant_id": "NO2", "avg_value": "40"},
        {"station": "bad", "latitude": "NA", "longitude": "77.2", "pollutant_id": "PM2.5", "avg_value": "NA"},
    ]
    st = cpcb.parse_records(recs)
    assert len(st) == 2 and st["pm25"].tolist() == [120.0, 100.0]
    s = national.load(prefer_live=False)["summary"]
    m = cpcb.match_cities(s, st)
    d = m[m.city == "Delhi"].iloc[0]
    assert d["cpcb_pm25"] == 110 and d["cpcb_stations"] == 2
    assert cpcb.validation(m)["n"] >= 1


def test_cpcb_missing_key(monkeypatch):
    monkeypatch.delenv("DATA_GOV_IN_KEY", raising=False)
    assert cpcb.load()["mode"] == "none"


def test_pcm_to_wav_header():
    wav = gemini._pcm_to_wav(b"\x00\x00" * 2400)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
