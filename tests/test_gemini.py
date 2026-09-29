import io
import pandas as pd
from PIL import Image
from core import config, gemini


def _ctx():
    src = dict(config.SOURCES[0], confidence=0.49, evidence=["upwind", "ratio"])
    return {"top_source": src, "pm25": 137, "excess": 41, "ratio": 0.63, "node_name": "Peenya Phase 2",
            "lat": 13.03, "lon": 77.52, "wind": {"from_deg": 316, "speed_ms": 0.9},
            "forecast_peak": {"time": "29 Sep 22:00", "value": 141}}


def test_alert_uses_gemini_json(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    fake = {"authority": "KSPCB", "subject": "s", "authority_message": "m",
            "public_advisory": {"en": "a", "kn": "b", "hi": "c"}, "priority": "high", "_model": "gemini-x"}
    monkeypatch.setattr(gemini, "_generate", lambda *a, **k: fake)
    assert gemini.write_alert(_ctx())["_model"] == "gemini-x"


def test_alert_falls_back_on_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "invalid-key")   # real SDK call, fails -> template
    al = gemini.write_alert(_ctx())
    assert al["_model"].startswith("template")
    assert set(al["public_advisory"]) == {"en", "kn", "hi"}


def test_photo_classification_parsing(monkeypatch):
    monkeypatch.setattr(gemini, "_generate", lambda *a, **k: {"category": "smoke??", "confidence": "0.7"})
    buf = io.BytesIO(); Image.new("RGB", (8, 8), "gray").save(buf, "JPEG")
    out = gemini.classify_photo(buf.getvalue())
    assert out["category"] == "none" and out["confidence"] == 0.7
