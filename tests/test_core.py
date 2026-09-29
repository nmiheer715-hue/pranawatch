import pandas as pd
from core import calibration, config, data, forecast, gemini, hotspots, satellite


def _ds():
    ds = data.demo_dataset(days=30, now=pd.Timestamp("2026-09-29 00:00"))
    ds.obs, m = calibration.calibrate(ds.obs, ds.weather)
    return ds, m


def test_aqi():
    assert config.pm25_to_aqi(25)[1] == "Good"
    assert config.pm25_to_aqi(95)[1] == "Poor"
    assert config.pm25_to_aqi(300)[1] == "Severe"


def test_calibration_reduces_error():
    _, m = _ds()
    assert (m["mae_cal"] < m["mae_raw"] * 0.3).all()


def test_hotspot_and_attribution():
    ds, _ = _ds()
    hs = hotspots.detect_hotspots(ds.obs, ds.now)
    hot = hs[hs.hotspot]
    assert "PW-01" in hot["id"].tolist()
    r = hot[hot.id == "PW-01"].iloc[0]
    wx = ds.weather.loc[ds.now]
    no2 = satellite.simulated_no2()
    ranked = hotspots.attribute(r.to_dict(), wx.wind_dir, wx.wind_speed, r.ratio,
                                satellite.no2_enhancement_at(no2, r.lat, r.lon), satellite.simulated_fires())
    assert ranked[0]["kind"] == "industrial"


def test_forecast_beats_persistence():
    ds, _ = _ds()
    ser = ds.obs[ds.obs.node_id == "PW-04"].set_index("time")["pm25_cal"].asfreq("h").interpolate()
    band, bt = forecast.forecast_node(ser, ds.weather, horizon=24)
    assert len(band) == 24
    assert bt["mae_model"] < bt["mae_persistence"]


def test_template_alert_multilingual():
    src = dict(config.SOURCES[0], confidence=0.5, evidence=["x"])
    al = gemini.template_alert({"top_source": src, "pm25": 140, "excess": 40, "node_name": "Peenya"})
    assert set(al["public_advisory"]) == {"en", "kn", "hi"}


def test_national_summary_and_smoke():
    from core import national
    d = national.load(prefer_live=False)
    s = d["summary"]
    assert len(s) == len(national.CITIES)
    assert s.loc[s.city == "Delhi", "smoke_flag"].iloc[0]
    assert not s.loc[s.city == "Chennai", "smoke_flag"].iloc[0]
    ctx = national.alert_context(s[s.city == "Delhi"].iloc[0], s)
    assert "CAQM" in ctx["top_source"]["authority"] or "Punjab" in ctx["top_source"]["authority"]
