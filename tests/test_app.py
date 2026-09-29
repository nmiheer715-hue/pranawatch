"""Headless UI test: loads the app, checks every tab renders and the alert button works."""
from streamlit.testing.v1 import AppTest


def test_app_runs_and_alert():
    at = AppTest.from_file("../app.py", default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 8
    alert_buttons = [b for b in at.button if b.key and b.key.startswith("alert-")]
    assert alert_buttons, "expected at least one hotspot alert button"
    alert_buttons[0].click().run()
    assert not at.exception, at.exception
    text = " ".join(i.value for i in at.info)
    assert "PranaWatch" in text or "Subject" in text


def test_india_overview_alert():
    at = AppTest.from_file("../app.py", default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    btn = [b for b in at.button if b.key and b.key.startswith("nat-alert-")]
    assert btn, "expected the national alert button"
    btn[0].click().run()
    assert not at.exception, at.exception
    assert any("Requested actions" in i.value for i in at.info)
