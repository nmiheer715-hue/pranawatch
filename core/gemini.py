"""Gemini integration (google-genai SDK).

  * classify_photo()  - multimodal: what pollution source is visible in a citizen photo?
  * write_alert()     - turns hotspot evidence into an authority alert + public advisory
                        in English, Kannada and Hindi.

Set GEMINI_API_KEY (free key from https://aistudio.google.com/apikey).
Optionally set GEMINI_MODEL (default below). Without a key, deterministic templates
are used so the pipeline still runs end to end.
"""
from __future__ import annotations

import json
import os

from . import config

DEFAULT_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]
CATEGORIES = ["industrial", "traffic", "construction", "waste_burning", "crop_burning", "vehicle", "none"]


def _key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def available() -> bool:
    if not _key():
        return False
    try:
        import google.genai  # noqa: F401
        return True
    except ImportError:
        return False


def _models() -> list[str]:
    m = os.environ.get("GEMINI_MODEL")
    return ([m] if m else []) + [x for x in DEFAULT_MODELS if x != m]


def _generate(contents, schema_hint: str) -> dict:
    from google import genai
    from google.genai import types

    # AI Studio key first; then Vertex AI "express mode" (some newer "AQ." keys are Vertex keys)
    clients = [("gemini-api", lambda: genai.Client(api_key=_key())),
               ("vertex", lambda: genai.Client(vertexai=True, api_key=_key()))]
    last_err = None
    for backend, make in clients:
        try:
            client = make()
        except Exception as e:
            last_err = e
            continue
        for model in _models():
            try:
                resp = client.models.generate_content(
                    model=model, contents=contents,
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.3),
                )
                data = json.loads(resp.text)
                if isinstance(data, list):
                    data = data[0] if data else {}
                data["_model"] = f"{model} ({backend})"
                return data
            except Exception as e:  # try next model / backend
                last_err = e
                continue
    raise RuntimeError(f"Gemini call failed ({schema_hint}): {last_err}")


# ---------------------------------------------------------------------------
# Photo classification
# ---------------------------------------------------------------------------

PHOTO_PROMPT = f"""You are an air-pollution field analyst in Bengaluru, India.
A citizen uploaded this photo as a pollution report. Identify the most likely visible
pollution source. Respond ONLY with JSON:
{{
 "category": one of {CATEGORIES},
 "confidence": number 0-1,
 "severity": integer 1-5 (5 = thick visible smoke/dust),
 "visible_evidence": short list of what you see (e.g. "grey plume from chimney"),
 "description": one sentence summary,
 "is_pollution_photo": true/false
}}
If the photo does not show pollution, use category "none"."""


def classify_photo(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    from google.genai import types
    data = _generate([types.Part.from_bytes(data=image_bytes, mime_type=mime_type), PHOTO_PROMPT], "photo")
    if data.get("category") not in CATEGORIES:
        data["category"] = "none"
    data["confidence"] = float(data.get("confidence", 0.5))
    return data


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

ALERT_PROMPT = """You are PranaWatch, an AI air-quality early-warning system for Bengaluru.
Write an alert from the evidence below. Be specific, factual, and do not invent data.
Respond ONLY with JSON:
{
 "authority": name of the agency to notify,
 "subject": short email subject line,
 "authority_message": 90-140 word formal message to that agency with location, readings,
     likely source, the evidence, forecast, and 2-3 concrete requested actions,
 "public_advisory": {"en": "...", "kn": "... (Kannada)", "hi": "... (Hindi)"}
     each 1-2 short sentences a resident can act on,
 "priority": "high" | "medium" | "low"
}
EVIDENCE:
"""


_CAT_KN = {"Good": "ಉತ್ತಮ", "Satisfactory": "ತೃಪ್ತಿಕರ", "Moderate": "ಮಧ್ಯಮ", "Poor": "ಕಳಪೆ",
           "Very Poor": "ತುಂಬಾ ಕಳಪೆ", "Severe": "ಗಂಭೀರ"}
_CAT_HI = {"Good": "अच्छी", "Satisfactory": "संतोषजनक", "Moderate": "मध्यम", "Poor": "खराब",
           "Very Poor": "बहुत खराब", "Severe": "गंभीर"}


def template_alert(ctx: dict) -> dict:
    """Deterministic fallback used when no Gemini key is configured."""
    src = ctx["top_source"]
    pm = ctx["pm25"]
    aqi, cat = config.pm25_to_aqi(pm)
    peak = ctx.get("forecast_peak")
    ev_list = src["evidence"][:4]
    fc_line = (f" Our model forecasts PM2.5 to peak near {peak['value']:.0f} ug/m3 at {peak['time']}."
               if peak and not any("forecast" in e for e in ev_list) else "")
    ev = "; ".join(ev_list) or "network anomaly"
    actions = {
        "crop_burning": "(1) deploy field teams and enforce the stubble-burning ban in the upwind districts, "
                        "(2) pre-emptively activate the GRAP stage (NCR) or the city clean-air action plan, "
                        "(3) issue a health advisory and review school outdoor activities",
        "unknown": "(1) intensify road-dust, construction and traffic-emission enforcement in hotspot wards, "
                   "(2) prepare the city's air-pollution action plan measures for the forecast peak, "
                   "(3) issue a public health advisory",
    }.get(src["kind"], "(1) field inspection of the source within 24 hours, "
                       "(2) verify emission-control compliance, (3) issue a local health advisory for nearby schools")
    msg = (f"PranaWatch has detected a pollution hotspot at {ctx['node_name']} "
           f"(PM2.5 {pm:.0f} ug/m3, AQI {aqi} '{cat}', {ctx['excess']:+.0f} ug/m3 vs the "
           f"{ctx.get('baseline_label', 'corridor median')}). "
           f"The most likely source is {src['name']} ({src['confidence']*100:.0f}% attribution confidence). "
           f"Evidence: {ev}.{fc_line} Requested actions: {actions}.")
    node = ctx["node_name"]
    return {
        "authority": src["authority"],
        "subject": f"[PranaWatch] PM2.5 hotspot at {node} - likely {src['name'] if src['kind'] == 'unknown' else src['kind'].replace('_', ' ')} source",
        "authority_message": msg,
        "public_advisory": {
            "en": f"Air quality near {node} is '{cat}' (PM2.5 {pm:.0f}). Limit outdoor activity; children, "
                  f"elderly and asthma patients should stay indoors and wear an N95 mask outside.",
            "kn": f"{node} ಬಳಿ ಗಾಳಿಯ ಗುಣಮಟ್ಟ '{_CAT_KN.get(cat, cat)}' ಇದೆ (PM2.5 {pm:.0f}). ಹೊರಾಂಗಣ ಚಟುವಟಿಕೆ ಕಡಿಮೆ ಮಾಡಿ; "
                  f"ಮಕ್ಕಳು, ಹಿರಿಯರು ಮತ್ತು ಅಸ್ತಮಾ ರೋಗಿಗಳು ಮನೆಯೊಳಗೆ ಇರಿ, ಹೊರಗೆ N95 ಮಾಸ್ಕ್ ಧರಿಸಿ.",
            "hi": f"{node} के पास वायु गुणवत्ता '{_CAT_HI.get(cat, cat)}' है (PM2.5 {pm:.0f})। बाहर की गतिविधियाँ कम करें; "
                  f"बच्चे, बुज़ुर्ग और अस्थमा के मरीज़ घर के अंदर रहें और बाहर N95 मास्क पहनें।",
        },
        "priority": "high" if pm >= 120 else "medium",
        "_model": "template (no GEMINI_API_KEY)",
    }


def write_alert(ctx: dict, use_gemini: bool = True) -> dict:
    if use_gemini and available():
        payload = {
            "location": ctx["node_name"], "lat": ctx["lat"], "lon": ctx["lon"],
            "pm25_ugm3": round(ctx["pm25"], 1), "aqi": config.pm25_to_aqi(ctx["pm25"]),
            "excess_over_corridor_median": round(ctx["excess"], 1),
            "pm25_pm10_ratio": round(ctx["ratio"], 2),
            "wind": ctx.get("wind"),
            "likely_source": {k: ctx["top_source"][k] for k in ("name", "kind", "authority", "evidence")},
            "attribution_confidence": round(ctx["top_source"]["confidence"], 2),
            "forecast_peak": ctx.get("forecast_peak"),
            "citizen_reports": ctx.get("reports", [])[:3],
        }
        try:
            out = _generate([ALERT_PROMPT + json.dumps(payload, default=str, ensure_ascii=False)], "alert")
            if "authority_message" in out and "public_advisory" in out:
                return out
        except Exception as e:
            fb = template_alert(ctx)
            fb["_model"] = f"template (Gemini error: {str(e)[:120]})"
            return fb
    return template_alert(ctx)
