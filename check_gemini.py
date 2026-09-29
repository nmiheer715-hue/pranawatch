"""Quick check that the Gemini key in .streamlit/secrets.toml works.  Run: python check_gemini.py"""
import os
from pathlib import Path

sec = Path(__file__).parent / ".streamlit" / "secrets.toml"
if sec.exists() and not os.environ.get("GEMINI_API_KEY"):
    for line in sec.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

from core import gemini  # noqa: E402

if not gemini.available():
    print("[X] No GEMINI_API_KEY found in .streamlit/secrets.toml")
else:
    try:
        out = gemini._generate(['Return JSON {"status": "ok"}'], "check")
        print(f"[OK] Gemini works  ->  {out.get('_model')}")
    except Exception as e:
        print(f"[X] Gemini call failed: {str(e)[:400]}")
        print("    Fix: create a key at https://aistudio.google.com/apikey and paste it into .streamlit/secrets.toml")
