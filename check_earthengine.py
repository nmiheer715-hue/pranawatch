"""Check Earth Engine setup.  Run:  python check_earthengine.py"""
import os
from pathlib import Path

sec = Path(__file__).parent / ".streamlit" / "secrets.toml"
if sec.exists():
    for line in sec.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

try:
    import ee
except ImportError:
    raise SystemExit("[X] earthengine-api not installed: run  pip install earthengine-api")

project = os.environ.get("EE_PROJECT")
if not project:
    raise SystemExit('[X] Add  EE_PROJECT = "your-cloud-project-id"  to .streamlit/secrets.toml')
try:
    ee.Initialize(project=project)
    img = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2").filterDate("2026-09-01", "2026-09-28").first()
    print("[OK] Earth Engine works. Sentinel-5P image:", img.get("system:index").getInfo())
except Exception as e:
    print(f"[X] {type(e).__name__}: {str(e)[:400]}")
    print("    Fix: run  earthengine authenticate  and make sure the project is registered for Earth Engine.")
