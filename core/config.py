"""Corridor configuration: Peenya industrial belt, Bengaluru.

Node coordinates are approximate points along the corridor. Source locations are
illustrative categories of real emitters in the area (industrial estate, highway
junction, construction, waste burning) used for the demo attribution engine.
"""

CITY = "Bengaluru"
CORRIDOR_NAME = "Peenya Industrial Corridor"
CENTER = (13.030, 77.520)

# Ground sensor nodes (PranaWatch low-cost nodes + one CPCB reference station)
NODES = [
    {"id": "PW-01", "name": "Peenya Phase 2", "lat": 13.0285, "lon": 77.5190, "type": "pranawatch"},
    {"id": "PW-02", "name": "Peenya Phase 4", "lat": 13.0395, "lon": 77.5125, "type": "pranawatch"},
    {"id": "PW-03", "name": "Jalahalli Cross", "lat": 13.0440, "lon": 77.5305, "type": "pranawatch"},
    {"id": "PW-04", "name": "Yeshwanthpur", "lat": 13.0230, "lon": 77.5400, "type": "pranawatch"},
    {"id": "PW-05", "name": "Dasarahalli", "lat": 13.0470, "lon": 77.5140, "type": "pranawatch"},
    {"id": "PW-06", "name": "Goraguntepalya", "lat": 13.0290, "lon": 77.5320, "type": "pranawatch"},
    {"id": "PW-07", "name": "Nagasandra", "lat": 13.0480, "lon": 77.5000, "type": "pranawatch"},
    {"id": "PW-08", "name": "Rajajinagar Ind. Area", "lat": 13.0150, "lon": 77.5470, "type": "pranawatch"},
    {"id": "CPCB-REF", "name": "Peenya CPCB station (reference)", "lat": 13.0270, "lon": 77.4950, "type": "reference"},
]

# Candidate emission sources used for upwind attribution
SOURCES = [
    {"id": "S1", "name": "Peenya Industrial Estate (foundries, forging)", "lat": 13.0330, "lon": 77.5160,
     "kind": "industrial", "authority": "KSPCB (Karnataka State Pollution Control Board)"},
    {"id": "S2", "name": "Tumkur Road / Goraguntepalya junction", "lat": 13.0300, "lon": 77.5300,
     "kind": "traffic", "authority": "Bengaluru Traffic Police & BMTC"},
    {"id": "S3", "name": "Metro / flyover construction zone", "lat": 13.0420, "lon": 77.5260,
     "kind": "construction", "authority": "BBMP (Bruhat Bengaluru Mahanagara Palike)"},
    {"id": "S4", "name": "Open waste burning site near Nagasandra", "lat": 13.0500, "lon": 77.5040,
     "kind": "waste_burning", "authority": "BBMP Solid Waste Management"},
]

AUTHORITY_BY_KIND = {
    "industrial": "KSPCB (Karnataka State Pollution Control Board)",
    "traffic": "Bengaluru Traffic Police & BMTC",
    "construction": "BBMP (Bruhat Bengaluru Mahanagara Palike)",
    "waste_burning": "BBMP Solid Waste Management",
    "crop_burning": "Karnataka Department of Agriculture",
    "vehicle": "Transport Department (RTO)",
    "unknown": "BBMP Ward Office",
}

# Indian National AQI breakpoints for PM2.5 (24h, ug/m3)
AQI_PM25 = [
    (0, 30, 0, 50, "Good"),
    (31, 60, 51, 100, "Satisfactory"),
    (61, 90, 101, 200, "Moderate"),
    (91, 120, 201, 300, "Poor"),
    (121, 250, 301, 400, "Very Poor"),
    (251, 500, 401, 500, "Severe"),
]

ALERT_THRESHOLD_PM25 = 90.0  # "Poor" and above triggers an alert


def pm25_to_aqi(pm25: float):
    """Return (aqi, category) using India's National AQI PM2.5 sub-index."""
    if pm25 is None:
        return None, "Unknown"
    pm = max(0.0, float(pm25))
    for c_lo, c_hi, i_lo, i_hi, cat in AQI_PM25:
        if pm <= c_hi + 0.999:
            aqi = i_lo + (i_hi - i_lo) * (min(pm, c_hi) - c_lo) / max(c_hi - c_lo, 1)
            return int(round(max(aqi, i_lo))), cat
    return 500, "Severe"


AQI_COLORS = {
    "Good": "#2e7d32", "Satisfactory": "#8bc34a", "Moderate": "#fbc02d",
    "Poor": "#f57c00", "Very Poor": "#d32f2f", "Severe": "#7b1fa2", "Unknown": "#9e9e9e",
}
