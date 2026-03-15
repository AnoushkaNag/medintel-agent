import pandas as pd
import random

df = pd.read_csv("data/structured_capabilities.csv")

region_coords = {
    "greater accra": (5.6037, -0.1870),
    "ashanti": (6.6885, -1.6244),
    "western": (4.8960, -1.7831),
    "central": (5.1053, -1.2466),
    "eastern": (6.0918, -0.2606),
    "northern": (9.4034, -0.8424),
    "upper east": (10.7082, -0.9821),
    "upper west": (10.0601, -2.5099),
    "volta": (6.5781, 0.4502),
    "bono": (7.6500, -2.5000),
    "bono east": (7.8000, -1.9000),
    "ahafo": (7.0000, -2.5000),
    "oti": (7.9000, 0.4000),
    "savannah": (9.0000, -1.5000),
    "western north": (6.2000, -2.6000)
}

latitudes = []
longitudes = []

for _, row in df.iterrows():

    region = str(row["region"]).lower()

    region = region.replace("region", "").strip()

    found = None

    for key in region_coords:
        if key in region:
            found = region_coords[key]
            break

    if found:
        base_lat, base_lon = found
        lat = base_lat + random.uniform(-0.07, 0.07)
        lon = base_lon + random.uniform(-0.07, 0.07)
    else:
        lat = None
        lon = None

    latitudes.append(lat)
    longitudes.append(lon)

df["lat"] = latitudes
df["lon"] = longitudes

df.to_csv("data/structured_capabilities_geo.csv", index=False)

print("Coordinates regenerated successfully")
