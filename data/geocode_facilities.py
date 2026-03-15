import pandas as pd
import time
from geopy.geocoders import Nominatim

print("Starting geocoding...")

df = pd.read_csv("data/structured_capabilities.csv")

geolocator = Nominatim(user_agent="medintel_mapper", timeout=10)

latitudes = []
longitudes = []

total = len(df)

for i, row in df.iterrows():

    facility = str(row["facility"])
    region = str(row["region"])

    query = f"{facility}, {region}, Ghana"

    location = None

    # retry mechanism
    for attempt in range(3):

        try:
            location = geolocator.geocode(query)

            if location:
                break

        except Exception:
            time.sleep(2)

    if location:

        latitudes.append(location.latitude)
        longitudes.append(location.longitude)

        print(f"[{i+1}/{total}] Found:", facility)

    else:

        latitudes.append(None)
        longitudes.append(None)

        print(f"[{i+1}/{total}] Not found:", facility)

    # Respect API limits
    time.sleep(1.2)

df["lat"] = latitudes
df["lon"] = longitudes

df.to_csv("data/structured_capabilities_geo.csv", index=False)

print("\nGeocoding finished.")
print("Saved: data/structured_capabilities_geo.csv")
