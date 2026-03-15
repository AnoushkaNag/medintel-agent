import pandas as pd
import numpy as np


def enrich_specialties(df):

    def infer_specialty(row):

        name = str(row["facility"]).lower()

        specialties = []

        if "maternity" in name:
            specialties.append("gynecologyAndObstetrics")

        if "children" in name or "pediatric" in name:
            specialties.append("pediatrics")

        if "eye" in name:
            specialties.append("ophthalmology")

        if "dental" in name:
            specialties.append("dentistry")

        if "surgical" in name or "surgery" in name:
            specialties.append("generalSurgery")

        if "hospital" in name:
            specialties.extend([
                "internalMedicine",
                "emergencyMedicine",
                "generalSurgery"
            ])

        if "clinic" in name:
            specialties.append("internalMedicine")

        if len(specialties) == 0:
            specialties.append("internalMedicine")

        return list(set(specialties))

    df["specialties"] = df.apply(infer_specialty, axis=1)

    return df


# Detect medical deserts using grid density
def detect_medical_deserts(df):

    grid_size = 1.0

    lat_bins = np.arange(df["lat"].min(), df["lat"].max(), grid_size)
    lon_bins = np.arange(df["lon"].min(), df["lon"].max(), grid_size)

    deserts = []

    for lat in lat_bins:
        for lon in lon_bins:

            subset = df[
                (df["lat"] > lat) &
                (df["lat"] <= lat + grid_size) &
                (df["lon"] > lon) &
                (df["lon"] <= lon + grid_size)
            ]

            if len(subset) < 3:

                deserts.append({
                    "lat": lat + grid_size / 2,
                    "lon": lon + grid_size / 2,
                    "facility_count": len(subset)
                })

    return pd.DataFrame(deserts)
