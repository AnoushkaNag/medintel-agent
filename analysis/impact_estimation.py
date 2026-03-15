import pandas as pd

DATA_PATH = "data/structured_capabilities_geo.csv"

def estimate_impact(region):

    df = pd.read_csv(DATA_PATH)

    region_df = df[df["region"] == region]

    facility_count = len(region_df)

    # rough healthcare catchment estimate
    population_estimate = facility_count * 25000

    return population_estimate