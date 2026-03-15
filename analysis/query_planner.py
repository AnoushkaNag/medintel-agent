import sys
import os
import pandas as pd

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.retrieval import search


# ------------------------------------------------
# VALID GHANA REGIONS
# ------------------------------------------------

VALID_REGIONS = [
    "Ashanti",
    "Greater Accra",
    "Central",
    "Western",
    "Eastern",
    "Northern",
    "Savannah",
    "Volta",
    "Upper East",
    "Upper West",
    "Bono",
    "Bono East",
    "Ahafo",
    "Oti",
    "Western North",
    "North East",
]


# ------------------------------------------------
# REGION CLEANING
# ------------------------------------------------

def normalize_region(region):

    if pd.isna(region):
        return None

    r = str(region).lower().strip()

    r = r.replace(" region", "")
    r = r.title()

    for valid in VALID_REGIONS:
        if valid.lower() in r.lower():
            return valid

    return None


def clean_regions(df):

    df["region"] = df["region"].apply(normalize_region)

    df = df.dropna(subset=["region"])

    return df


# ------------------------------------------------
# RAG SEARCH
# ------------------------------------------------

def find_facilities_by_capability(query):

    results = search(query)

    facilities = []

    for r in results:

        text = r["document"]

        lines = text.split("\n")

        for line in lines:
            if "Facility:" in line:
                facility = line.replace("Facility:", "").strip()
                facilities.append(facility)

    return facilities


# ------------------------------------------------
# GAP DETECTION
# ------------------------------------------------

def find_regions_missing_specialty(specialty):

    df = pd.read_csv("data/structured_capabilities_geo.csv")

    df = clean_regions(df)

    regions = df["region"].unique()

    missing = []

    for r in regions:

        region_df = df[df["region"] == r]

        specialties = region_df["specialties"].astype(str).str.lower()

        if not specialties.str.contains(specialty.lower()).any():
            missing.append(r)

    return sorted(missing)


# ------------------------------------------------
# DEPLOYMENT SUGGESTIONS
# ------------------------------------------------

def suggest_deployment(specialty):

    regions = find_regions_missing_specialty(specialty)

    suggestions = []

    for r in regions:
        suggestions.append(f"Deploy {specialty} specialist to {r}")

    return suggestions


# ------------------------------------------------
# TEST RUN
# ------------------------------------------------

if __name__ == "__main__":

    print("\nFacilities with cardiology:\n")

    facilities = find_facilities_by_capability("cardiology")

    for f in facilities:
        print("-", f)

    print("\nRegions lacking cardiology:\n")

    regions = find_regions_missing_specialty("cardiology")

    for r in regions:
        print("-", r)

    print("\nDeployment suggestions:\n")

    suggestions = suggest_deployment("cardiology")

    for s in suggestions:
        print("-", s)