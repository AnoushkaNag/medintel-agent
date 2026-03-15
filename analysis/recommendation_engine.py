import pandas as pd
import ast

df = pd.read_csv("data/structured_capabilities.csv")


def parse_list(x):
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except:
        return []


df["specialties"] = df["specialties"].apply(parse_list)


def clean_region(r):
    if pd.isna(r):
        return None
    r = str(r).lower()
    r = r.replace(" region", "")
    return r.title()


df["region_clean"] = df["region"].apply(clean_region)


def safe_capacity(x):

    try:
        return int(x)
    except:
        return 0


df["capacity_num"] = df["capacity"].apply(safe_capacity)


CORE_SPECIALTIES = [
    "internalMedicine",
    "emergencyMedicine",
    "pediatrics",
    "gynecologyAndObstetrics",
    "generalSurgery",
    "cardiology"
]


print("\n=== Healthcare Intervention Recommendations ===\n")

regions = df["region_clean"].dropna().unique()

for region in regions:

    region_df = df[df["region_clean"] == region]

    available_specialties = set()

    for s in region_df["specialties"]:
        available_specialties.update(s)

    missing = [sp for sp in CORE_SPECIALTIES if sp not in available_specialties]

    if not missing:
        continue

    print("Region:", region)
    print("Missing:", ", ".join(missing))

    best_facility = None
    best_capacity = -1

    for _, row in region_df.iterrows():

        capacity = row["capacity_num"]

        if capacity > best_capacity:

            best_capacity = capacity
            best_facility = row["facility"]

    if best_facility:

        print("Recommended facility:", best_facility)

    print()
