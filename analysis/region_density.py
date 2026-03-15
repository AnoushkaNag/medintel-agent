import pandas as pd

df = pd.read_csv("data/structured_capabilities.csv")

density = df.groupby("region").size().reset_index(name="facility_count")

density.to_csv("data/region_density.csv", index=False)

print("Region density computed")
