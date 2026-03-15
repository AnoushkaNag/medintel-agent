import pandas as pd
import json

# Load extracted data
df = pd.read_csv("data/extracted_capabilities.csv")

# Parse JSON safely
def parse_json(cell):
    if pd.isna(cell):
        return {"specialties": [], "procedures": [], "equipment": [], "capabilities": []}
    try:
        data = json.loads(cell)
        return {
            "specialties": data.get("specialties", []),
            "procedures": data.get("procedures", []),
            "equipment": data.get("equipment", []),
            "capabilities": data.get("capabilities", []),
        }
    except:
        return {"specialties": [], "procedures": [], "equipment": [], "capabilities": []}

parsed = df["extraction"].apply(parse_json)

df["specialties_parsed"] = parsed.apply(lambda x: x["specialties"])
df["procedures_parsed"] = parsed.apply(lambda x: x["procedures"])
df["equipment_parsed"] = parsed.apply(lambda x: x["equipment"])
df["capabilities_parsed"] = parsed.apply(lambda x: x["capabilities"])

# Save structured dataset
df.to_csv("data/structured_capabilities.csv", index=False)

print("Structured dataset saved to data/structured_capabilities.csv")
