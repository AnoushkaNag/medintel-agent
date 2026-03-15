import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/clean_facilities.csv")

print("Facilities loaded:", len(df))


# Combine medical text fields
def combine_medical_text(row):

    text_parts = []

    fields = [
        "specialties",
        "procedure",
        "equipment",
        "capability",
        "description"
    ]

    for field in fields:
        value = row[field]

        if pd.notna(value):
            text_parts.append(str(value))

    return " ".join(text_parts)


df["medical_text"] = df.apply(combine_medical_text, axis=1)


print("\nExample medical text:\n")
print(df["medical_text"].iloc[0])


# Save processed dataset
df.to_csv("data/facilities_with_text.csv", index=False)

print("\nMedical text column created.")