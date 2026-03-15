import pandas as pd

DATA_PATH = "data/structured_capabilities_geo.csv"


def detect_anomalies():

    df = pd.read_csv(DATA_PATH)

    anomalies = []

    for _, row in df.iterrows():

        facility = row.get("facility", "Unknown")
        doctors = row.get("numberDoctors", 0)
        specialties = str(row.get("specialties", "")).lower()
        equipment = str(row.get("equipment", "")).lower()

        try:
            doctors = int(doctors)
        except:
            doctors = 0

        # -------------------------
        # Rule 1: Advanced surgery
        # -------------------------

        if "neurosurgery" in specialties and doctors < 3:

            anomalies.append({
                "facility": facility,
                "issue": "Claims neurosurgery but has very few doctors"
            })

        # -------------------------
        # Rule 2: Cardiology without equipment
        # -------------------------

        if "cardiology" in specialties and len(equipment.strip()) < 5:

            anomalies.append({
                "facility": facility,
                "issue": "Claims cardiology but no diagnostic equipment listed"
            })

        # -------------------------
        # Rule 3: Surgery with 1 doctor
        # -------------------------

        if "surgery" in specialties and doctors <= 1:

            anomalies.append({
                "facility": facility,
                "issue": "Claims surgical capability with only one doctor"
            })

        # -------------------------
        # Rule 4: Large hospital with no doctors
        # -------------------------

        if "hospital" in facility.lower() and doctors == 0:

            anomalies.append({
                "facility": facility,
                "issue": "Hospital listing but no doctors recorded"
            })
            

    return anomalies
if __name__ == "__main__":

    anomalies = detect_anomalies()

    print("\nHealthcare Capability Anomalies\n")

    for a in anomalies[:10]:
        print(a["facility"], "→", a["issue"])