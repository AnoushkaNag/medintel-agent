import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.retrieval import search
from analysis.query_planner import find_regions_missing_specialty, suggest_deployment

DATA_PATH = "data/structured_capabilities_geo.csv"


SPECIALTY_MAP = {
    "cardiology": "cardiology",
    "cardiologist": "cardiology",
    "heart": "cardiology",
    "dentist": "dentistry",
    "dental": "dentistry",
    "pediatric": "pediatrics",
    "child": "pediatrics",
    "radiology": "radiology",
    "imaging": "radiology",
    "surgery": "generalsurgery",
    "obstetric": "gynecologyandobstetrics",
    "maternal": "gynecologyandobstetrics"
}


def detect_specialty(query):

    for k in SPECIALTY_MAP:
        if k in query:
            return SPECIALTY_MAP[k]

    return None


def answer_query(query):

    query = query.lower()

    try:
        df = pd.read_csv(DATA_PATH)
    except:
        return {"type": "error", "message": "Dataset not found"}

    reasoning = []
    citations = []

    specialty = detect_specialty(query)

    # ---------------------------------------------------
    # GAP QUERIES
    # ---------------------------------------------------

    if "lack" in query or "missing" in query:

        if specialty:

            reasoning.append("Step 1 — Detect specialty requested")

            regions = find_regions_missing_specialty(specialty)

            reasoning.append("Step 2 — Identify regions lacking this specialty")

            return {
                "type": "gap",
                "specialty": specialty,
                "regions": regions,
                "reasoning": reasoning
            }

    # ---------------------------------------------------
    # DEPLOYMENT QUERIES
    # ---------------------------------------------------

    if (
        "deploy" in query
        or "deployed" in query
        or "deployment" in query
        or "where should" in query
    ):

        if specialty:

            reasoning.append("Step 1 — Detect specialty for deployment")

            regions = find_regions_missing_specialty(specialty)

            reasoning.append("Step 2 — Identify regions lacking the specialty")

            suggestions = suggest_deployment(specialty)

            reasoning.append("Step 3 — Generate deployment recommendations")

            return {
                "type": "deployment",
                "specialty": specialty,
                "regions": regions,
                "suggestions": suggestions,
                "reasoning": reasoning
            }

    # ---------------------------------------------------
    # FACILITY SEARCH
    # ---------------------------------------------------

    reasoning.append("Step 1 — Perform semantic facility search")

    results = search(query)

    facilities = []

    for r in results:

        text = r["document"]

        lines = text.split("\n")

        facility_name = "Unknown"

        for line in lines:

            if "Facility:" in line:
                facility_name = line.replace("Facility:", "").strip()

        facilities.append(facility_name)

    reasoning.append("Step 2 — Match facilities with dataset records")

    matched = df[df["facility"].isin(facilities)]

    for _, row in matched.head(5).iterrows():

        citations.append({
            "facility": row["facility"],
            "region": row["region"],
            "specialties": row["specialties"]
        })

    reasoning.append("Step 3 — Return most relevant facilities")

    return {
        "type": "facility_search",
        "facilities": facilities[:5],
        "reasoning": reasoning,
        "citations": citations
    }