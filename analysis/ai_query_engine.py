import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.healthcare_agent import run_healthcare_agent
from analysis.query_planner import find_regions_missing_specialty, suggest_deployment
from rag.retrieval import search

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
    "maternal": "gynecologyandobstetrics",
}


def detect_specialty(query):

    for keyword, specialty in SPECIALTY_MAP.items():
        if keyword in query:
            return specialty

    return None


def heuristic_answer_query(query):

    query = query.lower()

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception:
        return {"type": "error", "message": "Dataset not found"}

    reasoning = []
    citations = []

    specialty = detect_specialty(query)

    if "lack" in query or "missing" in query:

        if specialty:

            reasoning.append("Used heuristic specialty detection")
            regions = find_regions_missing_specialty(specialty)
            reasoning.append("Checked dataset coverage by region")

            return {
                "type": "gap",
                "specialty": specialty,
                "regions": regions,
                "reasoning": reasoning,
            }

    if (
        "deploy" in query
        or "deployed" in query
        or "deployment" in query
        or "where should" in query
    ):

        if specialty:

            reasoning.append("Used heuristic specialty detection")
            regions = find_regions_missing_specialty(specialty)
            suggestions = suggest_deployment(specialty)
            reasoning.append("Generated rule-based deployment suggestions")

            return {
                "type": "deployment",
                "specialty": specialty,
                "regions": regions,
                "suggestions": suggestions,
                "reasoning": reasoning,
            }

    reasoning.append("Ran facility search over the capability index")

    results = search(query)
    facilities = []

    for result in results:

        text = result["document"]
        facility_name = "Unknown"

        for line in text.split("\n"):
            if "Facility:" in line:
                facility_name = line.replace("Facility:", "").strip()
                break

        facilities.append(facility_name)

    matched = df[df["facility"].isin(facilities)]
    reasoning.append("Matched retrieved facilities back to dataset rows")

    for _, row in matched.head(5).iterrows():
        citations.append({
            "facility": row["facility"],
            "region": row["region"],
            "specialties": row["specialties"],
        })

    return {
        "type": "facility_search",
        "facilities": facilities[:5],
        "reasoning": reasoning,
        "citations": citations,
    }


def answer_query(query):

    agent_result = run_healthcare_agent(query)

    if agent_result.get("ok"):
        return agent_result["response"]

    fallback = heuristic_answer_query(query)

    if agent_result.get("error"):
        fallback.setdefault("reasoning", []).insert(
            0,
            f"ReAct agent unavailable, fell back to heuristic planner: {agent_result['error']}",
        )

    return fallback
