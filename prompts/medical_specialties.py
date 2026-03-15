from typing import List, Optional
from pydantic import BaseModel, Field


MEDICAL_SPECIALTIES = [
    "internalMedicine",
    "familyMedicine",
    "emergencyMedicine",
    "generalSurgery",
    "cardiology",
    "pediatrics",
    "gynecologyAndObstetrics",
    "radiology",
    "pathology",
    "nephrology",
    "orthopedicSurgery",
    "ophthalmology",
    "dentistry",
    "medicalOncology",
    "infectiousDiseases",
    "anesthesia",
    "criticalCareMedicine",
    "plasticSurgery"
]


MEDICAL_SPECIALTIES_SYSTEM_PROMPT = """
You are a medical specialty classifier.

Your task is to extract which specialties a healthcare facility provides.

Facility name: {organization}

You will be given facility description text.

From the following allowed specialties list, identify which specialties the facility offers.

Allowed specialties:

""" + "\n".join(MEDICAL_SPECIALTIES) + """

Rules:

- Only return specialties from the allowed list.
- If maternity, obstetric, or women's health is mentioned → gynecologyAndObstetrics
- If pediatric or children → pediatrics
- If emergency / ER → emergencyMedicine
- If surgery → generalSurgery
- If heart or cardiac → cardiology
- If kidney or dialysis → nephrology
- If imaging / x-ray / CT → radiology
- If lab or laboratory → pathology

Return JSON format:

{{
 "specialties": ["specialty1","specialty2"]
}}
"""


class MedicalSpecialties(BaseModel):

    specialties: Optional[List[str]] = Field(
        default_factory=list,
        description="List of specialties offered by the facility"
    )
