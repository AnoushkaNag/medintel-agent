import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import json
import time
from openai import OpenAI

from prompts.medical_specialties import MEDICAL_SPECIALTIES_SYSTEM_PROMPT
from prompts.free_form import FREE_FORM_SYSTEM_PROMPT
from prompts.facility_and_ngo_fields import ORGANIZATION_INFORMATION_SYSTEM_PROMPT

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FILE = "data/facilities_with_text.csv"
OUTPUT_FILE = "data/structured_capabilities.csv"


def call_llm(system_prompt, user_text):

    # Force JSON instruction so API accepts json_object mode
    system_prompt = system_prompt + "\n\nReturn your response strictly as JSON."

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:

        print("LLM parsing error:", e)
        return {}


def extract_specialties(name, text):

    prompt = MEDICAL_SPECIALTIES_SYSTEM_PROMPT.format(
        organization=name
    )

    data = call_llm(prompt, text)

    return data.get("specialties", [])


def extract_capabilities(name, text):

    prompt = FREE_FORM_SYSTEM_PROMPT.format(
        organization=name
    )

    data = call_llm(prompt, text)

    return {
        "procedure": data.get("procedure", []),
        "equipment": data.get("equipment", []),
        "capability": data.get("capability", [])
    }


def extract_facility_fields(name, text):

    prompt = ORGANIZATION_INFORMATION_SYSTEM_PROMPT.format(
        organization=name
    )

    data = call_llm(prompt, text)

    return data


def run_extraction():

    df = pd.read_csv(INPUT_FILE)

    results = []

    print("Starting multi-agent IDP pipeline\n")

    for i, row in df.iterrows():

        name = row["name"]
        text = row["medical_text"]
        region = row.get("address_stateOrRegion", "Unknown")

        print("Processing:", name)

        specialties = extract_specialties(name, text)

        capabilities = extract_capabilities(name, text)

        facility_data = extract_facility_fields(name, text)

        results.append({

            "facility": name,
            "region": region,

            "specialties": specialties,

            "procedures": capabilities["procedure"],
            "equipment": capabilities["equipment"],
            "capabilities": capabilities["capability"],

            "numberDoctors": facility_data.get("numberDoctors"),
            "capacity": facility_data.get("capacity")

        })

        time.sleep(0.1)

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    print("\nExtraction finished")
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    run_extraction()
