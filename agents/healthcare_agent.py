import ast
import json
import os
import re
from functools import lru_cache

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.anomaly_detection import detect_anomalies
from analysis.query_planner import find_regions_missing_specialty
from rag.retrieval import search

DATA_PATH = "data/structured_capabilities_geo.csv"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

LOCAL_MODEL_CANDIDATES = [
    (
        "Qwen2.5-0.5B-Instruct",
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"),
    ),
    (
        "TinyLlama-1.1B-Chat-v1.0",
        os.path.expanduser("~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"),
    ),
]
DEFAULT_MODEL = os.getenv("MEDINTEL_LOCAL_MODEL", "Qwen2.5-0.5B-Instruct")
MAX_AGENT_STEPS = 5


@lru_cache(maxsize=1)
def load_dataset():

    df = pd.read_csv(DATA_PATH)

    df["facility"] = df["facility"].fillna("").astype(str)
    df["region"] = df["region"].fillna("").astype(str)

    return df


def normalize_region(region):

    if not isinstance(region, str):
        return "Unknown"

    region = region.lower().strip()
    region = region.replace(" region", "")
    region = region.replace(" municipality", "")
    region = region.replace(" district", "")

    return region.title() or "Unknown"


def parse_list(value):

    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass

    return [text]


def row_citation(row_index, row):

    return {
        "facility": row["facility"],
        "region": row["region"] or "Unknown",
        "specialties": parse_list(row.get("specialties")),
        "source_row": int(row_index) + 2,
    }


def extract_document_field(document, field_name):

    match = re.search(rf"{re.escape(field_name)}:\s*(.+)", document)

    if not match:
        return ""

    return match.group(1).strip()


def build_toolset():

    def search_facilities(arguments):

        query = str(arguments.get("query", "")).strip()
        top_k = int(arguments.get("top_k", 5))

        results = search(query, top_k=max(1, min(top_k, 8)))
        df = load_dataset()

        facilities = []
        citations = []

        for result in results:

            facility_name = extract_document_field(result["document"], "Facility")
            region = extract_document_field(result["document"], "Region")

            matched = df[df["facility"].str.lower() == facility_name.lower()]

            if matched.empty:
                facilities.append({
                    "facility": facility_name or "Unknown",
                    "region": region or "Unknown",
                    "score": round(result["score"], 4),
                })
                continue

            row_index = matched.index[0]
            row = matched.iloc[0]

            facility_record = {
                "facility": row["facility"],
                "region": row["region"] or region or "Unknown",
                "specialties": parse_list(row.get("specialties")),
                "procedures": parse_list(row.get("procedures"))[:3],
                "equipment": parse_list(row.get("equipment"))[:3],
                "score": round(result["score"], 4),
                "source_row": int(row_index) + 2,
            }

            facilities.append(facility_record)
            citations.append(row_citation(row_index, row))

        return {
            "query": query,
            "results": facilities,
            "citations": citations,
        }

    def find_missing_regions(arguments):

        specialty = str(arguments.get("specialty", "")).strip().lower()

        if not specialty:
            return {"specialty": "", "regions": [], "citations": []}

        regions = find_regions_missing_specialty(specialty)
        df = load_dataset().copy()
        df["region_clean"] = df["region"].apply(normalize_region)

        summaries = []
        citations = []

        for region in regions:

            region_df = df[df["region_clean"] == region]
            examples = region_df["facility"].head(3).tolist()

            summaries.append({
                "region": region,
                "facility_count": int(len(region_df)),
                "example_facilities": examples,
            })

            if not region_df.empty:
                row_index = region_df.index[0]
                citations.append(row_citation(row_index, region_df.iloc[0]))

        return {
            "specialty": specialty,
            "regions": summaries,
            "citations": citations,
        }

    def recommend_deployment(arguments):

        specialty = str(arguments.get("specialty", "")).strip().lower()
        limit = int(arguments.get("limit", 5))

        missing = find_missing_regions({"specialty": specialty})
        df = load_dataset().copy()
        df["region_clean"] = df["region"].apply(normalize_region)

        suggestions = []
        citations = []

        for region_info in missing["regions"][:max(1, min(limit, 8))]:

            region = region_info["region"]
            region_df = df[df["region_clean"] == region].copy()

            if region_df.empty:
                continue

            capacity_series = pd.to_numeric(region_df["capacity"], errors="coerce")
            region_df["capacity_num"] = capacity_series.fillna(-1)
            region_df = region_df.sort_values(
                by=["capacity_num", "facility"],
                ascending=[False, True],
            )

            facility_row = region_df.iloc[0]
            facility_name = facility_row["facility"]

            suggestion = {
                "region": region,
                "specialty": specialty,
                "facility": facility_name,
                "recommendation": f"Deploy {specialty} specialist to {region}",
                "expansion_site": facility_name,
            }

            suggestions.append(suggestion)
            citations.append(row_citation(region_df.index[0], facility_row))

        return {
            "specialty": specialty,
            "suggestions": suggestions,
            "citations": citations,
        }

    def audit_anomalies(arguments):

        limit = int(arguments.get("limit", 10))
        anomalies = detect_anomalies()[:max(1, min(limit, 25))]
        df = load_dataset()
        citations = []

        for anomaly in anomalies:

            matched = df[df["facility"].str.lower() == anomaly["facility"].lower()]

            if matched.empty:
                continue

            row_index = matched.index[0]
            citations.append(row_citation(row_index, matched.iloc[0]))

        return {
            "anomalies": anomalies,
            "citations": citations,
        }

    def inspect_facility(arguments):

        facility = str(arguments.get("facility", "")).strip().lower()
        df = load_dataset()

        if not facility:
            return {"matches": [], "citations": []}

        matched = df[df["facility"].str.lower().str.contains(facility, na=False, regex=False)]

        matches = []
        citations = []

        for row_index, row in matched.head(5).iterrows():

            matches.append({
                "facility": row["facility"],
                "region": row["region"] or "Unknown",
                "specialties": parse_list(row.get("specialties")),
                "procedures": parse_list(row.get("procedures"))[:3],
                "equipment": parse_list(row.get("equipment"))[:3],
                "capabilities": parse_list(row.get("capabilities"))[:3],
                "source_row": int(row_index) + 2,
            })
            citations.append(row_citation(row_index, row))

        return {
            "matches": matches,
            "citations": citations,
        }

    return {
        "search_facilities": {
            "description": "Semantic search over healthcare facilities and capabilities.",
            "arguments": {"query": "string", "top_k": "integer"},
            "handler": search_facilities,
        },
        "find_missing_regions": {
            "description": "Find regions missing a specialty and summarize facility coverage there.",
            "arguments": {"specialty": "string"},
            "handler": find_missing_regions,
        },
        "recommend_deployment": {
            "description": "Recommend regions and facilities for specialist deployment.",
            "arguments": {"specialty": "string", "limit": "integer"},
            "handler": recommend_deployment,
        },
        "audit_anomalies": {
            "description": "Return suspicious facility records from the anomaly detector.",
            "arguments": {"limit": "integer"},
            "handler": audit_anomalies,
        },
        "inspect_facility": {
            "description": "Look up a facility directly by name fragment.",
            "arguments": {"facility": "string"},
            "handler": inspect_facility,
        },
    }


def tool_catalog_text(toolset):

    lines = []

    for name, tool in toolset.items():
        lines.append(
            f"- {name}: {tool['description']} | args={json.dumps(tool['arguments'])}"
        )

    return "\n".join(lines)


def extract_json(text):

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start == -1:
            raise

        depth = 0
        in_string = False
        escape = False

        for index in range(start, len(cleaned)):
            char = cleaned[index]

            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == "\"":
                    in_string = False
                continue

            if char == "\"":
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(cleaned[start:index + 1])

        raise


def preview_observation(result):

    preview = json.dumps(result, ensure_ascii=True)

    if len(preview) > 1200:
        preview = preview[:1200] + "..."

    return preview


def resolve_snapshot_path(cache_root):

    refs_path = os.path.join(cache_root, "refs", "main")
    snapshots_root = os.path.join(cache_root, "snapshots")

    if os.path.exists(refs_path):
        snapshot = open(refs_path, "r", encoding="utf-8").read().strip()
        candidate = os.path.join(snapshots_root, snapshot)
        if os.path.isdir(candidate):
            return candidate

    if os.path.isdir(snapshots_root):
        snapshots = sorted(os.listdir(snapshots_root))
        if snapshots:
            return os.path.join(snapshots_root, snapshots[0])

    return None


def resolve_local_model(model_name):

    requested_path = os.getenv("MEDINTEL_LOCAL_MODEL_PATH")
    if requested_path and os.path.isdir(requested_path):
        return ("custom-local-model", requested_path)

    for label, cache_root in LOCAL_MODEL_CANDIDATES:
        if model_name and model_name not in {label, cache_root}:
            continue

        snapshot_path = resolve_snapshot_path(cache_root)
        if snapshot_path:
            return (label, snapshot_path)

    if model_name:
        for label, cache_root in LOCAL_MODEL_CANDIDATES:
            snapshot_path = resolve_snapshot_path(cache_root)
            if snapshot_path:
                return (label, snapshot_path)

    return None


@lru_cache(maxsize=2)
def load_local_llm(model_name):

    resolved = resolve_local_model(model_name)

    if not resolved:
        raise RuntimeError(
            "No cached local instruct model found. Expected one of: "
            + ", ".join(label for label, _ in LOCAL_MODEL_CANDIDATES)
        )

    label, model_path = resolved
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    model.eval()

    return label, tokenizer, model


def render_chat_prompt(tokenizer, messages):

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    lines = []

    for message in messages:
        role = message["role"].upper()
        lines.append(f"{role}:\n{message['content']}")

    lines.append("ASSISTANT:\n")

    return "\n\n".join(lines)


def call_local_text(messages, model_name, max_new_tokens=220):

    resolved_label, tokenizer, model = load_local_llm(model_name)
    prompt = render_chat_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    content = tokenizer.decode(generated, skip_special_tokens=True)

    return resolved_label, content


def call_local_json(messages, model_name):

    resolved_label, content = call_local_text(messages, model_name)

    return resolved_label, extract_json(content)


def result_type_from_tools(final_type, tool_outputs):

    if "recommend_deployment" in tool_outputs:
        return "deployment"

    if "find_missing_regions" in tool_outputs:
        return "gap"

    if "audit_anomalies" in tool_outputs:
        return "anomaly"

    if "search_facilities" in tool_outputs or "inspect_facility" in tool_outputs:
        return "facility_search"

    if final_type in {"gap", "deployment", "facility_search", "anomaly", "answer"}:
        return final_type

    return "answer"


def merge_response(final_payload, reasoning, tool_outputs, tool_trace, model):

    response_type = result_type_from_tools(
        final_payload.get("result_type", "answer"),
        tool_outputs,
    )

    response = {
        "type": response_type,
        "answer": final_payload.get("final_answer", "").strip(),
        "reasoning": reasoning,
        "citations": final_payload.get("citations", []),
        "agent_mode": "react",
        "model": model,
        "tool_trace": tool_trace,
    }

    if response_type == "gap" and "find_missing_regions" in tool_outputs:
        region_data = tool_outputs["find_missing_regions"]["regions"]
        response["specialty"] = tool_outputs["find_missing_regions"]["specialty"]
        response["regions"] = [item["region"] for item in region_data]
        response["citations"] = response["citations"] or tool_outputs["find_missing_regions"]["citations"]
        if response["regions"]:
            response["answer"] = (
                f"Regions lacking {response['specialty']}: "
                + ", ".join(response["regions"])
                + "."
            )
        else:
            response["answer"] = f"No regional gaps found for {response['specialty']}."

    if response_type == "deployment" and "recommend_deployment" in tool_outputs:
        suggestions = tool_outputs["recommend_deployment"]["suggestions"]
        response["specialty"] = tool_outputs["recommend_deployment"]["specialty"]
        response["suggestions"] = [item["recommendation"] for item in suggestions]
        response["recommended_facilities"] = suggestions
        response["citations"] = response["citations"] or tool_outputs["recommend_deployment"]["citations"]
        if response["suggestions"]:
            response["answer"] = "Deployment recommendations: " + "; ".join(response["suggestions"]) + "."
        else:
            response["answer"] = f"No deployment recommendations available for {response['specialty']}."

    if response_type == "facility_search":
        facility_payload = tool_outputs.get("search_facilities") or tool_outputs.get("inspect_facility")
        if facility_payload:
            entries = facility_payload.get("results") or facility_payload.get("matches") or []
            response["facilities"] = [item["facility"] for item in entries]
            response["facility_records"] = entries
            response["citations"] = response["citations"] or facility_payload.get("citations", [])
            if response["facilities"]:
                response["answer"] = "Top matching facilities: " + ", ".join(response["facilities"][:5]) + "."
            else:
                response["answer"] = "No matching facilities found."

    if response_type == "anomaly" and "audit_anomalies" in tool_outputs:
        response["anomalies"] = tool_outputs["audit_anomalies"]["anomalies"]
        response["citations"] = response["citations"] or tool_outputs["audit_anomalies"]["citations"]
        count = len(response["anomalies"])
        response["answer"] = f"Found {count} suspicious facility records."

    return response


def run_healthcare_agent(query, model=DEFAULT_MODEL, max_steps=MAX_AGENT_STEPS):

    toolset = build_toolset()

    planning_prompt = f"""
You are MedIntel's healthcare planning agent.
Use a compact ReAct workflow: think briefly, choose the single best tool, observe its result, then finish.

Available tools:
{tool_catalog_text(toolset)}

Return strict JSON only:
{{
  "step_summary": "short user-safe trace of what you are about to do",
  "action": "one tool name above OR finish",
  "action_input": {{}},
  "result_type": "gap | deployment | facility_search | anomaly | answer",
  "final_answer": ""
}}

Rules:
- Prefer exactly one tool call for this query.
- Use finish only if no tool is needed.
- Keep action_input minimal and valid JSON.
- Never invent facts or citations.
""".strip()

    try:
        resolved_model_label, plan_payload = call_local_json(
            [
                {"role": "system", "content": planning_prompt},
                {"role": "user", "content": f"User question: {query}\nReturn JSON only."},
            ],
            model,
        )

        step_summary = str(plan_payload.get("step_summary", "")).strip() or "Planned the next step"
        action = str(plan_payload.get("action", "")).strip()
        action_input = plan_payload.get("action_input", {}) or {}
        result_type = str(plan_payload.get("result_type", "answer")).strip() or "answer"

        reasoning = [step_summary]

        if action == "finish":
            response = merge_response(
                plan_payload,
                reasoning,
                {},
                [],
                resolved_model_label,
            )
            if not response["answer"]:
                response["answer"] = "The agent finished without a summary."
            return {"ok": True, "response": response}

        if action not in toolset:
            return {
                "ok": False,
                "error": f"Agent selected an unknown action: {action}",
            }

        tool_result = toolset[action]["handler"](action_input)
        tool_outputs = {action: tool_result}
        tool_trace = [{
            "tool": action,
            "input": action_input,
            "output_preview": preview_observation(tool_result),
        }]
        reasoning.append(f"Reviewed the output from {action}")

        synthesis_prompt = """
You are MedIntel's healthcare planning agent finishing a ReAct turn.
Write a concise user-facing answer grounded only in the tool observation.
Do not mention hidden reasoning or JSON.
Do not invent facts beyond the observation.
""".strip()

        resolved_model_label, final_answer = call_local_text(
            [
                {"role": "system", "content": synthesis_prompt},
                {
                    "role": "user",
                    "content": (
                        f"User question: {query}\n"
                        f"Tool used: {action}\n"
                        f"Action input: {json.dumps(action_input, ensure_ascii=True)}\n"
                        f"Observation: {json.dumps(tool_result, ensure_ascii=True)}\n"
                        "Write the final answer only."
                    ),
                },
            ],
            model,
        )

        final_step = "Summarized the tool result with the local model"
        reasoning.append(final_step)
        final_payload = {
            "result_type": result_type,
            "final_answer": final_answer.strip(),
            "citations": [],
        }

        response = merge_response(
            final_payload,
            reasoning,
            tool_outputs,
            tool_trace,
            resolved_model_label,
        )
        if not response["answer"]:
            response["answer"] = "The agent completed the ReAct turn but did not provide a summary."
        return {"ok": True, "response": response}
    except Exception as exc:
        return {
            "ok": False,
            "error": f"ReAct agent failed: {exc}",
        }
