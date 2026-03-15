MedIntel — Bridging Medical Deserts

MedIntel is an AI-powered healthcare intelligence platform that analyzes healthcare facility data to detect medical deserts, identify capability gaps, and support healthcare planning decisions.

The system transforms messy healthcare facility reports into structured intelligence and enables NGOs, policymakers, and planners to answer questions like:

Why MedIntel Matters

Healthcare planning often relies on manual analysis of fragmented reports.

MedIntel automates this process by transforming facility descriptions into structured intelligence and identifying healthcare gaps across regions.

This enables faster decisions on:
• resource allocation
• specialist deployment
• infrastructure investment

Which regions lack cardiology services?

Where should doctors be deployed?

Which facilities provide dialysis?

Which healthcare claims look suspicious?

The project was built for the Accenture Databricks Hackathon — Bridging Medical Deserts Challenge.

Live Demo

Streamlit App:

https://medintel-agent.streamlit.app/
Problem

Many regions worldwide suffer from medical deserts — areas where people have inadequate access to healthcare services, facilities, or medical professionals.

These areas often experience:

Shortage of doctors and specialists

Long travel distances to hospitals

Limited diagnostic equipment

Delayed treatment and poorer health outcomes

Healthcare organizations often struggle to coordinate resources because data about facilities is unstructured, incomplete, and scattered across reports.

MedIntel solves this problem by converting raw facility descriptions into actionable healthcare intelligence.

System Architecture
Healthcare Dataset
        ↓
LLM-based Information Extraction (IDP)
        ↓
Structured Healthcare Capability Database
        ↓
Vector Embeddings + Semantic Search
        ↓
AI Planning Engine
        ↓
Healthcare Gap Detection
        ↓
Interactive Dashboard + AI Query Interface
Core Capabilities
1. Intelligent Document Parsing (IDP)

Healthcare facility descriptions often contain unstructured text such as:

“District hospital providing maternal care, ultrasound diagnostics and emergency obstetric services.”

MedIntel extracts structured information including:

medical specialties

procedures

equipment

facility capabilities

number of doctors

bed capacity

2. Healthcare Gap Detection

The system analyzes coverage across regions to detect missing services such as:

cardiology

trauma care

maternal healthcare

diagnostic imaging

This allows identification of medical deserts.

3. AI Healthcare Planning Assistant

Users can ask natural language questions such as:

Which regions lack cardiology?
Where should cardiologists be deployed?
Dental clinics in northern Ghana

The system uses semantic retrieval (RAG) to find relevant facilities and generate insights.

4. Medical Desert Detection

The platform detects underserved regions by analyzing:

facility density

specialty availability

regional coverage gaps

These regions are visualized on the interactive map.

5. Doctor Deployment Recommendations

The system recommends where specialists should be deployed to maximize healthcare coverage.

Example:

Deploy cardiology specialist to Northern Region
Deploy cardiology specialist to Volta Region
6. AI Data Auditor (Anomaly Detection)

Healthcare data often contains inconsistencies.

MedIntel automatically flags suspicious records such as:

hospital with zero doctors recorded

clinic claiming advanced surgery with one doctor

facility listing capabilities without equipment

Example output:

Accra Newtown Islamic Hospital → Claims surgery with only one doctor
A&A Medlove Medical Centre → Claims cardiology but no diagnostic equipment
Dashboard Features
Facility Coverage Map

Interactive map showing:

healthcare facility distribution

heatmap of healthcare density

underserved regions

Includes light/dark map modes.

Healthcare Gap Analysis

Visualizes specialties missing across regions.

Example:

Regions lacking cardiology:
Ashanti
Northern
Savannah
Upper West
Intervention Recommendations

Suggests facilities that could be expanded to reduce healthcare gaps.

AI Planning Interface

Natural language interface that answers healthcare planning questions.

Includes:

AI reasoning transparency

dataset citations

semantic facility search

Dataset Explorer

Allows users to explore the structured healthcare capability database.

Technologies Used
Core Stack

Python

Streamlit

Pandas

Plotly

Streamlit is an open-source Python framework that allows developers to build interactive data applications quickly using only Python code.

AI Components

Sentence Transformers (embeddings)

FAISS vector search

Retrieval Augmented Generation (RAG)

Agent-style query planning

Data Processing

Intelligent Document Parsing

Capability extraction

Specialty classification

anomaly detection

Repository Structure
medintel-agent
│
├── analysis
│   ├── query_planner.py
│   ├── anomaly_detection.py
│
├── extraction
│   ├── parse_extractions.py
│
├── prompts
│   ├── free_form.py
│   ├── medical_specialties.py
│
├── rag
│   ├── embeddings.py
│   ├── retrieval.py
│
├── data
│   ├── structured_capabilities_geo.csv
│
├── ui
│   └── app.py
│
├── requirements.txt
└── README.md
Example Queries

Try asking the AI planner:

Which regions lack cardiology?
Where should cardiologists be deployed?
Dental clinics in northern region
Hospitals with radiology equipment
Installation

Clone the repository:

git clone https://github.com/AnoushkaNag/medintel-agent.git
cd medintel-agent

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run ui/app.py
Deployment

The project is deployed using Streamlit Community Cloud, which allows rapid deployment of Python data apps directly from GitHub repositories.

Future Improvements

Potential future features include:

population impact estimation

healthcare referral network analysis

hospital failure simulation

doctor workforce optimization

real-time healthcare data integration

Acknowledgments

Accenture Databricks Hackathon
Virtue Foundation — Bridging Medical Deserts Challenge

Author

Team MedIntel Core
KIIT University

GitHub repo link:
https://github.com/AnoushkaNag/medintel-agent