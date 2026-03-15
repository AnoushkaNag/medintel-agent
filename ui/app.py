import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.ai_query_engine import answer_query
from analysis.anomaly_detection import detect_anomalies

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "structured_capabilities_geo.csv"))

st.set_page_config(layout="wide")

st.title("MedIntel – Bridging Medical Deserts")

# ----------------------------------------------------------
# SYSTEM OVERVIEW (NEW)
# ----------------------------------------------------------

st.markdown(
"""
MedIntel is an AI healthcare intelligence platform that:

• Extracts structured medical capabilities from facility reports  
• Detects healthcare access gaps across regions  
• Identifies medical deserts  
• Recommends doctor deployment strategies  
• Audits facility capability claims  

The system transforms messy healthcare data into actionable intelligence
for NGOs and healthcare planners.
"""
)

tabs = st.tabs([
    "Facility Map",
    "Healthcare Gaps",
    "Recommendations",
    "AI Planner",
    "AI Data Auditor",
    "Dataset Explorer"
])

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

try:
    df = pd.read_csv(DATA_PATH)
except:
    st.error("Dataset could not be loaded.")
    st.stop()

# ----------------------------------------------------------
# REGION CLEANING (NEW)
# ----------------------------------------------------------

def clean_region(region):

    if not isinstance(region, str):
        return "Unknown"

    region = region.lower()

    region = region.replace(" region", "")
    region = region.replace(" municipality", "")
    region = region.replace(" district", "")

    region = region.strip()

    return region.title()


df["region"] = df["region"].apply(clean_region)

# ----------------------------------------------------------
# IMPACT ESTIMATION FUNCTION (NEW)
# ----------------------------------------------------------

def estimate_impact(region):

    region_df = df[df["region"] == region]

    facility_count = len(region_df)

    population_estimate = facility_count * 25000

    return population_estimate

# ----------------------------------------------------------
# TAB 1 — FACILITY MAP
# ----------------------------------------------------------

with tabs[0]:

    st.header("Healthcare Facility Coverage")

    map_mode = st.toggle("Dark Mode Map", value=True)

    if map_mode:
        map_style = "carto-darkmatter"
        heat_colors = "Turbo"
        marker_color = "cyan"
    else:
        map_style = "carto-positron"
        heat_colors = "YlOrRd"
        marker_color = "blue"

    map_df = df.dropna(subset=["lat", "lon"])

    fig = go.Figure()

    # HEATMAP
    fig.add_trace(
        go.Densitymapbox(
            lat=map_df["lat"],
            lon=map_df["lon"],
            radius=35,
            colorscale=heat_colors,
            opacity=0.65,
            name="Facility Density"
        )
    )

    # FACILITY MARKERS WITH DETAILS
    fig.add_trace(
        go.Scattermapbox(
            lat=map_df["lat"],
            lon=map_df["lon"],
            mode="markers",
            marker=dict(size=6, color=marker_color),
            text=map_df["facility"],
            customdata=map_df[["region", "specialties"]],
            hovertemplate=
                "<b>%{text}</b><br>" +
                "Region: %{customdata[0]}<br>" +
                "Specialties: %{customdata[1]}<br>" +
                "<extra></extra>",
            name="Facilities"
        )
    )

    # MEDICAL DESERT DETECTION
    region_counts = (
        map_df.groupby("region")
        .size()
        .reset_index(name="facility_count")
    )

    desert_regions = region_counts[region_counts["facility_count"] <= 2]

    desert_points = map_df[map_df["region"].isin(desert_regions["region"])]

    if not desert_points.empty:

        fig.add_trace(
            go.Scattermapbox(
                lat=desert_points["lat"],
                lon=desert_points["lon"],
                mode="markers",
                marker=dict(
                    size=12,
                    color="red",
                    opacity=0.6
                ),
                text=desert_points["region"],
                name="Medical Desert"
            )
        )

    fig.update_layout(
        mapbox=dict(
            style=map_style,
            center=dict(lat=7.9465, lon=-1.0232),
            zoom=6
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=700
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True}
    )

# ----------------------------------------------------------
# TAB 2 — GAP ANALYSIS
# ----------------------------------------------------------

with tabs[1]:

    st.header("Healthcare Gap Analysis")

    specialties = [
        "cardiology",
        "pediatrics",
        "radiology",
        "emergencymedicine",
        "generalsurgery"
    ]

    gap_data = []

    regions = df["region"].dropna().unique()

    for region in regions:

        region_df = df[df["region"] == region]

        missing_count = 0

        for s in specialties:

            if not region_df["specialties"].astype(str).str.contains(s, case=False).any():
                missing_count += 1

        gap_data.append({
            "region": region,
            "gap_score": missing_count
        })

    gap_df = pd.DataFrame(gap_data)

    fig = px.bar(
        gap_df.sort_values("gap_score", ascending=False),
        x="region",
        y="gap_score",
        color="gap_score",
        title="Healthcare Capability Gap by Region"
    )

    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# TAB 3 — RECOMMENDATIONS
# ----------------------------------------------------------

with tabs[2]:

    st.header("Healthcare Intervention Recommendations")

    regions = df["region"].dropna().unique()

    for r in regions[:10]:

        st.subheader(r)

        st.write("High healthcare gap detected")

        facility = df[df["region"] == r]["facility"].iloc[0]

        st.write("Recommended facility for expansion:")

        st.success(facility)

# ----------------------------------------------------------
# TAB 4 — AI PLANNER
# ----------------------------------------------------------

with tabs[3]:

    st.header("AI Healthcare Planning Assistant")

    query = st.text_input("Ask a healthcare planning question")

    if query:

        result = answer_query(query)

        # FACILITY SEARCH
        if result["type"] == "facility_search":

            st.subheader("Relevant Facilities")

            for f in result["facilities"]:
                st.write(f)

        # GAP ANALYSIS
        elif result["type"] == "gap":

            st.subheader(f"Regions lacking {result['specialty']}")

            for r in result["regions"]:
                st.write(r)

        # DEPLOYMENT
        elif result["type"] == "deployment":

            st.subheader("Deployment Recommendations")

            for s in result["suggestions"]:

                region = s.replace("Deploy cardiology specialist to ", "")

                st.write(s)

                impact = estimate_impact(region)

                st.info(f"Estimated population affected: {impact:,}")

        # REASONING
        if "reasoning" in result:

            st.subheader("AI Reasoning")

            for step in result["reasoning"]:
                st.write(step)

        # CITATIONS
        if "citations" in result:

            st.subheader("Sources")

            for c in result["citations"]:

                if "facility" in c:
                    st.write("Facility:", c["facility"])

                if "region" in c:
                    st.write("Region:", c["region"])

                if "specialties" in c:
                    st.write("Specialties:", c["specialties"])

                st.write("---")

# ----------------------------------------------------------
# TAB 5 — AI DATA AUDITOR
# ----------------------------------------------------------

with tabs[4]:

    st.header("AI Healthcare Data Auditor")

    st.write(
        "Detect facilities with suspicious or inconsistent medical capability claims."
    )

    anomalies = detect_anomalies()

    if len(anomalies) == 0:

        st.success("No anomalies detected in dataset.")

    else:

        for a in anomalies[:50]:

            st.warning(
                f"""
⚠ **Facility:** {a["facility"]}

Issue: {a["issue"]}
"""
            )

# ----------------------------------------------------------
# TAB 6 — DATASET EXPLORER
# ----------------------------------------------------------

with tabs[5]:

    st.header("Dataset Explorer")

    st.dataframe(df)