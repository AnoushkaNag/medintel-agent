import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.helpers import enrich_specialties


# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(layout="wide")

st.title("MedIntel – Bridging Medical Deserts")

tabs = st.tabs([
    "Facility Map",
    "Healthcare Gaps",
    "Recommendations",
    "Dataset Explorer"
])


# ==================================================
# TAB 1 — FACILITY MAP
# ==================================================

with tabs[0]:

    st.header("Healthcare Facility Coverage")
    st.caption("Facility density across Ghana")

    df = pd.read_csv("data/structured_capabilities_geo.csv")

    df = enrich_specialties(df)

    df = df.dropna(subset=["lat", "lon"])

    # -------------------------------
    # Density Heatmap
    # -------------------------------

    heatmap = go.Densitymapbox(
        lat=df["lat"],
        lon=df["lon"],
        z=[1] * len(df),
        radius=65,
        colorscale="Turbo",
        opacity=0.75,
        hoverinfo="skip"
    )

    # -------------------------------
    # Hover Details Layer
    # -------------------------------

    hover_layer = go.Scattermapbox(
        lat=df["lat"],
        lon=df["lon"],
        mode="markers",
        marker=dict(
            size=6,
            color="rgba(0,0,0,0)"
        ),
        text=df.apply(
            lambda row: (
                f"<b>{row['facility']}</b><br>"
                f"Region: {row['region']}<br>"
                f"Specialties: {row['specialties']}"
            ),
            axis=1
        ),
        hoverinfo="text",
        showlegend=False
    )

    fig = go.Figure()

    fig.add_trace(heatmap)
    fig.add_trace(hover_layer)

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=7.9, lon=-1.0),
            zoom=6
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=650
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True}
    )


# ==================================================
# TAB 2 — HEALTHCARE GAPS
# ==================================================

with tabs[1]:

    st.header("Healthcare Gap Analysis")

    gap_df = pd.read_csv("data/region_density.csv")

    gap_df["gap_score"] = gap_df["facility_count"].max() - gap_df["facility_count"]

    fig = px.bar(
        gap_df.sort_values("gap_score", ascending=False),
        x="region",
        y="gap_score",
        color="gap_score",
        title="Regions with Largest Healthcare Gaps",
        color_continuous_scale="Turbo"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True}
    )


# ==================================================
# TAB 3 — RECOMMENDATIONS
# ==================================================

with tabs[2]:

    st.header("Healthcare Intervention Recommendations")

    gap_df = pd.read_csv("data/region_density.csv")
    facilities_df = pd.read_csv("data/structured_capabilities_geo.csv")

    gap_df["gap_score"] = gap_df["facility_count"].max() - gap_df["facility_count"]

    high_gap = gap_df.sort_values("gap_score", ascending=False).head(5)

    for _, row in high_gap.iterrows():

        region = row["region"]

        st.subheader(region.upper())

        st.write("High healthcare gap detected")

        region_facilities = facilities_df[
            facilities_df["region"].str.contains(region, case=False, na=False)
        ]

        if len(region_facilities) > 0:

            suggested = region_facilities.sample(1).iloc[0]["facility"]

            st.write("Recommended facility for expansion:")
            st.write(suggested)

        else:

            st.write("Recommended facility for expansion:")
            st.write("New facility recommended")


# ==================================================
# TAB 4 — DATASET EXPLORER
# ==================================================

with tabs[3]:

    st.header("Dataset Explorer")

    df = pd.read_csv("data/structured_capabilities_geo.csv")

    st.dataframe(df)
