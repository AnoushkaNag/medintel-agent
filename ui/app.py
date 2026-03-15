import ast
import os
import re
import sys
from collections import Counter
from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.ai_query_engine import answer_query
from analysis.anomaly_detection import detect_anomalies

DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "structured_capabilities_geo.csv")
)

COLORS = {
    "bg": "#F6F8FB",
    "card": "rgba(255,255,255,0.72)",
    "card_solid": "#FFFFFF",
    "border": "rgba(20,20,20,0.06)",
    "text": "#111827",
    "muted": "#6B7280",
    "blue": "#4F7CFF",
    "teal": "#2EC5B6",
    "coral": "#FF7A59",
    "red": "#FF5A5F",
    "yellow": "#F6C453",
    "mist": "#E8EEF8",
}

CORE_SPECIALTIES = [
    ("cardiology", "Cardiology"),
    ("pediatrics", "Pediatrics"),
    ("radiology", "Radiology"),
    ("emergencymedicine", "Emergency Medicine"),
    ("generalsurgery", "General Surgery"),
]

SPECIALTY_LABEL_TO_TOKEN = {label: token for token, label in CORE_SPECIALTIES}

PROMPT_SUGGESTIONS = [
    {"label": "Need cardiology support", "prompt": "Which regions need cardiology support?"},
    {"label": "Surgery expansion fit", "prompt": "Show facilities suitable for surgery expansion"},
    {"label": "Highest urgency regions", "prompt": "Where is deployment urgency highest?"},
    {"label": "Audit suspicious claims", "prompt": "Find suspicious healthcare claims in the dataset"},
]

NAV_ITEMS = [
    "Overview",
    "Coverage Map",
    "Gap Intelligence",
    "Interventions",
    "AI Planner",
    "Audit Center",
    "Data Explorer",
]


st.set_page_config(
    page_title="MedIntel",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles():

    st.markdown(
        f"""
        <style>
            :root {{
                --bg: {COLORS["bg"]};
                --card: {COLORS["card"]};
                --card-solid: {COLORS["card_solid"]};
                --border: {COLORS["border"]};
                --text: {COLORS["text"]};
                --muted: {COLORS["muted"]};
                --blue: {COLORS["blue"]};
                --teal: {COLORS["teal"]};
                --coral: {COLORS["coral"]};
                --red: {COLORS["red"]};
                --yellow: {COLORS["yellow"]};
                --mist: {COLORS["mist"]};
                --radius-lg: 28px;
                --radius-md: 22px;
                --radius-pill: 999px;
                --shadow-soft: 0 18px 44px rgba(15, 23, 42, 0.06);
                --shadow-hover: 0 22px 54px rgba(79, 124, 255, 0.10);
            }}

            html, body, [class*="css"] {{
                font-family: "Avenir Next", "SF Pro Display", "Segoe UI", "Helvetica Neue", sans-serif;
            }}

            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(79,124,255,0.08), transparent 32%),
                    radial-gradient(circle at top right, rgba(46,197,182,0.10), transparent 28%),
                    linear-gradient(180deg, #FBFCFE 0%, var(--bg) 50%, #F1F5FA 100%);
                color: var(--text);
            }}

            .block-container {{
                max-width: 1420px;
                padding-top: 2rem;
                padding-bottom: 4rem;
            }}

            header[data-testid="stHeader"] {{
                background: rgba(246, 248, 251, 0.75);
                backdrop-filter: blur(14px);
            }}

            #MainMenu, footer {{
                visibility: hidden;
            }}

            @keyframes fadeUp {{
                from {{
                    opacity: 0;
                    transform: translateY(14px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}

            .fade-up {{
                animation: fadeUp 0.55s ease both;
            }}

            .hero-panel,
            .metric-card,
            .compact-stat,
            .mini-insight-card,
            .glass-card,
            .recommendation-card,
            .anomaly-card,
            .result-card,
            .empty-state,
            .timeline-card,
            .source-card {{
                background: var(--card);
                border: 1px solid var(--border);
                backdrop-filter: blur(14px);
                box-shadow: var(--shadow-soft);
            }}

            .hero-panel {{
                border-radius: 34px;
                padding: 2rem 2.1rem;
                position: relative;
                overflow: hidden;
            }}

            .hero-panel::before {{
                content: "";
                position: absolute;
                inset: 0;
                background: linear-gradient(135deg, rgba(79,124,255,0.08), rgba(46,197,182,0.05) 48%, rgba(255,122,89,0.06));
                pointer-events: none;
            }}

            .hero-kicker {{
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.42rem 0.8rem;
                border-radius: var(--radius-pill);
                background: rgba(79,124,255,0.10);
                color: var(--blue);
                font-size: 0.82rem;
                font-weight: 600;
                letter-spacing: 0.02em;
            }}

            .hero-title {{
                margin: 1rem 0 0.35rem;
                font-size: clamp(2.5rem, 3vw, 4rem);
                font-weight: 700;
                letter-spacing: -0.04em;
                color: var(--text);
                line-height: 0.95;
            }}

            .hero-subtitle {{
                margin: 0;
                font-size: 1.1rem;
                color: var(--muted);
                max-width: 40rem;
                line-height: 1.65;
            }}

            .chip-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin-top: 1.2rem;
            }}

            .product-chip,
            .pill-chip {{
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.48rem 0.82rem;
                border-radius: var(--radius-pill);
                font-size: 0.84rem;
                font-weight: 600;
                border: 1px solid rgba(79,124,255,0.08);
                background: rgba(255,255,255,0.85);
                color: var(--text);
            }}

            .pill-blue {{
                background: rgba(79,124,255,0.10);
                color: var(--blue);
            }}

            .pill-teal {{
                background: rgba(46,197,182,0.12);
                color: #139889;
            }}

            .pill-coral {{
                background: rgba(255,122,89,0.14);
                color: #D65B3A;
            }}

            .pill-amber {{
                background: rgba(246,196,83,0.18);
                color: #A46A00;
            }}

            .pill-red {{
                background: rgba(255,90,95,0.12);
                color: #D53E45;
            }}

            .compact-stat {{
                border-radius: 24px;
                padding: 1rem 1.05rem;
                min-height: 124px;
                transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
            }}

            .compact-stat:hover,
            .metric-card:hover,
            .mini-insight-card:hover,
            .recommendation-card:hover,
            .anomaly-card:hover,
            .result-card:hover,
            .timeline-card:hover,
            .source-card:hover {{
                transform: translateY(-4px);
                box-shadow: var(--shadow-hover);
                border-color: rgba(79,124,255,0.16);
            }}

            .stat-label,
            .metric-label,
            .card-eyebrow {{
                font-size: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--muted);
                font-weight: 700;
            }}

            .stat-value,
            .metric-value {{
                margin-top: 0.45rem;
                font-size: 1.9rem;
                font-weight: 700;
                letter-spacing: -0.04em;
                color: var(--text);
            }}

            .stat-note,
            .metric-note,
            .card-note {{
                margin-top: 0.35rem;
                color: var(--muted);
                font-size: 0.9rem;
                line-height: 1.45;
            }}

            .metric-card {{
                border-radius: 26px;
                padding: 1.15rem 1.15rem 1.05rem;
                min-height: 148px;
                transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
            }}

            .metric-dot {{
                width: 10px;
                height: 10px;
                border-radius: 999px;
                display: inline-block;
                margin-right: 0.45rem;
                vertical-align: middle;
            }}

            .dot-blue {{ background: var(--blue); }}
            .dot-teal {{ background: var(--teal); }}
            .dot-coral {{ background: var(--coral); }}
            .dot-amber {{ background: var(--yellow); }}
            .dot-red {{ background: var(--red); }}

            .section-header {{
                display: flex;
                justify-content: space-between;
                align-items: flex-end;
                gap: 1rem;
                margin: 0.3rem 0 1rem;
            }}

            .section-title {{
                margin: 0;
                font-size: 1.45rem;
                font-weight: 700;
                letter-spacing: -0.03em;
                color: var(--text);
            }}

            .section-subtitle {{
                margin: 0.3rem 0 0;
                color: var(--muted);
                max-width: 46rem;
                line-height: 1.6;
            }}

            .status-pill {{
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.5rem 0.82rem;
                border-radius: var(--radius-pill);
                background: rgba(46,197,182,0.12);
                color: #139889;
                font-size: 0.84rem;
                font-weight: 700;
                white-space: nowrap;
            }}

            .glass-card,
            .mini-insight-card,
            .result-card,
            .timeline-card,
            .source-card {{
                border-radius: 28px;
                padding: 1.15rem 1.2rem;
            }}

            .mini-insight-card {{
                min-height: 180px;
            }}

            .mini-insight-card h4,
            .result-card h4,
            .timeline-card h4,
            .source-card h4 {{
                margin: 0.1rem 0 0.55rem;
                font-size: 1.05rem;
                color: var(--text);
            }}

            .region-row,
            .recommendation-card,
            .anomaly-card {{
                border-radius: 28px;
                padding: 1.1rem 1.15rem;
                margin-bottom: 0.85rem;
            }}

            .region-row .region-top,
            .recommendation-top,
            .anomaly-top {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1rem;
                margin-bottom: 0.7rem;
            }}

            .region-name,
            .recommendation-title,
            .anomaly-title {{
                font-size: 1.08rem;
                font-weight: 700;
                color: var(--text);
                margin: 0;
            }}

            .region-meta,
            .recommendation-meta,
            .anomaly-meta {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.65rem;
                margin-top: 0.9rem;
            }}

            .meta-item {{
                padding: 0.72rem 0.85rem;
                border-radius: 18px;
                background: rgba(255,255,255,0.82);
                border: 1px solid rgba(79,124,255,0.08);
            }}

            .meta-label {{
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--muted);
                font-weight: 700;
            }}

            .meta-value {{
                margin-top: 0.28rem;
                font-size: 0.95rem;
                color: var(--text);
                line-height: 1.45;
            }}

            .result-answer {{
                font-size: 1rem;
                color: var(--text);
                line-height: 1.7;
            }}

            .empty-state {{
                border-radius: 28px;
                padding: 1.7rem;
                text-align: left;
            }}

            .empty-title {{
                margin: 0 0 0.4rem;
                font-size: 1.05rem;
                font-weight: 700;
                color: var(--text);
            }}

            .empty-copy {{
                margin: 0;
                color: var(--muted);
                line-height: 1.7;
            }}

            .timeline-step {{
                padding: 0.78rem 0.85rem;
                border-radius: 18px;
                background: rgba(255,255,255,0.82);
                border: 1px solid rgba(79,124,255,0.08);
                margin-top: 0.55rem;
                color: var(--text);
                line-height: 1.5;
            }}

            .source-card {{
                margin-bottom: 0.65rem;
            }}

            .source-grid {{
                display: grid;
                gap: 0.3rem;
            }}

            div[role="radiogroup"] {{
                gap: 0.55rem !important;
                padding: 0.4rem;
                background: rgba(255,255,255,0.74);
                border: 1px solid var(--border);
                border-radius: 999px;
                box-shadow: var(--shadow-soft);
            }}

            div[role="radiogroup"] label {{
                border-radius: 999px !important;
                padding: 0.32rem 0.25rem !important;
            }}

            div[role="radiogroup"] p {{
                font-size: 0.92rem !important;
                font-weight: 600 !important;
                color: var(--muted) !important;
            }}

            .stButton button,
            .stDownloadButton button {{
                border-radius: 999px;
                padding: 0.72rem 1.1rem;
                border: 1px solid rgba(79,124,255,0.14);
                background: rgba(255,255,255,0.88);
                color: var(--text);
                font-weight: 600;
                transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
            }}

            .stButton button:hover,
            .stDownloadButton button:hover {{
                transform: translateY(-2px) scale(1.01);
                border-color: rgba(79,124,255,0.22);
                box-shadow: var(--shadow-hover);
            }}

            .stButton button[kind="primary"] {{
                background: linear-gradient(135deg, rgba(79,124,255,0.95), rgba(46,197,182,0.90));
                color: white;
                border: none;
            }}

            .stTextInput input,
            .stTextArea textarea,
            .stMultiSelect div[data-baseweb="select"],
            .stSelectbox div[data-baseweb="select"] {{
                border-radius: 20px !important;
                border: 1px solid rgba(79,124,255,0.10) !important;
                background: rgba(255,255,255,0.84) !important;
                color: var(--text) !important;
                box-shadow: inset 0 0 0 1px rgba(255,255,255,0.45);
            }}

            .stTextArea textarea {{
                min-height: 200px;
                padding-top: 1rem;
                line-height: 1.7;
            }}

            .stToggle label, .stCheckbox label {{
                color: var(--text) !important;
                font-weight: 600;
            }}

            div[data-testid="stPlotlyChart"] {{
                border-radius: 28px;
                overflow: hidden;
            }}

            .soft-note {{
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.6;
            }}

            .stExpander {{
                background: rgba(255,255,255,0.74);
                border: 1px solid var(--border);
                border-radius: 24px;
                overflow: hidden;
            }}

            .planner-helper {{
                margin: 0.55rem 0 0.9rem;
                color: var(--muted);
                font-size: 0.94rem;
                line-height: 1.6;
            }}

            .st-key-planner-suggestions .stButton button {{
                min-height: 4.4rem;
                padding: 0.95rem 1rem;
                justify-content: flex-start;
                align-items: center;
                text-align: left;
                white-space: normal;
                line-height: 1.35;
                background: rgba(255,255,255,0.76);
                border: 1px solid rgba(79,124,255,0.10);
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.045);
            }}

            .st-key-planner-suggestions .stButton button:hover {{
                background: rgba(255,255,255,0.92);
                border-color: rgba(79,124,255,0.18);
            }}

            .st-key-planner-input-shell {{
                margin-top: 1rem;
                padding: 1.05rem 1.05rem 1.15rem;
                border-radius: 30px;
                background: rgba(255,255,255,0.78);
                border: 1px solid var(--border);
                backdrop-filter: blur(14px);
                box-shadow: var(--shadow-soft);
            }}

            .st-key-planner-input-shell .stTextArea {{
                margin-top: 0.4rem;
            }}

            .st-key-planner-input-shell [data-testid="stTextArea"] {{
                background: transparent !important;
            }}

            .st-key-planner-input-shell .stTextArea > div {{
                background: transparent !important;
            }}

            .st-key-planner-input-shell .stTextArea [data-baseweb="base-input"],
            .st-key-planner-input-shell .stTextArea [data-baseweb="textarea"] {{
                border-radius: 28px !important;
                border: 1px solid rgba(79,124,255,0.14) !important;
                background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,253,0.96)) !important;
                color-scheme: light !important;
                box-shadow:
                    inset 0 1px 0 rgba(255,255,255,0.72),
                    0 14px 34px rgba(15, 23, 42, 0.05) !important;
                overflow: hidden !important;
            }}

            .st-key-planner-input-shell .stTextArea [data-baseweb="base-input"]:focus-within,
            .st-key-planner-input-shell .stTextArea [data-baseweb="textarea"]:focus-within {{
                border-color: rgba(79,124,255,0.28) !important;
                box-shadow:
                    0 0 0 4px rgba(79,124,255,0.10),
                    0 18px 38px rgba(79,124,255,0.10) !important;
            }}

            .st-key-planner-input-shell textarea,
            .st-key-planner-input-shell .stTextArea textarea {{
                min-height: 180px;
                padding: 1.08rem 1.12rem 1.02rem !important;
                background: #F8FAFD !important;
                color: var(--text) !important;
                -webkit-text-fill-color: var(--text) !important;
                -webkit-box-shadow: 0 0 0 1000px #F8FAFD inset !important;
                caret-color: var(--blue) !important;
                line-height: 1.68;
                font-size: 1rem;
                border: none !important;
                text-shadow: none !important;
            }}

            .st-key-planner-input-shell textarea::placeholder,
            .st-key-planner-input-shell .stTextArea textarea::placeholder {{
                color: rgba(107,114,128,0.88) !important;
                opacity: 1 !important;
                -webkit-text-fill-color: rgba(107,114,128,0.88) !important;
            }}

            .st-key-planner-input-shell .stButton button[kind="primary"] {{
                min-height: 3.35rem;
                margin-top: 0.25rem;
            }}

            .planner-stat-card {{
                min-height: 112px;
                padding: 1rem 1rem 0.92rem;
            }}

            .planner-stat-card h4 {{
                margin-top: 0.38rem;
                font-size: 1.05rem;
                line-height: 1.22;
                overflow-wrap: anywhere;
            }}

            .planner-stat-card--wide {{
                margin-top: 0.7rem;
            }}

            .planner-stat-card--wide h4 {{
                font-size: 1.32rem;
                line-height: 1.15;
            }}

            .signal-panel {{
                padding: 1.2rem 1.2rem 1.15rem;
            }}

            .signal-panel h4 {{
                margin: 0.2rem 0 0.45rem;
                font-size: 1.28rem;
                color: var(--text);
            }}

            .signal-panel .chip-row {{
                margin-top: 0.95rem;
            }}

            .signal-kpi {{
                min-height: 122px;
                padding: 1rem 1rem 0.9rem;
            }}

            .signal-kpi h4 {{
                margin: 0.35rem 0 0.25rem;
                font-size: 1.65rem;
                line-height: 1;
                color: var(--text);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass

    return [text]


def clean_region(region):

    if not isinstance(region, str):
        return "Unknown"

    value = region.lower().strip()
    value = value.replace(" region", "")
    value = value.replace(" municipality", "")
    value = value.replace(" district", "")

    return value.title() if value else "Unknown"


def normalize_specialty(token):

    if not isinstance(token, str):
        return ""

    return re.sub(r"[^a-z]", "", token.lower())


def humanize_specialty(token):

    explicit = {
        "cardiology": "Cardiology",
        "pediatrics": "Pediatrics",
        "radiology": "Radiology",
        "emergencymedicine": "Emergency Medicine",
        "generalsurgery": "General Surgery",
        "gynecologyandobstetrics": "Maternal Care",
        "internalmedicine": "Internal Medicine",
        "criticalcaremedicine": "Critical Care",
        "dentistry": "Dentistry",
    }

    normalized = normalize_specialty(token)

    if normalized in explicit:
        return explicit[normalized]

    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(token))
    return text.replace("_", " ").strip().title()


def estimate_impact(facility_count, gap_score):

    baseline = max(1, gap_score) * 25000
    density_bonus = max(0, 3 - int(facility_count)) * 8000

    return int(baseline + density_bonus)


def priority_from_gap(gap_score, facility_count):

    if gap_score >= 4 or facility_count <= 2:
        return "High"
    if gap_score >= 2:
        return "Medium"
    return "Low"


def priority_tone(priority):

    return {
        "High": "coral",
        "Medium": "amber",
        "Low": "blue",
    }.get(priority, "blue")


def chip_html(text, tone="blue"):

    return f'<span class="pill-chip pill-{tone}">{escape(str(text))}</span>'


def metric_card_html(label, value, subtitle, accent):

    return f"""
    <div class="metric-card fade-up">
        <div class="metric-label"><span class="metric-dot dot-{accent}"></span>{escape(label)}</div>
        <div class="metric-value">{escape(str(value))}</div>
        <div class="metric-note">{escape(subtitle)}</div>
    </div>
    """


def compact_stat_html(label, value, note):

    return f"""
    <div class="compact-stat fade-up">
        <div class="stat-label">{escape(label)}</div>
        <div class="stat-value">{escape(str(value))}</div>
        <div class="stat-note">{escape(note)}</div>
    </div>
    """


def section_header(title, subtitle, badge=None):

    badge_html = f'<div class="status-pill">{escape(badge)}</div>' if badge else ""

    st.markdown(
        f"""
        <div class="section-header fade-up">
            <div>
                <h2 class="section-title">{escape(title)}</h2>
                <p class="section-subtitle">{escape(subtitle)}</p>
            </div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_chart_style(fig, height=420):

    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Avenir Next, SF Pro Display, Segoe UI, Helvetica Neue, sans-serif",
            color=COLORS["text"],
        ),
        margin=dict(l=10, r=10, t=20, b=10),
        colorway=[COLORS["blue"], COLORS["teal"], COLORS["coral"], COLORS["yellow"]],
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(17,24,39,0.06)", zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)

    return fig


def apply_intelligence_bar_style(fig, height, x_title):

    fig = apply_chart_style(fig, height=height)
    fig.update_layout(
        paper_bgcolor="rgba(255,255,255,0.76)",
        plot_bgcolor="rgba(255,255,255,0.96)",
        bargap=0.2,
        xaxis_title=x_title,
        yaxis_title=None,
        margin=dict(l=110, r=42, t=24, b=48),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(17,24,39,0.12)",
        tickfont=dict(size=12, color=COLORS["text"]),
        title_font=dict(size=12, color=COLORS["muted"]),
        showline=True,
        linecolor="rgba(17,24,39,0.08)",
        automargin=True,
    )
    fig.update_yaxes(
        tickfont=dict(size=12, color=COLORS["text"]),
        showline=False,
        automargin=True,
    )

    return fig


@st.cache_data(show_spinner=False)
def load_prepared_data():

    df = pd.read_csv(DATA_PATH)
    df["region_clean"] = df["region"].apply(clean_region)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["capacity_num"] = pd.to_numeric(df["capacity"], errors="coerce")
    df["doctor_count"] = pd.to_numeric(df["numberDoctors"], errors="coerce")
    df["mapped"] = df["lat"].notna() & df["lon"].notna()

    for column in ["specialties", "procedures", "equipment", "capabilities"]:
        df[f"{column}_list"] = df[column].apply(parse_list)

    df["specialty_lookup"] = df["specialties_list"].apply(
        lambda items: [normalize_specialty(item) for item in items if normalize_specialty(item)]
    )
    df["specialty_count"] = df["specialty_lookup"].apply(len)

    region_rows = []

    for region in sorted(
        [name for name in df["region_clean"].dropna().unique().tolist() if name and name != "Unknown"]
    ):

        region_df = df[df["region_clean"] == region]
        available_specialties = set()

        for specialty_list in region_df["specialty_lookup"]:
            available_specialties.update(specialty_list)

        missing_tokens = [token for token, _ in CORE_SPECIALTIES if token not in available_specialties]
        missing_labels = [label for token, label in CORE_SPECIALTIES if token in missing_tokens]
        facility_count = int(len(region_df))
        mapped_count = int(region_df["mapped"].sum())
        gap_score = int(len(missing_tokens))
        priority = priority_from_gap(gap_score, facility_count)

        region_rows.append(
            {
                "region": region,
                "facility_count": facility_count,
                "mapped_count": mapped_count,
                "gap_score": gap_score,
                "missing_tokens": missing_tokens,
                "missing_specialties": missing_labels,
                "impact_estimate": estimate_impact(facility_count, gap_score),
                "priority": priority,
                "density_desert": facility_count <= 2,
                "desert_risk": facility_count <= 2 or gap_score >= 4,
            }
        )

    region_summary = pd.DataFrame(region_rows)
    region_summary = region_summary.sort_values(
        by=["gap_score", "facility_count", "region"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    missing_counter = Counter()
    for tokens in region_summary["missing_tokens"]:
        missing_counter.update(tokens)

    specialty_summary = pd.DataFrame(
        [
            {"token": token, "label": humanize_specialty(token), "count": count}
            for token, count in missing_counter.items()
        ]
    )

    if not specialty_summary.empty:
        specialty_summary = specialty_summary.sort_values(
            by=["count", "label"],
            ascending=[False, True],
        ).reset_index(drop=True)

    return df, region_summary, specialty_summary


@st.cache_data(show_spinner=False)
def load_anomaly_records():

    return detect_anomalies()


def select_best_facility(region_df):

    if region_df.empty:
        return None

    ranked = region_df.copy()
    ranked["capacity_rank"] = ranked["capacity_num"].fillna(-1)
    ranked["doctor_rank"] = ranked["doctor_count"].fillna(-1)
    ranked["mapped_rank"] = ranked["mapped"].astype(int)

    ranked = ranked.sort_values(
        by=["capacity_rank", "doctor_rank", "specialty_count", "mapped_rank", "facility"],
        ascending=[False, False, False, False, True],
    )

    return ranked.iloc[0]


def build_map_figure(map_df, desert_regions, view_mode, map_style):

    fig = go.Figure()

    if map_df.empty:
        fig.add_annotation(
            text="No facilities match the selected filters.",
            showarrow=False,
            font=dict(size=16, color=COLORS["muted"]),
        )
        return apply_chart_style(fig, height=620)

    if view_mode in {"Density", "Hybrid"}:
        fig.add_trace(
            go.Densitymapbox(
                lat=map_df["lat"],
                lon=map_df["lon"],
                radius=34,
                colorscale=[
                    [0.0, "#E7F8F5"],
                    [0.35, "#7DD8CB"],
                    [0.7, "#84A4FF"],
                    [1.0, "#4F7CFF"],
                ],
                opacity=0.48,
                name="Density",
                hoverinfo="skip",
                showscale=False,
            )
        )

    if view_mode in {"Markers", "Hybrid"}:
        fig.add_trace(
            go.Scattermapbox(
                lat=map_df["lat"],
                lon=map_df["lon"],
                mode="markers",
                marker=dict(size=9, color=COLORS["blue"], opacity=0.88),
                text=map_df["facility"],
                customdata=map_df[["region_clean", "specialties"]],
                hovertemplate="<b>%{text}</b><br>Region: %{customdata[0]}<br>Specialties: %{customdata[1]}<extra></extra>",
                name="Facilities",
            )
        )

    desert_points = map_df[map_df["region_clean"].isin(desert_regions)]

    if not desert_points.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=desert_points["lat"],
                lon=desert_points["lon"],
                mode="markers",
                marker=dict(size=13, color=COLORS["coral"], opacity=0.76),
                text=desert_points["region_clean"],
                hovertemplate="Desert-risk region: %{text}<extra></extra>",
                name="Desert Risk",
            )
        )

    fig.update_layout(
        mapbox=dict(
            style=map_style,
            center=dict(lat=7.9465, lon=-1.0232),
            zoom=5.8,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=620,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Avenir Next, SF Pro Display, Segoe UI, Helvetica Neue, sans-serif",
            color=COLORS["text"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.75)",
        ),
    )

    return fig


def render_region_row(row):

    chip_markup = "".join(
        chip_html(specialty, "teal" if idx % 2 == 0 else "blue")
        for idx, specialty in enumerate(row["missing_specialties"][:4])
    )

    st.markdown(
        f"""
        <div class="region-row fade-up">
            <div class="region-top">
                <div>
                    <p class="region-name">{escape(row["region"])}</p>
                    <div class="card-note">Priority corridor for service expansion</div>
                </div>
                {chip_html(f'{row["gap_score"]}/5 gap score', priority_tone(row["priority"]))}
            </div>
            <div class="chip-row">{chip_markup}</div>
            <div class="region-meta">
                <div class="meta-item">
                    <div class="meta-label">Urgency</div>
                    <div class="meta-value">{escape(row["priority"])}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Population impact</div>
                    <div class="meta-value">{row["impact_estimate"]:,}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Facilities</div>
                    <div class="meta-value">{row["facility_count"]} in-region sites</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Mapped coverage</div>
                    <div class="meta-value">{row["mapped_count"]} geocoded facilities</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_card(region_row, facility_row):

    missing_markup = "".join(
        chip_html(specialty, priority_tone(region_row["priority"]))
        for specialty in region_row["missing_specialties"][:4]
    )
    candidate_name = facility_row["facility"] if facility_row is not None else "No clear candidate"
    action_copy = (
        f"Expand capability depth at {candidate_name}"
        if facility_row is not None
        else f"Deploy a specialist outreach pod to {region_row['region']}"
    )

    st.markdown(
        f"""
        <div class="recommendation-card fade-up">
            <div class="recommendation-top">
                <div>
                    <p class="recommendation-title">{escape(region_row["region"])}</p>
                    <div class="card-note">Recommended intervention board</div>
                </div>
                {chip_html(region_row["priority"] + " Priority", priority_tone(region_row["priority"]))}
            </div>
            <div class="meta-item">
                <div class="meta-label">Suggested action</div>
                <div class="meta-value">{escape(action_copy)}</div>
            </div>
            <div class="recommendation-meta">
                <div class="meta-item">
                    <div class="meta-label">Best facility candidate</div>
                    <div class="meta-value">{escape(candidate_name)}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Estimated people impacted</div>
                    <div class="meta-value">{region_row["impact_estimate"]:,}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Missing specialties</div>
                    <div class="meta-value">{", ".join(region_row["missing_specialties"][:3]) or "None"}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Coverage profile</div>
                    <div class="meta-value">{region_row["facility_count"]} facilities / {region_row["gap_score"]} critical gaps</div>
                </div>
            </div>
            <div class="chip-row">{missing_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def anomaly_severity(issue):

    issue_lower = issue.lower()

    if "only one doctor" in issue_lower or "very few doctors" in issue_lower:
        return "High"
    if "no doctors recorded" in issue_lower:
        return "Medium"
    if "no diagnostic equipment" in issue_lower:
        return "Medium"
    return "Low"


def anomaly_issue_type(issue):

    issue_lower = issue.lower()

    if "doctor" in issue_lower:
        return "Staffing risk"
    if "equipment" in issue_lower:
        return "Equipment mismatch"
    if "surgery" in issue_lower or "cardiology" in issue_lower:
        return "Capability mismatch"
    return "Review required"


def anomaly_tone(severity):

    return {
        "High": "red",
        "Medium": "amber",
        "Low": "blue",
    }.get(severity, "blue")


def render_anomaly_card(record):

    st.markdown(
        f"""
        <div class="anomaly-card fade-up">
            <div class="anomaly-top">
                <div>
                    <p class="anomaly-title">{escape(record["facility"])}</p>
                    <div class="card-note">{escape(record["issue_type"])}</div>
                </div>
                {chip_html(record["severity"] + " Severity", anomaly_tone(record["severity"]))}
            </div>
            <div class="meta-item">
                <div class="meta-label">Explanation</div>
                <div class="meta-value">{escape(record["issue"])}</div>
            </div>
            <div class="recommendation-meta">
                <div class="meta-item">
                    <div class="meta-label">Region</div>
                    <div class="meta-value">{escape(record["region"])}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Source row</div>
                    <div class="meta-value">{escape(str(record["source_row"]))}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Review lane</div>
                    <div class="meta-value">Manual verification queue</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Suggested next step</div>
                    <div class="meta-value">Compare claim against staffing and equipment fields</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title, copy):

    st.markdown(
        f"""
        <div class="empty-state fade-up">
            <p class="empty-title">{escape(title)}</p>
            <p class="empty-copy">{escape(copy)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_card(citation):

    specialty_text = citation.get("specialties")
    if isinstance(specialty_text, list):
        specialty_text = ", ".join(specialty_text)

    st.markdown(
        f"""
        <div class="source-card fade-up">
            <h4>{escape(citation.get("facility", "Facility source"))}</h4>
            <div class="source-grid">
                <div class="soft-note"><strong>Region:</strong> {escape(str(citation.get("region", "Unknown")))}</div>
                <div class="soft-note"><strong>Specialties:</strong> {escape(str(specialty_text or "Unknown"))}</div>
                <div class="soft-note"><strong>Source row:</strong> {escape(str(citation.get("source_row", "N/A")))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_styles()

df, region_summary, specialty_summary = load_prepared_data()
anomalies = load_anomaly_records()

if "active_view" not in st.session_state:
    st.session_state.active_view = "Overview"

if "planner_query" not in st.session_state:
    st.session_state.planner_query = "Which regions need cardiology support?"

if "planner_result" not in st.session_state:
    st.session_state.planner_result = None

region_lookup = {}
for row_index, row in df.iterrows():
    name = str(row["facility"]).strip().lower()
    if name and name not in region_lookup:
        region_lookup[name] = {
            "region": row["region_clean"],
            "source_row": row_index + 2,
        }

anomaly_records = []
for anomaly in anomalies:
    lookup = region_lookup.get(anomaly["facility"].strip().lower(), {})
    anomaly_records.append(
        {
            "facility": anomaly["facility"],
            "issue": anomaly["issue"],
            "severity": anomaly_severity(anomaly["issue"]),
            "issue_type": anomaly_issue_type(anomaly["issue"]),
            "region": lookup.get("region", "Unknown"),
            "source_row": lookup.get("source_row", "N/A"),
        }
    )

anomaly_df = pd.DataFrame(anomaly_records)

total_facilities = len(df)
mapped_facilities = int(df["mapped"].sum())
regions_analyzed = int(len(region_summary))
desert_risk_regions = int(region_summary["desert_risk"].sum())
density_desert_regions = int(region_summary["density_desert"].sum())
avg_gap_score = round(float(region_summary["gap_score"].mean()), 1) if not region_summary.empty else 0
estimated_population_impact = int(region_summary.head(5)["impact_estimate"].sum()) if not region_summary.empty else 0
highest_urgency_region = region_summary.iloc[0]["region"] if not region_summary.empty else "Unknown"
top_missing_specialty = specialty_summary.iloc[0]["label"] if not specialty_summary.empty else "None detected"

hero_left, hero_right = st.columns([1.45, 1.0], gap="large")

with hero_left:
    with st.container(key="hero-panel-shell"):
        st.markdown(
            """
            <div class="hero-panel fade-up">
                <div class="hero-kicker">Soft intelligence for equitable care access</div>
                <h1 class="hero-title">MedIntel</h1>
                <p class="hero-subtitle">
                    Healthcare intelligence for underserved regions. Detect care gaps,
                    surface medical deserts, and plan interventions with AI.
                </p>
                <div class="chip-row">
                    <span class="product-chip">Coverage intelligence</span>
                    <span class="product-chip">Gap detection</span>
                    <span class="product-chip">Intervention design</span>
                    <span class="product-chip">AI copilot</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cta_primary, cta_secondary = st.columns([0.9, 1.0], gap="small")
        with cta_primary:
            if st.button("Open AI Planner", key="hero-open-planner", type="primary", use_container_width=True):
                st.session_state.active_view = "AI Planner"
                st.rerun()
        with cta_secondary:
            if st.button("View High-Risk Regions", key="hero-view-risks", use_container_width=True):
                st.session_state.active_view = "Gap Intelligence"
                st.rerun()

with hero_right:
    stat_cols = st.columns(3, gap="small")
    stat_payload = [
        ("Total facilities", f"{total_facilities:,}", "Structured capability records"),
        ("Regions analyzed", f"{regions_analyzed}", "Coverage intelligence ready"),
        ("Desert-risk regions", f"{desert_risk_regions}", "Priority intervention zones"),
    ]

    for column, payload in zip(stat_cols, stat_payload):
        with column:
            st.markdown(compact_stat_html(*payload), unsafe_allow_html=True)


kpi_cols = st.columns(4, gap="medium")
kpi_cards = [
    ("Facilities Mapped", f"{mapped_facilities:,}", "Geocoded care sites visible on the map", "blue"),
    ("Avg. Gap Score", f"{avg_gap_score:.1f}", "Average missing critical specialties by region", "teal"),
    ("Medical Desert Regions", f"{density_desert_regions}", "Low-density coverage footprints", "coral"),
    ("Estimated Population Impact", f"{estimated_population_impact:,}", "Top risk regions in current intervention stack", "amber"),
]

for column, card in zip(kpi_cols, kpi_cards):
    with column:
        st.markdown(metric_card_html(*card), unsafe_allow_html=True)


with st.container(key="nav-shell"):
    st.radio(
        "Workspace",
        NAV_ITEMS,
        horizontal=True,
        key="active_view",
        label_visibility="collapsed",
    )

selected_view = st.session_state.active_view


if selected_view == "Overview":

    section_header(
        "Overview",
        "A high-level intelligence snapshot with current access risk, top gaps, and immediate directions for action.",
        "Live system snapshot",
    )

    overview_left, overview_right = st.columns([1.0, 1.1], gap="large")

    with overview_left:
        top_regions = region_summary.head(4).to_dict("records")

        for region_row in top_regions:
            render_region_row(region_row)

    with overview_right:
        preview_regions = region_summary.head(5)["region"].tolist()
        preview_map_df = df[df["region_clean"].isin(preview_regions) & df["mapped"]]
        preview_fig = build_map_figure(
            preview_map_df,
            region_summary[region_summary["desert_risk"]]["region"].tolist(),
            "Hybrid",
            "carto-positron",
        )
        preview_fig.update_layout(height=430)
        st.plotly_chart(preview_fig, use_container_width=True, config={"displayModeBar": False})

        quick_cols = st.columns(3, gap="small")
        quick_cards = [
            (
                "Highest urgency",
                highest_urgency_region,
                "Region currently carrying the heaviest missing-service load",
            ),
            (
                "Most missing specialty",
                top_missing_specialty,
                "Most common deficit across the regional capability graph",
            ),
            (
                "Audit signal count",
                f"{len(anomaly_records)}",
                "Records currently flagged for operational review",
            ),
        ]

        for column, card in zip(quick_cols, quick_cards):
            with column:
                st.markdown(
                    f"""
                    <div class="mini-insight-card fade-up">
                        <div class="card-eyebrow">{escape(card[0])}</div>
                        <h4>{escape(str(card[1]))}</h4>
                        <p class="card-note">{escape(card[2])}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

elif selected_view == "Coverage Map":

    section_header(
        "Coverage Intelligence",
        "Explore facility density, surface desert-risk regions, and filter the map by specialty or geography.",
        "Live facility view",
    )

    filter_cols = st.columns([1.0, 1.0, 0.8, 1.05], gap="small")
    region_options = ["All regions"] + sorted(region_summary["region"].tolist())
    specialty_options = ["All specialties"] + [label for _, label in CORE_SPECIALTIES]
    specialty_lookup = {label: token for token, label in CORE_SPECIALTIES}

    with filter_cols[0]:
        selected_region = st.selectbox("Region", region_options, index=0)
    with filter_cols[1]:
        selected_specialty_label = st.selectbox("Specialty", specialty_options, index=0)
    with filter_cols[2]:
        show_deserts_only = st.toggle("Show deserts only", value=False)
    with filter_cols[3]:
        view_mode = st.radio(
            "View mode",
            ["Density", "Markers", "Hybrid"],
            horizontal=True,
            label_visibility="collapsed",
        )

    with st.expander("Display settings"):
        map_theme = st.selectbox("Map style", ["Soft light", "Night"], index=0)

    filtered_map_df = df[df["mapped"]].copy()
    filtered_region_summary = region_summary.copy()

    if selected_region != "All regions":
        filtered_map_df = filtered_map_df[filtered_map_df["region_clean"] == selected_region]
        filtered_region_summary = filtered_region_summary[filtered_region_summary["region"] == selected_region]

    if selected_specialty_label != "All specialties":
        selected_token = specialty_lookup[selected_specialty_label]
        filtered_map_df = filtered_map_df[
            filtered_map_df["specialty_lookup"].apply(lambda items: selected_token in items)
        ]
        filtered_region_summary = filtered_region_summary[
            filtered_region_summary["missing_tokens"].apply(lambda items: selected_token in items)
            | filtered_region_summary["region"].isin(filtered_map_df["region_clean"])
        ]

    desert_region_names = filtered_region_summary[filtered_region_summary["desert_risk"]]["region"].tolist()

    if show_deserts_only:
        filtered_map_df = filtered_map_df[filtered_map_df["region_clean"].isin(desert_region_names)]

    map_fig = build_map_figure(
        filtered_map_df,
        desert_region_names,
        view_mode,
        "carto-positron" if map_theme == "Soft light" else "carto-darkmatter",
    )
    st.plotly_chart(
        map_fig,
        use_container_width=True,
        config={"displayModeBar": False, "scrollZoom": True},
    )

    if selected_specialty_label != "All specialties":
        dominant_missing_specialty = selected_specialty_label
    elif not specialty_summary.empty:
        dominant_missing_specialty = specialty_summary.iloc[0]["label"]
    else:
        dominant_missing_specialty = "None"

    footer_cols = st.columns(3, gap="small")
    footer_cards = [
        ("Visible facilities", f"{len(filtered_map_df):,}", "Facilities currently matching the filter stack"),
        ("Underserved regions", f"{len(set(desert_region_names))}", "Regions flagged for low coverage or gap intensity"),
        ("Most missing specialty", dominant_missing_specialty, "Current highest-volume care deficit"),
    ]

    for column, card in zip(footer_cols, footer_cards):
        with column:
            st.markdown(
                f"""
                <div class="glass-card fade-up">
                    <div class="card-eyebrow">{escape(card[0])}</div>
                    <h4>{escape(str(card[1]))}</h4>
                    <p class="card-note">{escape(card[2])}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

elif selected_view == "Gap Intelligence":

    section_header(
        "Gap Intelligence",
        "Rank underserved regions, compare specialty deficits, and quickly see where intervention urgency is highest.",
        "Regional deficit ranking",
    )

    gap_left, gap_right = st.columns([1.0, 1.1], gap="large")

    with gap_left:
        for region_row in region_summary.head(6).to_dict("records"):
            render_region_row(region_row)

    with gap_right:
        top_gap_chart = region_summary.head(10).sort_values("gap_score", ascending=True)
        gap_fig = px.bar(
            top_gap_chart,
            x="gap_score",
            y="region",
            orientation="h",
            text="gap_score",
        )
        gap_fig.update_traces(
            marker_color=COLORS["blue"],
            marker_line_color="rgba(79,124,255,0.16)",
            marker_line_width=1.2,
            texttemplate="%{x:g}",
            textposition="outside",
            textfont=dict(size=12, color=COLORS["text"]),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Gap score: %{x}<extra></extra>",
        )
        st.plotly_chart(
            apply_intelligence_bar_style(gap_fig, height=470, x_title="Gap score"),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    specialty_col, urgency_col = st.columns([1.0, 1.0], gap="large")

    with specialty_col:
        if specialty_summary.empty:
            render_empty_state("No missing specialties detected", "Every core specialty is represented across the analyzed regions.")
        else:
            deficit_fig = px.bar(
                specialty_summary.head(6).sort_values("count", ascending=True),
                x="count",
                y="label",
                orientation="h",
                text="count",
            )
            deficit_fig.update_traces(
                marker_color=COLORS["teal"],
                marker_line_color="rgba(46,197,182,0.18)",
                marker_line_width=1.2,
                texttemplate="%{x}",
                textposition="outside",
                textfont=dict(size=12, color=COLORS["text"]),
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>Missing in %{x} regions<extra></extra>",
            )
            st.plotly_chart(
                apply_intelligence_bar_style(deficit_fig, height=390, x_title="Regions affected"),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    with urgency_col:
        chip_markup = "".join(
            chip_html(row["label"], "coral" if idx < 2 else "teal")
            for idx, row in specialty_summary.head(8).iterrows()
        )
        st.markdown(
            f"""
            <div class="glass-card signal-panel fade-up">
                <div class="card-eyebrow">Specialty deficit signals</div>
                <h4>Most common gap patterns</h4>
                <p class="card-note">Quick-scan the specialties showing up most often across underserved regions, then use the urgency tiles to prioritize intervention pace.</p>
                <div class="chip-row">{chip_markup}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        urgency_breakdown = region_summary["priority"].value_counts()
        urgency_cols = st.columns(3, gap="small")
        for column, priority in zip(urgency_cols, ["High", "Medium", "Low"]):
            count = int(urgency_breakdown.get(priority, 0))
            with column:
                st.markdown(
                    f"""
                    <div class="glass-card signal-kpi fade-up" style="margin-top: 0.75rem;">
                        <div class="card-eyebrow">{escape(priority)} urgency</div>
                        <h4>{count}</h4>
                        <p class="card-note">Regions in the {priority.lower()} intervention lane.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

elif selected_view == "Interventions":

    section_header(
        "Interventions",
        "Turn deficit signals into concrete action cards with facility candidates, impact estimates, and rationale.",
        "Action board",
    )

    recommendation_cols = st.columns(2, gap="large")
    intervention_rows = region_summary[region_summary["gap_score"] > 0].head(6).to_dict("records")

    for index, region_row in enumerate(intervention_rows):
        region_df = df[df["region_clean"] == region_row["region"]]
        best_candidate = select_best_facility(region_df)

        with recommendation_cols[index % 2]:
            render_recommendation_card(region_row, best_candidate)
            with st.expander(f"Why this recommendation for {region_row['region']}?"):
                missing_copy = ", ".join(region_row["missing_specialties"]) or "no major specialties"
                candidate_copy = best_candidate["facility"] if best_candidate is not None else "a specialist outreach unit"
                st.write(
                    f"{region_row['region']} is carrying a gap score of {region_row['gap_score']} with missing coverage in {missing_copy}. "
                    f"The recommendation surfaces {candidate_copy} because it is the strongest current site in-region based on "
                    "existing capacity, mapped visibility, and specialty depth."
                )

elif selected_view == "AI Planner":

    section_header(
        "AI Planning Copilot",
        "A calmer assistant workspace for planning questions, operational trace, and grounded evidence.",
        "Local ReAct workspace",
    )

    planner_left, planner_center, planner_right = st.columns([0.9, 1.2, 0.8], gap="large")

    with planner_left:
        with st.container(key="planner-compose"):
            st.markdown(
                """
                <div class="glass-card fade-up">
                    <div class="card-eyebrow">Prompt composer</div>
                    <h4>Ask the care-access copilot</h4>
                    <p class="card-note">Frame a planning question, then review recommendations, trace, and citations without leaving the workspace.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<p class="planner-helper fade-up">Start with a quick prompt, or write your own planning question below.</p>',
                unsafe_allow_html=True,
            )

            with st.container(key="planner-suggestions"):
                suggestion_cols = st.columns(2, gap="medium")
                for index, suggestion in enumerate(PROMPT_SUGGESTIONS):
                    with suggestion_cols[index % 2]:
                        if st.button(
                            suggestion["label"],
                            key=f"planner-suggestion-{index}",
                            help=suggestion["prompt"],
                            use_container_width=True,
                        ):
                            st.session_state.planner_query = suggestion["prompt"]
                            st.rerun()

            with st.container(key="planner-input-shell"):
                st.markdown(
                    """
                    <div class="card-eyebrow">Planning prompt</div>
                    <p class="planner-helper">Describe a region, specialty, intervention, or audit question. The copilot will return the answer, reasoning trace, and evidence panel together.</p>
                    """,
                    unsafe_allow_html=True,
                )
                with st.form("planner-form", clear_on_submit=False):
                    st.text_area(
                        "Prompt",
                        key="planner_query",
                        height=190,
                        label_visibility="collapsed",
                        placeholder="Ask about missing specialties, deployment urgency, facility matches, or suspicious claims...",
                    )
                    submitted = st.form_submit_button("Run planning analysis", type="primary", use_container_width=True)

            if submitted and st.session_state.planner_query.strip():
                with st.spinner("Mapping intelligence through the local planner..."):
                    st.session_state.planner_result = answer_query(st.session_state.planner_query.strip())

    planner_result = st.session_state.planner_result

    with planner_center:
        with st.container(key="planner-answer"):
            if not planner_result:
                render_empty_state(
                    "AI answer will appear here",
                    "Use one of the suggested prompts or ask your own planning question to generate grounded recommendations and evidence.",
                )
            elif planner_result.get("type") == "error":
                render_empty_state("Planner error", planner_result.get("message", "The planner could not answer this request."))
            else:
                mode_text = planner_result.get("model", "Local planner")
                st.markdown(
                    f"""
                    <div class="result-card fade-up">
                        <div class="card-eyebrow">Primary answer</div>
                        <h4>{escape(planner_result.get("type", "answer").replace("_", " ").title())}</h4>
                        <p class="result-answer">{escape(planner_result.get("answer", "No answer generated."))}</p>
                        <p class="card-note">Planner mode: {escape(planner_result.get("agent_mode", "heuristic").title())} via {escape(mode_text)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if planner_result["type"] == "deployment":
                    for suggestion in planner_result.get("suggestions", []):
                        st.markdown(
                            f"""
                            <div class="result-card fade-up" style="margin-top: 0.75rem;">
                                <div class="card-eyebrow">Actionable suggestion</div>
                                <h4>{escape(suggestion)}</h4>
                                <p class="card-note">Designed to move quickly from capability gap to deployment planning.</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                elif planner_result["type"] == "gap":
                    gap_chip_markup = "".join(
                        chip_html(region, "coral" if idx < 3 else "blue")
                        for idx, region in enumerate(planner_result.get("regions", []))
                    )
                    st.markdown(
                        f"""
                        <div class="result-card fade-up" style="margin-top: 0.75rem;">
                            <div class="card-eyebrow">Region highlights</div>
                            <div class="chip-row">{gap_chip_markup}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                elif planner_result["type"] == "facility_search":
                    for record in planner_result.get("facility_records", [])[:5]:
                        specialty_copy = ", ".join(record.get("specialties", [])) or "Unknown"
                        st.markdown(
                            f"""
                            <div class="result-card fade-up" style="margin-top: 0.75rem;">
                                <div class="card-eyebrow">Facility match</div>
                                <h4>{escape(record["facility"])}</h4>
                                <p class="card-note">Region: {escape(record.get("region", "Unknown"))}</p>
                                <p class="card-note">Specialties: {escape(specialty_copy)}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                elif planner_result["type"] == "anomaly":
                    for anomaly in planner_result.get("anomalies", [])[:5]:
                        st.markdown(
                            f"""
                            <div class="result-card fade-up" style="margin-top: 0.75rem;">
                                <div class="card-eyebrow">Audit signal</div>
                                <h4>{escape(anomaly["facility"])}</h4>
                                <p class="card-note">{escape(anomaly["issue"])}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    with planner_right:
        with st.container(key="planner-side"):
            if not planner_result or planner_result.get("type") == "error":
                render_empty_state(
                    "Trace, evidence, and mode",
                    "Once a planner response is available, you will see the execution trace, evidence cards, and result metadata here.",
                )
            else:
                stat_line = [
                    ("Mode", planner_result.get("agent_mode", "heuristic").title(), "blue"),
                    ("Type", planner_result.get("type", "answer").replace("_", " ").title(), "teal"),
                    ("Citations", str(len(planner_result.get("citations", []))), "coral"),
                ]

                stat_cols = st.columns(2, gap="small")
                for column, stat in zip(stat_cols, stat_line[:2]):
                    with column:
                        st.markdown(
                            f"""
                            <div class="glass-card planner-stat-card fade-up">
                                <div class="card-eyebrow">{escape(stat[0])}</div>
                                <h4>{escape(stat[1])}</h4>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.markdown(
                    f"""
                    <div class="glass-card planner-stat-card planner-stat-card--wide fade-up">
                        <div class="card-eyebrow">{escape(stat_line[2][0])}</div>
                        <h4>{escape(stat_line[2][1])}</h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    """
                    <div class="timeline-card fade-up" style="margin-top: 0.9rem;">
                        <div class="card-eyebrow">Planner trace</div>
                        <h4>Observed reasoning steps</h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                for step in planner_result.get("reasoning", []):
                    st.markdown(f'<div class="timeline-step fade-up">{escape(step)}</div>', unsafe_allow_html=True)

                st.markdown(
                    """
                    <div class="source-card fade-up" style="margin-top: 0.9rem;">
                        <div class="card-eyebrow">Evidence</div>
                        <h4>Facility references</h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                for citation in planner_result.get("citations", [])[:4]:
                    render_source_card(citation)

elif selected_view == "Audit Center":

    section_header(
        "Audit Center",
        "Review suspicious capability claims, sort by severity, and prepare manual follow-up for operational review.",
        "Risk review dashboard",
    )

    severity_counts = anomaly_df["severity"].value_counts() if not anomaly_df.empty else Counter()
    low_confidence_claims = int(
        anomaly_df["issue"].str.contains("no doctors recorded|no diagnostic equipment", case=False, regex=True).sum()
    ) if not anomaly_df.empty else 0

    audit_kpis = st.columns(4, gap="medium")
    audit_cards = [
        ("Total anomalies", len(anomaly_df), "Current records flagged by the auditor", "coral"),
        ("High severity", int(severity_counts.get("High", 0)), "Likely staffing or surgical credibility risks", "red"),
        ("Medium severity", int(severity_counts.get("Medium", 0)), "Capability mismatch needing review", "amber"),
        ("Low-confidence claims", low_confidence_claims, "Signals that rely on partial evidence fields", "blue"),
    ]

    for column, card in zip(audit_kpis, audit_cards):
        with column:
            st.markdown(metric_card_html(*card), unsafe_allow_html=True)

    severity_filter = st.selectbox("Severity filter", ["All", "High", "Medium", "Low"], index=0)
    filtered_anomaly_df = anomaly_df.copy()

    if severity_filter != "All":
        filtered_anomaly_df = filtered_anomaly_df[filtered_anomaly_df["severity"] == severity_filter]

    anomaly_cols = st.columns(2, gap="large")
    for index, record in enumerate(filtered_anomaly_df.head(10).to_dict("records")):
        with anomaly_cols[index % 2]:
            render_anomaly_card(record)

    if filtered_anomaly_df.empty:
        render_empty_state("No anomalies in this filter", "Try a different severity level to review other flagged facilities.")

elif selected_view == "Data Explorer":

    section_header(
        "Data Explorer",
        "Browse the structured healthcare capability base with filters for region, specialty, anomalies, and name search.",
        "Structured dataset browser",
    )

    search_col, region_col, specialty_col, toggle_col = st.columns([1.2, 0.9, 0.9, 0.8], gap="small")

    with search_col:
        search_query = st.text_input("Search facility", placeholder="Search by facility name")
    with region_col:
        explorer_region = st.selectbox("Region", ["All regions"] + sorted(region_summary["region"].tolist()), index=0)
    with specialty_col:
        explorer_specialty = st.selectbox("Specialty", ["All specialties"] + [label for _, label in CORE_SPECIALTIES], index=0)
    with toggle_col:
        anomaly_only = st.toggle("Anomalies only", value=False)

    explorer_df = df.copy()

    if search_query:
        explorer_df = explorer_df[explorer_df["facility"].str.contains(search_query, case=False, na=False)]

    if explorer_region != "All regions":
        explorer_df = explorer_df[explorer_df["region_clean"] == explorer_region]

    if explorer_specialty != "All specialties":
        explorer_token = SPECIALTY_LABEL_TO_TOKEN[explorer_specialty]
        explorer_df = explorer_df[
            explorer_df["specialty_lookup"].apply(lambda items: explorer_token in items)
        ]

    if anomaly_only:
        anomaly_names = {record["facility"] for record in anomaly_records}
        explorer_df = explorer_df[explorer_df["facility"].isin(anomaly_names)]

    display_df = explorer_df.copy()
    display_df["Region"] = display_df["region_clean"]
    display_df["Specialties"] = display_df["specialties_list"].apply(
        lambda items: ", ".join(humanize_specialty(item) for item in items[:6]) or "Unknown"
    )
    display_df["Procedures"] = display_df["procedures_list"].apply(
        lambda items: ", ".join(items[:3]) or "Not available"
    )
    display_df["Equipment"] = display_df["equipment_list"].apply(
        lambda items: ", ".join(items[:3]) or "Not available"
    )
    display_df["Capabilities"] = display_df["capabilities_list"].apply(
        lambda items: ", ".join(items[:3]) or "Not available"
    )
    display_df["Doctors"] = display_df["doctor_count"].fillna("Unknown")
    display_df["Capacity"] = display_df["capacity_num"].fillna("Unknown")
    display_df["Mapped"] = display_df["mapped"].apply(lambda value: "Yes" if value else "No")

    display_df = display_df[
        [
            "facility",
            "Region",
            "Specialties",
            "Procedures",
            "Equipment",
            "Capabilities",
            "Doctors",
            "Capacity",
            "Mapped",
        ]
    ].rename(columns={"facility": "Facility"})

    st.markdown(
        f"""
        <div class="glass-card fade-up" style="margin-bottom: 1rem;">
            <div class="card-eyebrow">Current result set</div>
            <h4>{len(display_df):,} matching facilities</h4>
            <p class="card-note">Use the filters above to move from broad exploration to precise operational review.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_export = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download current view as CSV",
        data=csv_export,
        file_name="medintel_data_explorer.csv",
        mime="text/csv",
        use_container_width=False,
    )
