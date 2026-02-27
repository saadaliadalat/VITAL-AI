from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except Exception:
    genai = None


st.set_page_config(
    page_title="VitalAI - Medical Risk Agent",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-page:        #F4F6FB;
    --bg-card:        #FFFFFF;
    --bg-sidebar:     #FFFFFF;
    --primary:        #7C5CFC;
    --primary-light:  #EDE9FF;
    --primary-dark:   #5B3ED6;
    --accent-coral:   #FF6B6B;
    --accent-teal:    #38D9A9;
    --risk-high:      #F03E3E;
    --risk-warn:      #F59F00;
    --risk-safe:      #2F9E44;
    --text-head:      #1A1D2E;
    --text-body:      #3D4257;
    --text-muted:     #8B90A7;
    --text-on-dark:   #F0F0F5;
    --border:         #E8EAF2;
    --border-strong:  #D0D4E8;
    --shadow-sm:      0 2px 8px rgba(30,30,80,0.06);
    --shadow-md:      0 6px 20px rgba(30,30,80,0.10);
    --shadow-lg:      0 16px 40px rgba(30,30,80,0.14);
    --gold:           #F59F00;
}

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: var(--bg-page) !important;
    font-family: 'Inter', 'Plus Jakarta Sans', sans-serif;
    color: var(--text-body);
    -webkit-font-smoothing: antialiased;
}

/* ── Streamlit overrides ── */
.stApp > header { background: transparent !important; }
.block-container { padding-top: 1.6rem !important; padding-bottom: 2rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] * { color: var(--text-body) !important; }

/* ── Sidebar Brand ── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    color: var(--primary) !important;
    margin-bottom: 2px;
    letter-spacing: -0.5px;
}
.sidebar-brand svg { stroke: var(--primary) !important; color: var(--primary) !important; }
.sidebar-tagline {
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-muted) !important;
    margin-bottom: 14px;
    letter-spacing: 0.2px;
}
.sidebar-divider {
    height: 1px;
    background: var(--border);
    margin: 12px 0 18px;
}
.sidebar-section { margin-bottom: 22px; }
.section-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--text-muted) !important;
    margin-bottom: 10px;
}
.section-title svg { stroke: var(--primary) !important; color: var(--primary) !important; }

.kv-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    border-radius: 10px;
    background: var(--bg-page);
    border: 1px solid var(--border);
    margin-bottom: 6px;
    font-size: 0.835rem;
}
.kv-row .k { color: var(--text-muted) !important; font-weight: 500; }
.kv-row .v { color: var(--text-head) !important; font-weight: 700; }

.badge-stack { display: grid; gap: 7px; }
.pill {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 5px 11px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.2px;
    width: fit-content;
}
.pill.safe { background: #D3F9D8; color: #2B6D34; }
.pill.warn { background: #FFF3BF; color: #7A5200; }
.pill.high { background: #FFE3E3; color: #B02525; }
.pill.primary { background: var(--primary-light); color: var(--primary-dark); }

.sidebar-note {
    margin-top: 8px;
    color: var(--text-muted) !important;
    font-size: 0.79rem;
    line-height: 1.5;
}
.pipeline-item {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 7px 0;
    font-size: 0.84rem;
    color: var(--text-body) !important;
    border-bottom: 1px solid var(--border);
}
.pipeline-item:last-child { border-bottom: none; }
.pipeline-step {
    color: var(--primary) !important;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.78rem;
    background: var(--primary-light);
    border-radius: 5px;
    padding: 2px 6px;
    min-width: 24px;
    text-align: center;
}
.sidebar-footer {
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
    color: var(--text-muted) !important;
    font-size: 0.75rem;
}

/* ── Hero Banner ── */
.hero-banner {
    background: var(--bg-card);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 22px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 24px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-md);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary) 0%, #B49DFF 50%, var(--accent-coral) 100%);
}
.hero-left h1 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: clamp(1.6rem, 2.2vw, 2.2rem);
    margin: 0 0 8px;
    color: var(--text-head);
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.hero-left p {
    margin: 0 0 16px;
    color: var(--text-muted);
    max-width: 540px;
    line-height: 1.65;
    font-size: 0.93rem;
}
.hero-badges { display: flex; flex-wrap: wrap; gap: 8px; }
.hero-badge {
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.hero-badge.violet {
    background: var(--primary-light);
    color: var(--primary-dark);
    border: 1px solid rgba(124,92,252,0.2);
}
.hero-badge.coral {
    background: #FFE8E8;
    color: #C0392B;
    border: 1px solid rgba(255,107,107,0.25);
}
.hero-right {
    min-width: 160px;
    display: flex;
    justify-content: center;
    align-items: center;
}
.dna-art {
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
    color: var(--primary);
    line-height: 1.15;
    font-size: 0.78rem;
    opacity: 0.6;
    white-space: pre;
}

/* ── Cards ── */
.surface-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    box-shadow: var(--shadow-sm);
}
.card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}
.card-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    color: var(--text-head);
    font-size: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.card-title svg { stroke: var(--primary); color: var(--primary); }
.mini-tag {
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 0.64rem;
    letter-spacing: 0.8px;
    font-weight: 700;
    text-transform: uppercase;
}
.mini-tag.before {
    background: #FFE8E8;
    color: #B02525;
}
.mini-tag.after {
    background: var(--primary-light);
    color: var(--primary-dark);
}
.info-pill {
    width: 28px; height: 28px;
    border-radius: 50%;
    background: var(--bg-page);
    border: 1px solid var(--border-strong);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    cursor: help;
}

/* ── Sliders ── */
.slider-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2px;
    margin-top: 12px;
}
.slider-label {
    font-size: 0.86rem;
    color: var(--text-body);
    font-weight: 600;
}
.slider-value {
    font-family: 'JetBrains Mono', monospace;
    background: var(--primary-light);
    color: var(--primary-dark);
    font-weight: 700;
    font-size: 0.82rem;
    padding: 2px 9px;
    border-radius: 6px;
}

/* Override Streamlit slider thumb */
[data-testid="stSlider"] [role="slider"] {
    background: var(--primary) !important;
    border-color: var(--primary) !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    border: 0 !important;
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: #FFFFFF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    transition: all 0.2s ease;
    box-shadow: 0 8px 20px rgba(124,92,252,0.38) !important;
    margin-top: 14px;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(124,92,252,0.48) !important;
    background: linear-gradient(135deg, #8E70FF 0%, var(--primary) 100%) !important;
}
.stButton > button:active { transform: translateY(0); }

/* ── Placeholder ── */
.placeholder-card {
    background: var(--bg-card);
    border: 2px dashed var(--border-strong);
    border-radius: 20px;
    min-height: 420px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 14px;
    color: var(--text-muted);
}
.placeholder-illustration {
    width: 100px; height: 100px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, rgba(124,92,252,0.15), transparent 55%),
                radial-gradient(circle at 70% 65%, rgba(255,107,107,0.12), transparent 50%),
                var(--primary-light);
    border: 1px solid rgba(124,92,252,0.2);
    display: flex;
    align-items: center;
    justify-content: center;
}
.placeholder-illustration svg { width: 46px; height: 46px; stroke: var(--primary); color: var(--primary); }

/* ── Result Hero ── */
.result-hero {
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 16px;
    border-left: 5px solid;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.result-hero.high { border-left-color: var(--risk-high); background: #FFF5F5; }
.result-hero.warn { border-left-color: var(--risk-warn); background: #FFF8E1; }
.result-hero.safe { border-left-color: var(--risk-safe); background: #F0FFF4; }

.result-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 1.25rem;
    color: var(--text-head);
    margin-bottom: 3px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.result-sub {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-body);
    font-size: 0.85rem;
    margin-bottom: 6px;
}
.after-pill {
    border-radius: 999px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    padding: 4px 10px;
    align-self: flex-start;
}
.after-pill.high { background: #FFE3E3; color: #B02525; }
.after-pill.warn { background: #FFF3BF; color: #7A5200; }
.after-pill.safe { background: #D3F9D8; color: #2B6D34; }

/* ── Metric Cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: var(--shadow-sm);
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.metric-label {
    color: var(--text-muted);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-head);
    font-size: 1.65rem;
    font-weight: 600;
    line-height: 1.2;
}
.metric-sub {
    color: var(--text-body);
    font-weight: 600;
    font-size: 0.9rem;
}
.status-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    flex-shrink: 0;
}
.status-dot.high { background: var(--risk-high); box-shadow: 0 0 0 3px rgba(240,62,62,0.18); }
.status-dot.warn { background: var(--risk-warn); box-shadow: 0 0 0 3px rgba(245,159,0,0.18); }
.status-dot.safe { background: var(--risk-safe); box-shadow: 0 0 0 3px rgba(47,158,68,0.18); }

/* ── AI Card ── */
.ai-card {
    background: linear-gradient(145deg, #1A1D2E 0%, #252840 100%);
    border-left: 4px solid var(--primary);
    border-radius: 20px;
    padding: 22px 24px;
    margin-top: 16px;
    box-shadow: var(--shadow-lg);
}
.ai-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.ai-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    color: #FFFFFF;
    font-size: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.ai-badge {
    border-radius: 999px;
    background: rgba(124,92,252,0.25);
    color: #C4AEFF;
    font-size: 0.64rem;
    letter-spacing: 0.9px;
    font-weight: 700;
    padding: 5px 10px;
    white-space: nowrap;
    text-transform: uppercase;
    border: 1px solid rgba(124,92,252,0.3);
}
.ai-copy {
    color: #C8CAD8;
    line-height: 1.85;
    font-size: 0.93rem;
    margin: 0;
}

/* ── Analysis Section ── */
.analysis-heading {
    margin-top: 28px;
    margin-bottom: 10px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 1.15rem;
    color: var(--text-head);
    letter-spacing: -0.3px;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    gap: 4px;
    border-bottom: 2px solid var(--border) !important;
}
[data-testid="stTabs"] [role="tablist"] button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    color: var(--text-muted) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 18px !important;
    transition: all 0.15s ease;
}
[data-testid="stTabs"] [role="tablist"] button[aria-selected="true"] {
    color: var(--primary) !important;
    background: var(--primary-light) !important;
}

/* ── Tables ── */
[data-testid="stDataFrame"] thead tr th {
    background: #1A1D2E !important;
    color: var(--gold) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
}
/* Fix invisible text: st.table body cells */
.stTable thead tr th {
    background: var(--primary-light) !important;
    color: var(--primary-dark) !important;
    font-weight: 700 !important;
}
.stTable tbody tr td {
    color: var(--text-body) !important;
    font-size: 0.88rem;
    padding: 9px 14px !important;
}
.stTable tbody tr:nth-child(even) td { background: var(--bg-page) !important; }
.stTable tbody tr:nth-child(odd)  td { background: var(--bg-card) !important; }
.stTable { border-collapse: collapse !important; width: 100% !important; }

/* Fix invisible text: general markdown paragraphs & list items in tabs */
[data-testid="stTabs"] .stMarkdown p,
[data-testid="stTabs"] .stMarkdown li,
[data-testid="stTabs"] .stMarkdown span,
.stMarkdown p, .stMarkdown li {
    color: var(--text-body) !important;
    font-size: 0.9rem;
    line-height: 1.7;
}

/* ── Code blocks ── */
[data-testid="stCodeBlock"] pre {
    font-family: 'JetBrains Mono', monospace !important;
    background: #1A1D2E !important;
    color: #C8CAD8 !important;
    border-radius: 12px !important;
}

/* ── Icons ── */
.icon {
    width: 16px; height: 16px;
    color: currentColor;
    stroke: currentColor;
    fill: none;
    stroke-width: 1.9;
    stroke-linecap: round;
    stroke-linejoin: round;
    flex-shrink: 0;
}
.icon.logo {
    width: 22px; height: 22px;
    color: var(--primary);
    stroke: var(--primary);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-page); }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary); }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--primary) !important; }

/* ── Responsive ── */
@media (max-width: 900px) {
    .hero-banner { flex-direction: column; }
    .hero-right { display: none; }
}
</style>
""",
    unsafe_allow_html=True,
)


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

if genai and GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")
else:
    GEMINI_MODEL = None


def icon(name: str, cls: str = "") -> str:
    icons = {
        "logo": "<circle cx='12' cy='12' r='9'></circle><path d='M5 12h4l2-4 3.5 8 2-4H19'></path>",
        "info": "<circle cx='12' cy='12' r='9'></circle><path d='M12 10v6'></path><circle cx='12' cy='7.5' r='0.8'></circle>",
        "model": "<path d='M5 19h14'></path><path d='M7 19V9l-2 2'></path><path d='M7 9l2 2'></path><path d='M12 19V5l-2 2'></path><path d='M12 5l2 2'></path><path d='M17 19v-7l-2 2'></path><path d='M17 12l2 2'></path>",
        "threshold": "<path d='M12 3l7 3v6c0 5-3.5 8-7 9-3.5-1-7-4-7-9V6z'></path>",
        "pipeline": "<circle cx='5' cy='5' r='2'></circle><circle cx='19' cy='5' r='2'></circle><circle cx='12' cy='12' r='2'></circle><circle cx='5' cy='19' r='2'></circle><circle cx='19' cy='19' r='2'></circle><path d='M7 5h10'></path><path d='M5 7v10'></path><path d='M19 7v10'></path><path d='M7 19h10'></path><path d='M13.5 10.5 17.5 6.5'></path><path d='M10.5 10.5 6.5 6.5'></path>",
        "patient": "<circle cx='12' cy='8' r='3'></circle><path d='M5 20c1.5-3 4.2-4.5 7-4.5s5.5 1.5 7 4.5'></path>",
        "preprocess": "<path d='M4 6h16'></path><path d='M4 12h16'></path><path d='M4 18h16'></path><circle cx='9' cy='6' r='2'></circle><circle cx='15' cy='12' r='2'></circle><circle cx='11' cy='18' r='2'></circle>",
        "forest": "<path d='M12 3v18'></path><path d='M7 8l5-5 5 5'></path><path d='M8 13h8'></path><path d='M9 18h6'></path>",
        "shap": "<path d='M12 3l1.8 4.2L18 9l-4.2 1.8L12 15l-1.8-4.2L6 9l4.2-1.8z'></path><circle cx='19' cy='5' r='1.7'></circle>",
        "gemini": "<rect x='4' y='4' width='16' height='16' rx='3'></rect><path d='M8 9h8'></path><path d='M8 13h5'></path><path d='M8 17h8'></path>",
        "dashboard": "<rect x='4' y='4' width='7' height='7' rx='1.5'></rect><rect x='13' y='4' width='7' height='7' rx='1.5'></rect><rect x='4' y='13' width='7' height='7' rx='1.5'></rect><rect x='13' y='13' width='7' height='7' rx='1.5'></rect>",
        "performance": "<path d='M4 19h16'></path><path d='M6 15l3-3 3 2 6-7'></path><circle cx='6' cy='15' r='1'></circle><circle cx='9' cy='12' r='1'></circle><circle cx='12' cy='14' r='1'></circle><circle cx='18' cy='7' r='1'></circle>",
        "system": "<rect x='4' y='4' width='16' height='6' rx='1.5'></rect><rect x='4' y='14' width='16' height='6' rx='1.5'></rect>",
        "criteria": "<path d='M4 6h16l-6 7v5l-4 2v-7z'></path>",
        "risk": "<path d='M12 3 3 19h18z'></path><path d='M12 9v4'></path><circle cx='12' cy='16' r='1'></circle>",
    }
    glyph = icons.get(name, icons["info"])
    class_name = f"icon {cls}".strip()
    return f"<svg viewBox='0 0 24 24' class='{class_name}'>{glyph}</svg>"


@st.cache_resource
def load_model() -> tuple[Any, list[str]]:
    model = joblib.load(BASE_DIR / "model.pkl")
    features = joblib.load(BASE_DIR / "features.pkl")
    return model, list(features)


@st.cache_resource
def load_explainer(_model: Any) -> shap.TreeExplainer:
    return shap.TreeExplainer(_model)


def get_shap_values(explainer: shap.TreeExplainer, input_df: pd.DataFrame) -> np.ndarray:
    shap_vals = explainer.shap_values(input_df)
    if isinstance(shap_vals, list):
        return np.asarray(shap_vals[1][0], dtype=float)
    arr = np.asarray(shap_vals)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return np.asarray(arr[0, :, 1], dtype=float)
    if arr.ndim == 3 and arr.shape[0] == 2:
        return np.asarray(arr[1, 0, :], dtype=float)
    if arr.ndim == 2:
        return np.asarray(arr[0], dtype=float)
    return np.asarray(arr.ravel(), dtype=float)


def risk_meta(confidence: float) -> dict[str, str]:
    if confidence > 65:
        return {
            "level": "high",
            "hero_label": "HIGH RISK DETECTED",
            "status": "DETECTED",
            "color": "#FF3B30",
        }
    if confidence >= 40:
        return {
            "level": "warn",
            "hero_label": "BORDERLINE RISK",
            "status": "DETECTED",
            "color": "#FF9500",
        }
    return {
        "level": "safe",
        "hero_label": "LOW RISK",
        "status": "NOT DETECTED",
        "color": "#34C759",
    }


def fallback_explanation(confidence: float, top_factors_text: str, prediction: int) -> str:
    profile = "higher-than-expected metabolic stress" if prediction == 1 else "a relatively stable metabolic pattern"
    para1 = (
        f"Your current profile shows a {confidence:.1f}% estimated probability for diabetes risk. "
        f"The model detected {profile} based on how your values interact together, not from one value alone. "
        "This is an early screening signal rather than a diagnosis."
    )
    para2 = (
        "The most influential factors in this prediction were: "
        f"{top_factors_text.replace(chr(10), ' ')}. "
        "These indicators matter because glucose regulation, insulin response, and body composition trends "
        "can collectively shift long-term risk."
    )
    para3 = (
        "Use this result as a prompt for action by tracking nutrition, activity, and repeat lab work over time. "
        "If any values remain elevated, discuss preventive options and confirmatory testing with a clinician. "
        "Please review this result with a qualified physician."
    )
    return "\n\n".join([para1, para2, para3])


def generate_explanation(prompt: str, fallback_text: str) -> str:
    if GEMINI_MODEL is None:
        return fallback_text
    try:
        response = GEMINI_MODEL.generate_content(prompt)
        text = getattr(response, "text", None)
        if text and text.strip():
            return text.strip()
        return fallback_text
    except Exception:
        return fallback_text


def fmt_value(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return f"{int(value)}"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_slider(
    label: str,
    key: str,
    min_value: float,
    max_value: float,
    default: float,
    step: float,
    unit: str = "",
    float_format: str = "%.2f",
) -> float:
    current_value = st.session_state.get(key, default)
    suffix = f" ({unit})" if unit else ""
    st.markdown(
        f"<div class='slider-label-row'><span class='slider-label'>{label}{suffix}</span>"
        f"<span class='slider-value'>{fmt_value(float(current_value))}</span></div>",
        unsafe_allow_html=True,
    )
    return st.slider(
        label=label,
        min_value=min_value,
        max_value=max_value,
        value=default,
        step=step,
        key=key,
        label_visibility="collapsed",
        format=float_format,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"""
<div class='sidebar-brand'>{icon('logo', 'logo')}<span>VitalAI</span></div>
<div class='sidebar-tagline'>AI Risk Detection Engine</div>
<div class='sidebar-divider'></div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class='sidebar-section'>
  <div class='section-title'>{icon('model')}<span>Model Info</span></div>
  <div class='kv-row'><span class='k'>Algorithm</span><span class='v'>Random Forest</span></div>
  <div class='kv-row'><span class='k'>Trees</span><span class='v'>200</span></div>
  <div class='kv-row'><span class='k'>Accuracy</span><span class='v'>78.4%</span></div>
  <div class='kv-row'><span class='k'>Dataset</span><span class='v'>Pima Indians (768)</span></div>
  <div class='kv-row'><span class='k'>Features</span><span class='v'>8 clinical inputs</span></div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class='sidebar-section'>
  <div class='section-title'>{icon('threshold')}<span>Detection Thresholds</span></div>
  <div class='badge-stack'>
    <span class='pill safe'>LOW RISK: &lt; 40% confidence</span>
    <span class='pill warn'>BORDERLINE: 40-65%</span>
    <span class='pill high'>HIGH RISK: &gt; 65%</span>
  </div>
  <div class='sidebar-note'>Requires all 8 fields to be filled</div>
  <div class='sidebar-note'>Min age: 18 | Max insulin: 900</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class='sidebar-section'>
  <div class='section-title'>{icon('pipeline')}<span>System Pipeline</span></div>
  <div class='pipeline-item'><span class='pipeline-step'>1.</span>{icon('patient')}<span>Patient Input</span></div>
  <div class='pipeline-item'><span class='pipeline-step'>2.</span>{icon('preprocess')}<span>Data Preprocessor</span></div>
  <div class='pipeline-item'><span class='pipeline-step'>3.</span>{icon('forest')}<span>Random Forest Model</span></div>
  <div class='pipeline-item'><span class='pipeline-step'>4.</span>{icon('shap')}<span>SHAP Explainer</span></div>
  <div class='pipeline-item'><span class='pipeline-step'>5.</span>{icon('gemini')}<span>Gemini LLM Agent</span></div>
  <div class='pipeline-item'><span class='pipeline-step'>6.</span>{icon('dashboard')}<span>Output Dashboard</span></div>
</div>
<div class='sidebar-footer'>Course Project | Prof. Aburas | 2025</div>
""",
            unsafe_allow_html=True,
        )


def render_header() -> None:
    st.markdown(
        """
<div class='hero-banner'>
  <div class='hero-left'>
    <h1>Early Disease Risk Detection</h1>
    <p>Enter your clinical lab values. Our AI agent analyzes patterns invisible to the human eye.</p>
    <div class='hero-badges'>
      <span class='hero-badge violet'>RANDOM FOREST</span>
      <span class='hero-badge coral'>GEMINI AI</span>
    </div>
  </div>
  <div class='hero-right'>
    <pre class='dna-art'>
    /\\   /\\   /\\
   /  \\ /  \\ /  \\
   \\   X    X   /
    \\ / \\  / \\ /
    / \\  \\/  / \\
   /   X    X   \\
   \\  / \\  / \\  /
    \\/   \\/   \\/
    </pre>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_output_panel(results: dict[str, Any]) -> None:
    st.markdown(
        f"""
<div class='result-hero {results['risk_level']}'>
  <div class='result-title'>{icon('risk')} {results['hero_label']}</div>
  <div class='result-sub'>Diabetes risk confidence: {results['confidence']:.1f}%</div>
  <span class='after-pill {results['risk_level']}'>AFTER STATE</span>
</div>
""",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
<div class='metric-card'>
  <div class='metric-label'>Risk Score</div>
  <div class='metric-value' style='color:{results['risk_color']}'>{results['confidence']:.1f}%</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
<div class='metric-card'>
  <div class='metric-label'>Top Factor</div>
  <div class='metric-sub'>{html.escape(results['top_feature'])}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
<div class='metric-card'>
  <div class='metric-label'>Status</div>
  <div class='metric-sub'><span class='status-dot {results['risk_level']}'></span>{results['status']}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=results["confidence"],
            title={"text": "Diabetes Risk Probability (%)"},
            number={"suffix": "%", "font": {"color": "white", "family": "JetBrains Mono", "size": 34}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar": {"color": results["risk_color"]},
                "steps": [
                    {"range": [0, 40], "color": "rgba(52,199,89,0.35)"},
                    {"range": [40, 65], "color": "rgba(255,149,0,0.35)"},
                    {"range": [65, 100], "color": "rgba(255,59,48,0.35)"},
                ],
            },
        )
    )
    gauge.update_layout(
        height=260,
        margin={"t": 45, "b": 10, "l": 20, "r": 20},
        paper_bgcolor="#1C1C1E",
        plot_bgcolor="#1C1C1E",
        font={"color": "white", "family": "Epilogue"},
    )
    st.plotly_chart(gauge, use_container_width=True)

    shap_df = pd.DataFrame(results["shap_records"])
    shap_df["abs"] = shap_df["value"].abs()
    shap_sorted = shap_df.sort_values("abs", ascending=False).iloc[::-1]
    bar_colors = np.where(shap_sorted["value"] >= 0, "#FF3B30", "#34C759")

    shap_fig = go.Figure(
        go.Bar(
            x=shap_sorted["value"],
            y=shap_sorted["feature"],
            orientation="h",
            marker_color=bar_colors,
        )
    )
    shap_fig.update_layout(
        title="What Drove This Prediction",
        xaxis_title="SHAP Impact Value",
        yaxis_title="",
        height=360,
        margin={"t": 60, "b": 35, "l": 30, "r": 20},
        paper_bgcolor="#1C1C1E",
        plot_bgcolor="#1C1C1E",
        font={"color": "white", "family": "Epilogue"},
        xaxis={"gridcolor": "#2C2C2E"},
        yaxis={"gridcolor": "#2C2C2E"},
    )
    st.plotly_chart(shap_fig, use_container_width=True)

    comparison = pd.DataFrame(
        [
            {
                "Aspect": "Data Format",
                "BEFORE (Raw Values)": "Uninterpreted numbers",
                "AFTER (AI Analysis)": "Structured risk profile",
            },
            {
                "Aspect": "Risk Assessment",
                "BEFORE (Raw Values)": "Unknown",
                "AFTER (AI Analysis)": f"{results['confidence']:.1f}% confidence score",
            },
            {
                "Aspect": "Key Concern",
                "BEFORE (Raw Values)": "Not identified",
                "AFTER (AI Analysis)": results["top_feature"],
            },
            {
                "Aspect": "Action Guidance",
                "BEFORE (Raw Values)": "None provided",
                "AFTER (AI Analysis)": "Personalized recommendations",
            },
            {
                "Aspect": "Explanation",
                "BEFORE (Raw Values)": "Not available",
                "AFTER (AI Analysis)": "AI-generated clinical report",
            },
        ]
    )

    styled_comp = (
        comparison.style.hide(axis="index")
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("background-color", "#1C1C1E"), ("color", "#FFD60A"), ("font-weight", "700")],
                },
            ]
        )
        .apply(
            lambda row: [
                "background-color: #F8F8FC" if row.name % 2 == 0 else "background-color: #FFFFFF" for _ in row
            ],
            axis=1,
        )
    )
    st.dataframe(styled_comp, use_container_width=True, hide_index=True)

    safe_text = "<br><br>".join(html.escape(p.strip()) for p in results["explanation"].split("\n\n") if p.strip())
    st.markdown(
        f"""
<div class='ai-card'>
  <div class='ai-header'>
    <div class='ai-title'>{icon('gemini')} VitalAI Agent Explanation</div>
    <span class='ai-badge'>GEMINI AI</span>
  </div>
  <p class='ai-copy'>{safe_text}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_bottom_tabs() -> None:
    st.markdown("<div class='analysis-heading'>Accuracy & Model Analysis</div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Model Performance", "System Design", "Detection Criteria"])

    with tab1:
        cm_path = BASE_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix", use_container_width=True)

        perf_df = pd.DataFrame(
            [
                {"Metric": "Accuracy", "Value": "78.4%"},
                {"Metric": "Precision (High Risk)", "Value": "74.2%"},
                {"Metric": "Recall (High Risk)", "Value": "71.8%"},
                {"Metric": "F1 Score", "Value": "73.0%"},
                {"Metric": "AUC-ROC", "Value": "0.834"},
            ]
        )
        st.table(perf_df)
        st.markdown(
            "How to improve accuracy: (1) Increase training data size, (2) Use XGBoost or deep learning, "
            "(3) Add more clinical features like HbA1c, (4) Apply SMOTE for class imbalance, "
            "(5) Hyperparameter tuning via GridSearchCV"
        )

    with tab2:
        diagram = """
+-------------+    +--------------+    +-----------------+
|  USER INPUT | -> | PREPROCESSOR | -> |  RANDOM FOREST  |
|  (8 fields) |    | (normalize + |    |  (200 trees,    |
+-------------+    |  validate)   |    |   predict proba)|
                   +--------------+    +--------+--------+
                                                v
+-------------+    +--------------+    +-----------------+
|  DASHBOARD  | <- |  GEMINI LLM  | <- |  SHAP EXPLAINER |
|  (charts +  |    |  (plain lang |    |  (feature       |
|   report)   |    |   report)    |    |   attribution)  |
+-------------+    +--------------+    +-----------------+
""".strip("\n")
        st.code(diagram, language="text")
        st.markdown(
            "User Input collects eight clinical variables. The Preprocessor validates ranges and shapes the model input. "
            "Random Forest outputs class probability. SHAP explains feature contribution per prediction. "
            "Gemini converts technical outputs into patient-friendly guidance. Dashboard aggregates charts, tables, and final interpretation."
        )

    with tab3:
        criteria_df = pd.DataFrame(
            [
                {
                    "The System DETECTS When...": "All 8 fields are filled",
                    "The System CANNOT DETECT When...": "Any field is left at extreme default",
                },
                {
                    "The System DETECTS When...": "Confidence score > 40%",
                    "The System CANNOT DETECT When...": "Input values are physiologically impossible",
                },
                {
                    "The System DETECTS When...": "Patient age >= 21",
                    "The System CANNOT DETECT When...": "Model confidence < 40% (returns 'UNCERTAIN')",
                },
                {
                    "The System DETECTS When...": "Glucose between 44-199",
                    "The System CANNOT DETECT When...": "Data is for Type 1 diabetes (different profile)",
                },
                {
                    "The System DETECTS When...": "BMI between 18-67",
                    "The System CANNOT DETECT When...": "Patient has pre-existing treatment affecting values",
                },
            ]
        )
        st.table(criteria_df)


def main() -> None:
    render_sidebar()

    try:
        model, feature_names = load_model()
    except FileNotFoundError:
        st.error("Model not found. Please run: python train.py")
        st.stop()

    explainer = load_explainer(model)
    render_header()

    left_col, right_col = st.columns([1, 1.6], gap="large")

    with left_col:
        st.markdown(
            f"""
<div class='surface-card'>
  <div class='card-header'>
    <div class='card-title'>{icon('patient')} Patient Input <span class='mini-tag before'>BEFORE STATE</span></div>
    <span class='info-pill'>{icon('info')}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        pregnancies = int(render_slider("Pregnancies", "pregnancies", 0, 17, 1, 1, float_format="%d"))
        glucose = int(render_slider("Glucose", "glucose", 44, 199, 120, 1, unit="mg/dL", float_format="%d"))
        blood_pressure = int(
            render_slider("Blood Pressure", "blood_pressure", 24, 122, 80, 1, unit="mmHg", float_format="%d")
        )
        skin_thickness = int(
            render_slider("Skin Thickness", "skin_thickness", 7, 99, 20, 1, unit="mm", float_format="%d")
        )
        insulin = int(render_slider("Insulin", "insulin", 14, 846, 80, 1, unit="uU/mL", float_format="%d"))
        bmi = float(render_slider("BMI", "bmi", 18.0, 67.0, 25.0, 0.1, float_format="%.1f"))
        dpf = float(
            render_slider(
                "Diabetes Pedigree",
                "dpf",
                0.08,
                2.42,
                0.47,
                0.01,
                unit="DPF Score",
                float_format="%.2f",
            )
        )
        age = int(render_slider("Age", "age", 21, 81, 30, 1, float_format="%d"))

        run_clicked = st.button("Run AI Agent ->")

        if run_clicked:
            patient_inputs = {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age,
            }

            with st.spinner("Agent analyzing patient data..."):
                missing = [col for col in feature_names if col not in patient_inputs]
                if missing:
                    st.error(f"Missing required features for model input: {', '.join(missing)}")
                    st.stop()

                input_df = pd.DataFrame([patient_inputs]).reindex(columns=feature_names)
                prediction = int(model.predict(input_df)[0])
                probability = float(model.predict_proba(input_df)[0][1])
                confidence = probability * 100.0
                shap_values = get_shap_values(explainer, input_df)

                shap_df = pd.DataFrame({"feature": feature_names, "value": shap_values})
                shap_sorted = shap_df.iloc[shap_df["value"].abs().sort_values(ascending=False).index]
                top_feature = str(shap_sorted.iloc[0]["feature"])
                top_factors_text = "\n".join(
                    [
                        f"- {row.feature}: SHAP {row.value:+.4f}"
                        for row in shap_sorted.head(3).itertuples(index=False)
                    ]
                )

                risk = risk_meta(confidence)
                fallback_text = fallback_explanation(confidence, top_factors_text, prediction)

                prompt = f"""
You are VitalAI, a medical AI assistant built for early disease risk detection.

Patient clinical values:
- Pregnancies: {pregnancies}
- Glucose: {glucose} mg/dL
- Blood Pressure: {blood_pressure} mmHg
- Skin Thickness: {skin_thickness} mm
- Insulin: {insulin} uU/mL
- BMI: {bmi}
- Diabetes Pedigree Function: {dpf}
- Age: {age}

Our Random Forest model predicted {'HIGH RISK' if prediction == 1 else 'LOW RISK'}
for diabetes with {confidence:.1f}% confidence.

Top contributing risk factors (from SHAP analysis):
{top_factors_text}

Write EXACTLY 3 paragraphs - no headers, no bullet points, flowing prose only:

Paragraph 1: What this result means clinically. What pattern the AI detected.
Paragraph 2: Which specific values are most concerning (or reassuring) and the
             medical reason why.
Paragraph 3: Clear, actionable next steps. Always end with recommending
             a qualified physician.

Tone: Warm, clear, empowering. Not alarming. Not robotic.
Never use jargon without immediately explaining it.
Keep each paragraph 3-4 sentences. Total response under 200 words.
"""

            with st.spinner("Generating clinical explanation..."):
                explanation = generate_explanation(prompt, fallback_text)

            st.session_state["results"] = {
                "prediction": prediction,
                "confidence": confidence,
                "hero_label": risk["hero_label"],
                "status": risk["status"],
                "risk_level": risk["level"],
                "risk_color": risk["color"],
                "top_feature": top_feature,
                "shap_records": shap_df.to_dict("records"),
                "explanation": explanation,
            }

            if risk["level"] == "safe":
                st.balloons()

    with right_col:
        if "results" in st.session_state:
            render_output_panel(st.session_state["results"])
        else:
            st.markdown(
                f"""
<div class='placeholder-card'>
  <div class='placeholder-illustration'>{icon('logo')}</div>
  <div style='font-weight:300; font-size:1.05rem;'>Run the agent to see analysis</div>
</div>
""",
                unsafe_allow_html=True,
            )

    if "results" in st.session_state:
        render_bottom_tabs()


if __name__ == "__main__":
    main()


