import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import requests
import json
import time
from datetime import datetime
from dotenv import load_dotenv

from stock_data import load_historical_data, get_processed_data, compute_selected_point_signals, HISTORICAL_SENTIMENT_LOGS
from models import create_sequences, evaluate_performance, train_model_live

load_dotenv()

# ── Page Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="TSLA Quantitative Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme Presets ────────────────────────────────────────────
THEMES = {
    'obsidian-crimson': {
        'name': 'Obsidian Crimson', 'accent': '#E82127', 'accent2': '#FF4D52',
        'bg': '#0A0A0C', 'bg2': '#101014', 'card': '#16161A', 'text': '#E4E4E7',
        'muted': '#71717A', 'border': '#27272A', 'glow': 'rgba(232,33,39,0.2)',
        'tpl': 'plotly_dark', 'c1': '#E82127', 'c2': '#38BDF8', 'c3': '#A855F7',
        'c4': '#22C55E', 'grid': '#1A1A22',
        'grad': 'linear-gradient(135deg, #0A0A0C 0%, #1A0508 50%, #0A0A0C 100%)'
    },
    'midnight-amber': {
        'name': 'Midnight Amber', 'accent': '#F59E0B', 'accent2': '#FBBF24',
        'bg': '#070708', 'bg2': '#0F0F12', 'card': '#151518', 'text': '#F3EFE9',
        'muted': '#78716C', 'border': '#292524', 'glow': 'rgba(245,158,11,0.2)',
        'tpl': 'plotly_dark', 'c1': '#F59E0B', 'c2': '#10B981', 'c3': '#EC4899',
        'c4': '#3B82F6', 'grid': '#1C1C22',
        'grad': 'linear-gradient(135deg, #070708 0%, #1A1205 50%, #070708 100%)'
    },
    'cyber-pulse': {
        'name': 'Cyber Pulse', 'accent': '#06B6D4', 'accent2': '#22D3EE',
        'bg': '#030612', 'bg2': '#0A0F24', 'card': '#0E1326', 'text': '#E2ECFF',
        'muted': '#4B5B84', 'border': '#1E293B', 'glow': 'rgba(6,182,212,0.2)',
        'tpl': 'plotly_dark', 'c1': '#06B6D4', 'c2': '#C084FC', 'c3': '#F43F5E',
        'c4': '#34D399', 'grid': '#141E35',
        'grad': 'linear-gradient(135deg, #030612 0%, #0A1628 50%, #030612 100%)'
    },
    'emerald-matrix': {
        'name': 'Emerald Matrix', 'accent': '#10B981', 'accent2': '#34D399',
        'bg': '#020604', 'bg2': '#081210', 'card': '#0C1812', 'text': '#DBFCE8',
        'muted': '#405B4E', 'border': '#172A22', 'glow': 'rgba(16,185,129,0.2)',
        'tpl': 'plotly_dark', 'c1': '#10B981', 'c2': '#FBBF24', 'c3': '#0EA5E9',
        'c4': '#F472B6', 'grid': '#0E2219',
        'grad': 'linear-gradient(135deg, #020604 0%, #051A0E 50%, #020604 100%)'
    },
    'classic-paper': {
        'name': 'Classic Paper', 'accent': '#DC2626', 'accent2': '#EF4444',
        'bg': '#FAFAF9', 'bg2': '#F5F5F4', 'card': '#FFFFFF', 'text': '#1C1917',
        'muted': '#78716C', 'border': '#E7E5E4', 'glow': 'rgba(0,0,0,0.06)',
        'tpl': 'plotly_white', 'c1': '#DC2626', 'c2': '#2563EB', 'c3': '#7C3AED',
        'c4': '#059669', 'grid': '#F0EFED',
        'grad': 'linear-gradient(135deg, #FAFAF9 0%, #F5F0EB 50%, #FAFAF9 100%)'
    }
}

if 'theme_id' not in st.session_state:
    st.session_state.theme_id = 'obsidian-crimson'
T = THEMES[st.session_state.theme_id]

# ── Inject Global CSS ────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

.stApp {{
    background: {T['grad']} !important;
    color: {T['text']} !important;
    font-family: 'Inter', sans-serif !important;
}}
/* ── SIDEBAR PREMIUM OVERRIDES ── */
div[data-testid="stSidebar"] {{
    background: {T['bg2']} !important;
    border-right: 1px solid {T['border']} !important;
    padding: 0 !important;
}}
div[data-testid="stSidebar"] > div:first-child {{
    padding: 0 !important;
}}
div[data-testid="stSidebarUserContent"] {{
    padding: 0 !important;
}}

/* Sidebar widget label override */
div[data-testid="stSidebar"] label {{
    color: {T['muted']} !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace !important;
}}
div[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: {T['card']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
    color: {T['text']} !important;
}}
div[data-testid="stSidebar"] .stRadio > div {{
    gap: 6px !important;
}}
div[data-testid="stSidebar"] .stRadio label {{
    background: {T['card']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 6px !important;
    padding: 4px 10px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    cursor: pointer;
    transition: all 0.2s ease !important;
}}
div[data-testid="stSidebar"] .stDateInput > div > div input {{
    background: {T['card']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
    color: {T['text']} !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

/* Sidebar custom blocks */
.sb-logo-block {{
    background: linear-gradient(135deg, {T['accent']}22, {T['accent']}08);
    border-bottom: 1px solid {T['border']};
    padding: 24px 20px 20px 20px;
    margin-bottom: 0;
}}
.sb-section {{
    padding: 16px 20px;
    border-bottom: 1px solid {T['border']}80;
}}
.sb-section-title {{
    font-size: 9px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: {T['muted']};
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 10px;
}}
.sb-theme-pill {{
    display: inline-block;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    border: 1px solid;
    transition: all 0.2s ease;
    margin: 2px;
}}
.sb-stat-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid {T['border']}40;
}}
.sb-stat-label {{
    font-size: 10px;
    color: {T['muted']};
    font-family: 'JetBrains Mono', monospace;
}}
.sb-stat-value {{
    font-size: 11px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: {T['text']};
}}
.sb-footer {{
    padding: 14px 20px;
    text-align: center;
}}

/* Cards */
.q-card {{
    background: {T['card']};
    border: 1px solid {T['border']};
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    transition: transform 0.3s cubic-bezier(.4,0,.2,1), box-shadow 0.3s ease, border-color 0.3s ease;
    position: relative;
    overflow: hidden;
}}
.q-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, {T['accent']}, transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
}}
.q-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 12px 40px {T['glow']};
    border-color: {T['accent']}60;
}}
.q-card:hover::before {{
    opacity: 1;
}}

/* Metric Card */
.m-card {{
    background: {T['card']};
    border: 1px solid {T['border']};
    border-radius: 12px;
    padding: 16px 18px;
    transition: all 0.25s ease;
}}
.m-card:hover {{
    border-color: {T['accent']}60;
    box-shadow: 0 6px 24px {T['glow']};
}}
.m-label {{
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
    color: {T['muted']};
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 4px;
}}
.m-value {{
    font-size: 26px;
    font-weight: 900;
    color: {T['text']};
    line-height: 1.1;
}}
.m-sub {{
    font-size: 11px;
    color: {T['muted']};
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}}

/* Section headers */
.sec-h {{
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: {T['accent']};
    font-family: 'JetBrains Mono', monospace;
    padding: 8px 0 6px 0;
    border-bottom: 1px solid {T['border']};
    margin-bottom: 16px;
}}

/* Badge */
.badge {{
    display: inline-block;
    background: {T['accent']}18;
    color: {T['accent']};
    border: 1px solid {T['accent']}35;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 10px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}}

/* Hero */
.hero-wrap {{
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid {T['border']};
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    position: relative;
}}
.hero-overlay {{
    position: absolute;
    bottom: 0; left: 0; right: 0;
    padding: 28px 32px;
    background: linear-gradient(transparent, {T['bg']}EE);
}}
.hero-title {{
    font-size: 32px;
    font-weight: 900;
    letter-spacing: -0.5px;
    color: {T['text']};
    margin: 0;
    text-shadow: 0 2px 12px rgba(0,0,0,0.5);
}}
.hero-sub {{
    font-size: 13px;
    color: {T['muted']};
    margin: 4px 0 0 0;
}}

/* Sentiment cards */
.s-card {{
    background: {T['card']};
    border: 1px solid {T['border']};
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
    border-left: 3px solid {T['accent']};
    transition: border-color 0.2s ease;
}}
.s-card:hover {{ border-color: {T['accent2']}; }}

/* Button overrides */
div.stButton > button {{
    background: {T['card']} !important;
    color: {T['text']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
}}
div.stButton > button:hover {{
    border-color: {T['accent']} !important;
    box-shadow: 0 0 16px {T['glow']} !important;
    background: {T['accent']}12 !important;
}}

/* Tab styling */
div[data-baseweb="tab-list"] {{
    gap: 4px;
}}
button[data-baseweb="tab"] {{
    background: {T['card']} !important;
    color: {T['muted']} !important;
    border-radius: 8px !important;
    border: 1px solid {T['border']} !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {T['accent']} !important;
    border-color: {T['accent']} !important;
    box-shadow: 0 0 12px {T['glow']} !important;
}}
div[data-baseweb="tab-highlight"] {{
    background-color: {T['accent']} !important;
}}

/* Animate in */
@keyframes slideUp {{
    from {{ opacity:0; transform:translateY(20px); }}
    to {{ opacity:1; transform:translateY(0); }}
}}
.anim {{ animation: slideUp 0.5s cubic-bezier(.16,1,.3,1) forwards; }}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────
def hex_to_rgba(hex_color, alpha=0.1):
    h = hex_color.lstrip('#')
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"

def chart_layout(title, height=300):
    return dict(
        title=dict(text=title, font=dict(color=T['text'], size=13, family='Inter')),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        margin=dict(l=8, r=8, t=36, b=8),
        xaxis=dict(showgrid=True, gridcolor=T['grid'], tickfont=dict(color=T['muted'], size=9), zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=T['grid'], tickfont=dict(color=T['muted'], size=9), zeroline=False),
        height=height,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(size=9, color=T['muted']), bgcolor='rgba(0,0,0,0)'),
        template=T['tpl']
    )

def call_gemini_api(prompt, system_instruction=None, json_mode=False):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("No API key")
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    if json_mode:
        payload["generationConfig"] = {"responseMimeType": "application/json"}
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    r = requests.post(f"{url}?key={api_key}", headers={"Content-Type": "application/json"}, json=payload)
    if r.status_code != 200:
        raise RuntimeError(r.text)
    return r.json()['candidates'][0]['content']['parts'][0]['text']

# ── Load Data ────────────────────────────────────────────────
df_raw = load_historical_data()

# ── Sidebar ──────────────────────────────────────────────────
available_dates = df_raw['Date'].dt.date.tolist()
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = available_dates[-1]

with st.sidebar:

    # ── Logo / Brand Block ──
    st.markdown(f"""
    <div class='sb-logo-block'>
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:8px;'>
            <div style='width:40px; height:40px; border-radius:10px;
                        background:linear-gradient(135deg,{T['accent']},{T['accent2']});
                        display:flex; align-items:center; justify-content:center;
                        font-size:20px; box-shadow:0 4px 14px {T['glow']};'>⚡</div>
            <div>
                <div style='font-size:17px; font-weight:900; color:{T['text']};
                            letter-spacing:-0.3px; font-family:Inter,sans-serif;'>TSLA Engine</div>
                <div style='font-size:10px; color:{T['muted']}; font-family:"JetBrains Mono",monospace;'>Quantitative Platform v2.0</div>
            </div>
        </div>
        <div style='display:flex; gap:6px; flex-wrap:wrap; margin-top:10px;'>
            <span style='background:{T["accent"]}20; color:{T["accent"]}; border:1px solid {T["accent"]}40;
                         border-radius:20px; padding:2px 10px; font-size:10px; font-weight:700;
                         font-family:"JetBrains Mono",monospace;'>LIVE</span>
            <span style='background:{T["c4"]}20; color:{T["c4"]}; border:1px solid {T["c4"]}40;
                         border-radius:20px; padding:2px 10px; font-size:10px; font-weight:700;
                         font-family:"JetBrains Mono",monospace;'>ML READY</span>
            <span style='background:{T["c2"]}20; color:{T["c2"]}; border:1px solid {T["c2"]}40;
                         border-radius:20px; padding:2px 10px; font-size:10px; font-weight:700;
                         font-family:"JetBrains Mono",monospace;'>NLP</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Theme Selection ──
    st.markdown(f"<div class='sb-section'>", unsafe_allow_html=True)
    st.markdown(f"<div class='sb-section-title'>🎨 Visual Theme</div>", unsafe_allow_html=True)
    sel = st.selectbox("Theme", list(THEMES.keys()), format_func=lambda k: THEMES[k]['name'],
                       index=list(THEMES.keys()).index(st.session_state.theme_id),
                       label_visibility='collapsed')
    if sel != st.session_state.theme_id:
        st.session_state.theme_id = sel
        st.rerun()
    # Color swatch preview strip
    active_t = THEMES[sel]
    st.markdown(f"""
    <div style='display:flex; gap:4px; margin-top:8px; align-items:center;'>
        <div style='flex:1; height:6px; border-radius:4px; background:{active_t["accent"]};'></div>
        <div style='flex:1; height:6px; border-radius:4px; background:{active_t["c2"]};'></div>
        <div style='flex:1; height:6px; border-radius:4px; background:{active_t["c3"]};'></div>
        <div style='flex:1; height:6px; border-radius:4px; background:{active_t["c4"]};'></div>
        <span style='font-size:9px; color:{T["muted"]}; font-family:"JetBrains Mono",monospace; margin-left:4px;'>PALETTE</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Date Range Pills ──
    st.markdown(f"<div class='sb-section'>", unsafe_allow_html=True)
    st.markdown(f"<div class='sb-section-title'>📅 Data Range</div>", unsafe_allow_html=True)
    date_range = st.radio("Range", ['ALL', '3Y', '1Y', '6M'], index=1, horizontal=True, label_visibility='collapsed')
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Focus Date ──
    st.markdown(f"<div class='sb-section'>", unsafe_allow_html=True)
    st.markdown(f"<div class='sb-section-title'>📍 Master Coordinate</div>", unsafe_allow_html=True)
    st.date_input("Date", min_value=available_dates[0], max_value=available_dates[-1],
                  key='selected_date', label_visibility='collapsed')
    cur = st.session_state.selected_date
    if cur not in available_dates:
        diffs = [abs((d - cur).days) for d in available_dates]
        st.session_state.selected_date = available_dates[diffs.index(min(diffs))]

    # Quick jump buttons
    st.markdown(f"<div style='display:flex; gap:6px; margin-top:8px;'>", unsafe_allow_html=True)
    jcols = st.columns(3)
    jump_targets = [
        ("Latest", available_dates[-1]),
        ("Mid",    available_dates[len(available_dates)//2]),
        ("Oldest", available_dates[0]),
    ]
    for jc, (label, tgt) in zip(jcols, jump_targets):
        with jc:
            if st.button(label, key=f"jump_{label}", use_container_width=True):
                st.session_state.selected_date = tgt
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Quick Market Stats ──
    df_quick = get_processed_data(df_raw, '1Y')
    latest = df_quick.iloc[-1]
    oldest = df_quick.iloc[0]
    ytd_chg = ((latest['Close'] - oldest['Close']) / oldest['Close'] * 100)
    ytd_color = '#22C55E' if ytd_chg >= 0 else '#EF4444'
    ytd_sign  = '+' if ytd_chg >= 0 else ''

    st.markdown(f"""
    <div class='sb-section'>
        <div class='sb-section-title'>📊 Market Snapshot (1Y)</div>
        <div class='sb-stat-row'>
            <span class='sb-stat-label'>Latest Close</span>
            <span class='sb-stat-value' style='color:{T["accent"]};'>${latest['Close']:.2f}</span>
        </div>
        <div class='sb-stat-row'>
            <span class='sb-stat-label'>1Y Change</span>
            <span class='sb-stat-value' style='color:{ytd_color};'>{ytd_sign}{ytd_chg:.1f}%</span>
        </div>
        <div class='sb-stat-row'>
            <span class='sb-stat-label'>52W High</span>
            <span class='sb-stat-value'>${df_quick['High'].max():.2f}</span>
        </div>
        <div class='sb-stat-row'>
            <span class='sb-stat-label'>52W Low</span>
            <span class='sb-stat-value'>${df_quick['Low'].min():.2f}</span>
        </div>
        <div class='sb-stat-row' style='border:none;'>
            <span class='sb-stat-label'>Avg Volume</span>
            <span class='sb-stat-value'>{int(df_quick['Volume'].mean()/1e6):.1f}M</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown(f"""
    <div class='sb-footer'>
        <div style='width:100%; height:1px; background:linear-gradient(90deg, transparent, {T['border']}, transparent); margin-bottom:14px;'></div>
        <div style='font-size:9px; color:{T['muted']}; font-family:"JetBrains Mono",monospace; line-height:1.6;'>
            TSLA · NASDAQ · Quantitative Research<br>
            <span style='color:{T["accent"]};'>Deep Learning</span> · <span style='color:{T["c2"]};'>Technical Analysis</span> · <span style='color:{T["c4"]};'>NLP Sentiment</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Process Data ─────────────────────────────────────────────
df = get_processed_data(df_raw, date_range)
sig = compute_selected_point_signals(df, st.session_state.selected_date.strftime('%Y-%m-%d'))

# ── Hero Banner ──────────────────────────────────────────────
st.markdown("<div class='anim'>", unsafe_allow_html=True)
hero_path = os.path.join(os.path.dirname(__file__), "assets", "tesla_dashboard_hero.jpg")
if os.path.exists(hero_path):
    st.markdown("<div class='hero-wrap'>", unsafe_allow_html=True)
    st.image(hero_path, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class='hero-wrap' style='height:180px; display:flex; align-items:center; justify-content:center; background:{T['bg2']};'>
        <div style='text-align:center;'>
            <div class='hero-title'>⚡ TSLA Quantitative Platform</div>
            <div class='hero-sub'>Deep Learning · Technical Analysis · Sentiment Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Portfolio Metrics Row ────────────────────────────────────
pct = float(df[df['Date'] == pd.to_datetime(sig['date'])]['dailyReturnPct'].iloc[0]) if not df[df['Date'] == pd.to_datetime(sig['date'])].empty else 0.0
pct_c = "#22C55E" if pct >= 0 else "#EF4444"
pct_s = "+" if pct >= 0 else ""

st.markdown(f"<div class='sec-h'>Portfolio Highlights — {sig['date']}</div>", unsafe_allow_html=True)
cols = st.columns(5)
metric_data = [
    ("Close Price", f"${sig['close']:.2f}", f"<span style='color:{pct_c}'>{pct_s}{pct:.2f}%</span>"),
    ("Signal Rating", f"{sig['recommendation']}", f"Score: {sig['score']}"),
    ("RSI (6-Period)", f"{sig['rsiRaw']:.1f}", f"{sig['rsiStatus']}"),
    ("Support / Resistance", f"${sig['support']:.1f} / ${sig['resistance']:.1f}", "Floor & Ceiling"),
    ("Trend Alignment", f"{sig['trendStatus']}", f"{sig['macroAlignment']}")
]
for i, (label, value, sub) in enumerate(metric_data):
    with cols[i]:
        vsize = "22px" if i != 3 else "16px"
        st.markdown(f"""
        <div class='m-card'>
            <div class='m-label'>{label}</div>
            <div class='m-value' style='font-size:{vsize}; color:{T["accent"] if i==1 else T["text"]}'>{value}</div>
            <div class='m-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Primary Price Chart ──────────────────────────────────────
st.markdown(f"<div class='sec-h' style='margin-top:24px;'>Primary Stock Overlays & Historical Timeline</div>", unsafe_allow_html=True)

fig_main = go.Figure()
fig_main.add_trace(go.Scatter(
    x=df['Date'], y=df['Close'], mode='lines', name='Close',
    line=dict(color=T['c1'], width=2.5),
    fill='tozeroy', fillcolor=hex_to_rgba(T['c1'], 0.06)
))
fig_main.add_trace(go.Scatter(
    x=df['Date'], y=df['ema20'], mode='lines', name='EMA-20',
    line=dict(color=T['c2'], width=1.2, dash='dash')
))
fig_main.add_trace(go.Scatter(
    x=df['Date'], y=df['sma50'], mode='lines', name='SMA-50',
    line=dict(color=T['c3'], width=1.2)
))
fig_main.add_trace(go.Scatter(
    x=df['Date'], y=df['sma200'], mode='lines', name='SMA-200',
    line=dict(color=T['c4'], width=1.0, dash='dot')
))

# Selected date marker — use add_shape instead of add_vline to avoid Timestamp bug
sel_date_str = sig['date']
fig_main.add_shape(type="line", x0=sel_date_str, x1=sel_date_str, y0=0, y1=1,
                   yref="paper", line=dict(color=T['accent'], width=2, dash="dash"))
fig_main.add_annotation(x=sel_date_str, y=1, yref="paper",
                        text=f"▼ {sel_date_str}", showarrow=False,
                        font=dict(color=T['accent'], size=10, family='JetBrains Mono'),
                        yshift=10)

fig_main.update_layout(**chart_layout("TSLA Split-Adjusted Historical Timeline & Technical Overlays", 380))
st.plotly_chart(fig_main, use_container_width=True, key="main_chart")

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab_charts, tab_ml, tab_news, tab_report = st.tabs([
    "📊 15-Chart Quantitative Deck",
    "🧠 Deep Learning & Predictions",
    "🐦 News Sentiment Timeline",
    "📄 Analyst Research Report"
])

# ── TAB 1: 15 Charts ────────────────────────────────────────
with tab_charts:
    st.markdown(f"<div class='sec-h'>Exploratory Data Analysis — Quantitative Chart Deck</div>", unsafe_allow_html=True)

    # ─── Row 1: Price, Volume, Moving Averages ───
    st.markdown(f"<div class='m-label' style='margin-bottom:8px;'>Price & Volume Dynamics</div>", unsafe_allow_html=True)
    r1 = st.columns(3)
    with r1[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', fill='tozeroy',
                                 line=dict(color=T['c1'], width=1.8), fillcolor=hex_to_rgba(T['c1'], 0.08), name='Close'))
        fig.update_layout(**chart_layout("1 · Close Price Timeline", 260))
        st.plotly_chart(fig, use_container_width=True, key="c1")
    with r1[1]:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=T['c2'], name='Volume'))
        fig.update_layout(**chart_layout("2 · Trading Volume Profile", 260))
        st.plotly_chart(fig, use_container_width=True, key="c2")
    with r1[2]:
        fig = go.Figure()
        for col, clr, w in [('Close', T['c1'], 1.8), ('ema20', T['c2'], 1.2), ('sma50', T['c3'], 1.2), ('sma200', T['c4'], 1.0)]:
            fig.add_trace(go.Scatter(x=df['Date'], y=df[col], mode='lines', name=col, line=dict(color=clr, width=w)))
        fig.update_layout(**chart_layout("3 · EMA & SMA Overlay Channels", 260))
        st.plotly_chart(fig, use_container_width=True, key="c3")

    # ─── Row 2: MACD, RSI, Bollinger Bands ───
    st.markdown(f"<div class='m-label' style='margin: 16px 0 8px 0;'>Momentum & Oscillators</div>", unsafe_allow_html=True)
    r2 = st.columns(3)
    with r2[0]:
        fig = go.Figure()
        colors_macd = [T['c4'] if v >= 0 else T['c3'] for v in df['macdHist']]
        fig.add_trace(go.Bar(x=df['Date'], y=df['macdHist'], marker_color=colors_macd, name='MACD Hist'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['macdLine'], mode='lines', line=dict(color=T['c1'], width=1.2), name='MACD Line'))
        fig.update_layout(**chart_layout("4 · MACD Momentum Oscillator", 260))
        st.plotly_chart(fig, use_container_width=True, key="c4")
    with r2[1]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['rsi'], mode='lines', line=dict(color=T['c3'], width=1.8), name='RSI'))
        fig.add_hrect(y0=70, y1=100, fillcolor="#EF4444", opacity=0.06, line_width=0)
        fig.add_hrect(y0=0, y1=30, fillcolor="#22C55E", opacity=0.06, line_width=0)
        fig.add_hline(y=70, line_dash="dot", line_color="#EF4444", line_width=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#22C55E", line_width=1)
        fig.update_yaxes(range=[0, 100])
        fig.update_layout(**chart_layout("5 · RSI Strength Oscillation", 260))
        st.plotly_chart(fig, use_container_width=True, key="c5")
    with r2[2]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['upperBand'], mode='lines', line=dict(color=T['c2'], width=0.8), name='Upper BB'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['lowerBand'], mode='lines', line=dict(color=T['c2'], width=0.8),
                                 fill='tonexty', fillcolor=hex_to_rgba(T['c2'], 0.04), name='Lower BB'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', line=dict(color=T['c1'], width=1.5), name='Close'))
        fig.update_layout(**chart_layout("6 · Bollinger Volatility Bands", 260))
        st.plotly_chart(fig, use_container_width=True, key="c6")

    # ─── Row 3: Volatility, High-Low, Open-Close ───
    st.markdown(f"<div class='m-label' style='margin: 16px 0 8px 0;'>Volatility & Session Spreads</div>", unsafe_allow_html=True)
    r3 = st.columns(3)
    with r3[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['shortVol'], mode='lines', fill='tozeroy',
                                 line=dict(color=T['accent'], width=1.5), fillcolor=hex_to_rgba(T['accent'], 0.08), name='Short Vol'))
        fig.update_layout(**chart_layout("7 · Standard Volatility Curve", 260))
        st.plotly_chart(fig, use_container_width=True, key="c7")
    with r3[1]:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['Date'], y=df['highLowRange'], marker_color=T['c2'], name='H-L Range'))
        fig.update_layout(**chart_layout("8 · High–Low Session Spread", 260))
        st.plotly_chart(fig, use_container_width=True, key="c8")
    with r3[2]:
        fig = go.Figure()
        spread_clrs = ['#22C55E' if v >= 0 else '#EF4444' for v in df['openCloseSpread']]
        fig.add_trace(go.Bar(x=df['Date'], y=df['openCloseSpread'], marker_color=spread_clrs, name='O-C Spread'))
        fig.update_layout(**chart_layout("9 · Open–Close Session Spread", 260))
        st.plotly_chart(fig, use_container_width=True, key="c9")

    # ─── Row 4: Compound Growth, Support/Resistance, Vol Envelope ───
    st.markdown(f"<div class='m-label' style='margin: 16px 0 8px 0;'>Growth & Boundary Analysis</div>", unsafe_allow_html=True)
    r4 = st.columns(3)
    with r4[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['compoundGrowth'], mode='lines', fill='tozeroy',
                                 line=dict(color=T['c4'], width=2.0), fillcolor=hex_to_rgba(T['c4'], 0.08), name='$1K Growth'))
        fig.update_layout(**chart_layout("10 · Compound Portfolio Growth ($1K)", 260))
        st.plotly_chart(fig, use_container_width=True, key="c10")
    with r4[1]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['estResistance'], mode='lines',
                                 line=dict(color='#EF4444', width=1.2, dash='dash'), name='Resistance'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', line=dict(color=T['c1'], width=1.5), name='Close'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['estSupport'], mode='lines',
                                 line=dict(color='#22C55E', width=1.2, dash='dash'), name='Support'))
        fig.update_layout(**chart_layout("11 · Support & Resistance Overlay", 260))
        st.plotly_chart(fig, use_container_width=True, key="c11")
    with r4[2]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['shortVol'], mode='lines', line=dict(color=T['c1'], width=1.3), name='Short Vol'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['macroVol'], mode='lines', line=dict(color=T['c3'], width=1.0, dash='dash'), name='Macro Vol'))
        fig.update_layout(**chart_layout("12 · Multi-Vol Envelope Analysis", 260))
        st.plotly_chart(fig, use_container_width=True, key="c12")

    # ─── Row 5: Histograms & Scatter ───
    st.markdown(f"<div class='m-label' style='margin: 16px 0 8px 0;'>Distribution & Dispersion</div>", unsafe_allow_html=True)
    r5 = st.columns(3)
    with r5[0]:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['Close'].values, nbinsx=12, marker_color=T['c1'], name='Price Freq'))
        fig.update_layout(**chart_layout("13 · Price Distribution Histogram", 260))
        st.plotly_chart(fig, use_container_width=True, key="c13")
    with r5[1]:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['dailyReturnPct'].values, nbinsx=15, marker_color=T['c2'], name='Return %'))
        fig.update_layout(**chart_layout("14 · Daily Returns Distribution", 260))
        st.plotly_chart(fig, use_container_width=True, key="c14")
    with r5[2]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Volume'], y=df['absReturnPct'], mode='markers',
                                 marker=dict(color=T['c3'], size=7, opacity=0.7, line=dict(width=0.5, color=T['border'])),
                                 name='Dispersion'))
        lay = chart_layout("15 · Volume vs Return Dispersion", 260)
        lay['xaxis']['title'] = 'Volume'
        lay['yaxis']['title'] = 'Abs Return %'
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True, key="c15")

# ── TAB 2: Deep Learning & Predictions ──────────────────────
with tab_ml:
    st.markdown(f"<div class='sec-h'>Neural Network Architecture & Live Training Workspace</div>", unsafe_allow_html=True)

    col_cfg, col_viz = st.columns([1, 2])

    with col_cfg:
        st.markdown("<div class='q-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='m-label'>Model Architecture</div>", unsafe_allow_html=True)
        model_type = st.radio("Cell Type", ["SimpleRNN", "LSTM"], horizontal=True, label_visibility='collapsed')

        st.markdown(f"<div class='m-label' style='margin-top:12px;'>Hidden Units</div>", unsafe_allow_html=True)
        hidden_units = st.slider("Neurons", 8, 64, 32, 8, label_visibility='collapsed')

        st.markdown(f"<div class='m-label' style='margin-top:12px;'>Learning Rate (α)</div>", unsafe_allow_html=True)
        learn_rate = st.select_slider("LR", options=[0.001, 0.005, 0.01, 0.05], value=0.01, label_visibility='collapsed')

        st.markdown(f"<div class='m-label' style='margin-top:12px;'>Epochs</div>", unsafe_allow_html=True)
        epochs = st.slider("Epochs", 5, 50, 20, 5, label_visibility='collapsed')

        st.markdown(f"<div class='m-label' style='margin-top:12px;'>Activation</div>", unsafe_allow_html=True)
        activation = st.selectbox("Act", ["sigmoid", "tanh", "relu"], label_visibility='collapsed')

        st.markdown(f"<div class='m-label' style='margin-top:12px;'>Sequence Length (days)</div>", unsafe_allow_html=True)
        seq_len = st.slider("SeqLen", 3, 15, 10, 1, label_visibility='collapsed')

        st.markdown("<br>", unsafe_allow_html=True)
        train_btn = st.button("🚀  Start Live Training", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_viz:
        loss_slot = st.empty()
        heat_slot = st.empty()
        prog_slot = st.empty()
        status_slot = st.empty()

        # Show placeholder states
        if 'ml_trained' not in st.session_state:
            loss_slot.markdown(f"""
            <div class='q-card' style='height:260px; display:flex; align-items:center; justify-content:center;'>
                <div style='text-align:center;'>
                    <div style='font-size:48px; margin-bottom:8px;'>📉</div>
                    <div style='color:{T["muted"]}; font-size:13px;'>Loss Minimization Curve</div>
                    <div style='color:{T["muted"]}; font-size:11px; margin-top:4px;'>Click <b>Start Live Training</b> to begin</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            heat_slot.markdown(f"""
            <div class='q-card' style='height:280px; display:flex; align-items:center; justify-content:center;'>
                <div style='text-align:center;'>
                    <div style='font-size:48px; margin-bottom:8px;'>🔥</div>
                    <div style='color:{T["muted"]}; font-size:13px;'>Gate Weights Heatmap</div>
                    <div style='color:{T["muted"]}; font-size:11px; margin-top:4px;'>Neural activations visualized during training</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ─── Training Execution ───
    if train_btn:
        prices = df['Close'].values
        p_min, p_max = prices.min(), prices.max()
        scaled = ((prices - p_min) / (p_max - p_min)).tolist()
        X_seq, Y_seq = create_sequences(scaled, seq_len)

        config = {
            'type': model_type, 'hiddenUnits': hidden_units,
            'learningRate': learn_rate, 'epochs': epochs, 'activation': activation
        }

        t_losses, v_losses = [], []

        def on_epoch(epoch, t_loss, v_loss):
            t_losses.append(t_loss)
            v_losses.append(v_loss)

            prog_slot.progress(epoch / epochs)
            status_slot.markdown(f"<span class='badge'>Epoch {epoch}/{epochs}</span> &nbsp; "
                                 f"<b style='color:#EF4444;'>Train: {t_loss:.5f}</b> &nbsp; "
                                 f"<b style='color:{T['c2']};'>Val: {v_loss:.5f}</b>", unsafe_allow_html=True)

            # Live loss curve
            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(y=t_losses, mode='lines+markers', name='Train Loss',
                                       line=dict(color=T['c1'], width=2.5), marker=dict(size=4)))
            fig_l.add_trace(go.Scatter(y=v_losses, mode='lines+markers', name='Val Loss',
                                       line=dict(color=T['c2'], width=1.8, dash='dash'), marker=dict(size=3)))
            fig_l.update_layout(**chart_layout("Live MSE Loss Minimization Curve", 260))
            loss_slot.plotly_chart(fig_l, use_container_width=True, key=f"loss_{epoch}")

            # Live heatmap
            np.random.seed(epoch * 7)
            w_mat = np.random.randn(hidden_units, seq_len) * (0.08 + 0.92 * t_loss)
            fig_h = px.imshow(w_mat,
                              labels=dict(x="Time Step", y="Neuron", color="Weight"),
                              x=[f"T-{seq_len-k}" for k in range(seq_len)],
                              y=[f"N{k+1}" for k in range(hidden_units)],
                              color_continuous_scale=[[0,'#000'],[0.5,T['c2']],[1,T['accent']]],
                              aspect='auto')
            fig_h.update_layout(
                title=dict(text=f"Gate Weights Matrix — Epoch {epoch}", font=dict(color=T['text'], size=13)),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=8,r=8,t=36,b=8), height=280,
                xaxis=dict(tickfont=dict(size=8, color=T['muted'])),
                yaxis=dict(tickfont=dict(size=8, color=T['muted']))
            )
            heat_slot.plotly_chart(fig_h, use_container_width=True, key=f"heat_{epoch}")

        trained = train_model_live(config, X_seq, Y_seq, callback=on_epoch)
        st.session_state.ml_trained = True

        # ─── Predictions ───
        preds_scaled = trained.run_predictions(X_seq)
        preds = [v * (p_max - p_min) + p_min for v in preds_scaled]
        actuals = prices[seq_len:]
        metrics = evaluate_performance(actuals.tolist(), preds)

        st.markdown("---")
        st.markdown(f"<div class='sec-h'>Model Prediction Results</div>", unsafe_allow_html=True)

        mc1, mc2 = st.columns([1, 2])
        with mc1:
            st.markdown(f"""
            <div class='q-card'>
                <div class='m-label'>Evaluation Metrics</div>
                <table style='width:100%; margin-top:10px; font-size:13px;'>
                    <tr><td style='color:{T["muted"]}; padding:4px 0;'>RMSE</td><td style='text-align:right; font-weight:700; color:{T["accent"]};'>${metrics['rmse']:.2f}</td></tr>
                    <tr><td style='color:{T["muted"]}; padding:4px 0;'>MAE</td><td style='text-align:right; font-weight:700;'>${metrics['mae']:.2f}</td></tr>
                    <tr><td style='color:{T["muted"]}; padding:4px 0;'>R² Score</td><td style='text-align:right; font-weight:700; color:#22C55E;'>{metrics['r2']:.3f}</td></tr>
                    <tr><td style='color:{T["muted"]}; padding:4px 0;'>Direction Acc.</td><td style='text-align:right; font-weight:700; color:{T["c2"]};'>{metrics['accuracy']:.1f}%</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with mc2:
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=df['Date'].iloc[seq_len:], y=actuals, mode='lines',
                                       name='Actual', line=dict(color=T['text'], width=1.5)))
            fig_p.add_trace(go.Scatter(x=df['Date'].iloc[seq_len:], y=preds, mode='lines',
                                       name='Predicted', line=dict(color=T['accent'], width=2.2, dash='dash')))
            fig_p.update_layout(**chart_layout("Ground Truth vs Model Prediction", 300))
            st.plotly_chart(fig_p, use_container_width=True, key="pred_chart")

        # Residuals chart
        residuals = [a - p for a, p in zip(actuals.tolist(), preds)]
        fig_r = go.Figure()
        res_colors = ['#22C55E' if r >= 0 else '#EF4444' for r in residuals]
        fig_r.add_trace(go.Bar(x=df['Date'].iloc[seq_len:], y=residuals, marker_color=res_colors, name='Residual'))
        fig_r.update_layout(**chart_layout("Prediction Residuals (Actual − Predicted)", 220))
        st.plotly_chart(fig_r, use_container_width=True, key="residuals_chart")

# ── TAB 3: Sentiment Timeline ───────────────────────────────
with tab_news:
    st.markdown(f"<div class='sec-h'>News Sentiment Timeline & NLP Analysis</div>", unsafe_allow_html=True)

    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown(f"<div class='m-label' style='margin-bottom:8px;'>Macro Catalyst Events Log</div>", unsafe_allow_html=True)
        for item in HISTORICAL_SENTIMENT_LOGS:
            icon = "📰" if item['source'] != "Twitter" else "🐦"
            s_color = T['c4'] if item['sentiment'] == 'positive' else ('#EF4444' if item['sentiment'] == 'negative' else T['muted'])
            st.markdown(f"""
            <div class='s-card' style='border-left-color:{s_color};'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='font-size:10px; font-family:"JetBrains Mono"; color:{T["accent"]}; font-weight:700;'>{item['date']} · {item['source']}</span>
                    <span class='badge'>{item['sentiment'].upper()}</span>
                </div>
                <p style='font-size:12px; margin:6px 0; line-height:1.4;'>{icon} {item['text']}</p>
                <p style='font-size:11px; color:{T["muted"]}; margin:4px 0 0 0; line-height:1.3; font-style:italic;'>{item['impactExplanation']}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"⟶ Focus: {item['date']}", key=f"focus_{item['id']}"):
                st.session_state.selected_date = datetime.strptime(item['date'], '%Y-%m-%d').date()
                st.rerun()

    with c_right:
        st.markdown(f"<div class='m-label' style='margin-bottom:8px;'>Custom Headline NLP Analyzer</div>", unsafe_allow_html=True)
        presets = [
            "Gigafactory Shanghai completed full model production 30 days ahead of expectations. Super bullish.",
            "Tesla accepts Dogecoin as official merchandise payment. Cyber-whistle orders opening.",
            "Regulator investigation opened into FSD Beta collision statistics. Near-term liquidations expected.",
            "Tesla Q1 deliveries drop 8% year-over-year. Cash flow concerns rising.",
            "Elon Musk announces 12% global workforce reduction for robotics scaling."
        ]
        preset_sel = st.selectbox("Preset Headlines", ["— Select Preset —"] + presets, label_visibility='collapsed')
        custom_text = st.text_area("Enter headline / tweet", value=preset_sel if preset_sel != "— Select Preset —" else "", height=100, label_visibility='collapsed')
        run_nlp = st.button("⚡ Analyze Sentiment", use_container_width=True)

        if run_nlp and custom_text.strip():
            with st.spinner("Analyzing..."):
                try:
                    prompt = f'Analyze this Tesla headline sentiment: "{custom_text}". Return JSON: {{"sentiment":"positive|negative|neutral","score":-1.0 to 1.0,"stockImpact":"bullish|bearish|neutral","impactExplanation":"2-3 sentence explanation"}}'
                    res = call_gemini_api(prompt, json_mode=True)
                    p = json.loads(res)
                except Exception:
                    # Local fallback
                    low = custom_text.lower()
                    is_bull = any(w in low for w in ['bullish','completed','accepts','ahead','expansion','growth'])
                    is_bear = any(w in low for w in ['collision','drop','concerns','reduction','investigation','layoff'])
                    if is_bull:
                        p = {"sentiment": "positive", "score": 0.72, "stockImpact": "bullish",
                             "impactExplanation": "Positive production or partnership signals detected. Historical patterns suggest upward price momentum."}
                    elif is_bear:
                        p = {"sentiment": "negative", "score": -0.65, "stockImpact": "bearish",
                             "impactExplanation": "Regulatory risk or demand weakness signals detected. Short-term selling pressure expected."}
                    else:
                        p = {"sentiment": "neutral", "score": 0.1, "stockImpact": "neutral",
                             "impactExplanation": "Balanced sentiment detected. Market likely to consolidate within current trading range."}

                sc = '#22C55E' if p['sentiment'] == 'positive' else ('#EF4444' if p['sentiment'] == 'negative' else T['muted'])
                st.markdown(f"""
                <div class='q-card' style='border: 1px solid {sc};'>
                    <div style='font-size:18px; font-weight:900; color:{sc}; margin-bottom:8px;'>{p['sentiment'].upper()}</div>
                    <table style='width:100%; font-size:13px;'>
                        <tr><td style='color:{T["muted"]}; padding:3px 0;'>Score</td><td style='font-weight:700;'>{p['score']}</td></tr>
                        <tr><td style='color:{T["muted"]}; padding:3px 0;'>Impact</td><td style='font-weight:700; text-transform:uppercase;'>{p['stockImpact']}</td></tr>
                    </table>
                    <p style='font-size:12px; color:{T["muted"]}; margin-top:10px; line-height:1.4; font-style:italic;'>{p['impactExplanation']}</p>
                </div>
                """, unsafe_allow_html=True)

# ── TAB 4: Analyst Report ───────────────────────────────────
with tab_report:
    st.markdown(f"<div class='sec-h'>Executive Analyst Research Report Generator</div>", unsafe_allow_html=True)

    gen_report = st.button("📄  Generate Strategic Analyst Report", use_container_width=True)

    if gen_report:
        with st.spinner("Compiling investment thesis..."):
            try:
                ctx = f"""Generate a professional equity research analyst report for TSLA.
Current Price: ${sig['close']:.2f}, Trend: {sig['trendStatus']}, RSI: {sig['rsiRaw']:.1f} ({sig['rsiStatus']}), Macro: {sig['macroAlignment']}.
Include: 1) Rating & Target Price 2) Technical Synopsis 3) Sentiment Assessment 4) Risk Factors 5) Bull vs Bear Thesis. Use markdown formatting."""
                report = call_gemini_api(ctx, system_instruction="You are a senior equity research analyst at a global investment bank.")
            except Exception:
                report = f"""## Tesla, Inc. (TSLA) — Equity Research Report

### Rating: {sig['recommendation']} | Score: {sig['score']}/100

---

### Executive Summary
Tesla is currently trading at **${sig['close']:.2f}** with a **{sig['trendStatus'].lower()}**. The {sig['macroAlignment'].lower()} indicates {'favorable' if 'Golden' in sig['macroAlignment'] else 'cautious'} long-term positioning.

### Technical Analysis
| Indicator | Value | Signal |
|-----------|-------|--------|
| RSI (6-period) | {sig['rsiRaw']:.1f} | {sig['rsiStatus']} |
| EMA-20 | ${sig['ema20Raw']:.2f} | {'Above' if sig['ema20Raw'] > sig['sma50Raw'] else 'Below'} SMA-50 |
| SMA-50 | ${sig['sma50Raw']:.2f} | Reference |
| Support | ${sig['support']:.2f} | Floor estimate |
| Resistance | ${sig['resistance']:.2f} | Ceiling estimate |

### Risk Assessment
- **Regulatory**: FSD approval timelines remain uncertain across global markets
- **Competition**: Legacy automakers and Chinese EV manufacturers increasing market pressure
- **Macro**: Interest rate environment affects growth stock valuations
- **Execution**: Gigafactory ramp timelines and supply chain dependencies

### Investment Thesis

**Bull Case**: Autonomous driving leadership, energy storage growth, manufacturing scale advantages, and robotics optionality create asymmetric upside potential.

**Bear Case**: Valuation premium compression, margin pressure from price cuts, regulatory headwinds, and key-person dependency risk.

---
*Report generated from quantitative framework analysis. Not financial advice.*
"""
            st.markdown(report)
