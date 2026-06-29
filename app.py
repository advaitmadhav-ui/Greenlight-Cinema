import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import json
import uuid

# IMPORT YOUR REAL MULTI-AGENT ENGINE
from agent_engine import build_graph, quick_revise

# ==========================================
# 0. ICON LIBRARY (inline SVG, no emoji/images)
# ==========================================
# Minimal line-icon set in the Feather/Lucide style. Rendered as raw SVG so
# they inherit the theme via currentColor and stay crisp at any size.
_ICONS = {
    "clapper": '<path d="M2.5 9.5 L21.5 9.5 L20 20.5 L4 20.5 Z"/><path d="M3.2 9.5 L4.5 4 L9 4 L7.7 9.5"/><path d="M9.5 9.5 L10.8 4 L15.3 4 L14 9.5"/><path d="M15.8 9.5 L17.1 4 L21 4 L21.5 9.5"/>',
    "chart": '<path d="M4 19 L4 5"/><path d="M4 19 L20 19"/><rect x="7" y="12" width="2.6" height="7" rx="0.5"/><rect x="11.7" y="8" width="2.6" height="11" rx="0.5"/><rect x="16.4" y="14.5" width="2.6" height="4.5" rx="0.5"/>',
    "chat": '<path d="M21 11.5a8.38 8.38 0 0 1-1.9 5.4 8.5 8.5 0 0 1-6.6 3.1 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 3.1-6.6 8.38 8.38 0 0 1 5.4-1.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>',
    "plus": '<path d="M12 5 L12 19"/><path d="M5 12 L19 12"/>',
    "pin": '<path d="M12 17 L12 22"/><path d="M7.5 14.5 L16.5 14.5 L15 9.5 L17 7 L15.5 5.5 L8.5 5.5 L7 7 L9 9.5 Z"/>',
    "pin_off": '<path d="M2 2 L22 22"/><path d="M9 9 L7 11.5 L9 14 L15 14"/><path d="M12 17 L12 22"/><path d="M15 5.5 L8.5 5.5 L7 7 L8 8.2"/><path d="M17 7 L15.5 5.5"/>',
    "trash": '<path d="M4 7 L20 7"/><path d="M9 7 L9 4.5 C9 4 9.4 3.5 10 3.5 L14 3.5 C14.6 3.5 15 4 15 4.5 L15 7"/><path d="M6.5 7 L7.3 19.5 C7.35 20.3 8 21 8.8 21 L15.2 21 C16 21 16.65 20.3 16.7 19.5 L17.5 7"/><path d="M10.3 11 L10.3 17"/><path d="M13.7 11 L13.7 17"/>',
    "filter": '<path d="M4 5 L20 5 L13.5 12.5 L13.5 18.5 L10.5 20 L10.5 12.5 Z"/>',
    "doc": '<path d="M7 3.5 L15 3.5 L18 6.5 L18 20.5 L7 20.5 Z"/><path d="M15 3.5 L15 6.5 L18 6.5"/><path d="M9.5 11 L15.5 11"/><path d="M9.5 14 L15.5 14"/><path d="M9.5 17 L13 17"/>',
    "arrow_right": '<path d="M4 12 L20 12"/><path d="M14 6 L20 12 L14 18"/>',
}

def icon(name, size=16, color="currentColor", stroke=1.8, extra_style=""):
    """Render an inline SVG line-icon as an HTML string (no emoji, no network fetch)."""
    body = _ICONS.get(name, "")
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:-3px;{extra_style}">{body}</svg>'
    )

# ==========================================
# 1. GLOBAL CONFIG & INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="Greenlight Cinema - AI Script Engine",
    page_icon=":clapper:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------
# STATE MANAGEMENT (UPGRADED FOR CHAT HISTORY)
# ------------------------------------------
if "chats" not in st.session_state:
    # Initialize the master dictionary to hold all chat sessions
    default_id = str(uuid.uuid4())
    st.session_state.chats = {
        default_id: {
            "title": "New Pitch", 
            "messages": [], 
            "pinned": False, 
            "updated_at": time.time(),
            "latest_state": None  # Added to track raw AgentState
        }
    }
    st.session_state.current_chat_id = default_id

# Helper to easily grab the active chat
def get_current_chat():
    return st.session_state.chats[st.session_state.current_chat_id]

# ==========================================
# 2. CINEMATIC THEME ENGINE (Streaming / Marquee style)
# ==========================================
def apply_theme():
    font_imports = "@import url('https://fonts.googleapis.com/css2?family=Anton&family=Archivo+Black&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');"

    st.markdown(f"""
        <style>
        {font_imports}

        :root {{
            --bg: #0A0A0A;
            --bg-raised: #141414;
            --bg-card: #181818;
            --bg-hover: #232323;
            --border: #2A2A2A;
            --border-soft: #1F1F1F;
            --text-primary: #F5F5F1;
            --text-secondary: #B3B3B3;
            --text-muted: #6E6E6E;
            --accent: #E50914;
            --accent-hover: #FF1A28;
            --accent-dim: rgba(229, 9, 20, 0.14);
            --accent-border: rgba(229, 9, 20, 0.35);
            --gold: #D4AF37;
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.35);
            --shadow-md: 0 10px 28px rgba(0,0,0,0.45);
            --radius: 10px;
        }}

        * {{ scrollbar-width: thin; scrollbar-color: var(--border) transparent; }}
        ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        ::-webkit-scrollbar-thumb {{ background-color: var(--border); border-radius: 8px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}

        .stApp, .stApp > header, [data-testid="stBottom"] > div {{
            background-color: var(--bg) !important;
            background-image: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(229,9,20,0.08), transparent);
        }}

        header[data-testid="stHeader"] {{
            background: transparent !important;
            background-image: none !important;
            box-shadow: none !important;
            height: 2.5rem !important;
            overflow: visible !important;
        }}
        [data-testid="stToolbar"] {{
            background: transparent !important;
        }}
        [data-testid="stDecoration"] {{ display: none !important; }}
        #MainMenu {{ visibility: hidden !important; }}

        [data-testid="stExpandSidebarButton"] {{
            visibility: visible !important;
            opacity: 1 !important;
            display: flex !important;
            z-index: 999999 !important;
            pointer-events: auto !important;
        }}
        [data-testid="stExpandSidebarButton"] svg {{
            fill: var(--text-primary) !important;
        }}

        [data-testid="stAppViewContainer"] > .main .block-container {{
            padding-top: 2.5rem;
        }}
        [data-testid="stSidebar"] {{
            background-color: var(--bg-raised) !important;
            border-right: 1px solid var(--border-soft) !important;
        }}
        p, label, li, span {{ color: var(--text-secondary) !important; font-family: 'Inter', sans-serif; }}

        h1 {{
            color: var(--text-primary) !important;
            font-family: 'Anton', sans-serif;
            letter-spacing: 1.2px;
            text-transform: uppercase;
            font-size: 38px !important;
            margin-bottom: 4px !important;
        }}
        h2, h3, h4, h5, h6 {{
            color: var(--text-primary) !important;
            font-family: 'Anton', sans-serif;
            letter-spacing: 0.8px;
            text-transform: uppercase;
        }}

        .accent-header {{
            border-left: 3px solid var(--accent);
            padding-left: 14px;
            margin-top: 28px;
            margin-bottom: 20px;
            font-family: 'Anton', sans-serif;
            font-size: 19px;
            color: var(--text-primary);
            letter-spacing: 1.2px;
            text-transform: uppercase;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .accent-header svg {{ color: var(--accent); flex-shrink: 0; }}

        .marquee-brand {{
            font-family: 'Anton', sans-serif;
            font-size: 30px;
            letter-spacing: 2px;
            color: var(--text-primary);
            text-transform: uppercase;
            line-height: 1.1;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .marquee-brand .accent-letter {{ color: var(--accent); }}
        .marquee-brand svg {{ color: var(--accent); flex-shrink: 0; }}

        div[data-baseweb="select"] > div, .stTextArea textarea {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            color: var(--text-primary) !important;
            border-radius: 6px;
            transition: border-color 0.15s ease;
        }}
        div[data-baseweb="select"] > div:hover, .stTextArea textarea:hover {{
            border-color: var(--text-muted) !important;
        }}
        div[data-baseweb="select"] > div:focus-within, .stTextArea textarea:focus {{
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 1px var(--accent-border) !important;
        }}
        div[data-baseweb="select"] * {{ color: var(--text-primary) !important; font-family: 'Inter', sans-serif; }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            border: 1px solid var(--border) !important;
            border-radius: var(--radius);
            background-color: var(--bg-card) !important;
            padding: 16px;
            box-shadow: var(--shadow-sm);
        }}
        div[role="radiogroup"] > label {{
            padding: 10px 14px; border-radius: 6px; transition: background 0.15s ease; margin-bottom: 3px;
        }}
        div[role="radiogroup"] > label:hover {{ background-color: var(--bg-hover); }}
        div[data-baseweb="radio"] > div:first-child {{ display: none; }}
        div[role="radiogroup"] p {{
            font-weight: 600; font-size: 13.5px; color: var(--text-primary) !important; font-family: 'Inter', sans-serif;
            text-transform: uppercase; letter-spacing: 0.3px;
        }}

        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {{ background: var(--border) !important; }}
        div[data-testid="stSlider"] [role="slider"] {{ background-color: var(--accent) !important; border: 3px solid var(--text-primary) !important; }}
        div[data-testid="stSliderTickBar"] {{ color: var(--text-muted) !important; font-family: 'JetBrains Mono', monospace; }}
        div[data-testid="stThumbValue"] {{ color: var(--text-primary) !important; font-family: 'JetBrains Mono', monospace !important; background-color: transparent !important; }}

        .stButton>button {{
            background-color: transparent !important;
            color: var(--text-secondary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            transition: all 0.15s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 0.3rem 0.6rem;
        }}
        .stButton>button:hover {{ border-color: var(--accent) !important; color: var(--text-primary) !important; background-color: var(--accent-dim) !important; }}
        .stButton>button:active {{ transform: scale(0.98); }}
        .stButton>button [data-testid="stIconMaterial"] {{
            font-size: 16px !important;
            vertical-align: -3px;
        }}
        .stButton>button:hover [data-testid="stIconMaterial"] {{ color: var(--accent) !important; }}

        /* Compact square icon-only buttons (sidebar pin/delete) */
        .icon-btn-row .stButton>button {{
            padding: 0.35rem !important;
            min-height: 2.1rem;
        }}
        .icon-btn-row .stButton>button p {{ display: none; }}

        [data-testid="stChatMessage"] {{
            background-color: var(--bg-card) !important;
            border-radius: var(--radius);
            padding: 1.2rem 1.3rem;
            margin-bottom: 1rem;
            border: 1px solid var(--border) !important;
            box-shadow: var(--shadow-sm);
        }}
        [data-testid="stChatMessage"]:nth-child(even) {{ background-color: var(--bg-raised) !important; border-color: var(--border) !important; }}
        [data-testid="stChatInput"] {{ background-color: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 12px; }}
        [data-testid="stChatInput"]:focus-within {{ border-color: var(--accent-border) !important; }}
        [data-testid="stChatInputSubmitButton"] {{ background-color: var(--accent) !important; color: white !important; border-radius: 8px; }}
        [data-testid="stChatInputSubmitButton"]:hover {{ background-color: var(--accent-hover) !important; }}

        .custom-table {{
            width: 100%; border-collapse: collapse; font-family: 'Inter', sans-serif; font-size: 14px;
            background-color: var(--bg-card); border-radius: var(--radius); overflow: hidden; border: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
        }}
        .custom-table th {{ background-color: var(--bg-hover); color: var(--text-muted); text-align: left; padding: 13px 18px; font-weight: 700; text-transform: uppercase; font-size: 11.5px; letter-spacing: 0.6px; }}
        .custom-table td {{ padding: 12px 18px; color: var(--text-primary); border-bottom: 1px solid var(--border); }}
        .custom-table tr:last-child td {{ border-bottom: none; }}
        .custom-table tr:hover td {{ background-color: var(--bg-hover); }}
        hr {{ border-color: var(--border) !important; }}

        .kpi-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-top: 3px solid var(--accent);
            border-radius: var(--radius);
            padding: 18px 20px;
            transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
            box-shadow: var(--shadow-sm);
        }}
        .kpi-card:hover {{ transform: translateY(-3px); border-color: var(--text-muted); box-shadow: var(--shadow-md); }}
        .kpi-label {{ font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 700; letter-spacing: 0.6px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px; }}
        .kpi-value {{ font-family: 'Anton', sans-serif; font-size: 27px; color: var(--text-primary); letter-spacing: 0.4px; }}
        .kpi-sub {{ font-family: 'Inter', sans-serif; font-size: 12px; color: var(--text-muted); margin-top: 5px; }}

        .tier-pill {{
            display:inline-block; font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 5px 13px;
            border-radius: 999px; background-color: var(--accent-dim); color: #FF6B6B; border: 1px solid var(--accent-border);
            text-transform: uppercase; letter-spacing: 0.5px;
        }}

        .status-dot {{
            display: inline-block; width: 6px; height: 6px; border-radius: 50%;
            background-color: #7DC4A6; margin-right: 6px; box-shadow: 0 0 6px rgba(125,196,166,0.6);
        }}

        .agent-chip {{
            display:inline-flex; align-items:center; gap:7px; font-family: 'JetBrains Mono', monospace;
            font-size: 11px; padding: 5px 12px; border-radius: 5px; text-transform: uppercase; letter-spacing: 0.6px;
            border: 1px solid var(--border); color: var(--text-secondary); background-color: var(--bg-hover);
            margin-bottom: 10px; margin-right: 6px;
        }}
        .agent-chip::before {{
            content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex-shrink: 0;
        }}
        .agent-chip.writer {{ border-color: rgba(212,175,55,0.4); color: var(--gold); }}
        .agent-chip.critic {{ border-color: var(--accent-border); color: #FF6B6B; }}
        .agent-chip.refiner {{ border-color: rgba(111,168,140,0.4); color: #7DC4A6; }}

        .script-card {{
            background-color: #101010;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 30px 34px;
            font-family: 'JetBrains Mono', monospace;
            color: #E8E6E1;
            line-height: 1.7;
            position: relative;
            box-shadow: var(--shadow-sm);
            background-image:
                repeating-linear-gradient(0deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 3px);
        }}
        .script-card::before {{
            content: "PRODUCTION DRAFT";
            position: absolute; top: 16px; right: 22px;
            font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 1.5px;
            color: var(--text-muted); opacity: 0.55;
        }}
        .script-slug {{ color: var(--accent); font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }}

        .marquee-divider {{
            display:flex; align-items:center; gap:12px; margin: 30px 0 20px 0;
            font-family: 'Anton', sans-serif; font-size: 12px; letter-spacing: 1.8px;
            color: var(--text-muted); text-transform: uppercase;
        }}
        .marquee-divider::before, .marquee-divider::after {{
            content: ""; flex: 1; height: 1px; background: var(--border);
        }}
        </style>
    """, unsafe_allow_html=True)

apply_theme()

# ==========================================
# 3. DATA LOADERS & AI CACHE
# ==========================================
@st.cache_resource
def get_ai_engine():
    return build_graph()

def load_market_data():
    try:
        with open("market_constraints.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("market_constraints.json not found. Please run constraints.py first.")
        return None

def load_face_data():
    try:
        with open("face.constraint", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def call_langgraph_agent(genre, prompt):
    engine = get_ai_engine()
    # Give the writer explicit framing about what the critic will actually grade
    # against (genre ROI fit + budget honesty), so the first draft is more likely
    # to land an ACCEPT instead of cycling through 2-3 rejections before converging.
    master_prompt = (
        f"Write a {genre} movie pitch. "
        f"Core idea: {prompt}\n\n"
        f"Note for the writer: this pitch will be evaluated by a studio executive "
        f"against real market ROI data for the '{genre}' genre. Make sure the scale, "
        f"tone, and stakes of the story honestly match any budget mentioned above "
        f"(or, if no budget is mentioned, keep the scope modest and contained) — "
        f"a mismatch between budget and story scale is the most common reason scripts "
        f"get rejected."
    )
    
    initial_state = {
        "prompt": master_prompt,
        "draft": "", 
        "feedback": "None yet", 
        "revision_count": 0, 
        "status": "PENDING",
        "cast": "",
        "score": None,
        "quarter": None,
    }
    
    final_state = engine.invoke(initial_state)
    return {
        "pitch": final_state["draft"],
        "score": final_state.get("score") or "N/A",
        "quarter": final_state.get("quarter") or "N/A",
        "revisions": final_state["revision_count"],
        "constraints": {
            "passed": [genre, "Market ROI Check"],
            "failed": [] if final_state["status"] == "ACCEPT" else ["Max Revisions Hit"]
        },
        "feedback": final_state["feedback"],
        "cast": final_state.get("cast", "No casting recommendations available."),
        "raw_state": final_state # Added to export complete state map
    }

# ==========================================
# 4. SIDEBAR NAVIGATION & CHAT HISTORY
# ==========================================
def render_sidebar():
    with st.sidebar:
        st.markdown(
            f"<div class='marquee-brand'>{icon('clapper', size=26, stroke=2)}GREEN<span class='accent-letter'>LIGHT</span></div>",
            unsafe_allow_html=True
        )
        st.caption("A I   S C R I P T   S T U D I O")
        st.divider()
        
        # --- PAGE NAVIGATION ---
        st.markdown("<p style='font-size: 11.5px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: var(--text-muted) !important;'>Browse</p>", unsafe_allow_html=True)
        nav_options = {
            ":material/bar_chart: Market Analytics": "Market Analytics",
            ":material/chat: Pitch Chatbot": "Pitch Chatbot",
        }
        nav_choice = st.radio("Go to:", list(nav_options.keys()), label_visibility="collapsed")
        selected_page = nav_options[nav_choice]
        
        st.divider()

        # --- CHAT HISTORY MANAGER ---
        if selected_page == "Pitch Chatbot":
            st.markdown("<p style='font-size: 11.5px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: var(--text-muted) !important;'>Pitch Meetings</p>", unsafe_allow_html=True)
            
            if st.button("NEW PITCH MEETING", use_container_width=True, icon=":material/add:"):
                new_id = str(uuid.uuid4())
                st.session_state.chats[new_id] = {
                    "title": "New Pitch", "messages": [], "pinned": False, "updated_at": time.time(), "latest_state": None
                }
                st.session_state.current_chat_id = new_id
                st.rerun()

            st.write("") 

            sorted_chats = sorted(
                st.session_state.chats.items(),
                key=lambda item: (item[1]["pinned"], item[1]["updated_at"]),
                reverse=True
            )

            for chat_id, chat_data in sorted_chats:
                is_active = (chat_id == st.session_state.current_chat_id)
                row_label = chat_data['title']
                row_icon = ":material/play_arrow:" if is_active else (":material/push_pin:" if chat_data["pinned"] else ":material/movie:")

                st.markdown("<div class='icon-btn-row'>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([0.65, 0.16, 0.19])

                with col1:
                    if st.button(row_label, key=f"sel_{chat_id}", use_container_width=True, icon=row_icon):
                        st.session_state.current_chat_id = chat_id
                        st.rerun()
                with col2:
                    pin_icon = ":material/keep_off:" if chat_data["pinned"] else ":material/push_pin:"
                    if st.button("", key=f"pin_{chat_id}", use_container_width=True, icon=pin_icon, help="Unpin" if chat_data["pinned"] else "Pin"):
                        st.session_state.chats[chat_id]["pinned"] = not chat_data["pinned"]
                        st.rerun()
                with col3:
                    if st.button("", key=f"del_{chat_id}", use_container_width=True, icon=":material/delete:", help="Delete"):
                        del st.session_state.chats[chat_id]
                        if st.session_state.current_chat_id == chat_id:
                            if len(st.session_state.chats) > 0:
                                st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                            else:
                                new_id = str(uuid.uuid4())
                                st.session_state.chats[new_id] = {"title": "New Pitch", "messages": [], "pinned": False, "updated_at": time.time(), "latest_state": None}
                                st.session_state.current_chat_id = new_id
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            st.divider()

        st.markdown(
            "<p style='font-size: 11px; color: var(--text-muted) !important; text-align:center; margin-top: 8px;'>"
            "<span class='status-dot'></span>STUDIO SYSTEMS LIVE</p>",
            unsafe_allow_html=True
        )
                
    return selected_page

# ==========================================
# 5. PAGE: MARKET Analytics
# ==========================================
def render_analytics_tab():
    st.markdown(f"<h1 style='display:flex;align-items:center;gap:12px;'>{icon('chart', size=34, stroke=2.2)}Market Analytics</h1>", unsafe_allow_html=True)
    st.write("Live data extracted from TMDB via DuckDB.")
    st.write("")
    
    market_data = load_market_data()
    if not market_data:
        return

    plotly_template = 'plotly_dark'
    font_color = '#B3B3B3'
    bar_color = '#E50914'
    line_colors = ['#E50914', '#D4AF37', "#7BC4A6", '#6FA8DC', '#B3B3B3']

    top_genres = pd.DataFrame(market_data.get("top_genres", []))

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Genres Ranked</div><div class='kpi-value'>{len(top_genres)}</div><div class='kpi-sub'>by revenue : budget ratio</div></div>", unsafe_allow_html=True)
    with kpi_cols[1]:
        if not top_genres.empty:
            top_row = top_genres.iloc[0]
            st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Top Genre ROI</div><div class='kpi-value' style='color:#E50914;'>{top_row['revenue_to_budget_ratio']}×</div><div class='kpi-sub'>{top_row['genre']} leads the field</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='kpi-card'><div class='kpi-label'>Top Genre ROI</div><div class='kpi-value'>—</div></div>", unsafe_allow_html=True)
    with kpi_cols[2]:
        seasonal_fit_preview = market_data.get("seasonal_fit", {})
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Genres Tracked</div><div class='kpi-value'>{len(seasonal_fit_preview)}</div><div class='kpi-sub'>seasonal ROI series</div></div>", unsafe_allow_html=True)
    with kpi_cols[3]:
        talent_count = len(market_data.get("emerging_actors", [])) + len(market_data.get("top_actors", []))
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Tracked Talent</div><div class='kpi-value'>{talent_count}</div><div class='kpi-sub'>{len(market_data.get('emerging_actors', []))} emerging · {len(market_data.get('top_actors', []))} mainstream</div></div>", unsafe_allow_html=True)

    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            if not top_genres.empty:
                fig_bar = px.bar(top_genres, x='genre', y='revenue_to_budget_ratio', title='Top Profitable Genres by ROI', color_discrete_sequence=[bar_color])
                fig_bar.update_layout(template=plotly_template, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color=font_color), title_font=dict(family="Anton", color="#F5F5F1", size=26))
                st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        with st.container(border=True):
            seasonal_fit = market_data.get("seasonal_fit", {})
            seasonal_rows = []
            for genre, quarters in seasonal_fit.items():
                for quarter, roi in quarters.items():
                    seasonal_rows.append({"Quarter": quarter, "Genre": genre, "ROI": roi})
            
            if seasonal_rows:
                seasonal_df = pd.DataFrame(seasonal_rows).sort_values(by=["Quarter"])
                heatmap_pivot = seasonal_df.pivot(index="Genre", columns="Quarter", values="ROI")
                fig_line = px.imshow(
                    heatmap_pivot,
                    title='Seasonal ROI Trends (Top 5 Genres)',
                    color_continuous_scale=[[0, '#181818'], [0.5, '#8A1014'], [1, '#E50914']],
                    text_auto='.1f',
                    aspect='auto'
                )
                fig_line.update_traces(xgap=4, ygap=4)
                fig_line.update_layout(template=plotly_template, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter", color=font_color), title_font=dict(family="Anton", color="#F5F5F1", size=26), xaxis_title="Quarter", yaxis_title="", coloraxis_colorbar=dict(title="ROI (x)"))
                st.plotly_chart(fig_line, use_container_width=True)

    st.markdown(f"<div class='accent-header'>{icon('chart', size=18)}GENRE-SPECIFIC CASTING OPPORTUNITIES</div>", unsafe_allow_html=True)
    
    face_data = load_face_data()
    all_genres = list(market_data.get("top_actors_by_genre", {}).keys())
    
    if all_genres:
        selected_genre_filter = st.selectbox("Filter Casting Opportunities by Genre:", all_genres, index=0)
        genre_actors = market_data.get("top_actors_by_genre", {}).get(selected_genre_filter, [])
        
        if genre_actors:
            cols = st.columns(3)
            for idx, actor_str in enumerate(genre_actors[:3]):
                if " (" in actor_str:
                    name = actor_str.split(" (")[0]
                    roi = actor_str.split("(")[1].replace("x ROI)", "")
                else:
                    name = actor_str
                    roi = "N/A"
                    
                wiki_link = f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}"
                img_url = face_data.get(name, "https://upload.wikimedia.org/wikipedia/commons/8/89/Portrait_Placeholder.png")
                
                with cols[idx]:
                    st.markdown(f"""
                    <a href="{wiki_link}" target="_blank" style="text-decoration: none;">
                        <div class='kpi-card' style='border-top: 3px solid #7DC4A6; cursor: pointer; padding: 0; overflow: hidden;'>
                            <div style="position:relative;">
                                <img src="{img_url}" style="width: 100%; height: 100%; object-fit: cover; display:block;">
                                <div style="position:absolute; top:10px; right:10px; background: rgba(0,0,0,0.78); border: 1px solid rgba(125,196,166,0.5); border-radius: 5px; padding: 3px 10px; font-family:'JetBrains Mono', monospace; font-size:11px; color:#7DC4A6; font-weight:700; letter-spacing:0.4px;">{roi}x ROI</div>
                            </div>
                            <div style="padding: 14px 16px;">
                                <div class='kpi-label'>{selected_genre_filter} Lead</div>
                                <div class='kpi-value' style='font-size: 18px;'>{name}</div>
                                <div class='kpi-sub' style='color:#7DC4A6; font-weight:600;'>{roi}x Historical ROI</div>
                            </div>
                        </div>
                    </a>
                    """, unsafe_allow_html=True)
        else:
            st.info(f"No specific casting data found for {selected_genre_filter}.")
    else:
        st.info("No genre-specific casting data available in constraints.")

    st.write("")
    with st.expander("Current Live AI Constraints (JSON Payload)", icon=":material/description:"):
        st.json(market_data)

# ==========================================
# 6. PAGE: CHATBOT PITCH GENERATOR
# ==========================================
def render_generator_tab():
    current_chat = get_current_chat()
    
    header_title = "AI Pitch Assistant" if current_chat["title"] == "New Pitch" else current_chat['title']
    st.markdown(f"<h1 style='display:flex;align-items:center;gap:12px;'>{icon('chat', size=32, stroke=2.1)}{header_title}</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: var(--text-secondary) !important; font-size: 15px;'>"
        "Step into the room. Pitch your idea to the Writer, Critic, and Refiner — "
        "three AI agents working the table until the concept earns its greenlight."
        "</p>",
        unsafe_allow_html=True
    )
    
    st.markdown(f"<div class='accent-header'>{icon('clapper', size=18)}STUDIO PARAMETERS</div>", unsafe_allow_html=True)
    with st.container(border=True):
        genre_list = [
            "Action", "Adventure", "Animation", "Comedy", "Crime", 
            "Documentary", "Drama", "Family", "Fantasy", "History", 
            "Horror", "Musical", "Mystery", "Romance", "Sci-Fi Thriller", 
            "Psychological Horror", "Thriller", "War", "Western"
        ]
        genre = st.selectbox("TARGET GENRE", genre_list)
        st.caption("Mention your budget directly in the pitch below (e.g. \"budget of $80 million\") — the room will read it from your prompt.")

    st.markdown(
        "<div class='marquee-divider'>NOW IN THE ROOM</div>",
        unsafe_allow_html=True
    )

    if not current_chat["messages"]:
        st.markdown(
            "<p style='text-align:center; color: var(--text-muted) !important; "
            "font-family: \"JetBrains Mono\", monospace; font-size: 13px; padding: 24px 0;'>"
            "— the room is quiet. pitch your idea below to get started —"
            "</p>",
            unsafe_allow_html=True
        )

    for message in current_chat["messages"]:
        avatar = ":material/person:" if message["role"] == "user" else ":material/movie:"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"], unsafe_allow_html=True)

    # Chat Input Box
    if prompt := st.chat_input("Pitch your core story idea or ask for changes..."):
        
        # Check if this is a brand new pitch or a request to edit the current one
        is_revision = len(current_chat["messages"]) > 0
        
        # Auto-update the chat title only if it's the first message
        if current_chat["title"] == "New Pitch" and not is_revision:
            current_chat["title"] = f"{genre} Pitch"
            
        current_chat["updated_at"] = time.time()
        current_chat["messages"].append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=":material/movie:"):
            if not is_revision:
                # --- RUN FULL ENGINE (NEW PITCH) ---
                with st.status(f"Pitching {genre} to the room...", expanded=True) as status:
                    st.markdown("<span class='agent-chip writer'>WRITER — drafting concept</span>", unsafe_allow_html=True)
                    st.markdown("<span class='agent-chip critic'>CRITIC — stress-testing market fit</span>", unsafe_allow_html=True)
                    st.markdown("<span class='agent-chip refiner'>REFINER — polishing the draft</span>", unsafe_allow_html=True)
                    
                    result = call_langgraph_agent(genre, prompt)
                    current_chat["latest_state"] = result["raw_state"] # SAVE STATE HERE
                    status.update(label="Pitch Finalized — Greenlit by the Room", state="complete", expanded=False)
                
                response_markdown = f"""
<div class="script-card">
<span class="script-slug">FINAL APPROVED PITCH — {genre.upper()}</span>

{result['pitch']}

---

**Casting Recommendations**

{result['cast']}

---

**Executive Validation Report**

* **Score:** {result['score']}
* **Recommended Quarter:** {result['quarter']}
* **Revisions Taken:** {result['revisions']}
* **Constraints Passed:** {len(result['constraints']['passed'])} / 2
* **Failed Checks:** {len(result['constraints']['failed'])}

**Critic Feedback:**

> {result['feedback']}

</div>
"""
            else:
                # --- RUN QUICK REVISION (DYNAMIC SUPERVISOR) ---
                with st.status("Supervisor is routing your request...", expanded=True) as status:
                    
                    # Fetch the saved state, with a fallback just in case
                    state_to_edit = current_chat.get("latest_state")
                    if not state_to_edit:
                        state_to_edit = {
                            "prompt": prompt, "draft": "", "revision_count": 0, 
                            "status": "PENDING", "cast": "", "cast_list": [], "score": None, "quarter": None
                        }

                    # Execute the upgraded routing engine
                    new_state = quick_revise(state_to_edit, prompt)
                    
                    # Save the newly updated state back to the chat history
                    current_chat["latest_state"] = new_state
                    status.update(label="Studio Revisions Complete", state="complete", expanded=False)
                
                # Render the dynamically updated components
                response_markdown = f"""
<div class="script-card">
<span class="script-slug">REVISED PITCH — {genre.upper()}</span>

{new_state['draft']}

---

**Updated Casting Recommendations**

{new_state.get('cast', 'No changes to casting.')}

---

**Executive Validation Report**

* **Score:** {new_state.get('score', 'N/A')}
* **Recommended Quarter:** {new_state.get('quarter', 'N/A')}

**Latest Critic Feedback:**

> {new_state.get('feedback', 'No new critique requested.')}

</div>
"""

            st.markdown(response_markdown, unsafe_allow_html=True)
            current_chat["messages"].append({"role": "assistant", "content": response_markdown})
            st.rerun()

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    selected_page = render_sidebar()
    if selected_page == "Market Analytics":
        render_analytics_tab()
    elif selected_page == "Pitch Chatbot":
        render_generator_tab()

if __name__ == "__main__":
    main()