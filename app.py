"""
VideoGenius AI - Streamlit UI (AI Video Summarizer & Meeting Assistant)

Run with:   streamlit run app.py
Requires:   streamlit >= 1.40

On first run this file creates .streamlit/config.toml (light theme) in the folder
you launch it from, if one doesn't exist yet. Restart the app once after that.
"""

import json
import os
import tempfile
import uuid
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.audio_processors import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

NAVY, SKY, CREAM, PINK = "#425B9A", "#76C0EC", "#FFF6DC", "#FF95A5"

LANGUAGES = {"English": "english", "Hindi": "hindi"}
SUPPORTED_TYPES = ["mp3", "wav", "m4a", "aac", "ogg", "flac", "mp4", "mkv", "mov", "avi", "webm"]
UPLOAD_LIMIT_LABEL = "200MB"

DATA_FILE = Path(__file__).parent / "data" / "history.json"

# True = show 3 sample cards + baseline stats (looks like the design preview).
# Set to False to show only your real data.
DEMO_MODE = True

NAV_ITEMS = [
    ("dashboard", "Dashboard", ":material/dashboard:"),
    ("new", "New Summary", ":material/videocam:"),
    ("library", "My Library", ":material/folder_open:"),
    ("analytics", "Analytics", ":material/bar_chart:"),
    ("settings", "Settings", ":material/settings:"),
]

THEME_TOML = f"""[theme]
base = "light"
primaryColor = "{SKY}"
backgroundColor = "{CREAM}"
secondaryBackgroundColor = "#FFFFFF"
textColor = "{NAVY}"
"""


def ensure_light_theme() -> bool:
    """Create a light-theme config if the project has none. Returns True if created."""
    cfg = Path.cwd() / ".streamlit" / "config.toml"
    if cfg.exists():
        return False
    try:
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(THEME_TOML, encoding="utf-8")
        return True
    except OSError:
        return False


THEME_CREATED = ensure_light_theme()

st.set_page_config(
    page_title="VideoGenius AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# 1. ICONS + HTML HELPERS
# --------------------------------------------------

ICONS = {
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "bulb": '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
    "video": '<path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5"/><rect x="2" y="6" width="14" height="12" rx="2"/>',
    "list": '<path d="m3 17 2 2 4-4"/><path d="m3 7 2 2 4-4"/><path d="M13 6h8"/><path d="M13 12h8"/><path d="M13 18h8"/>',
    "chat": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "zap": '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>',
    "play": '<circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/>',
    "bell": '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/>',
}


def icon(name: str, size: int = 20, color: str = "currentColor") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0">{ICONS[name]}</svg>'
    )


def render_html(markup: str):
    """Render raw HTML (lines are joined so Markdown never treats indentation as code)."""
    flat = "".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


# --------------------------------------------------
# 2. CSS  (forces a light theme so text is always visible)
# --------------------------------------------------

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root { color-scheme: light; }
html, body, .stApp, button, input, textarea, select { font-family: 'Inter', sans-serif !important; }

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #FFF6DC !important; color: #425B9A; }
.block-container { padding: 1.4rem 2.5rem 4rem 2.5rem; max-width: 1400px; }

/* ---- Remove Streamlit chrome (Deploy button, menu, footer) ---- */
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
.stDeployButton, #MainMenu, footer { display: none !important; visibility: hidden !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ---- Text colours ---- */
:where(.stMarkdown, .stMarkdown *, label, label *) { color: #425B9A; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: #425B9A !important; }
[data-testid="stWidgetLabel"] *, [data-testid="stCaptionContainer"] * { color: #425B9A !important; }

/* ---- Inputs (always light, always readable) ---- */
div[data-baseweb="input"], div[data-baseweb="base-input"], div[data-baseweb="textarea"],
div[data-baseweb="select"] > div {
    background: #fff !important; border-radius: 12px !important;
}
div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="select"] > div {
    border: 1px solid rgba(66,91,154,.2) !important;
}
input, textarea {
    color: #425B9A !important; -webkit-text-fill-color: #425B9A !important;
    background: transparent !important; caret-color: #425B9A;
}
input::placeholder, textarea::placeholder {
    color: rgba(66,91,154,.5) !important; -webkit-text-fill-color: rgba(66,91,154,.5) !important; opacity: 1 !important;
}
div[data-baseweb="select"] * { color: #425B9A !important; }
div[data-baseweb="select"] svg { fill: #425B9A !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li, div[data-baseweb="popover"] [role="listbox"] {
    background: #fff !important; color: #425B9A !important;
}
div[data-baseweb="popover"] li:hover { background: rgba(118,192,236,.2) !important; }

.st-key-video_url div[data-baseweb="input"], .st-key-video_url div[data-baseweb="base-input"] {
    background: rgba(255,246,220,.6) !important;
}

/* ---- Buttons ---- */
.stButton button, .stDownloadButton button {
    border-radius: 12px; border: 1px solid rgba(118,192,236,.45) !important;
    background: #fff !important; color: #425B9A !important; font-weight: 600; transition: all .2s;
}
.stButton button p, .stDownloadButton button p { color: #425B9A !important; font-weight: 600; }
.stButton button:hover, .stDownloadButton button:hover { background: rgba(118,192,236,.18) !important; border-color: #76C0EC !important; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: #425B9A !important; border-right: 1px solid rgba(66,91,154,.9);
    box-shadow: 0 10px 25px rgba(0,0,0,.15);
}
section[data-testid="stSidebar"][aria-expanded="true"] { width: 16rem !important; min-width: 16rem !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding: 0 1rem 1.5rem 1rem; }

.brand { display:flex; align-items:center; gap:.75rem; padding:.25rem .5rem 1.5rem .5rem; }
.brand-icon { padding:.5rem; background:#76C0EC; border-radius:12px; display:flex; box-shadow: inset 0 2px 4px rgba(0,0,0,.1); }
.brand-name { font-size:1.2rem; font-weight:700; letter-spacing:.02em; color:#FFF6DC !important; line-height:1.2; }

section[data-testid="stSidebar"] .stButton button {
    width:100%; justify-content:flex-start !important; text-align:left; gap:.5rem;
    padding:.7rem 1rem; border-radius:12px; border:none !important;
    background:transparent !important; box-shadow:none; transition: all .2s;
}
section[data-testid="stSidebar"] .stButton button > div { justify-content:flex-start !important; width:100%; }
section[data-testid="stSidebar"] .stButton button p { color: rgba(255,246,220,.88) !important; font-weight:500; font-size:.9rem; }
section[data-testid="stSidebar"] .stButton button span[data-testid="stIconMaterial"] { color:#76C0EC !important; }
section[data-testid="stSidebar"] .stButton button:hover { background: rgba(118,192,236,.2) !important; }
section[data-testid="stSidebar"] .stButton button:hover p { color:#FFF6DC !important; }

.profile {
    display:flex; align-items:center; gap:.75rem; padding:1rem;
    border:1px solid rgba(118,192,236,.3); border-radius:16px; box-shadow: inset 0 2px 4px rgba(0,0,0,.1);
}
.avatar { width:40px; height:40px; border-radius:50%; background:#FF95A5; color:#425B9A !important;
          display:flex; align-items:center; justify-content:center; font-weight:700; flex-shrink:0; }
.profile-name { color:#FFF6DC !important; font-size:.92rem; font-weight:600; line-height:1.3; }
.profile-plan { color:#76C0EC !important; font-size:.75rem; }

/* ---- Page header ---- */
.pt-main { font-size:1.6rem; font-weight:700; color:#425B9A !important; line-height:1.25; }
.pt-sub  { font-size:.88rem; font-weight:500; color:rgba(66,91,154,.65) !important; margin-top:.25rem; }
.bell { position:relative; width:42px; height:42px; background:#fff; border-radius:50%;
        display:flex; align-items:center; justify-content:center; color:#425B9A;
        border:1px solid rgba(66,91,154,.05); box-shadow:0 1px 3px rgba(0,0,0,.08); }
.bell .dot { position:absolute; top:8px; right:9px; width:10px; height:10px; background:#FF95A5;
             border:2px solid #fff; border-radius:50%; }

.st-key-lib_search div[data-baseweb="input"] { border-radius:999px !important; box-shadow:0 1px 3px rgba(0,0,0,.06); }
.st-key-lib_search div[data-baseweb="base-input"] { border-radius:999px !important; }
.st-key-lib_search input {
    padding-left: 2.6rem !important;
    background-color: transparent !important;
    background-repeat: no-repeat !important; background-position: 1rem center !important; background-size: 16px !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2376C0EC' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.3-4.3'/%3E%3C/svg%3E") !important;
}

/* ---- Cards (bordered containers) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:#fff !important; border:1px solid rgba(66,91,154,.06) !important; border-radius:16px;
    box-shadow:0 1px 3px rgba(0,0,0,.05);
}

/* Hero */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-hero), .st-key-hero {
    border-radius:24px; box-shadow:0 10px 25px rgba(66,91,154,.12);
    background: linear-gradient(135deg,#ffffff 60%, rgba(118,192,236,.1)) !important;
}
.hero-title { display:flex; align-items:center; gap:.5rem; font-size:1.25rem; font-weight:700;
              color:#425B9A !important; margin-bottom:.9rem; }
.field-label { font-size:.88rem; font-weight:600; color:#425B9A !important; margin-bottom:.45rem; }
.upload-hint { text-align:center; color:rgba(66,91,154,.65) !important; font-size:.75rem; margin-top:.45rem; }

/* Uploader: centred cloud icon + single line of text, whole box clickable */
[data-testid="stFileUploaderDropzone"] {
    position:relative; min-height:170px; display:flex; flex-direction:column; align-items:center; justify-content:center;
    border:2px dashed #76C0EC; background:rgba(118,192,236,.06); border-radius:16px; padding:1rem; transition: background .2s;
}
[data-testid="stFileUploaderDropzone"]:hover { background:rgba(118,192,236,.14); }
[data-testid="stFileUploaderDropzoneInstructions"] { display:none !important; }
[data-testid="stFileUploaderDropzone"] button { position:absolute; inset:0; width:100%; height:100%; opacity:0; cursor:pointer; }
[data-testid="stFileUploaderDropzone"]::before {
    content:""; width:48px; height:48px; border-radius:50%; margin-bottom:.7rem; box-shadow:0 1px 3px rgba(0,0,0,.1);
    background:#fff center/24px no-repeat url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2376C0EC' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242'/%3E%3Cpath d='M12 12v9'/%3E%3Cpath d='m16 16-4-4-4 4'/%3E%3C/svg%3E");
}
[data-testid="stFileUploaderDropzone"]::after { content:"Click or drag file to upload"; font-weight:600; font-size:.9rem; color:#425B9A; }
[data-testid="stFileUploaderFile"] *, [data-testid="stFileUploaderFileName"] { color:#425B9A !important; }

.st-key-generate button {
    background:#FF95A5 !important; border:none !important; padding:.6rem 1.5rem; box-shadow:0 4px 10px rgba(255,149,165,.4);
}
.st-key-generate button p { font-weight:700; }
.st-key-generate button:hover { background:#ff8597 !important; box-shadow:0 8px 16px rgba(255,149,165,.45); }

.st-key-view_all button { background:transparent !important; border:none !important; box-shadow:none; justify-content:flex-end; }
.st-key-view_all button p { color:#76C0EC !important; }
.st-key-view_all button:hover { background:transparent !important; }
.st-key-view_all button:hover p { color:#425B9A !important; }

/* Summary cards */
.section-title { font-size:1.1rem; font-weight:700; color:#425B9A !important; }
div[data-testid="stVerticalBlockBorderWrapper"]:has([class*="st-key-card_"]) { transition: box-shadow .2s; }
div[data-testid="stVerticalBlockBorderWrapper"]:has([class*="st-key-card_"]):hover { box-shadow:0 12px 24px rgba(66,91,154,.15); }
.thumb { height:112px; border-radius:12px; position:relative; display:flex; align-items:center; justify-content:center; }
.thumb .dur { position:absolute; bottom:8px; right:8px; background:rgba(0,0,0,.6); color:#fff !important;
              font-size:.62rem; font-weight:600; padding:3px 8px; border-radius:6px; }
.card-title { font-weight:700; color:#425B9A !important; font-size:.92rem; margin-top:.7rem;
              white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.card-date { color:rgba(66,91,154,.55) !important; font-size:.75rem; font-weight:500; margin:.15rem 0 .6rem 0; }

/* Tabs, alerts, expanders, chat */
button[data-baseweb="tab"] p { color:#425B9A !important; font-weight:600; }
[data-baseweb="tab-highlight"] { background:#76C0EC !important; }
[data-testid="stAlert"] { background:rgba(118,192,236,.15) !important; border-radius:12px; }
[data-testid="stAlert"] * { color:#425B9A !important; }
[data-testid="stStatus"], [data-testid="stExpander"] details { background:#fff !important; border:1px solid rgba(66,91,154,.1) !important; border-radius:12px; }
[data-testid="stStatus"] *, [data-testid="stExpander"] * { color:#425B9A; }
[data-testid="stChatMessage"] { background:#fff !important; border-radius:16px; border:1px solid rgba(66,91,154,.06); }
[data-testid="stChatMessage"] * { color:#425B9A; }
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] { background:#FFF6DC !important; }
[data-testid="stChatInput"] { background:#fff !important; border-radius:16px; border:1px solid rgba(66,91,154,.15); }
[data-testid="stChatInput"] textarea { background:transparent !important; }

/* Result header */
.result-title { font-size:1.6rem; font-weight:700; color:#425B9A !important; line-height:1.25; }
.result-meta { color:rgba(66,91,154,.65) !important; font-size:.85rem; margin-top:.25rem; }

/* ================= Enhanced shadows (CSS only) ================= */

/* Bordered cards (results, settings, analytics, summary cards) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow: 0 6px 18px rgba(66,91,154,.14), 0 1px 4px rgba(66,91,154,.10) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has([class*="st-key-card_"]):hover {
    box-shadow: 0 16px 32px rgba(66,91,154,.24), 0 4px 10px rgba(66,91,154,.12) !important;
    transform: translateY(-2px);
}

/* Hero card */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-hero), .st-key-hero {
    box-shadow: 0 18px 40px rgba(66,91,154,.22), 0 4px 10px rgba(66,91,154,.10) !important;
}

/* Header: bell + search */
.bell { box-shadow: 0 4px 12px rgba(66,91,154,.18); }
.st-key-lib_search div[data-baseweb="input"] { box-shadow: 0 4px 12px rgba(66,91,154,.14) !important; }

/* Inputs */
.st-key-video_url div[data-baseweb="input"] { box-shadow: 0 3px 8px rgba(66,91,154,.12) !important; }
div[data-baseweb="select"] > div, div[data-baseweb="textarea"] { box-shadow: 0 3px 8px rgba(66,91,154,.10); }

/* Uploader + thumbnails */
[data-testid="stFileUploaderDropzone"] { box-shadow: 0 6px 16px rgba(118,192,236,.25); }
[data-testid="stFileUploaderDropzone"]::before { box-shadow: 0 4px 10px rgba(66,91,154,.20); }
.thumb { box-shadow: 0 6px 14px rgba(66,91,154,.18); }

/* Buttons (main area) */
.stButton button, .stDownloadButton button { box-shadow: 0 3px 8px rgba(66,91,154,.14); }
.stButton button:hover, .stDownloadButton button:hover { box-shadow: 0 8px 16px rgba(66,91,154,.22); }
.st-key-generate button { box-shadow: 0 8px 18px rgba(255,149,165,.55); }
.st-key-generate button:hover { box-shadow: 0 12px 24px rgba(255,149,165,.65); }
.st-key-view_all button, .st-key-view_all button:hover { box-shadow: none; }

/* Sidebar */
section[data-testid="stSidebar"] { box-shadow: 6px 0 24px rgba(66,91,154,.30); }
.profile { box-shadow: inset 0 2px 4px rgba(0,0,0,.10), 0 6px 14px rgba(0,0,0,.25); }
.brand-icon { box-shadow: 0 4px 10px rgba(0,0,0,.25), inset 0 2px 4px rgba(0,0,0,.10); }

/* Chat, alerts, status, expanders */
[data-testid="stChatMessage"] { box-shadow: 0 4px 12px rgba(66,91,154,.12); }
[data-testid="stChatInput"] { box-shadow: 0 6px 16px rgba(66,91,154,.16); }
[data-testid="stAlert"] { box-shadow: 0 4px 12px rgba(66,91,154,.12); }
[data-testid="stStatus"], [data-testid="stExpander"] details { box-shadow: 0 5px 14px rgba(66,91,154,.14); }
</style>
"""


def inject_css():
    st.markdown(BASE_CSS, unsafe_allow_html=True)
    active = st.session_state.page
    st.markdown(
        f"""
        <style>
        section[data-testid="stSidebar"] .st-key-nav_{active} button {{
            background:#76C0EC !important; box-shadow:0 4px 8px rgba(0,0,0,.15);
        }}
        section[data-testid="stSidebar"] .st-key-nav_{active} button p {{ color:#425B9A !important; font-weight:600; }}
        section[data-testid="stSidebar"] .st-key-nav_{active} button span[data-testid="stIconMaterial"] {{ color:#425B9A !important; }}
        section[data-testid="stSidebar"] .st-key-nav_{active} button:hover {{ background:#76C0EC !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# 3. SESSION STATE + NAVIGATION
# --------------------------------------------------

def init_state():
    defaults = {
        "page": "dashboard",
        "current": None,        # entry open on "New Summary"
        "rag_chains": {},       # entry_id -> rag_chain
        "chats": {},            # entry_id -> chat messages
        "user_name": "User",
        "user_plan": "Pro Plan",
        "default_language": "English",
        "theme_notice_shown": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def goto(page: str):
    st.session_state.page = page


def open_entry(entry_id: str):
    entry = find_entry(entry_id)
    if entry:
        st.session_state.current = entry
        st.session_state.page = "new"


# --------------------------------------------------
# 4. STORAGE (library saved to data/history.json)
# --------------------------------------------------

def load_history() -> list:
    try:
        if DATA_FILE.exists():
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def save_entry(entry: dict):
    history = load_history()
    history.insert(0, entry)
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def clear_history():
    if DATA_FILE.exists():
        DATA_FILE.unlink()


def demo_entries() -> list:
    now = datetime.now()
    note = "Sample entry shown for the design preview. Process a real video to see your own results."
    base = {"summary": note, "action_items": [], "key_decisions": [], "open_questions": [],
            "transcript": "", "source": "sample", "language": "english", "demo": True}
    return [
        {**base, "id": "demo1", "title": "Q3 Product Roadmap Review", "duration_sec": 45 * 60 + 20,
         "created": (now - timedelta(hours=2)).isoformat()},
        {**base, "id": "demo2", "title": "UX Research Synthesis", "duration_sec": 28 * 60 + 15,
         "created": (now - timedelta(hours=26)).isoformat()},
        {**base, "id": "demo3", "title": "All-Hands Meeting - Oct", "duration_sec": 55 * 60,
         "created": "2026-10-24T10:00:00"},
    ]


def all_entries() -> list:
    entries = load_history()
    if DEMO_MODE:
        entries = entries + demo_entries()
    return entries


def find_entry(entry_id: str):
    return next((e for e in all_entries() if e["id"] == entry_id), None)


# --------------------------------------------------
# 5. GENERAL HELPERS
# --------------------------------------------------

def to_text(value) -> str:
    if value is None:
        return "_Nothing found._"
    if isinstance(value, (list, tuple)):
        return "\n".join(f"- {item}" for item in value) if value else "_Nothing found._"
    if isinstance(value, dict):
        return "\n".join(f"- **{k}**: {v}" for k, v in value.items())
    text = str(value).strip()
    return text if text else "_Nothing found._"


def count_items(value) -> int:
    if not value:
        return 0
    if isinstance(value, (list, tuple)):
        return len(value)
    return len([line for line in str(value).splitlines() if line.strip()])


def fmt_duration(seconds: int) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02}:{s:02}" if h else f"{m}:{s:02}"


def relative_time(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
    except Exception:
        return ""
    delta = datetime.now() - dt
    if delta < timedelta(minutes=1):
        return "Just now"
    if delta < timedelta(hours=1):
        return f"{int(delta.total_seconds() // 60)} min ago"
    if delta < timedelta(days=1):
        hrs = int(delta.total_seconds() // 3600)
        return f"{hrs} hour{'s' if hrs != 1 else ''} ago"
    if delta < timedelta(days=2):
        return "Yesterday"
    return dt.strftime("%b %d, %Y")


def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def build_report(entry: dict) -> str:
    return f"""# {entry['title']}

**Source:** {entry.get('source', '')}
**Language:** {str(entry.get('language', '')).title()}
**Generated:** {entry.get('created', '')}

## Summary
{to_text(entry.get('summary'))}

## Action Items
{to_text(entry.get('action_items'))}

## Key Decisions
{to_text(entry.get('key_decisions'))}

## Open Questions
{to_text(entry.get('open_questions'))}

## Full Transcript
{entry.get('transcript', '')}
"""


# --------------------------------------------------
# 6. PIPELINE
# --------------------------------------------------

def run_pipeline_with_progress(source: str, language: str) -> dict:
    with st.status("Processing your video...", expanded=True) as status:
        bar = st.progress(0)

        st.write("🎧 Extracting audio and creating chunks...")
        chunks = process_input(source)
        st.write(f"Created **{len(chunks)}** audio chunks.")
        bar.progress(10)

        st.write("📝 Transcribing audio (this can take a while)...")
        transcript = transcribe_all(chunks, language=language)
        bar.progress(40)

        st.write("🏷️ Generating title...")
        title = generate_title(transcript)
        bar.progress(50)

        st.write("📄 Generating summary...")
        summary = summarize(transcript)
        bar.progress(60)

        st.write("✅ Extracting action items...")
        action_items = extract_action_items(transcript)
        bar.progress(70)

        st.write("⚖️ Extracting key decisions...")
        decisions = extract_key_decisions(transcript)
        bar.progress(80)

        st.write("❓ Extracting open questions...")
        questions = extract_questions(transcript)
        bar.progress(90)

        st.write("🧠 Building chat (RAG) system...")
        rag_chain = build_rag_chain(transcript)
        bar.progress(100)

        status.update(label="Processing complete!", state="complete", expanded=False)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


def process_source(source: str, language: str, label: str, cleanup_path: str = None) -> bool:
    try:
        result = run_pipeline_with_progress(source, language)
    except Exception as e:
        st.error(f"An error occurred while processing: {e}")
        return False
    finally:
        if cleanup_path and os.path.exists(cleanup_path):
            try:
                os.remove(cleanup_path)
            except OSError:
                pass

    words = len(result["transcript"].split())
    entry = {
        "id": uuid.uuid4().hex[:12],
        "title": str(result["title"]).strip().strip('"') or "Untitled video",
        "summary": result["summary"],
        "action_items": result["action_items"],
        "key_decisions": result["key_decisions"],
        "open_questions": result["open_questions"],
        "transcript": result["transcript"],
        "source": label,
        "language": language,
        "created": datetime.now().isoformat(),
        "duration_sec": int(words / 150 * 60),  # estimated at ~150 words per minute
        "demo": False,
    }
    save_entry(entry)
    st.session_state.current = entry
    st.session_state.rag_chains[entry["id"]] = result["rag_chain"]
    return True


# --------------------------------------------------
# 7. SIDEBAR + HEADER
# --------------------------------------------------

def render_sidebar():
    with st.sidebar:
        render_html(f"""
            <div class="brand">
                <div class="brand-icon">{icon("video", 24, NAVY)}</div>
                <div class="brand-name">VideoGenius AI</div>
            </div>
        """)

        for page_id, label, material_icon in NAV_ITEMS:
            st.button(label, key=f"nav_{page_id}", icon=material_icon,
                      use_container_width=True, on_click=goto, args=(page_id,))

        st.markdown("<div style='height:36vh'></div>", unsafe_allow_html=True)

        name = escape(st.session_state.user_name or "User")
        render_html(f"""
            <div class="profile">
                <div class="avatar">{name[:1].upper()}</div>
                <div style="flex:1;min-width:0">
                    <div class="profile-name">{name}</div>
                    <div class="profile-plan">{escape(st.session_state.user_plan)}</div>
                </div>
            </div>
        """)


def render_header(title: str, subtitle: str):
    c1, c2, c3 = st.columns([3, 1.6, 0.35], vertical_alignment="center")
    with c1:
        render_html(f'<div class="pt-main">{escape(title)}</div><div class="pt-sub">{escape(subtitle)}</div>')
    with c2:
        st.text_input("Search", key="lib_search", placeholder="Search summaries...",
                      label_visibility="collapsed", on_change=goto, args=("library",))
    with c3:
        render_html(f'<div class="bell">{icon("bell", 20)}<span class="dot"></span></div>')
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)


# --------------------------------------------------
# 8. DASHBOARD COMPONENTS
# --------------------------------------------------

def render_hero():
    with st.container(border=True, key="hero"):
        render_html(f'<div class="hero-title">{icon("video", 20, PINK)} Process a New Video</div>')
        left, right = st.columns(2, gap="large")

        with left:
            uploaded = st.file_uploader("Upload audio or video", type=SUPPORTED_TYPES,
                                        label_visibility="collapsed", key="uploader")
            render_html(f'<div class="upload-hint">MP4, MOV, Audio up to {UPLOAD_LIMIT_LABEL}</div>')

        with right:
            render_html('<div class="field-label">Or paste a video link</div>')
            url = st.text_input("Video link", key="video_url", label_visibility="collapsed",
                                placeholder="https://youtube.com/watch?v=...")
            st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
            clicked = st.button("Generate Summary", key="generate", icon=":material/bolt:",
                                use_container_width=True)

    if clicked:
        language = LANGUAGES[st.session_state.default_language]
        if uploaded is not None:
            path = save_uploaded_file(uploaded)
            ok = process_source(path, language, uploaded.name, cleanup_path=path)
        elif url.strip():
            ok = process_source(url.strip(), language, url.strip())
        else:
            st.warning("Please upload a file or paste a video link first.")
            return
        if ok:
            st.session_state.page = "new"
            st.rerun()


def render_summary_card(entry: dict, idx: int, prefix: str):
    thumb_colors = ["#DCEFFA", "#D5DBEA", "#FFE3E8"]
    with st.container(border=True, key=f"card_{prefix}_{entry['id']}"):
        render_html(f"""
            <div class="thumb" style="background:{thumb_colors[idx % 3]}">
                {icon("play", 40, "rgba(255,255,255,.95)")}
                <div class="dur">{fmt_duration(entry.get('duration_sec', 0))}</div>
            </div>
            <div class="card-title">{escape(str(entry['title']))}</div>
            <div class="card-date">{relative_time(entry.get('created', ''))}</div>
        """)
        st.button("View Summary", key=f"view_{prefix}_{entry['id']}", use_container_width=True,
                  on_click=open_entry, args=(entry["id"],))


def render_recent_summaries():
    c1, c2 = st.columns([5, 1], vertical_alignment="center")
    with c1:
        render_html('<div class="section-title">Recent Summaries</div>')
    with c2:
        st.button("View All ›", key="view_all", on_click=goto, args=("library",))

    entries = all_entries()[:3]
    if not entries:
        st.info("No summaries yet. Process your first video above!")
        return
    for i, (col, entry) in enumerate(zip(st.columns(3, gap="medium"), entries)):
        with col:
            render_summary_card(entry, i, "recent")


# --------------------------------------------------
# 9. PAGES
# --------------------------------------------------

def page_dashboard():
    render_header(f"Welcome back, {st.session_state.user_name}", "Ready to extract insights today?")
    render_hero()
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    render_recent_summaries()


def page_new():
    entry = st.session_state.current
    if entry is None:
        render_header("New Summary", "Upload a file or paste a link to get started")
        render_hero()
        return
    render_results(entry)


def page_library():
    render_header("My Library", "All your saved summaries")
    query = (st.session_state.get("lib_search") or "").strip().lower()
    entries = all_entries()
    if query:
        entries = [e for e in entries
                   if query in str(e["title"]).lower() or query in str(e.get("summary", "")).lower()]
        st.caption(f"{len(entries)} result(s) for “{query}”")

    if not entries:
        st.info("No summaries found.")
        return

    for row_start in range(0, len(entries), 3):
        cols = st.columns(3, gap="medium")
        for i, entry in enumerate(entries[row_start:row_start + 3]):
            with cols[i]:
                render_summary_card(entry, row_start + i, "lib")


def page_analytics():
    render_header("Analytics", "Your usage at a glance")
    real = load_history()

    if not real:
        st.info("Charts will appear here once you've processed at least one video.")
        return

    df = pd.DataFrame(real)
    df["date"] = pd.to_datetime(df["created"]).dt.date
    df["words"] = df["transcript"].fillna("").str.split().str.len()

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Videos per day")
            st.bar_chart(df.groupby("date").size(), color=SKY)
    with right:
        with st.container(border=True):
            st.subheader("Transcript length (words)")
            st.bar_chart(df.set_index("title")["words"], color=PINK)


def page_settings():
    render_header("Settings", "Personalize your workspace")
    with st.container(border=True):
        st.subheader("Profile")
        st.text_input("Display name", key="user_name")

    with st.container(border=True):
        st.subheader("Processing")
        st.selectbox("Video language", list(LANGUAGES.keys()), key="default_language")
        st.caption("Used when you click Generate Summary.")

    with st.container(border=True):
        st.subheader("Library")
        st.caption(f"Summaries are saved to `{DATA_FILE}`.")
        if st.button("Delete all saved summaries"):
            clear_history()
            st.session_state.current = None
            st.success("Library cleared.")
            st.rerun()


# --------------------------------------------------
# 10. RESULTS VIEW (summary / insights / transcript / chat)
# --------------------------------------------------

def get_rag_chain(entry: dict):
    chains = st.session_state.rag_chains
    if entry["id"] not in chains and entry.get("transcript"):
        with st.spinner("Preparing chat for this video..."):
            chains[entry["id"]] = build_rag_chain(entry["transcript"])
    return chains.get(entry["id"])


def render_results(entry: dict):
    c1, c2, c3 = st.columns([4, 1.2, 1.2], vertical_alignment="center")
    with c1:
        render_html(f"""
            <div class="result-title">{escape(str(entry['title']))}</div>
            <div class="result-meta">Source: {escape(str(entry.get('source', '')))} ·
            {str(entry.get('language', '')).title()} · {relative_time(entry.get('created', ''))}</div>
        """)
    with c2:
        st.download_button("Report", data=build_report(entry), file_name="video_report.md",
                           mime="text/markdown", icon=":material/download:", use_container_width=True)
    with c3:
        if st.button("New video", icon=":material/add:", use_container_width=True):
            st.session_state.current = None
            st.rerun()

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    tab_summary, tab_insights, tab_transcript, tab_chat = st.tabs(
        ["Summary", "Insights", "Transcript", "Chat"]
    )

    with tab_summary:
        with st.container(border=True):
            st.markdown(to_text(entry["summary"]))

    with tab_insights:
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            with st.container(border=True):
                st.subheader("Action Items")
                st.markdown(to_text(entry["action_items"]))
        with col2:
            with st.container(border=True):
                st.subheader("Key Decisions")
                st.markdown(to_text(entry["key_decisions"]))
        with st.container(border=True):
            st.subheader("Open Questions")
            st.markdown(to_text(entry["open_questions"]))

    with tab_transcript:
        transcript = entry.get("transcript", "")
        if not transcript:
            st.info("No transcript available for this entry.")
        else:
            search = st.text_input("Search transcript", placeholder="Type a keyword...", key="t_search")
            if search:
                hits = [ln for ln in transcript.split("\n") if search.lower() in ln.lower()]
                st.caption(f"{len(hits)} matching line(s)")
                for line in hits:
                    st.markdown(f"- {line}")
            else:
                st.text_area("Transcript", transcript, height=420, label_visibility="collapsed")
            st.download_button("Download transcript", data=transcript, file_name="transcript.txt",
                               mime="text/plain", icon=":material/download:")

    with tab_chat:
        render_chat(entry)


def render_chat(entry: dict):
    if not entry.get("transcript"):
        st.info("Chat is available for videos you've processed yourself.")
        return

    history = st.session_state.chats.setdefault(entry["id"], [])

    if history and st.button("Clear chat"):
        st.session_state.chats[entry["id"]] = []
        st.rerun()

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask something about the video...")
    if question:
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_question(get_rag_chain(entry), question)
                except Exception as e:
                    answer = f"Error generating answer: {e}"
            st.markdown(answer)
        history.append({"role": "assistant", "content": answer})


# --------------------------------------------------
# 11. MAIN
# --------------------------------------------------

PAGES = {
    "dashboard": page_dashboard,
    "new": page_new,
    "library": page_library,
    "analytics": page_analytics,
    "settings": page_settings,
}


def main():
    init_state()
    inject_css()
    render_sidebar()

    if THEME_CREATED and not st.session_state.theme_notice_shown:
        st.toast("Light theme config created. Restart the app once for the best look.", icon="🎨")
        st.session_state.theme_notice_shown = True

    PAGES[st.session_state.page]()


if __name__ == "__main__":
    main()