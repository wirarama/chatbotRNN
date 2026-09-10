"""
config.py — Page setup, constants, and CSS theme
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What lives here:
  • MODEL_DIR resolution (the --model CLI flag).
  • Every "magic" constant shared across modules: sensor keys/units, the
    Decision-Tree notification vocabulary (NOTIF_CLASSES / SEVERITY_MAP /
    NOTIF_MSG), cluster-name -> description text, and the chart palette.
  • configure_page(): calls st.set_page_config(...) and injects the CSS
    that forces a white theme with high-contrast text everywhere
    (navbar, chat bubbles, stat cards, tabs, dataframes, sidebar, etc).

When to edit this file:
  • Changing colors/branding/theme -> edit the CSS string in configure_page().
  • Adding a new sensor -> add it to SENSOR_KEYS / SENSOR_UNITS (then also
    update iot_train.py, since the model input size depends on it).
  • Changing what a Decision-Tree notification class means -> NOTIF_MSG /
    SEVERITY_MAP.

This file does NOT know about the model, the data, or the chat parser —
it only holds static configuration, so it can be imported by every other
module without creating circular imports.
"""

import sys
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL DIR ARGUMENT
# ─────────────────────────────────────────────────────────────────────────────
def get_model_dir():
    """Read --model argument passed after `--` in the streamlit run command."""
    args = sys.argv[1:]
    if "--model" in args:
        idx = args.index("--model")
        if idx + 1 < len(args):
            return args[idx + 1]
    return "./iot_model"


MODEL_DIR = get_model_dir()

# ─────────────────────────────────────────────────────────────────────────────
#  LIVE STREAMING DEFAULTS
#  Matches iot_stream_sender.py's own --db default. See iot_slm_app/
#  live_feed.py for the polling/ingestion logic these feed into.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_STREAM_DB       = "./iot_stream.db"
DEFAULT_POLL_SECONDS    = 5
DEFAULT_LIVE_CHART_POINTS = 200  # most recent live samples shown per sensor

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS (will be overridden by config.json)
# ─────────────────────────────────────────────────────────────────────────────
SENSOR_KEYS   = ["temperature", "pressure", "humidity", "light"]
SENSOR_UNITS  = {"temperature": "°C", "pressure": "hPa", "humidity": "% RH", "light": "lux"}
NOTIF_CLASSES = ["NO_ACTION", "INFO_DAY", "INFO_NIGHT", "WATCH_BORDERLINE",
                  "WARN_TEMP", "WARN_HUMIDITY", "WARN_LIGHT", "CRIT_PRESSURE", "CRIT_MULTI"]
SEVERITY_MAP  = {"NO_ACTION": "NOMINAL", "INFO_DAY": "INFO", "INFO_NIGHT": "INFO",
                  "WATCH_BORDERLINE": "WATCH", "WARN_TEMP": "WARNING", "WARN_HUMIDITY": "WARNING",
                  "WARN_LIGHT": "WARNING", "CRIT_PRESSURE": "CRITICAL", "CRIT_MULTI": "CRITICAL"}
NOTIF_MSG = {
    "NO_ACTION":        "All readings within normal operating range.",
    "INFO_DAY":         "Daytime conditions within expected parameters.",
    "INFO_NIGHT":       "Nighttime conditions within expected parameters.",
    "WATCH_BORDERLINE": "Values approaching threshold limits — increased monitoring recommended.",
    "WARN_TEMP":        "WARNING: Temperature exceeded upper bound. Check ventilation or cooling systems.",
    "WARN_HUMIDITY":    "WARNING: Humidity elevated. Inspect for condensation or seal failure.",
    "WARN_LIGHT":       "WARNING: Illuminance outside expected range. Check sensor integrity.",
    "CRIT_PRESSURE":    "CRITICAL: Pressure dropped sharply. Immediate inspection required.",
    "CRIT_MULTI":       "CRITICAL: Multiple sensors simultaneously abnormal. Immediate action required.",
}
CLUSTER_COND = {
    "Cool Night":    "comfortable nighttime conditions",
    "Warm Day":      "typical warm daytime conditions",
    "Borderline":    "borderline conditions requiring attention",
    "Cold Night":    "cool nighttime conditions",
    "Hot Afternoon": "hot afternoon with elevated temperature",
    "Warm Night":    "mild nighttime conditions",
    "Unstable":      "unstable conditions with high variability",
    "Sunny Day":     "sunny and hot daytime conditions",
    "Humid Day":     "humid daytime conditions",
    "Very Humid":    "very high humidity conditions",
    "Light Fault":   "light sensor fault conditions",
}
CLUSTER_DESCS = {
    "Cool Night":    "comfortable nighttime temperature and moderate humidity",
    "Warm Day":      "typical warm daytime conditions",
    "Borderline":    "values approaching but not crossing threshold limits",
    "Cold Night":    "cool nighttime temperatures with elevated humidity",
    "Hot Afternoon": "elevated temperature and reduced humidity — peak heat load",
    "Warm Night":    "mild overnight conditions with moderate humidity",
    "Unstable":      "high variability across sensors — possible compound fault",
    "Sunny Day":     "high light intensity with warm, dry conditions",
    "Humid Day":     "elevated humidity alongside warm daytime temperatures",
    "Very Humid":    "very high relative humidity — condensation risk",
    "Light Fault":   "abnormal illuminance readings — possible sensor fault",
}
COLORS = {
    "A": "#1A5276", "B": "#C0392B",
    "A_fill": "rgba(26,82,118,0.15)", "B_fill": "rgba(192,57,43,0.15)",
    "grid": "#E8EDF5", "bg": "#FAFBFF", "teal": "#0D7A6E", "navy": "#0E2A5C",
}


def cluster_desc(cname):
    """Human-readable one-liner for a cluster name (falls back to a generic phrase)."""
    return CLUSTER_DESCS.get(cname, "mixed environmental conditions")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG + WHITE THEME CSS
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
/* ── Bootstrap-aligned base — forced white theme, high-contrast text ── */
html, body, [class*="css"], [class*="st-"] {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 15px;
    color: #212529 !important;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
.main, .block-container {
    background-color: #ffffff !important;
    color: #212529 !important;
}
[data-testid="stHeader"] { background-color: rgba(255,255,255,0) !important; }
.block-container {
    padding-top: 1.25rem;
    padding-bottom: 1rem;
    max-width: 1400px;
}

/* Headings / plain markdown text / captions */
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stCaptionContainer"],
label, .stMarkdown, .stText, [data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #212529 !important;
}
[data-testid="stMarkdownContainer"] a { color: #0d6efd !important; }

/* Tabs */
[data-testid="stTabs"] button[role="tab"] { color: #495057 !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #0d6efd !important;
    border-bottom-color: #0d6efd !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: #0d6efd !important; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { background-color: #dee2e6 !important; }

/* Selectboxes / dropdowns / inputs */
[data-baseweb="select"] > div, .stSelectbox div[data-baseweb="select"] {
    background-color: #ffffff !important;
    color: #212529 !important;
    border-color: #ced4da !important;
}
[data-baseweb="popover"] li, [data-baseweb="menu"] li {
    background-color: #ffffff !important;
    color: #212529 !important;
}
input, textarea {
    background-color: #ffffff !important;
    color: #212529 !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: #212529 !important;
}

/* Dataframes / tables */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    color: #212529 !important;
    background-color: #ffffff !important;
}
[data-testid="stDataFrame"] * { color: #212529 !important; }

/* Expanders */
[data-testid="stExpander"] summary {
    color: #212529 !important;
    background-color: #f8f9fa !important;
}

/* Alerts (st.info / st.success / st.warning / st.error) keep their own
   accessible foreground/background pairs — just make sure text isn't
   overridden to something invisible. */
[data-testid="stAlert"] p, [data-testid="stAlertContentInfo"],
[data-testid="stAlertContentSuccess"], [data-testid="stAlertContentWarning"],
[data-testid="stAlertContentError"] {
    color: inherit !important;
}

/* Code blocks */
code, pre, [data-testid="stCodeBlock"] {
    color: #212529 !important;
    background-color: #f1f3f5 !important;
}

/* ── Page header (navbar style) ── */
.iot-navbar {
    background-color: #343a40;
    color: #fff;
    padding: 0.65rem 1.25rem;
    border-radius: 4px;
    margin-bottom: 1rem;
    display: flex;
    align-items: baseline;
    gap: 1rem;
    border-left: 4px solid #0d6efd;
}
.iot-navbar .brand {
    font-size: 1.1rem;
    font-weight: 600;
    color: #fff !important;
    letter-spacing: 0.02em;
}
.iot-navbar .meta {
    font-size: 0.82rem;
    color: #ced4da !important;
}

/* ── Chat bubbles ── */
.msg-user {
    background-color: #0d6efd;
    color: #fff;
    border-radius: 4px 4px 0 4px;
    padding: 0.55rem 0.9rem;
    margin: 0.35rem 0 0.35rem 22%;
    font-size: 0.93rem;
    line-height: 1.55;
}
.msg-bot {
    background-color: #fff;
    color: #212529;
    border-radius: 0 4px 4px 4px;
    border: 1px solid #dee2e6;
    border-left: 3px solid #0d6efd;
    padding: 0.65rem 0.95rem;
    margin: 0.35rem 18% 0.35rem 0;
    font-size: 0.92rem;
    line-height: 1.65;
    white-space: pre-wrap;
}
.msg-bot b { color: #0d6efd; }
.msg-bot i { color: #495057; }
.msg-notif {
    background-color: #fff3cd;
    color: #664d03;
    border-radius: 4px;
    border: 1px solid #ffda6a;
    border-left: 3px solid #d97706;
    padding: 0.65rem 0.95rem;
    margin: 0.35rem 18% 0.35rem 0;
    font-size: 0.92rem;
    line-height: 1.65;
    white-space: pre-wrap;
}

/* ── Stat cards ── */
.stat-card {
    background: #fff;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 0.9rem 0.75rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.stat-val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #212529;
    line-height: 1.2;
}
.stat-unit {
    font-size: 0.78rem;
    color: #6c757d;
    font-weight: 400;
}
.stat-lbl {
    font-size: 0.78rem;
    color: #6c757d;
    margin-top: 0.25rem;
}
.stat-sub {
    font-size: 0.75rem;
    color: #868e96;
    margin-top: 0.15rem;
}
.trend-up   { font-size: 0.75rem; color: #dc3545; font-weight: 600; }
.trend-dn   { font-size: 0.75rem; color: #198754; font-weight: 600; }
.trend-st   { font-size: 0.75rem; color: #6c757d; }

/* ── Notification badge ── */
.nbadge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 3px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.NOMINAL  { background: #d1e7dd; color: #0a3622; border: 1px solid #a3cfbb; }
.INFO     { background: #cfe2ff; color: #052c65; border: 1px solid #9ec5fe; }
.WATCH    { background: #fff3cd; color: #664d03; border: 1px solid #ffda6a; }
.WARNING  { background: #ffe5d0; color: #6e2d00; border: 1px solid #ffbd9b; }
.CRITICAL { background: #f8d7da; color: #58151c; border: 1px solid #f1aeb5; }

/* ── Rule cards (Rules tab) ── */
.rule-card {
    background: #fff;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.6rem;
}
.rule-card.disabled { opacity: 0.55; }
.rule-desc { font-weight: 600; color: #212529; font-size: 0.95rem; }
.rule-raw  { font-size: 0.8rem; color: #6c757d; font-style: italic; margin-top: 0.15rem; }
.rule-meta { font-size: 0.78rem; color: #868e96; margin-top: 0.3rem; }

/* ── Sidebar (white theme) ── */
section[data-testid="stSidebar"] {
    background-color: #f8f9fa !important;
    border-right: 1px solid #dee2e6;
}
section[data-testid="stSidebar"] * {
    color: #212529 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #dee2e6 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #212529 !important;
    border-color: #ced4da !important;
}

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    border-radius: 3px;
    font-size: 0.84rem;
    font-family: "Segoe UI", Arial, sans-serif;
    border: 1px solid #0d6efd;
    background: #0d6efd;
    color: #fff !important;
    padding: 0.3rem 0.6rem;
    text-align: left;
    font-weight: 400;
    letter-spacing: 0;
}
.stButton > button:hover {
    background: #0b5ed7;
    border-color: #0b5ed7;
    color: #fff !important;
}
.stButton > button p { color: #fff !important; }

/* ── Info / success boxes ── */
.status-ok  { background:#d1e7dd; border:1px solid #a3cfbb; border-radius:3px;
              padding:0.3rem 0.7rem; font-size:0.82rem; color:#0a3622; margin:2px 0; }
.status-err { background:#f8d7da; border:1px solid #f1aeb5; border-radius:3px;
              padding:0.3rem 0.7rem; font-size:0.82rem; color:#58151c; margin:2px 0; }

/* ── Restore Material Symbols icon font ──────────────────────────────
   Streamlit renders icons (sidebar collapse/expand arrow, expander
   chevrons, alert icons, etc.) as ligature text — e.g. the literal
   string "keyboard_double_arrow_right" — that a special icon font
   turns into a glyph via font-ligature substitution. The blanket
   `font-family` override at the top of this stylesheet (with
   !important) clobbers that font for those elements too, so the
   ligature never resolves and the raw text shows up instead of the
   arrow. This rule restores the icon font (and the font-feature
   properties the ligature needs) for every element Streamlit tags as
   a material icon, and — being declared after the earlier rule with
   equal selector specificity — wins the cascade. */
[data-testid="stIconMaterial"],
span[data-testid="stIconMaterial"],
[data-testid*="Icon"] span[class*="material"],
.material-symbols-outlined,
.material-icons {
    font-family: 'Material Symbols Outlined', 'Material Icons' !important;
    font-weight: normal !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    -webkit-font-feature-settings: 'liga' !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}
</style>
"""


def configure_page():
    """Call once, first thing in the entry-point script."""
    st.set_page_config(
        page_title="IoT-SLM Chatbot",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
