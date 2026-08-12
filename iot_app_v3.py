"""
iot_app.py  —  IoT-SLM Streamlit App  (model-loading only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Requires pre-trained model files from iot_train.py.
Does NOT train anything — pure inference + visualisation.

Usage:
  streamlit run iot_app.py                        # looks for ./iot_model/
  streamlit run iot_app.py -- --model /path/to/   # custom model dir

Dependencies (much lighter than training):
  pip install streamlit plotly numpy pandas scikit-learn torch
  (torch is only used for loading weights + forward pass, no training)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys, os, json, pickle, re, warnings, argparse
from datetime import datetime, timedelta
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ── Torch is only needed for inference ───────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IoT-SLM Chatbot",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.block-container{padding-top:1.1rem;padding-bottom:0.8rem;}
.header-bar{background:linear-gradient(135deg,#0E2A5C,#1A5276 60%,#0D7A6E);
  border-radius:12px;padding:14px 22px;margin-bottom:14px;display:flex;align-items:center;gap:14px;}
.header-bar h1{color:#fff;font-size:1.4rem;margin:0;font-weight:700;}
.header-bar p{color:#A8D8EA;font-size:0.80rem;margin:3px 0 0;}
.user-msg{background:#1A5276;color:#fff;border-radius:16px 16px 4px 16px;
  padding:9px 14px;margin:5px 0 5px 20%;font-size:0.88rem;line-height:1.5;}
.bot-msg{background:#F0F4FF;color:#0E2A5C;border-radius:16px 16px 16px 4px;
  border-left:4px solid #0D7A6E;padding:11px 16px;margin:5px 18% 5px 0;
  font-size:0.87rem;line-height:1.6;white-space:pre-wrap;}
.bot-msg b{color:#0D7A6E;}
.stat-card{background:#EEF3FA;border:1px solid #C8D8F0;border-radius:10px;
  padding:12px 10px;text-align:center;margin-bottom:6px;}
.stat-val{font-size:1.45rem;font-weight:700;color:#1A5276;}
.stat-lbl{font-size:0.75rem;color:#5A6E8C;margin-top:3px;}
.stat-trend-up{font-size:0.76rem;color:#C0392B;}
.stat-trend-dn{font-size:0.76rem;color:#1D8348;}
.stat-trend-st{font-size:0.76rem;color:#888;}
.nbadge{display:inline-block;padding:3px 10px;border-radius:16px;font-size:0.78rem;font-weight:600;}
.NOMINAL {background:#D5F5E3;color:#1D8348;}
.INFO    {background:#D6EAF8;color:#1A5276;}
.WATCH   {background:#FEF9E7;color:#B7770D;}
.WARNING {background:#FDEBD0;color:#BA4A00;}
.CRITICAL{background:#FADBD8;color:#922B21;}
.stButton>button{width:100%;border-radius:8px;font-size:0.82rem;
  border:1px solid #C8D8F0;background:#F0F4FF;color:#1A5276;padding:5px 8px;}
.stButton>button:hover{background:#1A5276;color:#fff;}
section[data-testid="stSidebar"]{background:#0E2A5C;}
section[data-testid="stSidebar"] *{color:#EEF3FA !important;}
.model-ok{background:#D5F5E3;border-radius:8px;padding:6px 12px;
  font-size:0.80rem;color:#1D8348;margin-bottom:6px;}
.model-err{background:#FADBD8;border-radius:8px;padding:6px 12px;
  font-size:0.80rem;color:#922B21;margin-bottom:6px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PARSE MODEL DIR ARGUMENT
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
#  CONSTANTS (will be overridden by config.json)
# ─────────────────────────────────────────────────────────────────────────────
SENSOR_KEYS   = ["temperature", "pressure", "humidity", "light"]
SENSOR_UNITS  = {"temperature":"°C","pressure":"hPa","humidity":"% RH","light":"lux"}
NOTIF_CLASSES = ["NO_ACTION","INFO_DAY","INFO_NIGHT","WATCH_BORDERLINE",
                 "WARN_TEMP","WARN_HUMIDITY","WARN_LIGHT","CRIT_PRESSURE","CRIT_MULTI"]
SEVERITY_MAP  = {"NO_ACTION":"NOMINAL","INFO_DAY":"INFO","INFO_NIGHT":"INFO",
                 "WATCH_BORDERLINE":"WATCH","WARN_TEMP":"WARNING","WARN_HUMIDITY":"WARNING",
                 "WARN_LIGHT":"WARNING","CRIT_PRESSURE":"CRITICAL","CRIT_MULTI":"CRITICAL"}
NOTIF_MSG = {
    "NO_ACTION":        "All readings within normal operating range.",
    "INFO_DAY":         "Daytime conditions within expected parameters.",
    "INFO_NIGHT":       "Nighttime conditions within expected parameters.",
    "WATCH_BORDERLINE": "⚠ Values approaching threshold limits — monitor closely.",
    "WARN_TEMP":        "⚠ Temperature exceeded upper bound. Check ventilation.",
    "WARN_HUMIDITY":    "⚠ Humidity elevated. Inspect for condensation or seal failure.",
    "WARN_LIGHT":       "⚠ Illuminance outside expected range. Check sensor.",
    "CRIT_PRESSURE":    "🚨 CRITICAL: Pressure dropped sharply. Immediate inspection required.",
    "CRIT_MULTI":       "🚨 CRITICAL: Multiple sensors abnormal. Immediate action required.",
}
CLUSTER_COND = {
    "Cool Night":"comfortable nighttime conditions",
    "Warm Day":"typical warm daytime conditions",
    "Borderline":"borderline conditions requiring attention",
    "Cold Night":"cool nighttime conditions",
    "Hot Afternoon":"hot afternoon with elevated temperature",
    "Warm Night":"mild nighttime conditions",
    "Unstable":"unstable conditions with high variability",
    "Sunny Day":"sunny and hot daytime conditions",
    "Humid Day":"humid daytime conditions",
    "Very Humid":"very high humidity conditions",
    "Light Fault":"light sensor fault conditions",
}
COLORS = {
    "A":"#1A5276","B":"#C0392B",
    "A_fill":"rgba(26,82,118,0.15)","B_fill":"rgba(192,57,43,0.15)",
    "grid":"#E8EDF5","bg":"#FAFBFF","teal":"#0D7A6E","navy":"#0E2A5C",
}

# ── RANKING QUERY ENGINE ──────────────────────────────────────────────────────
# Detects queries asking for highest/lowest values with temporal context
# Returns: sorted daily breakdown + rate-of-change analysis + cluster context

import re as _re
import numpy as _np
from collections import defaultdict as _dd

# Patterns for sensor extraction
_SENSOR_PATS = {
    "temperature": _re.compile(
        r"\b(temp(erature)?|suhu|panas|dingin|thermal)\b", _re.I),
    "humidity":    _re.compile(
        r"\b(humid(ity)?|kelembab(an)?|lembab|rh)\b", _re.I),
    "pressure":    _re.compile(
        r"\b(pressure|tekanan|baro(meter)?|hpa|atm)\b", _re.I),
    "light":       _re.compile(
        r"\b(light|cahaya|illu(minance)?|lux|cerah|terang|gelap)\b", _re.I),
}

# Patterns for direction (highest/lowest)
_HIGH_PAT = _re.compile(
    r"\b(high(est)?|max(imum)?|top|peak|tertinggi|terbesar|paling tinggi|paling panas|terpanas|terkuat|terbesar)\b",
    _re.I)
_LOW_PAT  = _re.compile(
    r"\b(low(est)?|min(imum)?|bottom|trough|terendah|terkecil|paling rendah|paling dingin|terdingin|terlemah)\b",
    _re.I)

# Rate of change words
_RATE_WORDS = {
    "sharp":    (0.08,  float("inf")),
    "gradual":  (0.02,  0.08),
    "stable":   (-0.02, 0.02),
    "gradual_fall": (-0.08, -0.02),
    "sharp_fall":   (float("-inf"), -0.08),
}

SENSOR_UNITS = {"temperature":"°C","pressure":"hPa","humidity":"% RH","light":"lux"}


def detect_ranking_query(text):
    """
    Returns (sensor, direction) if the query asks for highest/lowest,
    otherwise returns (None, None).
    direction: 'high' or 'low'
    """
    sensor = None
    for s, pat in _SENSOR_PATS.items():
        if pat.search(text):
            sensor = s
            break
    if sensor is None:
        return None, None

    if _HIGH_PAT.search(text):
        return sensor, "high"
    if _LOW_PAT.search(text):
        return sensor, "low"
    return None, None


def _rate_label(slope):
    """Convert slope to human-readable rate description."""
    if slope > 0.08:   return "rising sharply"
    if slope > 0.02:   return "gradually rising"
    if slope < -0.08:  return "dropping sharply"
    if slope < -0.02:  return "gradually falling"
    return "stable"


def _anomaly_note(val, all_vals):
    """Check if value is beyond 2.5σ — flag as anomaly."""
    mu, sd = _np.mean(all_vals), _np.std(all_vals)
    if sd < 1e-6:
        return ""
    z = abs(val - mu) / sd
    if z > 2.5:
        return (f" ⚠ This value is {z:.1f} standard deviations from the "
                f"period mean ({mu:.1f}) — classified as a statistical anomaly.")
    return ""


def ranking_query(engine, indices, sensor, direction, period_label,
                  top_n=5):
    """
    For a set of window indices, group readings by calendar date,
    compute daily aggregates (mean, max, min), sort by the target
    direction, and return a rich narrative dict.

    Returns dict with:
      ranked_days   : list of {date, mean, max, min, slope, cluster, n}
      overall_stats : {mean, std, global_min, global_max}
      direction     : 'high' or 'low'
      sensor        : sensor key
      period_label  : string
      narrative     : formatted multi-sentence narrative string
    """
    if not indices:
        return None

    # Group by date
    by_date = _dd(list)
    for i in indices:
        m = engine.metas[i]
        by_date[m["timestamp"].date()].append(i)

    all_vals = _np.array([engine.metas[i][sensor] for i in indices])
    unit = SENSOR_UNITS[sensor]

    # Build daily stats
    days = []
    for d, idxs in sorted(by_date.items()):
        vals   = _np.array([engine.metas[i][sensor] for i in idxs])
        t_axis = _np.arange(len(vals))
        slope  = float(_np.polyfit(t_axis, vals, 1)[0]) if len(vals) > 1 else 0.0

        # Dominant cluster for this day
        cl_counts = _dd(int)
        for i in idxs:
            if i < len(engine.cl):
                cl_counts[engine.cl[i]] += 1
        dom_cl = max(cl_counts, key=cl_counts.get) if cl_counts else 0
        cl_name = engine.cn.get(dom_cl, "Unknown")

        days.append({
            "date":    d,
            "mean":    round(float(vals.mean()), 2),
            "max":     round(float(vals.max()), 2),
            "min":     round(float(vals.min()), 2),
            "median":  round(float(_np.median(vals)), 2),
            "slope":   round(slope, 4),
            "rate":    _rate_label(slope),
            "cluster": cl_name,
            "n":       len(idxs),
            "anom":    _anomaly_note(vals.max() if direction=="high" else vals.min(), all_vals),
        })

    # Sort
    sort_key = "max" if direction == "high" else "min"
    reverse  = (direction == "high")
    ranked   = sorted(days, key=lambda d: d[sort_key], reverse=reverse)

    overall = {
        "mean":       round(float(all_vals.mean()), 2),
        "std":        round(float(all_vals.std()), 2),
        "global_max": round(float(all_vals.max()), 2),
        "global_min": round(float(all_vals.min()), 2),
    }

    # ── Build narrative ────────────────────────────────────────────────────
    dir_word = "highest" if direction == "high" else "lowest"
    target   = ranked[0]
    val_key  = "max" if direction == "high" else "min"
    peak_val = target[val_key]
    peak_dt  = target["date"].strftime("%A, %d %B %Y")

    # Sentence 1: answer the question directly
    s1 = (f"The {dir_word} {sensor} during **{period_label}** was recorded on "
          f"**{peak_dt}**, reaching **{peak_val:.1f}{unit}**.")

    # Sentence 2: rate of change
    rate = target["rate"]
    slope_val = target["slope"]
    if "sharp" in rate and "rising" in rate:
        s2 = (f"On that day, {sensor} was **{rate}** "
              f"(slope: +{abs(slope_val):.3f}{unit}/step), indicating a rapid increase "
              f"rather than a gradual diurnal progression.")
    elif "sharp" in rate and "fall" in rate:
        s2 = (f"On that day, {sensor} was **{rate}** "
              f"(slope: −{abs(slope_val):.3f}{unit}/step), indicating a rapid drop.")
    elif "gradual" in rate and "rising" in rate:
        s2 = (f"The rise was **gradual** (slope: +{abs(slope_val):.3f}{unit}/step), "
              f"consistent with a normal diurnal warming trend rather than a sudden fault.")
    elif "gradual" in rate and "fall" in rate:
        s2 = (f"The decline was **gradual** (slope: −{abs(slope_val):.3f}{unit}/step), "
              f"consistent with expected evening cooling.")
    else:
        s2 = (f"The {sensor} was **stable** on that day "
              f"(slope: {slope_val:+.3f}{unit}/step), showing little directional trend.")

    # Sentence 3: anomaly check
    s3 = target["anom"] if target["anom"] else (
        f" The value of {peak_val:.1f}{unit} is within "
        f"{abs(peak_val - overall['mean']):.1f}{unit} of the period mean "
        f"({overall['mean']:.1f}{unit}) — within normal operating range.")

    # Sentence 4: cluster context
    s4 = (f"The overall environmental condition on that day was classified as the "
          f"**{target['cluster']}** cluster, indicating "
          f"{_cluster_desc(target['cluster'])}.")

    # Sentence 5: ranked list (top_n days)
    top_list = "\n".join([
        f"  {ri+1}. {d['date'].strftime('%a %d %b')} — "
        f"{d[val_key]:.1f}{unit} "
        f"(mean {d['mean']:.1f}, {d['rate']}, cluster: {d['cluster']})"
        for ri, d in enumerate(ranked[:top_n])
    ])
    s5 = (f"\n**Top {min(top_n, len(ranked))} days by {dir_word} {sensor} "
          f"in {period_label}:**\n{top_list}")

    # Sentence 6: period summary
    s6 = (f"\nAcross the full {period_label}, {sensor} ranged from "
          f"**{overall['global_min']:.1f}{unit}** to "
          f"**{overall['global_max']:.1f}{unit}** "
          f"(mean: {overall['mean']:.1f}{unit}, "
          f"std: {overall['std']:.1f}{unit}).")

    narrative = "\n\n".join([s1, s2.strip(), s3.strip(), s4]) + s5 + s6

    return {
        "ranked_days":   ranked,
        "overall_stats": overall,
        "direction":     direction,
        "sensor":        sensor,
        "period_label":  period_label,
        "narrative":     narrative,
        "top_n":         top_n,
    }


def _cluster_desc(cname):
    descs = {
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
    return descs.get(cname, "mixed environmental conditions")


# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTER QUERY ENGINE
# ══════════════════════════════════════════════════════════════════════════════
_CLUSTER_Q_PAT = _re.compile(
    r"\b(clusters?|conditions?|kondisi|kelompok|grup|kategori|tipe hari|"
    r"what clusters?|what condition|jenis kondisi|nama cluster|list cluster|"
    r"tampilkan cluster|show cluster|available cluster|all cluster|"
    r"cluster apa|cluster mana|ada cluster|daftar cluster|semua cluster|"
    r"show all|list all)\b",
    _re.I)

def detect_cluster_query(text, cluster_names):
    """
    Returns ('list', None) if user asks to list all clusters,
    returns ('detail', cluster_name) if a specific cluster is mentioned,
    returns (None, None) otherwise.
    """
    tl = text.lower().strip()
    # Check for specific cluster name mention
    for cid, cname in cluster_names.items():
        if cname.lower() in tl:
            return "detail", cname
    # Check for general cluster listing question
    if _CLUSTER_Q_PAT.search(text):
        return "list", None
    return None, None


def cluster_list_query(engine):
    """
    List all clusters with: window count, date ranges, day-of-week profile,
    mean sensor values, and a plain-English description.
    """
    from collections import defaultdict as _ddict
    cl_arr = engine.cl
    cn     = engine.cn

    # Build per-cluster stats
    cluster_info = {}
    for cid, cname in sorted(cn.items()):
        members = [i for i,l in enumerate(cl_arr) if l == cid]
        if not members:
            continue

        dates   = sorted(set(engine.metas[i]["timestamp"].date() for i in members))
        by_dow  = _ddict(int)
        for i in members:
            by_dow[engine.metas[i]["timestamp"].strftime("%A")] += 1

        vals = {}
        for k in ["temperature","pressure","humidity","light"]:
            arr = _np.array([engine.metas[i][k] for i in members])
            vals[k] = {"mean": round(float(arr.mean()),1),
                       "std":  round(float(arr.std()),1)}

        peak_dow = max(by_dow, key=by_dow.get)
        cluster_info[cid] = {
            "name":     cname,
            "n":        len(members),
            "pct":      round(len(members)/len(cl_arr)*100, 1),
            "dates":    dates,
            "n_days":   len(dates),
            "peak_dow": peak_dow,
            "vals":     vals,
        }

    # Build narrative
    lines = [f"**{len(cluster_info)} environmental clusters** are currently defined "
             f"in the dataset:\n"]
    for cid, info in sorted(cluster_info.items(), key=lambda x: -x[1]["n"]):
        cname = info["name"]
        d0    = info["dates"][0].strftime("%d %b")
        d1    = info["dates"][-1].strftime("%d %b %Y")
        lines.append(
            f"---\n"
            f"**{cname}** — {info['n']} windows ({info['pct']}% of dataset)\n"
            f"• Occurs across **{info['n_days']} days** "
            f"(from {d0} to {d1})\n"
            f"• Most frequent on **{info['peak_dow']}s**\n"
            f"• Mean temperature: **{info['vals']['temperature']['mean']}°C** "
            f"(±{info['vals']['temperature']['std']}°C)  "
            f"humidity: **{info['vals']['humidity']['mean']}% RH**  "
            f"light: **{info['vals']['light']['mean']} lux**\n"
            f"• {_cluster_desc(cname).capitalize()}."
        )

    lines.append("\n---\nType a cluster name (e.g. **Warm Day** or **Hot Afternoon**) "
                 "to see which specific dates belong to it.")
    return "\n\n".join(lines)


def cluster_detail_query(engine, cluster_name):
    """
    For a named cluster, list every date (with day-of-week),
    mean sensor values per date, and overall cluster statistics.
    """
    from collections import defaultdict as _ddict
    cl_arr = engine.cl
    cn     = engine.cn

    # Find cluster id
    cid = None
    for k, v in cn.items():
        if v.lower() == cluster_name.lower():
            cid = k
            break
    if cid is None:
        return f"Cluster **{cluster_name}** not found. Type **clusters** to see the full list."

    members = [i for i, l in enumerate(cl_arr) if l == cid]
    if not members:
        return f"Cluster **{cluster_name}** has no windows in the current dataset."

    by_date = _ddict(list)
    for i in members:
        by_date[engine.metas[i]["timestamp"].date()].append(i)

    # Per-date stats
    day_rows = []
    for d, idxs in sorted(by_date.items()):
        vals = {k: _np.array([engine.metas[i][k] for i in idxs])
                for k in ["temperature","humidity","pressure","light"]}
        day_rows.append({
            "date":   d,
            "dow":    d.strftime("%A"),
            "n":      len(idxs),
            "t_mean": round(float(vals["temperature"].mean()),1),
            "h_mean": round(float(vals["humidity"].mean()),1),
            "p_mean": round(float(vals["pressure"].mean()),1),
            "l_mean": round(float(vals["light"].mean()),1),
        })

    # Overall stats
    all_t = _np.array([engine.metas[i]["temperature"] for i in members])
    all_h = _np.array([engine.metas[i]["humidity"]    for i in members])

    # Dow frequency
    dow_cnt = _ddict(int)
    for r in day_rows: dow_cnt[r["dow"]] += 1
    peak_dow = max(dow_cnt, key=dow_cnt.get)

    # Months breakdown
    month_cnt = _ddict(int)
    for r in day_rows: month_cnt[r["date"].strftime("%B %Y")] += 1

    lines = [
        f"**Cluster: {cluster_name}**\n"
        f"{_cluster_desc(cluster_name).capitalize()}.\n",

        f"**Overview:** {len(members)} windows across **{len(day_rows)} days** "
        f"({round(len(members)/len(cl_arr)*100,1)}% of total dataset). "
        f"Most common day of week: **{peak_dow}**.\n",

        f"**Sensor profile:**  "
        f"Temp {round(float(all_t.mean()),1)}°C (±{round(float(all_t.std()),1)})  |  "
        f"Humidity {round(float(all_h.mean()),1)}% RH\n",
    ]

    # Monthly breakdown
    month_lines = "  ".join([f"{m}: {c} days" for m,c in sorted(month_cnt.items())])
    lines.append(f"**By month:** {month_lines}\n")

    # Day list (grouped by month for readability)
    lines.append(f"**All dates in this cluster:**")
    cur_month = None
    month_block = []
    for r in day_rows:
        m = r["date"].strftime("%B %Y")
        if m != cur_month:
            if month_block:
                lines.append(f"*{cur_month}*: " + ", ".join(month_block))
            cur_month = m; month_block = []
        month_block.append(
            f"{r['dow'][:3]} {r['date'].day} "
            f"(T:{r['t_mean']}°C H:{r['h_mean']}%)")
    if month_block:
        lines.append(f"*{cur_month}*: " + ", ".join(month_block))

    return "\n\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  RATE-OF-CHANGE QUERY ENGINE
# ══════════════════════════════════════════════════════════════════════════════
_INCREASE_PAT = _re.compile(
    r"\b(increas|ris|naik|kenaikan|peningkatan|naiknya|meningkat)\w*\b", _re.I)
_DECREASE_PAT = _re.compile(
    r"\b(decreas|drop|fall|turun|penurunan|turunnya|menurun)\w*\b", _re.I)
_FASTEST_PAT  = _re.compile(
    r"\b(fastest|sharpest|steepest|tercepat|terbesar|paling cepat|"
    r"paling tajam|highest rate|maximum rate)\b", _re.I)
_SLOWEST_PAT  = _re.compile(
    r"\b(slowest|gentlest|most gradual|terlambat|terkecil|"
    r"paling lambat|paling landai|lowest rate|minimum rate)\b", _re.I)


def detect_rate_query(text):
    """
    Returns (sensor, change_dir, rate_dir) or (None, None, None).
    change_dir: 'increase' | 'decrease'
    rate_dir  : 'fastest'  | 'slowest'
    """
    sensor = None
    for s, pat in _SENSOR_PATS.items():
        if pat.search(text):
            sensor = s; break
    if sensor is None:
        return None, None, None

    change_dir = None
    if _INCREASE_PAT.search(text):
        change_dir = "increase"
    elif _DECREASE_PAT.search(text):
        change_dir = "decrease"
    if change_dir is None:
        return None, None, None

    rate_dir = "fastest"   # default
    if _SLOWEST_PAT.search(text):
        rate_dir = "slowest"
    elif _FASTEST_PAT.search(text):
        rate_dir = "fastest"

    return sensor, change_dir, rate_dir


def rate_of_change_query(engine, indices, sensor, change_dir, rate_dir,
                          period_label, top_n=5):
    """
    Find days with the fastest/slowest rate of increase/decrease for a sensor.
    change_dir: 'increase' (positive slope) | 'decrease' (negative slope)
    rate_dir  : 'fastest' | 'slowest'
    Returns narrative string.
    """
    if not indices:
        return "No data found for the requested period."

    from collections import defaultdict as _ddict
    by_date = _ddict(list)
    for i in indices:
        by_date[engine.metas[i]["timestamp"].date()].append(i)

    unit     = SENSOR_UNITS[sensor]
    all_vals = _np.array([engine.metas[i][sensor] for i in indices])

    days = []
    for d, idxs in sorted(by_date.items()):
        vals   = _np.array([engine.metas[i][sensor] for i in idxs])
        t_axis = _np.arange(len(vals))
        slope  = float(_np.polyfit(t_axis, vals, 1)[0]) if len(vals) > 1 else 0.0
        delta  = float(vals[-1] - vals[0])

        cl_counts = _ddict(int)
        for i in idxs:
            if i < len(engine.cl): cl_counts[engine.cl[i]] += 1
        dom_cl  = max(cl_counts, key=cl_counts.get) if cl_counts else 0
        cl_name = engine.cn.get(dom_cl, "Unknown")

        days.append({
            "date":    d,
            "slope":   round(slope, 4),
            "delta":   round(delta, 2),
            "mean":    round(float(vals.mean()), 2),
            "max":     round(float(vals.max()), 2),
            "min":     round(float(vals.min()), 2),
            "rate":    _rate_label(slope),
            "cluster": cl_name,
            "n":       len(idxs),
        })

    # Filter by direction — if no days match strictly, use all days
    # (the "fastest increase" is simply the least-negative slope on a falling day)
    if change_dir == "increase":
        relevant = [d for d in days if d["slope"] > 0]
        if not relevant:          # fallback: pick day with highest (least negative) slope
            relevant = sorted(days, key=lambda d: d["slope"], reverse=True)[:max(1,len(days))]
        slope_desc = "increase (positive slope)"
    else:
        relevant  = [d for d in days if d["slope"] < 0]
        if not relevant:          # fallback: pick day with lowest (least positive) slope
            relevant = sorted(days, key=lambda d: d["slope"])[:max(1,len(days))]
        slope_desc = "decrease (negative slope)"

    # Sort: fastest = largest abs slope; slowest = smallest abs slope
    reverse = (rate_dir == "fastest")
    ranked  = sorted(relevant, key=lambda d: abs(d["slope"]), reverse=reverse)

    target   = ranked[0]
    rate_word = "fastest" if rate_dir == "fastest" else "slowest"
    peak_dt  = target["date"].strftime("%A, %d %B %Y")
    slope_v  = target["slope"]
    delta_v  = target["delta"]
    sign     = "+" if delta_v >= 0 else ""

    # ── narrative ──────────────────────────────────────────────────────────
    s1 = (f"The **{rate_word} {change_dir}** in {sensor} during **{period_label}** "
          f"occurred on **{peak_dt}**.\n"
          f"The slope was **{slope_v:+.4f}{unit}/step** "
          f"({sign}{delta_v:.1f}{unit} net change across the day).")

    # Rate characterisation
    abs_slope = abs(slope_v)
    if abs_slope > 0.08:
        char = "This constitutes a **sharp** rate of change — likely driven by a sudden environmental event or sensor fault."
    elif abs_slope > 0.02:
        char = "This is a **gradual** rate of change — consistent with a steady diurnal or weather-driven trend."
    else:
        char = "Despite being the most extreme in the period, this rate is still relatively **gentle**."

    # Anomaly flag
    all_slopes = _np.array([d["slope"] for d in days])
    slope_mu, slope_sd = float(all_slopes.mean()), float(all_slopes.std())
    z = (abs(slope_v) - abs(slope_mu)) / (slope_sd + 1e-9)
    if z > 2.0:
        anom = (f"\n⚠ This slope is **{z:.1f} standard deviations** above the "
                f"average daily slope ({slope_mu:+.4f}{unit}/step) — "
                f"statistically anomalous rate of change.")
    else:
        anom = ""

    # Cluster
    s_cl = (f"The dominant environmental cluster on that day was "
            f"**{target['cluster']}** — {_cluster_desc(target['cluster'])}.")

    # Ranked list
    list_lines = []
    for ri, d in enumerate(ranked[:top_n]):
        list_lines.append(
            f"  {ri+1}. {d['date'].strftime('%a %d %b')} — "
            f"slope {d['slope']:+.4f}{unit}/step  "
            f"(net {d['delta']:+.1f}{unit}, {d['rate']}, "
            f"cluster: {d['cluster']})")
    ranked_block = (f"\n**Top {min(top_n,len(ranked))} days by "
                    f"{rate_word} {change_dir} of {sensor} "
                    f"in {period_label}:**\n" + "\n".join(list_lines))

    overall_block = (f"\nAcross {period_label}, {sensor} daily slopes ranged from "
                     f"**{all_slopes.min():+.4f}** to **{all_slopes.max():+.4f}**{unit}/step "
                     f"(mean: {slope_mu:+.4f}, std: {slope_sd:.4f}).")

    return "\n\n".join([s1, char + anom, s_cl]) + ranked_block + overall_block




# ─────────────────────────────────────────────────────────────────────────────
#  GRU MODEL CLASS  (must match iot_train.py exactly for state_dict loading)
# ─────────────────────────────────────────────────────────────────────────────
if TORCH_OK:
    class IoTSLM(nn.Module):
        def __init__(self, input_size=4, hidden=128, n_layers=2,
                     pred_len=6, dropout=0.2):
            super().__init__()
            self.pred_len    = pred_len
            self.input_size  = input_size
            self.proj        = nn.Linear(input_size, hidden)
            self.gru         = nn.GRU(hidden, hidden, n_layers, batch_first=True,
                                      dropout=dropout if n_layers > 1 else 0.0)
            self.attn_q = nn.Linear(hidden, hidden)
            self.attn_k = nn.Linear(hidden, hidden)
            self.attn_v = nn.Linear(hidden, hidden)
            self.scale   = hidden ** 0.5
            self.decoder = nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden//2),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden//2, pred_len * input_size),
            )
            self.anom_head = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(),
                nn.Linear(32, 1), nn.Sigmoid()
            )

        def forward(self, x):
            h      = self.proj(x)
            enc, _ = self.gru(h)
            Q, K, V = self.attn_q(enc), self.attn_k(enc), self.attn_v(enc)
            w = torch.softmax(torch.bmm(Q, K.transpose(1,2))/self.scale, dim=-1)
            ctx  = torch.bmm(w, V).mean(dim=1)
            pred = self.decoder(ctx).view(-1, self.pred_len, self.input_size)
            return pred, self.anom_head(ctx), w

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📂 Loading pre-trained models from disk...")
def load_models(model_dir):
    """
    Load all saved artifacts from iot_train.py output directory.
    Returns (engine, config, hist, eval_metrics, load_report).
    """
    load_report = {}

    def lp(fname):
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found — run iot_train.py first")
        with open(path, "rb") as f:
            return pickle.load(f)

    def lj(fname):
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        with open(path) as f:
            return json.load(f)

    # Config
    cfg = lj("config.json")
    load_report["config"] = "✓"

    # Data
    data_raw = lp("data.pkl")
    # Restore datetime timestamps
    data = dict(data_raw)
    data["timestamps"] = [datetime.fromisoformat(s)
                          for s in data_raw["timestamps_str"]]
    load_report["data"] = "✓"

    # Features + metas
    X_feat, metas = lp("features.pkl")
    load_report["features"] = "✓"

    # K-Means
    km    = lp("kmeans.pkl")
    km_sc = lp("kmeans_scaler.pkl")
    cl    = lp("cluster_labels.pkl")
    cn_raw= lj("cluster_names.json")
    cn    = {int(k): v for k, v in cn_raw.items()}
    load_report["kmeans"] = "✓"

    # Decision Tree
    dt    = lp("dt.pkl")
    dt_sc = lp("dt_scaler.pkl")
    load_report["decision_tree"] = "✓"

    # SLM scaler
    slm_sc = lp("slm_scaler.pkl")

    # GRU model
    slm = None
    if TORCH_OK:
        weights_path = os.path.join(model_dir, "slm_weights.pt")
        if os.path.exists(weights_path):
            slm = IoTSLM(
                input_size=cfg.get("n_sensors", 4),
                hidden    =cfg.get("hidden", 128),
                n_layers  =cfg.get("n_layers", 2),
                pred_len  =cfg.get("pred_len", 6),
                dropout   =cfg.get("dropout", 0.2),
            )
            slm.load_state_dict(
                torch.load(weights_path, map_location="cpu",
                           weights_only=True))
            slm.eval()
            load_report["slm_gru"] = "✓"
        else:
            load_report["slm_gru"] = "weights file not found"
    else:
        load_report["slm_gru"] = "⚠ torch not available — forecast disabled"

    # Training history
    hist = lp("slm_history.pkl")
    load_report["history"] = "✓"

    # Eval metrics
    eval_metrics = lj("eval_metrics.json")
    load_report["eval_metrics"] = "✓"

    # Build engine
    seq_len = cfg.get("seq_len", 24)
    engine  = Engine(data, X_feat, metas, cl, cn, slm, slm_sc,
                     km, km_sc, dt, dt_sc, seq_len)

    return engine, cfg, hist, eval_metrics, load_report


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE HELPERS  (no training — pure forward pass)
# ─────────────────────────────────────────────────────────────────────────────
def forecast_window(slm, slm_sc, data, data_idx, seq_len=24):
    """Run a single forward pass to get 6-step forecast + anomaly score."""
    if slm is None or not TORCH_OK:
        return None, 0.0
    if data_idx < seq_len or data_idx > len(data["temperature"]):
        return None, 0.0
    win = np.stack([data[k][data_idx-seq_len:data_idx]
                    for k in SENSOR_KEYS], axis=1)
    if win.shape[0] < seq_len:
        return None, 0.0
    sc  = slm_sc.transform(win)
    xt  = torch.tensor(sc[None], dtype=torch.float32)
    with torch.no_grad():
        pred, anom, _ = slm(xt)
    phys  = slm_sc.inverse_transform(pred.squeeze(0).numpy())
    score = float(anom.squeeze())
    return phys, score


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class Engine:
    def __init__(self, data, X_feat, metas, cl, cn,
                 slm, slm_sc, km, km_sc, dt, dt_sc, seq=24):
        self.data   = data;  self.X     = X_feat; self.metas = metas
        self.cl     = cl;    self.cn    = cn
        self.slm    = slm;   self.slm_sc= slm_sc
        self.km     = km;    self.km_sc = km_sc
        self.dt     = dt;    self.dt_sc = dt_sc
        self.seq    = seq
        self._idx   = defaultdict(list)
        for i, m in enumerate(metas):
            self._idx[m["timestamp"].date()].append(i)

    def indices(self, s, e):
        out = []; d = s
        while d <= e:
            out.extend(self._idx.get(d, []))
            d += timedelta(1)
        return sorted(out)

    def query(self, s, e, lbl="period"):
        idx = self.indices(s, e)
        if not idx:
            return None

        def sv(k): return np.array([self.metas[i][k] for i in idx])

        stats = {}
        for k in SENSOR_KEYS:
            v = sv(k); t = np.arange(len(v))
            slope = float(np.polyfit(t, v, 1)[0]) if len(v) > 1 else 0.0
            stats[k] = {
                "mean": float(v.mean()), "std":  float(v.std()),
                "min":  float(v.min()),  "max":  float(v.max()),
                "median": float(np.median(v)), "iqr": float(np.percentile(v,75)-np.percentile(v,25)),
                "first": float(v[0]),    "last": float(v[-1]),
                "trend": slope,
            }

        # Cluster
        cl_arr = [self.cl[i] for i in idx if i < len(self.cl)]
        c      = Counter(cl_arr)
        did    = max(c, key=c.get) if c else 0

        # DT notification
        fs    = self.dt_sc.transform(self.X[idx])
        preds = self.dt.predict(fs)
        nc    = Counter(preds)
        notif = "NO_ACTION"
        for ci in [8,7,4,5,6,3,1,2,0]:
            if nc[ci] > 0: notif = NOTIF_CLASSES[ci]; break

        # Forecast
        data_idx = min(max(idx) + self.seq, len(self.data["temperature"]))
        fore, ascore = forecast_window(self.slm, self.slm_sc,
                                       self.data, data_idx, self.seq)

        n_anom = sum(1 for i in idx if self.metas[i]["is_anomaly"])
        return {
            "label":        lbl,
            "start":        s, "end": e,
            "indices":      idx,
            "n_windows":    len(idx),
            "stats":        stats,
            "cluster_id":   did,
            "cluster_name": self.cn.get(did, "Unknown"),
            "notification": notif,
            "n_anomaly":    n_anom,
            "anomaly_rate": n_anom / max(len(idx), 1),
            "forecast":     fore,
            "anomaly_score":ascore,
        }

    def series(self, idx, sensor):
        ts = [self.metas[i]["timestamp"] for i in idx]
        vs = np.array([self.metas[i][sensor] for i in idx])
        return ts, vs

    def cluster_series(self, idx):
        return [self.cn.get(self.cl[i] if i < len(self.cl) else 0, "?")
                for i in idx]


# ─────────────────────────────────────────────────────────────────────────────
#  QUERY PARSER
# ─────────────────────────────────────────────────────────────────────────────
class Parser:
    PATS = {
        "today":      re.compile(r"\b(today|this day|hari ini)\b",    re.I),
        "yesterday":  re.compile(r"\b(yesterday|kemarin)\b",           re.I),
        "this_week":  re.compile(r"\b(this week|minggu ini)\b",        re.I),
        "last_week":  re.compile(r"\b(last week|minggu lalu|minggu kemarin)\b", re.I),
        "this_month": re.compile(r"\b(this month|bulan ini)\b",        re.I),
        "last_month": re.compile(r"\b(last month|bulan lalu|bulan kemarin)\b",  re.I),
        "2days_ago":  re.compile(r"\b(2 days? ago|2 hari lalu)\b",     re.I),
        "3days_ago":  re.compile(r"\b(3 days? ago|3 hari lalu)\b",     re.I),
    }
    def __init__(self, ref): self.ref = ref

    def resolve(self, key):
        r = self.ref
        if key == "today":      return r, r, "Today"
        if key == "yesterday":  d = r-timedelta(1); return d, d, "Yesterday"
        if key == "this_week":  s = r-timedelta(r.weekday()); return s, r, "This Week"
        if key == "last_week":
            e = r-timedelta(r.weekday()+1); return e-timedelta(6), e, "Last Week"
        if key == "this_month": return r.replace(day=1), r, "This Month"
        if key == "last_month":
            e = (r.replace(day=1)-timedelta(1))
            return e.replace(day=1), e, "Last Month"
        if key == "2days_ago":  d = r-timedelta(2); return d, d, "2 Days Ago"
        if key == "3days_ago":  d = r-timedelta(3); return d, d, "3 Days Ago"
        return None, None, None

    def parse(self, text):
        found = []
        for key, pat in self.PATS.items():
            if pat.search(text):
                s, e, lbl = self.resolve(key)
                if s:
                    found.append((s, e, lbl))
                if len(found) == 2:
                    break
        return found

    def parse_ranking(self, text):
        """Returns (sensor, direction, periods) if ranking query detected."""
        sensor, direction = detect_ranking_query(text)
        if sensor is None:
            return None, None, []
        periods = self.parse(text)
        return sensor, direction, periods

    def parse_cluster(self, text, cluster_names):
        """Returns (query_type, cluster_name_or_None)."""
        return detect_cluster_query(text, cluster_names)

    def parse_rate(self, text):
        """Returns (sensor, change_dir, rate_dir) for rate-of-change queries."""
        return detect_rate_query(text)


# ─────────────────────────────────────────────────────────────────────────────
#  NARRATION
# ─────────────────────────────────────────────────────────────────────────────
def _td(v):
    if v>=35: return "very hot"
    if v>=30: return "hot"
    if v>=26: return "warm"
    if v>=22: return "comfortable"
    if v>=18: return "cool"
    return "cold"
def _hd(v):
    if v>=85: return "very humid"
    if v>=70: return "humid"
    if v>=50: return "moderate"
    return "dry"
def _tw(s):
    if abs(s)<0.001: return "→ stable"
    if s>0.05:  return "↑ rising sharply"
    if s>0.01:  return "↑ gradually rising"
    if s<-0.05: return "↓ dropping sharply"
    if s<-0.01: return "↓ gradually falling"
    return "→ relatively stable"

def narrate(result, comp=None, comp_label=""):
    ss = result["stats"]; pl = result["label"]
    T  = ss["temperature"]; H = ss["humidity"]
    P  = ss["pressure"];    L = ss["light"]
    cname = result["cluster_name"]
    cond  = CLUSTER_COND.get(cname, "mixed conditions")

    p1 = (f"During **{pl}**, the sensor array recorded **{cond}** "
          f"(cluster: *{cname}*). "
          f"Temperature averaged **{T['mean']:.1f}°C** ({_td(T['mean'])}), "
          f"ranging {T['min']:.1f}–{T['max']:.1f}°C (median {T['median']:.1f}°C). "
          f"Humidity was **{H['mean']:.1f}% RH** ({_hd(H['mean'])}). "
          f"Pressure averaged **{P['mean']:.1f} hPa**. "
          f"Light averaged **{L['mean']:.0f} lux**.")

    p2 = (f"**Trend:** Temperature {_tw(T['trend'])} "
          f"({T['first']:.1f}→{T['last']:.1f}°C). "
          f"Humidity {_tw(H['trend'])}. "
          f"Pressure {_tw(P['trend'])}. "
          f"Light {_tw(L['trend'])}.")

    parts = [p1, p2]

    if comp and comp.get("stats"):
        cs = comp["stats"]
        def cmp(v1, v2, k):
            diff = v1 - v2; u = SENSOR_UNITS[k]
            if abs(diff) < 0.3:
                return f"{k.capitalize()} similar to {comp_label} ({v1:.1f}{u})"
            d = "higher" if diff > 0 else "lower"
            m = "significantly" if abs(diff) > 3 else "slightly"
            return (f"{k.capitalize()} {m} {d} than {comp_label} "
                    f"({v1:.1f}{u} vs {v2:.1f}{u}, Δ{diff:+.1f}{u})")
        lines = [cmp(T["mean"], cs["temperature"]["mean"], "temperature"),
                 cmp(H["mean"], cs["humidity"]["mean"],    "humidity"),
                 cmp(P["mean"], cs["pressure"]["mean"],    "pressure")]
        parts.append("**Comparison with " + comp_label + ":** " +
                      " | ".join(lines) + ".")

    na = result["n_anomaly"]; nt = result["n_windows"]
    anom_s = ("No anomalies detected." if na == 0
              else f"**{na} anomalies** detected "
                   f"({na/max(nt,1)*100:.1f}% of {nt} windows).")
    parts.append(anom_s + " " + NOTIF_MSG.get(result["notification"], ""))

    fore = result["forecast"]
    if fore is not None:
        ft = fore[:,0].mean(); fh = fore[:,2].mean(); fl = fore[:,3].mean()
        parts.append(f"**3-hour forecast:** {ft:.1f}°C ({_td(ft)}), "
                     f"humidity {fh:.1f}% ({_hd(fh)}), "
                     f"light {fl:.0f} lux.")
    elif not TORCH_OK:
        parts.append("*Forecast unavailable: torch not installed.*")

    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTLY CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def _sub(ts, vs, mx=400):
    n = len(vs)
    if n <= mx: return ts, vs
    step = n // mx; idx = list(range(0, n, step))
    return [ts[i] for i in idx], vs[idx]

def make_charts(engine, results):
    fig = make_subplots(rows=3, cols=2,
        subplot_titles=("Temperature (°C)", "Humidity (% RH)",
                        "Temp vs Light", "Temp vs Humidity (by Cluster)",
                        "Pressure (hPa)", "Hourly Mean Temp"),
        vertical_spacing=0.10, horizontal_spacing=0.08)

    pal  = [COLORS["A"], COLORS["B"]]
    fill = [COLORS["A_fill"], COLORS["B_fill"]]

    for ri, result in enumerate(results):
        if result is None: continue
        idx = result["indices"]; lbl = result["label"]
        col = pal[ri]; fl2 = fill[ri]

        # Temperature
        ts, vs = engine.series(idx, "temperature"); ts, vs = _sub(ts, vs)
        fig.add_trace(go.Scatter(x=ts, y=vs, name=lbl,
                                  line=dict(color=col, width=1.8),
                                  fill="tozeroy", fillcolor=fl2,
                                  legendgroup=lbl, showlegend=(ri==0)),
                      row=1, col=1)
        fore = result.get("forecast")
        if fore is not None and len(ts) > 0:
            lt  = ts[-1]
            fts = [lt + timedelta(minutes=30*(i+1)) for i in range(6)]
            fig.add_trace(go.Scatter(x=fts, y=fore[:,0],
                                      line=dict(color=col, width=1.4, dash="dash"),
                                      showlegend=False, legendgroup=lbl),
                          row=1, col=1)

        # Humidity
        ts2, vs2 = engine.series(idx, "humidity"); ts2, vs2 = _sub(ts2, vs2)
        fig.add_trace(go.Scatter(x=ts2, y=vs2, name=lbl,
                                  line=dict(color=col, width=1.8),
                                  fill="tozeroy", fillcolor=fl2,
                                  showlegend=False, legendgroup=lbl),
                      row=1, col=2)

        # Scatter T vs L
        T_v = np.array([engine.metas[i]["temperature"] for i in idx])
        L_v = np.array([engine.metas[i]["light"]       for i in idx])
        step = max(1, len(T_v)//300); T_s, L_s = T_v[::step], L_v[::step]
        fig.add_trace(go.Scatter(x=T_s, y=L_s, mode="markers",
                                  marker=dict(color=col, size=5, opacity=0.4),
                                  name=lbl, showlegend=False, legendgroup=lbl),
                      row=2, col=1)
        if len(T_s) > 2:
            m, b = np.polyfit(T_s, L_s, 1)
            xr   = np.array([T_s.min(), T_s.max()])
            fig.add_trace(go.Scatter(x=xr, y=m*xr+b, mode="lines",
                                      line=dict(color=col, width=2, dash="dash"),
                                      showlegend=False), row=2, col=1)

        # Scatter T vs H by cluster
        H_v = np.array([engine.metas[i]["humidity"] for i in idx])
        C_v = engine.cluster_series(idx)
        step2 = max(1, len(T_v)//250)
        T_sc, H_sc, C_sc = T_v[::step2], H_v[::step2], C_v[::step2]
        cmap = px.colors.qualitative.Set2
        for ci2, cname in enumerate(sorted(set(C_sc))):
            mask = np.array([c == cname for c in C_sc])
            fig.add_trace(go.Scatter(x=T_sc[mask], y=H_sc[mask], mode="markers",
                                      name=cname,
                                      marker=dict(color=cmap[ci2 % len(cmap)],
                                                  size=5, opacity=0.5),
                                      showlegend=(ri==0),
                                      legendgroup=f"cl_{cname}"),
                          row=2, col=2)

        # Pressure
        ts3, vs3 = engine.series(idx, "pressure"); ts3, vs3 = _sub(ts3, vs3)
        fig.add_trace(go.Scatter(x=ts3, y=vs3, name=lbl,
                                  line=dict(color=col, width=1.6),
                                  showlegend=False, legendgroup=lbl),
                      row=3, col=1)

        # Hourly bar
        by_h = defaultdict(list)
        for i in idx:
            by_h[engine.metas[i]["timestamp"].hour].append(
                engine.metas[i]["temperature"])
        hours  = list(range(24))
        means  = [np.mean(by_h[h]) if by_h[h] else float("nan") for h in hours]
        offset = ri * 0.4 - 0.2 * (len(results)-1)
        fig.add_trace(go.Bar(x=[h+offset for h in hours], y=means, width=0.38,
                              name=lbl, marker_color=col, opacity=0.82,
                              showlegend=False, legendgroup=lbl),
                      row=3, col=2)

    fig.add_hline(y=1013.25, line_dash="dot", line_color="#888",
                  annotation_text="ISA 1013", row=3, col=1)
    fig.update_layout(
        height=760, paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(family="Inter", size=11, color=COLORS["navy"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1),
        margin=dict(l=50, r=20, t=55, b=35),
    )
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].gridcolor = COLORS["grid"]
    return fig


def training_chart(hist):
    ep  = list(range(1, len(hist["train"])+1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=hist["train"], name="Train",
                              line=dict(color=COLORS["A"], width=2)))
    fig.add_trace(go.Scatter(x=ep, y=hist["val"], name="Validation",
                              line=dict(color=COLORS["B"], width=2, dash="dash")))
    fig.update_layout(height=260, paper_bgcolor=COLORS["bg"],
                      plot_bgcolor=COLORS["bg"],
                      font=dict(size=11, color=COLORS["navy"]),
                      title="Training Convergence (MSE)",
                      xaxis_title="Epoch", yaxis_title="MSE",
                      margin=dict(l=40, r=10, t=50, b=30),
                      legend=dict(orientation="h"))
    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  STAT CARDS
# ─────────────────────────────────────────────────────────────────────────────
def stat_cards(result):
    ss   = result["stats"]
    cols = st.columns(4)
    for k, col in zip(SENSOR_KEYS, cols):
        s = ss[k]; u = SENSOR_UNITS[k]
        tr = s["trend"]
        if abs(tr) < 0.001: tc, ta = "stat-trend-st", "→ stable"
        elif tr > 0:         tc, ta = "stat-trend-up", f"↑ +{abs(tr):.3f}/step"
        else:                tc, ta = "stat-trend-dn", f"↓ -{abs(tr):.3f}/step"
        col.markdown(f"""
        <div class="stat-card">
          <div class="stat-val">{s['mean']:.1f}<small style="font-size:.75rem"> {u}</small></div>
          <div class="stat-lbl">{k.capitalize()} — Mean</div>
          <div class="stat-lbl">Median {s['median']:.1f} · IQR {s['iqr']:.1f}</div>
          <div class="stat-lbl">Min {s['min']:.1f} · Max {s['max']:.1f}</div>
          <div class="{tc}">{ta}</div>
        </div>""", unsafe_allow_html=True)


def notif_badge(result):
    n   = result["notification"]
    sev = SEVERITY_MAP.get(n, "INFO")
    st.markdown(
        f'<span class="nbadge {sev}">{sev}</span> '
        f'<span style="font-size:.85rem;margin-left:6px">{n}</span>',
        unsafe_allow_html=True)
    st.caption(NOTIF_MSG.get(n, ""))


# ─────────────────────────────────────────────────────────────────────────────
#  HOURLY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def hourly_table(engine, result, sensor="temperature"):
    idx = result.get("indices", [])
    if not idx: return None
    by_h = defaultdict(list)
    for i in idx:
        by_h[engine.metas[i]["timestamp"].hour].append(engine.metas[i][sensor])
    rows = []
    for h in range(24):
        if by_h[h]:
            arr = np.array(by_h[h])
            rows.append({"Hour": f"{h:02d}:00",
                         "Mean": round(arr.mean(), 2),
                         "Median": round(float(np.median(arr)), 2),
                         "Min":  round(arr.min(), 2),
                         "Max":  round(arr.max(), 2),
                         "Std":  round(arr.std(), 2),
                         "Count": len(arr)})
    return pd.DataFrame(rows) if rows else None


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Session state
    for key, default in [("messages",[]), ("last_results",[])]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Load models ───────────────────────────────────────────────────────
    load_ok = True
    load_err = ""
    try:
        engine, cfg, hist, eval_metrics, load_report = load_models(MODEL_DIR)
    except Exception as e:
        load_ok  = False
        load_err = str(e)

    if not load_ok:
        st.error(f"**Cannot load models from `{MODEL_DIR}`**\n\n"
                 f"Error: `{load_err}`\n\n"
                 "Run the training script first:\n"
                 "```\npython iot_train.py\n```")
        st.stop()

    ref_date = datetime.fromisoformat(cfg["ref_date"]).date()
    parser   = Parser(ref_date)

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="header-bar">
      <div style="font-size:26px">🌡️</div>
      <div>
        <h1>IoT-SLM Environmental Chatbot</h1>
        <p>Pre-trained models loaded from <b>{MODEL_DIR}</b>  ·
           GRU-RNN · K-Means · Decision Tree  ·
           Ref date <b>{ref_date}</b>  ·
           {cfg.get('n_windows',0):,} windows</p>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Model Status")
        for k, v in load_report.items():
            icon = "✅" if v == "✓" else "⚠️"
            st.markdown(f"{icon} **{k}**: {v}")

        st.markdown("---")
        st.markdown("## 📊 Metrics (Validation)")
        for sensor, m in eval_metrics.items():
            st.markdown(f"**{sensor.capitalize()}**  "
                        f"R²={m['r2']}  MAE={m['mae']}")

        st.markdown("---")
        st.markdown("## 📋 Quick Queries")
        quick = ["today", "yesterday", "this week", "last week",
                 "this month", "last month",
                 "compare today and yesterday",
                 "compare this week and last week"]
        for qq in quick:
            if st.button(qq.title(), key=f"qq_{qq}"):
                st.session_state._pending = qq

        st.markdown("---")
        st.markdown("## 🌐 Query Language")
        st.markdown("""
| English | Indonesian |
|---|---|
| today | hari ini |
| yesterday | kemarin |
| this week | minggu ini |
| last week | minggu lalu |
| this month | bulan ini |
| last month | bulan lalu |
        """)

        st.markdown("---")
        st.markdown(f"## 📁 Model Directory\n`{MODEL_DIR}`")
        st.markdown(f"**Trained at:** {cfg.get('trained_at','?')}")
        st.markdown(f"**Days:** {cfg.get('days')}  ·  "
                    f"**Interval:** {cfg.get('interval_min')} min")
        st.markdown(f"**Epochs:** {cfg.get('epochs',60) if 'epochs' in cfg else '—'}")

    # ── Layout ────────────────────────────────────────────────────────────
    col_chat, col_vis = st.columns([1, 1.4], gap="medium")

    # ══ Chat ════════════════════════════════════════════════════════════
    with col_chat:
        st.markdown("### 💬 Chat")
        chat_box = st.container(height=440)
        with chat_box:
            if not st.session_state.messages:
                st.markdown("""<div class="bot-msg">
👋 Hi! I'm your <b>IoT-SLM Environmental Assistant</b>.<br>
All models are pre-loaded from disk — no training on startup!<br><br>
Ask me about sensor conditions using time expressions:<br>
<i>"today"</i>, <i>"this week"</i>, <i>"compare today and yesterday"</i>.<br>
Indonesian phrases also work: <i>"hari ini"</i>, <i>"minggu ini"</i>.
</div>""", unsafe_allow_html=True)
            for msg in st.session_state.messages:
                css = "user-msg" if msg["role"] == "user" else "bot-msg"
                icon = "👤" if msg["role"] == "user" else "🤖"
                st.markdown(
                    f'<div class="{css}">{icon} {msg["content"]}</div>',
                    unsafe_allow_html=True)

        # Input
        user_input = st.chat_input("Ask about sensor conditions...")
        if hasattr(st.session_state, "_pending"):
            user_input = st.session_state._pending
            del st.session_state._pending

        if user_input:
            st.session_state.messages.append(
                {"role": "user", "content": user_input})

            # ── 0. Cluster query (list all or detail one) ─────────────
            cl_qtype, cl_name = parser.parse_cluster(user_input, engine.cn)
            if cl_qtype == "list":
                reply   = cluster_list_query(engine)
                results = []

            elif cl_qtype == "detail" and cl_name:
                reply   = cluster_detail_query(engine, cl_name)
                results = []

            # ── 1. Rate-of-change query ───────────────────────────────────
            elif (lambda s,c,r: s is not None)(
                    *parser.parse_rate(user_input)):
                r_sensor, r_change, r_rate = parser.parse_rate(user_input)
                r_periods2 = parser.parse(user_input)
                if r_periods2:
                    s2, e2, lbl2 = r_periods2[0]
                    base2 = engine.query(s2, e2, lbl2)
                    if base2:
                        reply = rate_of_change_query(
                            engine, base2["indices"],
                            r_sensor, r_change, r_rate, lbl2, top_n=5)
                        results = [base2]
                    else:
                        reply = "No data found for the requested period."; results = []
                else:
                    reply = ("Please specify a time period. Example: "
                             "**fastest temperature increase this week**, "
                             "**slowest humidity decrease this month**."); results = []

            # ── 2. Ranking query (highest/lowest value) ───────────────────
            elif (lambda s,d,p: s is not None)(
                    *parser.parse_ranking(user_input)):
                sensor, direction, r_periods = parser.parse_ranking(user_input)
            direction = None
            sensor = None
            r_periods = None
            if sensor and direction and r_periods:
                s, e, lbl = r_periods[0]
                base = engine.query(s, e, lbl)
                if base:
                    rk = ranking_query(engine, base["indices"],
                                       sensor, direction, lbl, top_n=5)
                    reply   = rk["narrative"] if rk else "No data found."
                    results = [base]
                else:
                    reply   = "No data found for the requested period."
                    results = []

            else:
                # ── 3. Standard temporal query ────────────────────────────
                periods = parser.parse(user_input)
                if not periods:
                    reply   = (
                        "I didn't recognise the query. Here are examples:\n\n"
                        "**Temporal:** today · this week · compare today and yesterday\n"
                        "**Ranking:** highest temperature this week · suhu tertinggi hari ini\n"
                        "**Rate:** fastest temperature increase this week · "
                        "slowest humidity decrease this month\n"
                        "**Clusters:** clusters · what clusters exist · "
                        "Hot Afternoon · Warm Day")
                    results = []
                else:
                    results = []
                    for s, e, lbl in periods[:2]:
                        r = engine.query(s, e, lbl)
                        if r: results.append(r)

                    if not results:
                        reply = "No data found for the requested period."
                    elif len(results) == 1:
                        reply = narrate(results[0])
                    else:
                        r1, r2 = results[0], results[1]
                        reply = (f"**― {r1['label'].upper()} ―**\n\n"
                                 + narrate(r1, comp=r2, comp_label=r2["label"])
                                 + f"\n\n**― {r2['label'].upper()} ―**\n\n"
                                 + narrate(r2, comp=r1, comp_label=r1["label"]))

            st.session_state.messages.append(
                {"role": "bot", "content": reply})
            st.session_state.last_results = results
            st.rerun()

    # ══ Visualisation ════════════════════════════════════════════════════
    with col_vis:
        st.markdown("### 📈 Sensor Visualisation")
        t1, t2, t3, t4 = st.tabs(
            ["📊 Charts", "🧩 Stat Cards", "📋 Data Tables", "🔬 Model Info"])

        # Charts
        with t1:
            if st.session_state.last_results:
                fig = make_charts(engine, st.session_state.last_results)
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": True})
                for res in st.session_state.last_results:
                    notif_badge(res)
            else:
                st.info("💡 Ask a question in the chat to see charts.")
                st.plotly_chart(training_chart(hist),
                                use_container_width=True,key="training_chart")

        # Stat Cards
        with t2:
            if st.session_state.last_results:
                for res in st.session_state.last_results:
                    st.markdown(f"#### {res['label']} "
                                f"({res['start']} → {res['end']})")
                    stat_cards(res)
                    notif_badge(res)
                    st.markdown("---")
            else:
                st.info("Ask a question to see sensor statistics.")

        # Data Tables
        with t3:
            if st.session_state.last_results:
                for res in st.session_state.last_results:
                    st.markdown(f"#### {res['label']}")
                    sensor_t = st.selectbox(
                        "Sensor for hourly table", SENSOR_KEYS,
                        key=f"sel_{res['label']}",
                        format_func=str.capitalize)
                    df = hourly_table(engine, res, sensor_t)
                    if df is not None:
                        st.dataframe(df, use_container_width=True,
                                     height=300, hide_index=True)
                    # Full stats with median + IQR
                    ss = res["stats"]
                    sumdf = pd.DataFrame([
                        {"Sensor":  k.capitalize(),
                         "Mean":    round(ss[k]["mean"], 2),
                         "Median":  round(ss[k]["median"], 2),
                         "Std":     round(ss[k]["std"], 2),
                         "IQR":     round(ss[k]["iqr"], 2),
                         "Min":     round(ss[k]["min"], 2),
                         "Max":     round(ss[k]["max"], 2),
                         "Trend":   round(ss[k]["trend"], 4),
                         "Unit":    SENSOR_UNITS[k]}
                        for k in SENSOR_KEYS])
                    st.markdown("**Overall Statistics**")
                    st.dataframe(sumdf, use_container_width=True,
                                 hide_index=True)
                    st.markdown("---")
            else:
                st.info("Ask a question to see data tables.")

        # Model Info
        with t4:
            st.markdown("#### Training Convergence")
            st.plotly_chart(training_chart(hist),
                            use_container_width=True)

            st.markdown("#### Forecasting Metrics (Validation Set)")
            mdf = pd.DataFrame([
                {"Sensor":    sensor.capitalize(),
                 "MAE":       m["mae"],
                 "RMSE":      m["rmse"],
                 "R²":        m["r2"]}
                for sensor, m in eval_metrics.items()
            ])
            st.dataframe(mdf, use_container_width=True, hide_index=True)

            st.markdown("#### K-Means Cluster Summary")
            cl_arr = engine.cl
            cnt    = Counter(cl_arr.tolist())
            cldf   = pd.DataFrame([
                {"Cluster": k, "Label": engine.cn.get(k, f"C{k}"),
                 "Windows": cnt.get(k, 0)}
                for k in sorted(engine.cn.keys())
            ])
            st.dataframe(cldf, use_container_width=True, hide_index=True)

            st.markdown("#### GRU-RNN Architecture")
            arch = pd.DataFrame([
                {"Layer":"Input Projection","In":"(B,24,4)","Out":"(B,24,128)","Params":"640"},
                {"Layer":"GRU Layer 1","In":"(B,24,128)","Out":"(B,24,128)","Params":"99,072"},
                {"Layer":"GRU Layer 2","In":"(B,24,128)","Out":"(B,24,128)","Params":"99,072"},
                {"Layer":"Attention Q/K/V","In":"(B,24,128)","Out":"(B,24,128)","Params":"49,152"},
                {"Layer":"Pred Decoder","In":"(B,128)","Out":"(B,6,4)","Params":"12,388"},
                {"Layer":"Anomaly Head","In":"(B,128)","Out":"(B,1)","Params":"2,114"},
                {"Layer":"Total","In":"—","Out":"—","Params":"262,553"},
            ])
            st.dataframe(arch, use_container_width=True, hide_index=True)

            st.markdown("#### Model Load Report")
            for k, v in load_report.items():
                icon = "✅" if v == "✓" else "⚠️"
                st.markdown(f"{icon} `{k}`: {v}")

            st.markdown(f"#### Config\n```json\n"
                        + json.dumps(cfg, indent=2, default=str)
                        + "\n```")


if __name__ == "__main__":
    main()
