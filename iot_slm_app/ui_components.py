"""
ui_components.py — Small reusable Streamlit render helpers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What lives here:
  • stat_cards(result): the 4-column Temperature/Pressure/Humidity/Light
    "stat-card" row (mean, median, IQR, min/max, trend) used in the
    Statistics tab.
  • notif_badge(result): the colored NOMINAL/INFO/WATCH/WARNING/CRITICAL
    pill + message, used in both the Charts and Statistics tabs.
  • hourly_table(engine, result, sensor): builds the 24-row hourly
    mean/median/min/max/std/count DataFrame used in the Data Tables tab.

These are intentionally tiny and Streamlit-specific (unlike
nlp_queries.py / narration.py, which are pure functions) — they call
st.columns()/st.markdown() directly.
"""

import numpy as np
import pandas as pd
import streamlit as st
from collections import defaultdict

from .config import SENSOR_KEYS, SENSOR_UNITS, SEVERITY_MAP, NOTIF_MSG


def stat_cards(result):
    ss   = result["stats"]
    cols = st.columns(4)
    for k, col in zip(SENSOR_KEYS, cols):
        s = ss[k]; u = SENSOR_UNITS[k]
        tr = s["trend"]
        if abs(tr) < 0.001:  tc, ta = "trend-st", "stable"
        elif tr > 0.05:      tc, ta = "trend-up", f"rising sharply ({tr:+.3f}/step)"
        elif tr > 0:         tc, ta = "trend-up", f"rising ({tr:+.3f}/step)"
        elif tr < -0.05:     tc, ta = "trend-dn", f"dropping sharply ({tr:+.3f}/step)"
        else:                tc, ta = "trend-dn", f"falling ({tr:+.3f}/step)"
        col.markdown(f"""
        <div class="stat-card">
          <div class="stat-val">{s['mean']:.1f} <span class="stat-unit">{u}</span></div>
          <div class="stat-lbl"><strong>{k.capitalize()}</strong> — Mean</div>
          <div class="stat-sub">Median {s['median']:.1f} &nbsp;|&nbsp; IQR {s['iqr']:.1f}</div>
          <div class="stat-sub">Min {s['min']:.1f} &nbsp;|&nbsp; Max {s['max']:.1f}</div>
          <div class="{tc}">{ta}</div>
        </div>""", unsafe_allow_html=True)


def notif_badge(result):
    n   = result["notification"]
    sev = SEVERITY_MAP.get(n, "INFO")
    msg = NOTIF_MSG.get(n, "")
    st.markdown(
        f'<div style="margin:0.5rem 0">'
        f'<span class="nbadge {sev}">{sev}</span> '
        f'<span style="font-size:0.87rem;color:#212529;margin-left:0.5rem">{n}</span>'
        f'<br><span style="font-size:0.82rem;color:#6c757d">{msg}</span></div>',
        unsafe_allow_html=True)


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
