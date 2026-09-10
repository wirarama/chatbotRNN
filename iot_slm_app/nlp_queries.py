"""
nlp_queries.py — Regex-based intent detection + narrative generation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Three independent "mini chatbot engines" live in this file, each pairing
a `detect_*()` intent classifier with a `*_query()` narrative builder:

  1. RANKING   — "highest/lowest temperature this week" / "suhu tertinggi"
       detect_ranking_query(text)              -> (sensor, direction)
       ranking_query(engine, indices, ...)      -> dict incl. "narrative"

  2. CLUSTER   — "clusters" / "Hot Afternoon" / "cluster apa saja"
       detect_cluster_query(text, cluster_names) -> ("list"|"detail", name)
       cluster_list_query(engine)               -> narrative string
       cluster_detail_query(engine, name)       -> narrative string

  3. RATE OF CHANGE — "fastest temperature increase this month"
       detect_rate_query(text)                  -> (sensor, change_dir, rate_dir)
       rate_of_change_query(engine, indices,...) -> narrative string

All three are pure functions of (text, engine) — no Streamlit, no session
state — so they're easy to unit-test standalone.

When to edit this file:
  • Adding new phrasing/synonyms (English or Indonesian) -> the regex
    patterns near the top (_SENSOR_PATS, _HIGH_PAT, _INCREASE_PAT, etc).
  • Changing how a narrative reads -> the corresponding `*_query()` builder.

Note: sensor-word detection here is intentionally separate from the
notification-rule parser in rules.py, since rule conditions need
precise per-clause parsing (with numeric thresholds), while this file
only needs to classify the *overall* question being asked.
"""

import re as _re
from collections import defaultdict as _dd

import numpy as _np

from .config import SENSOR_UNITS, cluster_desc

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
            "anom":    _anomaly_note(vals.max() if direction == "high" else vals.min(), all_vals),
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
          f"{cluster_desc(target['cluster'])}.")

    # Sentence 5: ranked list (top_n days)
    top_list = "\n".join([
        f"  {ri + 1}. {d['date'].strftime('%a %d %b')} — "
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
    cl_arr = engine.cl
    cn     = engine.cn

    # Build per-cluster stats
    cluster_info = {}
    for cid, cname in sorted(cn.items()):
        members = [i for i, l in enumerate(cl_arr) if l == cid]
        if not members:
            continue

        dates  = sorted(set(engine.metas[i]["timestamp"].date() for i in members))
        by_dow = _dd(int)
        for i in members:
            by_dow[engine.metas[i]["timestamp"].strftime("%A")] += 1

        vals = {}
        for k in ["temperature", "pressure", "humidity", "light"]:
            arr = _np.array([engine.metas[i][k] for i in members])
            vals[k] = {"mean": round(float(arr.mean()), 1),
                       "std":  round(float(arr.std()), 1)}

        peak_dow = max(by_dow, key=by_dow.get)
        cluster_info[cid] = {
            "name":     cname,
            "n":        len(members),
            "pct":      round(len(members) / len(cl_arr) * 100, 1),
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
            f"• {cluster_desc(cname).capitalize()}."
        )

    lines.append("\n---\nType a cluster name (e.g. **Warm Day** or **Hot Afternoon**) "
                 "to see which specific dates belong to it.")
    return "\n\n".join(lines)


def cluster_detail_query(engine, cluster_name):
    """
    For a named cluster, list every date (with day-of-week),
    mean sensor values per date, and overall cluster statistics.
    """
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

    by_date = _dd(list)
    for i in members:
        by_date[engine.metas[i]["timestamp"].date()].append(i)

    # Per-date stats
    day_rows = []
    for d, idxs in sorted(by_date.items()):
        vals = {k: _np.array([engine.metas[i][k] for i in idxs])
                for k in ["temperature", "humidity", "pressure", "light"]}
        day_rows.append({
            "date":   d,
            "dow":    d.strftime("%A"),
            "n":      len(idxs),
            "t_mean": round(float(vals["temperature"].mean()), 1),
            "h_mean": round(float(vals["humidity"].mean()), 1),
            "p_mean": round(float(vals["pressure"].mean()), 1),
            "l_mean": round(float(vals["light"].mean()), 1),
        })

    # Overall stats
    all_t = _np.array([engine.metas[i]["temperature"] for i in members])
    all_h = _np.array([engine.metas[i]["humidity"]    for i in members])

    # Dow frequency
    dow_cnt = _dd(int)
    for r in day_rows: dow_cnt[r["dow"]] += 1
    peak_dow = max(dow_cnt, key=dow_cnt.get)

    # Months breakdown
    month_cnt = _dd(int)
    for r in day_rows: month_cnt[r["date"].strftime("%B %Y")] += 1

    lines = [
        f"**Cluster: {cluster_name}**\n"
        f"{cluster_desc(cluster_name).capitalize()}.\n",

        f"**Overview:** {len(members)} windows across **{len(day_rows)} days** "
        f"({round(len(members) / len(cl_arr) * 100, 1)}% of total dataset). "
        f"Most common day of week: **{peak_dow}**.\n",

        f"**Sensor profile:**  "
        f"Temp {round(float(all_t.mean()), 1)}°C (±{round(float(all_t.std()), 1)})  |  "
        f"Humidity {round(float(all_h.mean()), 1)}% RH\n",
    ]

    # Monthly breakdown
    month_lines = "  ".join([f"{m}: {c} days" for m, c in sorted(month_cnt.items())])
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

    by_date = _dd(list)
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

        cl_counts = _dd(int)
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
            relevant = sorted(days, key=lambda d: d["slope"], reverse=True)[:max(1, len(days))]
        slope_desc = "increase (positive slope)"
    else:
        relevant  = [d for d in days if d["slope"] < 0]
        if not relevant:          # fallback: pick day with lowest (least positive) slope
            relevant = sorted(days, key=lambda d: d["slope"])[:max(1, len(days))]
        slope_desc = "decrease (negative slope)"

    # Sort: fastest = largest abs slope; slowest = smallest abs slope
    reverse = (rate_dir == "fastest")
    ranked  = sorted(relevant, key=lambda d: abs(d["slope"]), reverse=reverse)

    target    = ranked[0]
    rate_word = "fastest" if rate_dir == "fastest" else "slowest"
    peak_dt   = target["date"].strftime("%A, %d %B %Y")
    slope_v   = target["slope"]
    delta_v   = target["delta"]
    sign      = "+" if delta_v >= 0 else ""

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
            f"**{target['cluster']}** — {cluster_desc(target['cluster'])}.")

    # Ranked list
    list_lines = []
    for ri, d in enumerate(ranked[:top_n]):
        list_lines.append(
            f"  {ri + 1}. {d['date'].strftime('%a %d %b')} — "
            f"slope {d['slope']:+.4f}{unit}/step  "
            f"(net {d['delta']:+.1f}{unit}, {d['rate']}, "
            f"cluster: {d['cluster']})")
    ranked_block = (f"\n**Top {min(top_n, len(ranked))} days by "
                    f"{rate_word} {change_dir} of {sensor} "
                    f"in {period_label}:**\n" + "\n".join(list_lines))

    overall_block = (f"\nAcross {period_label}, {sensor} daily slopes ranged from "
                      f"**{all_slopes.min():+.4f}** to **{all_slopes.max():+.4f}**{unit}/step "
                      f"(mean: {slope_mu:+.4f}, std: {slope_sd:.4f}).")

    return "\n\n".join([s1, char + anom, s_cl]) + ranked_block + overall_block
