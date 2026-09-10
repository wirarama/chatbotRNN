"""
charts.py — Plotly figure builders
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What lives here:
  • make_charts(engine, results): the 6-panel comparison figure (temp,
    humidity, temp-vs-light scatter, temp-vs-humidity-by-cluster scatter,
    pressure, hourly-mean-temp bar) used in the "Charts" tab whenever the
    user has an active chat result (or two, for a comparison query).
  • training_chart(hist): the train/validation MSE curve used in the
    "Charts" tab's empty state and in the "Model Info" tab.
  • live_stream_chart(engine, rules): 2x2 live LINE chart (one panel per
    sensor) over ONLY the live-streamed portion of engine.data (see
    engine._live_base_len, set by iot_slm_app/live_feed.py), with one
    horizontal threshold line per "level" condition (diatas/dibawah/
    antara) of every currently ENABLED notification rule touching that
    sensor — used by the "Live Stream" tab.
  • rate_of_change_chart(engine, rules, cfg): 2x2 live BAR chart (one
    panel per sensor) of the same rate-of-change quantity rules.py's
    "peningkatan"/"penurunan" conditions watch (value now vs ~60 min
    ago), with a threshold line per currently ENABLED rate-of-change
    condition touching that sensor. Deliberately a SEPARATE chart/function
    from live_stream_chart() — a rate-of-change threshold is not a level
    on the raw sensor value, so it never belongs on the same line chart
    or the same axis scale.
  • _sub(): downsamples a long time series for plotting performance.
  • _lookback_steps()/_raw_delta_condition(): small helpers used only by
    rate_of_change_chart() — see their docstrings.

When to edit this file:
  • Adding/removing a subplot -> make_charts() (remember to update
    `rows`/`cols`/`subplot_titles` in make_subplots(...) together).
  • Changing colors -> config.COLORS, not this file.
  • Changing how "level" rule thresholds are drawn on the live line
    chart -> live_stream_chart().
  • Changing how rate-of-change rule thresholds are drawn, or the
    lookback window used to compute the bars -> rate_of_change_chart()
    (and _lookback_steps() — keep it in sync with rules.py's own
    _lookback_steps(), which this deliberately duplicates rather than
    imports, same reasoning as live_feed.py's _extract_window_row()).
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict
from datetime import timedelta

from .config import COLORS, SENSOR_KEYS, SENSOR_UNITS


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
                                  legendgroup=lbl, showlegend=(ri == 0)),
                      row=1, col=1)
        fore = result.get("forecast")
        if fore is not None and len(ts) > 0:
            lt  = ts[-1]
            fts = [lt + timedelta(minutes=30 * (i + 1)) for i in range(6)]
            fig.add_trace(go.Scatter(x=fts, y=fore[:, 0],
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
        step = max(1, len(T_v) // 300); T_s, L_s = T_v[::step], L_v[::step]
        fig.add_trace(go.Scatter(x=T_s, y=L_s, mode="markers",
                                  marker=dict(color=col, size=5, opacity=0.4),
                                  name=lbl, showlegend=False, legendgroup=lbl),
                      row=2, col=1)
        if len(T_s) > 2:
            m, b = np.polyfit(T_s, L_s, 1)
            xr   = np.array([T_s.min(), T_s.max()])
            fig.add_trace(go.Scatter(x=xr, y=m * xr + b, mode="lines",
                                      line=dict(color=col, width=2, dash="dash"),
                                      showlegend=False), row=2, col=1)

        # Scatter T vs H by cluster
        H_v = np.array([engine.metas[i]["humidity"] for i in idx])
        C_v = engine.cluster_series(idx)
        step2 = max(1, len(T_v) // 250)
        T_sc, H_sc, C_sc = T_v[::step2], H_v[::step2], C_v[::step2]
        cmap = px.colors.qualitative.Set2
        for ci2, cname in enumerate(sorted(set(C_sc))):
            mask = np.array([c == cname for c in C_sc])
            fig.add_trace(go.Scatter(x=T_sc[mask], y=H_sc[mask], mode="markers",
                                      name=cname,
                                      marker=dict(color=cmap[ci2 % len(cmap)],
                                                  size=5, opacity=0.5),
                                      showlegend=(ri == 0),
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
        offset = ri * 0.4 - 0.2 * (len(results) - 1)
        fig.add_trace(go.Bar(x=[h + offset for h in hours], y=means, width=0.38,
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


def live_stream_chart(engine, rules, max_points=200):
    """
    2x2 grid, one LINE per sensor, plotting ONLY the live-streamed portion
    of engine.data — everything appended since live streaming first
    started (see engine._live_base_len, set once by
    iot_slm_app.live_feed.poll_and_ingest() the first time it runs) — not
    the full historical/offline-replay dataset. Capped to the most recent
    `max_points` samples so a long-running session stays readable.
    Samples flagged is_anomaly are marked with an X.

    Every "level" condition (diatas/dibawah/antara) of every currently
    ENABLED rule that targets a given sensor is drawn as a horizontal
    threshold line on that sensor's panel — exactly the boundary the rule
    is watching. Rate-of-change conditions (peningkatan/penurunan) are
    NOT drawn here at all — see rate_of_change_chart() below, a
    deliberately separate chart, since a rate threshold lives on a
    completely different axis (a delta, not a raw sensor value) and would
    be misleading drawn as a flat line across raw readings.

    Returns fig, or None if live streaming hasn't produced any samples
    yet (nothing to plot).
    """
    base   = getattr(engine, "_live_base_len", None)
    ts_all = engine.data.get("timestamps") or []
    if base is None or base >= len(ts_all):
        return None

    start = max(base, len(ts_all) - max_points)
    ts    = ts_all[start:]
    faults = engine.data.get("is_anomaly")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{k.capitalize()} ({SENSOR_UNITS[k]})" for k in SENSOR_KEYS],
        vertical_spacing=0.16, horizontal_spacing=0.08)
    pos = {SENSOR_KEYS[0]: (1, 1), SENSOR_KEYS[1]: (1, 2),
           SENSOR_KEYS[2]: (2, 1), SENSOR_KEYS[3]: (2, 2)}

    for sensor in SENSOR_KEYS:
        row, col = pos[sensor]
        vals = np.asarray(engine.data[sensor][start:])
        fig.add_trace(go.Scatter(x=ts, y=vals, mode="lines", name=sensor.capitalize(),
                                  line=dict(color=COLORS["A"], width=1.6),
                                  showlegend=False),
                      row=row, col=col)

        if faults is not None:
            fmask = np.asarray(faults[start:], dtype=bool)
            if fmask.any():
                fig.add_trace(go.Scatter(
                    x=[t for t, m in zip(ts, fmask) if m],
                    y=[v for v, m in zip(vals, fmask) if m],
                    mode="markers", marker=dict(color=COLORS["B"], size=8, symbol="x"),
                    name="anomali", showlegend=False), row=row, col=col)

        unit = SENSOR_UNITS.get(sensor, "")
        for rule in rules:
            if not rule.get("enabled", True):
                continue
            for cond in rule.get("conditions", []):
                if cond["sensor"] != sensor or cond["mode"] != "level":
                    continue
                if cond["op"] == "above":
                    fig.add_hline(y=cond["value"], line_dash="dash", line_color=COLORS["B"],
                                  annotation_text=f"> {cond['value']:g}{unit}",
                                  annotation_font_size=9, row=row, col=col)
                elif cond["op"] == "below":
                    fig.add_hline(y=cond["value"], line_dash="dash", line_color=COLORS["navy"],
                                  annotation_text=f"< {cond['value']:g}{unit}",
                                  annotation_font_size=9, row=row, col=col)
                elif cond["op"] == "between":
                    lo, hi = cond["value"]
                    fig.add_hline(y=lo, line_dash="dot", line_color=COLORS["teal"],
                                  annotation_text=f"min {lo:g}{unit}",
                                  annotation_font_size=9, row=row, col=col)
                    fig.add_hline(y=hi, line_dash="dot", line_color=COLORS["teal"],
                                  annotation_text=f"max {hi:g}{unit}",
                                  annotation_font_size=9, row=row, col=col)

    fig.update_layout(
        height=560, paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(family="Inter", size=11, color=COLORS["navy"]),
        margin=dict(l=50, r=20, t=45, b=30), showlegend=False)
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].gridcolor = COLORS["grid"]
    return fig


def _lookback_steps(interval_min, minutes=60):
    """How many raw samples back = `minutes` of simulated time — the same
    formula as rules.py's _lookback_steps(), duplicated here on purpose
    (see this file's module docstring) so charts.py doesn't reach into
    rules.py's underscore-prefixed internals."""
    try:
        interval_min = float(interval_min)
        if interval_min <= 0:
            raise ValueError
    except (TypeError, ValueError):
        interval_min = 30.0
    return max(1, int(round(minutes / interval_min)))


def _raw_delta_condition(cond):
    """
    Re-express a rate-of-change condition (mode "increase"/"decrease") in
    terms of the RAW delta rate_of_change_chart() plots — value now minus
    value ~60 min ago, positive meaning "rising". rules.py's
    evaluate_condition() negates that raw delta for "decrease" mode
    before comparing against op/value, so "peningkatan"/"penurunan" can
    share the same diatas/dibawah vocabulary internally; this function
    undoes exactly that negation, purely so a threshold line lands where
    the rule actually fires on this chart's axis. rules.py's own
    evaluation logic is untouched — this is a display-only translation.

    Returns (op, value) in raw-delta terms ("between" values sorted).
    """
    op, value = cond["op"], cond["value"]
    if cond["mode"] == "increase":
        return op, value
    if op == "above":
        return "below", -value
    if op == "below":
        return "above", -value
    if op == "between":
        lo, hi = value
        return "between", sorted([-hi, -lo])
    return op, value


def rate_of_change_chart(engine, rules, cfg, max_points=200):
    """
    2x2 grid, one BAR per sensor, showing rules.py's "peningkatan"/
    "penurunan" quantity directly: delta = value_now - value_~60min_ago
    (see rules.py's _lookback_steps()/_condition_value_at(); this mirrors
    that computation using the trained model's own interval_min from
    `cfg`). Bars are colored by direction (rising vs falling) so the sign
    reads at a glance. Only plots the live-streamed portion of
    engine.data (same engine._live_base_len cutoff live_stream_chart()
    uses), capped to the most recent `max_points` bars.

    Every rate-of-change condition of every currently ENABLED rule
    touching a sensor is drawn as a horizontal threshold line on that
    sensor's bar panel, translated into this raw-delta domain by
    _raw_delta_condition() so the line lands exactly where the rule
    actually fires — kept as a chart deliberately SEPARATE from
    live_stream_chart()'s level-threshold lines (see this file's module
    docstring for why).

    Assumes every such condition shares rules.py's default 60-minute
    lookback window — true for every rule the chat's rule grammar can
    currently produce (rules.py's CLAUSE_RE never sets a custom
    window_min). A future per-condition window_min would need this
    function to compute deltas per-condition instead of once per sensor.

    Returns fig, or None if there isn't yet enough live history for even
    one delta (need at least one lookback window of live samples).
    """
    base   = getattr(engine, "_live_base_len", None)
    ts_all = engine.data.get("timestamps") or []
    if base is None:
        return None

    interval_min = (cfg or {}).get("interval_min", 30)
    steps = _lookback_steps(interval_min, 60)

    start = max(base, steps)
    if start >= len(ts_all):
        return None
    start = max(start, len(ts_all) - max_points)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{k.capitalize()} — Δ/{steps * interval_min:g} menit"
                         for k in SENSOR_KEYS],
        vertical_spacing=0.18, horizontal_spacing=0.08)
    pos = {SENSOR_KEYS[0]: (1, 1), SENSOR_KEYS[1]: (1, 2),
           SENSOR_KEYS[2]: (2, 1), SENSOR_KEYS[3]: (2, 2)}

    any_bars = False
    for sensor in SENSOR_KEYS:
        row, col = pos[sensor]
        arr = np.asarray(engine.data[sensor])
        idxs = list(range(start, len(arr)))
        if not idxs:
            continue
        deltas = arr[idxs] - arr[[i - steps for i in idxs]]
        ts = [ts_all[i] for i in idxs]
        bar_colors = [COLORS["B"] if d >= 0 else COLORS["A"] for d in deltas]
        fig.add_trace(go.Bar(x=ts, y=deltas, marker_color=bar_colors,
                              showlegend=False, name=sensor.capitalize()),
                      row=row, col=col)
        fig.add_hline(y=0, line_color=COLORS["grid"], line_width=1, row=row, col=col)
        any_bars = True

        unit = SENSOR_UNITS.get(sensor, "")
        for rule in rules:
            if not rule.get("enabled", True):
                continue
            for cond in rule.get("conditions", []):
                if cond["sensor"] != sensor or cond["mode"] not in ("increase", "decrease"):
                    continue
                raw_op, raw_val = _raw_delta_condition(cond)
                if raw_op == "above":
                    fig.add_hline(y=raw_val, line_dash="dash", line_color=COLORS["navy"],
                                  annotation_text=f"> {raw_val:g}{unit}",
                                  annotation_font_size=9, row=row, col=col)
                elif raw_op == "below":
                    fig.add_hline(y=raw_val, line_dash="dash", line_color=COLORS["navy"],
                                  annotation_text=f"< {raw_val:g}{unit}",
                                  annotation_font_size=9, row=row, col=col)
                elif raw_op == "between":
                    lo, hi = raw_val
                    fig.add_hline(y=lo, line_dash="dot", line_color=COLORS["teal"],
                                  annotation_text=f"min {lo:g}{unit}",
                                  annotation_font_size=9, row=row, col=col)
                    fig.add_hline(y=hi, line_dash="dot", line_color=COLORS["teal"],
                                  annotation_text=f"max {hi:g}{unit}",
                                  annotation_font_size=9, row=row, col=col)

    if not any_bars:
        return None

    fig.update_layout(
        height=560, paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(family="Inter", size=11, color=COLORS["navy"]),
        margin=dict(l=50, r=20, t=45, b=30), showlegend=False)
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].gridcolor = COLORS["grid"]
    return fig


def training_chart(hist):
    ep  = list(range(1, len(hist["train"]) + 1))
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
