"""
live_feed.py — Polls the shared SQLite DB written by iot_stream_sender.py
and ingests new readings into a live Engine instance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is the receiving half of the streaming feature. iot_stream_sender.py
(a completely separate process — see its docstring) writes rows into a
SQLite DB; this module reads whatever is new since the last poll and folds
it into the running Engine so every existing feature keeps working with no
further changes:

  • rules.py's check_rules() always reads engine.data[sensor][-1] — the
    moment append_raw() adds a new sample, notification rules start firing
    on it on the very next rerun.
  • nlp_queries.py / narration.py / charts.py / stat_cards() all work off
    engine.metas / engine.X / engine.cl through Engine.query() — the
    moment append_window() adds a new window, "today" (and any other
    query touching today's date) includes it too.

WHY TWO LEVELS ("raw" vs "window")
  iot_train.py's extract_features() computes one 34-feature row per raw
  sample using a trailing window of `seq_len` prior samples (see its
  docstring). A brand-new raw reading can't become a window until at least
  `seq_len` earlier readings already exist — exactly mirrored here:
  append_raw() always runs immediately (so rules can react right away),
  while append_window() only runs once there's enough history.

WATERMARK / DEDUPE
  Engine is a single @st.cache_resource instance shared by every browser
  tab / user session hitting this Streamlit app — not per-session state.
  So the "how far have we read" watermark is stored as an attribute on the
  Engine instance itself (engine._live_since_id), never in
  st.session_state: if it lived in session state, two open tabs would each
  ingest every row once, doubling every reading.

  The same attribute-on-Engine trick marks where the live portion of
  engine.data begins (engine._live_base_len, set once, the first time
  poll_and_ingest() ever runs for this Engine) — iot_slm_app/charts.py's
  live_stream_chart() uses it to plot only what actually streamed in,
  not the full historical/offline-replay dataset underneath it.

When to edit this file:
  • iot_train.py's extract_features() formula changes -> update
    _extract_window_row() to match (kept as a plain, dependency-free
    re-implementation rather than importing iot_train.py, so this module
    doesn't drag in iot_train's torch/sklearn training-only imports just
    to read a shared DB).
  • The sender writes new/renamed columns -> update fetch_new().
"""

import os
import sqlite3
from datetime import datetime

import numpy as np

from .config import SENSOR_KEYS


# ─────────────────────────────────────────────────────────────────────────────
#  SQLITE SOURCE
# ─────────────────────────────────────────────────────────────────────────────
def connect(db_path):
    """Open the shared DB read-only-ish. Returns None (never raises) if the
    file doesn't exist yet — perfectly normal before the sender has run."""
    if not db_path or not os.path.exists(db_path):
        return None
    try:
        con = sqlite3.connect(db_path, check_same_thread=False, timeout=2.0)
        return con
    except sqlite3.Error:
        return None


def get_meta(con):
    """Key/value pairs iot_stream_sender.py records about itself (e.g.
    interval_min) — used to warn if it doesn't match the trained model's."""
    try:
        return dict(con.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.Error:
        return {}


def fetch_new(con, since_id):
    """Rows with id > since_id, ascending. Returns ([], since_id) on any
    error or when there's nothing new (e.g. table not created yet)."""
    try:
        cur = con.execute(
            "SELECT id, ts, temperature, pressure, humidity, light, "
            "fault_type, is_anomaly FROM readings WHERE id > ? ORDER BY id ASC",
            (since_id,))
        rows = cur.fetchall()
    except sqlite3.Error:
        return [], since_id
    if not rows:
        return [], since_id
    parsed = [{
        "id": r[0], "ts": datetime.fromisoformat(r[1]),
        "temperature": r[2], "pressure": r[3], "humidity": r[4], "light": r[5],
        "fault_type": r[6] or "normal", "is_anomaly": bool(r[7]),
    } for r in rows]
    return parsed, rows[-1][0]


# ─────────────────────────────────────────────────────────────────────────────
#  WINDOW FEATURE EXTRACTION — mirrors iot_train.extract_features(), see
#  the "When to edit this file" note above.
# ─────────────────────────────────────────────────────────────────────────────
def _extract_window_row(data, i, window):
    seg = {k: data[k][i - window:i] for k in SENSOR_KEYS}
    row = []
    for k in SENSOR_KEYS:
        s = seg[k]
        row += [s.mean(), s.std(), s.min(), s.max(),
                s.max() - s.min(),
                float(np.percentile(s, 75) - np.percentile(s, 25)),
                np.diff(s).mean(), np.abs(np.diff(s)).mean()]
    row += [float(np.corrcoef(seg["temperature"], seg["humidity"])[0, 1]),
            float(np.corrcoef(seg["temperature"], seg["light"])[0, 1])]
    return np.array(row, dtype=np.float32)


def ingest(engine, rows):
    """Fold `rows` (ascending, as returned by fetch_new) into `engine`.
    Returns (n_raw_appended, n_windows_appended)."""
    n_raw = n_windows = 0
    for r in rows:
        values = {k: r[k] for k in SENSOR_KEYS}
        engine.append_raw(r["ts"], values, r["fault_type"], r["is_anomaly"])
        n_raw += 1

        i = len(engine.data["temperature"]) - 1  # index of the sample just appended
        if i >= engine.seq:
            feat_row = _extract_window_row(engine.data, i, engine.seq)
            meta = {
                "timestamp":  r["ts"],
                "fault_type": r["fault_type"],
                "is_anomaly": r["is_anomaly"],
                **{k: float(r[k]) for k in SENSOR_KEYS},
            }
            engine.append_window(feat_row, meta)
            n_windows += 1
    return n_raw, n_windows


# ─────────────────────────────────────────────────────────────────────────────
#  ONE-CALL ENTRY POINT for iot_app_v3.py
# ─────────────────────────────────────────────────────────────────────────────
def poll_and_ingest(engine, db_path):
    """
    One polling cycle: connect, read whatever is new since this Engine's
    own watermark, ingest it, advance the watermark. Safe to call every
    rerun even if the sender isn't running yet (returns connected: False).
    """
    con = connect(db_path)
    if con is None:
        return {"connected": False}

    if not hasattr(engine, "_live_since_id"):
        engine._live_since_id = 0
        # Mark "where live data starts" in engine.data, BEFORE ingesting
        # anything this call — iot_app_v3.py's live chart (charts.py's
        # live_stream_chart()) uses this to show only what streamed in,
        # not the full historical/offline-replay dataset. Stays put even
        # if streaming is later paused/resumed via the sidebar toggle.
        engine._live_base_len = len(engine.data.get(SENSOR_KEYS[0], []))

    meta = get_meta(con)
    rows, last_id = fetch_new(con, engine._live_since_id)
    n_raw = n_windows = 0
    if rows:
        n_raw, n_windows = ingest(engine, rows)
        engine._live_since_id = last_id
    con.close()

    ts_list = engine.data.get("timestamps") or []
    return {
        "connected":           True,
        "n_new_raw":           n_raw,
        "n_new_windows":       n_windows,
        "total_rows_seen":     engine._live_since_id,
        "sender_interval_min": meta.get("interval_min"),
        "last_ts":             ts_list[-1] if ts_list else None,
    }
