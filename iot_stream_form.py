"""
iot_stream_form.py  —  Manual random-input web form for the shared stream DB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A second, completely separate way to feed live data into iot_app_v3.py,
alongside iot_stream_sender.py (the automatic realistic simulator). Where
iot_stream_sender.py generates readings on its own following a diurnal
pattern, this is a plain Flask form: you pick a min/max RANGE per sensor
yourself and either fire off a fixed number of random records at once, or
start a continuous background feed and stop it whenever you like. Useful
for deliberately pushing one sensor out of range to see a notification
rule or the Decision-Tree classifier react, without waiting for the
simulator to randomly produce a fault.

SAME DB, SAME SCHEMA, NO CHANGES NEEDED ELSEWHERE
  This reuses iot_stream_sender.py's own open_db()/insert_reading()/
  set_meta() (imported directly, not reimplemented — see iot_stream_sender.py
  docstring for the schema) so records land in the exact same
  `readings` table iot_slm_app/live_feed.py already knows how to read.
  You can alternate between running iot_stream_sender.py and this form
  against the same DB file — the timeline stays consistent because both
  always continue from whatever the newest row's timestamp already is.

RUNNING
  pip install flask          # if not already installed
  python iot_stream_form.py [--port 5050] [--db ./iot_stream.db]

Then open http://127.0.0.1:5050/ in a browser. Separate process from
iot_app_v3.py — run both at once (Streamlit + this Flask form) each in
their own terminal.

TWO SENDING MODES (either can be used any time, independently)
  • "Kirim jumlah tertentu": generate N random readings right now, one
    burst, then stop — good for a quick one-off nudge.
  • "Mode Start/Stop": spawn a background thread that keeps generating
    one random reading every `tick_seconds` (real time) until you click
    Stop — good for a sustained manual feed you control live from the
    browser. Only one continuous feed runs at a time per process.

When to edit this file:
  • Changing default sensor ranges -> DEFAULT_RANGES.
  • Adding a new sensor -> also update iot_train.py / iot_stream_sender.py
    / iot_slm_app/config.py's SENSOR_KEYS (all four must agree).
  • Changing the page's look -> the PAGE_TEMPLATE string below (plain
    inline HTML/CSS on purpose — no static files, no template folder, so
    this stays a single portable file).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import random
import threading
from datetime import datetime, timedelta

from flask import Flask, flash, redirect, render_template_string, request, url_for, jsonify

from iot_stream_sender import (
    SENSOR_KEYS, FAULT_TYPES, open_db, insert_reading, set_meta,
)

SENSOR_LABELS = {
    "temperature": ("Temperature", "°C"),
    "pressure":    ("Pressure", "hPa"),
    "humidity":    ("Humidity", "% RH"),
    "light":       ("Light", "lux"),
}
DEFAULT_RANGES = {
    "temperature": (20.0, 35.0),
    "pressure":    (1000.0, 1020.0),
    "humidity":    (30.0, 90.0),
    "light":       (0.0, 1000.0),
}
DEFAULT_DB           = "./iot_stream.db"
DEFAULT_INTERVAL_MIN = 30
DEFAULT_TICK_SECONDS = 5
MAX_BATCH_RECORDS    = 100_000

app = Flask(__name__)
app.secret_key = "iot-stream-form"  # local dev tool only, no real session data

# ─────────────────────────────────────────────────────────────────────────────
#  CONTINUOUS ("start/stop") FEED STATE — one background feed per process
# ─────────────────────────────────────────────────────────────────────────────
_lock = threading.Lock()
STATE = {
    "running":      False,
    "thread":       None,
    "stop_event":   None,
    "sent_count":   0,
    "last_reading": None,
    "started_at":   None,
    "db_path":      DEFAULT_DB,
    "error":        None,
}


def _next_start_time(con):
    """Continue right after the newest row already in the DB, so switching
    between this form and iot_stream_sender.py never overlaps timestamps."""
    row = con.execute("SELECT ts FROM readings ORDER BY id DESC LIMIT 1").fetchone()
    return datetime.fromisoformat(row[0]) if row else datetime.now()


def _random_reading(ranges, ts, fault_type="normal"):
    vals = {k: round(random.uniform(*ranges[k]), 2) for k in SENSOR_KEYS}
    return {
        "ts": ts.isoformat(),
        **vals,
        "fault_type": fault_type,
        "is_anomaly": int(fault_type != "normal"),
    }


def _continuous_worker(ranges, interval_min, tick_seconds, fault_type, db_path, stop_event):
    try:
        con = open_db(db_path)
        con.execute("PRAGMA busy_timeout=5000;")
        set_meta(con, interval_min=interval_min)
        t = _next_start_time(con)
        while not stop_event.is_set():
            r = _random_reading(ranges, t, fault_type)
            insert_reading(con, r)
            t += timedelta(minutes=interval_min)
            with _lock:
                STATE["sent_count"] += 1
                STATE["last_reading"] = r
            stop_event.wait(tick_seconds)  # sleeps, but wakes instantly on Stop
        con.close()
    except Exception as e:
        with _lock:
            STATE["error"] = str(e)
    finally:
        with _lock:
            STATE["running"] = False


# ─────────────────────────────────────────────────────────────────────────────
#  FORM HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _parse_ranges(form):
    """Read the 8 min/max fields from the submitted form. Raises ValueError
    with a human-readable message on anything invalid."""
    ranges = {}
    for k in SENSOR_KEYS:
        lo_raw = form.get(f"{k}_min", "")
        hi_raw = form.get(f"{k}_max", "")
        try:
            lo, hi = float(lo_raw), float(hi_raw)
        except (TypeError, ValueError):
            raise ValueError(f"Rentang {SENSOR_LABELS[k][0]} harus berupa angka.")
        if lo > hi:
            raise ValueError(f"Rentang {SENSOR_LABELS[k][0]}: nilai minimum ({lo}) "
                              f"lebih besar dari maksimum ({hi}).")
        ranges[k] = (lo, hi)
    return ranges


def _parse_common(form):
    fault_type = form.get("fault_type", "normal")
    if fault_type not in ("normal", *FAULT_TYPES):
        fault_type = "normal"
    try:
        interval_min = float(form.get("interval_min", DEFAULT_INTERVAL_MIN))
        if interval_min <= 0:
            raise ValueError
    except ValueError:
        raise ValueError("Interval simulasi (menit/record) harus angka positif.")
    db_path = form.get("db_path", "").strip() or DEFAULT_DB
    return fault_type, interval_min, db_path


def _last_form_values():
    """Values to pre-fill the form with — last submitted, else defaults."""
    with _lock:
        return dict(STATE.get("_last_form") or {})


def _remember_form_values(values):
    with _lock:
        STATE["_last_form"] = values


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    with _lock:
        status = {k: STATE[k] for k in
                   ("running", "sent_count", "last_reading", "started_at", "db_path", "error")}
    form_vals = _last_form_values()
    return render_template_string(
        PAGE_TEMPLATE,
        sensor_keys=SENSOR_KEYS, sensor_labels=SENSOR_LABELS,
        default_ranges=DEFAULT_RANGES, fault_types=FAULT_TYPES,
        default_db=DEFAULT_DB, default_interval=DEFAULT_INTERVAL_MIN,
        default_tick=DEFAULT_TICK_SECONDS, status=status, form_vals=form_vals)


@app.route("/send_batch", methods=["POST"])
def send_batch():
    try:
        ranges = _parse_ranges(request.form)
        fault_type, interval_min, db_path = _parse_common(request.form)
        n = int(request.form.get("num_records", 0))
        if n <= 0:
            raise ValueError("Jumlah records harus lebih besar dari 0.")
        if n > MAX_BATCH_RECORDS:
            raise ValueError(f"Jumlah records terlalu besar (maks {MAX_BATCH_RECORDS:,}).")
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("index"))

    _remember_form_values(dict(request.form))

    con = open_db(db_path)
    con.execute("PRAGMA busy_timeout=5000;")
    set_meta(con, interval_min=interval_min)
    t = _next_start_time(con)
    for _ in range(n):
        r = _random_reading(ranges, t, fault_type)
        insert_reading(con, r)
        t += timedelta(minutes=interval_min)
    con.close()

    with _lock:
        STATE["last_reading"] = r
        STATE["db_path"] = db_path
    flash(f"✅ Terkirim {n:,} record acak ke `{db_path}`.", "success")
    return redirect(url_for("index"))


@app.route("/start", methods=["POST"])
def start_stream():
    with _lock:
        already_running = STATE["running"]
    if already_running:
        flash("⚠ Mode Start/Stop sudah berjalan — hentikan dulu sebelum memulai lagi.", "error")
        return redirect(url_for("index"))

    try:
        ranges = _parse_ranges(request.form)
        fault_type, interval_min, db_path = _parse_common(request.form)
        tick_seconds = float(request.form.get("tick_seconds", DEFAULT_TICK_SECONDS))
        if tick_seconds <= 0:
            raise ValueError("Jeda antar record (detik) harus angka positif.")
    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for("index"))

    _remember_form_values(dict(request.form))

    stop_event = threading.Event()
    thread = threading.Thread(
        target=_continuous_worker,
        args=(ranges, interval_min, tick_seconds, fault_type, db_path, stop_event),
        daemon=True)
    with _lock:
        STATE.update(running=True, thread=thread, stop_event=stop_event,
                     sent_count=0, last_reading=None, started_at=datetime.now().isoformat(timespec="seconds"),
                     db_path=db_path, error=None)
    thread.start()
    flash(f"▶ Streaming kontinu dimulai ke `{db_path}` (tiap {tick_seconds:g} detik).", "success")
    return redirect(url_for("index"))


@app.route("/stop", methods=["POST"])
def stop_stream():
    with _lock:
        stop_event = STATE["stop_event"]
        running = STATE["running"]
    if not running or stop_event is None:
        flash("Mode Start/Stop sedang tidak berjalan.", "error")
        return redirect(url_for("index"))
    stop_event.set()
    with _lock:
        thread = STATE["thread"]
    if thread:
        thread.join(timeout=3)
    flash("⏹ Streaming kontinu dihentikan.", "success")
    return redirect(url_for("index"))


@app.route("/status")
def status():
    with _lock:
        return jsonify({k: STATE[k] for k in
                         ("running", "sent_count", "last_reading", "started_at", "db_path", "error")})


# ─────────────────────────────────────────────────────────────────────────────
#  TEMPLATE — plain inline HTML/CSS, no external assets
# ─────────────────────────────────────────────────────────────────────────────
PAGE_TEMPLATE = """
<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<title>IoT Random Input Form</title>
<style>
  body { font-family: Arial, Helvetica, sans-serif; background: #f4f5f7; color: #212529;
         max-width: 760px; margin: 2rem auto; padding: 0 1rem; }
  h1 { font-size: 1.4rem; margin-bottom: .25rem; }
  .sub { color: #6c757d; font-size: .88rem; margin-bottom: 1.25rem; }
  .card { background: #fff; border: 1px solid #dee2e6; border-radius: 6px;
          padding: 1rem 1.25rem; margin-bottom: 1.25rem; }
  .card h2 { font-size: 1.05rem; margin-top: 0; }
  .row { display: flex; gap: .75rem; flex-wrap: wrap; margin-bottom: .6rem; }
  .field { flex: 1 1 140px; }
  .field label { display: block; font-size: .8rem; color: #495057; margin-bottom: .2rem; }
  .field input, .field select { width: 100%; padding: .35rem .5rem; box-sizing: border-box;
                                  border: 1px solid #ced4da; border-radius: 4px; font-size: .9rem; }
  .sensor-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: .5rem .75rem;
                 align-items: end; margin-bottom: .5rem; }
  .sensor-grid .name { font-weight: 600; font-size: .88rem; padding-top: .4rem; }
  button { padding: .5rem 1rem; border-radius: 4px; border: 1px solid #0d6efd;
           background: #0d6efd; color: #fff; font-size: .9rem; cursor: pointer; }
  button:hover { background: #0b5ed7; }
  button.stop { border-color: #dc3545; background: #dc3545; }
  button.stop:hover { background: #bb2d3b; }
  .flash { padding: .55rem .8rem; border-radius: 4px; margin-bottom: 1rem; font-size: .88rem; }
  .flash.success { background: #d1e7dd; border: 1px solid #a3cfbb; color: #0a3622; }
  .flash.error   { background: #f8d7da; border: 1px solid #f1aeb5; color: #58151c; }
  .status { font-size: .88rem; }
  .status .dot { display: inline-block; width: .6rem; height: .6rem; border-radius: 50%;
                 margin-right: .4rem; background: #adb5bd; }
  .status.running .dot { background: #198754; }
  code { background: #f1f3f5; padding: .1rem .3rem; border-radius: 3px; }
  hr { border: none; border-top: 1px solid #e9ecef; margin: .9rem 0; }
</style>
</head>
<body>
  <h1>📋 IoT Random Input Form</h1>
  <p class="sub">Kirim pembacaan sensor acak (dalam rentang yang Anda tentukan) ke database
     streaming bersama yang dipantau <code>iot_app_v3.py</code>. Proses terpisah — jalankan
     di terminal lain sambil Streamlit tetap berjalan.</p>

  {% with messages = get_flashed_messages(with_categories=true) %}
    {% for category, msg in messages %}
      <div class="flash {{ category }}">{{ msg }}</div>
    {% endfor %}
  {% endwith %}

  <div class="card status {{ 'running' if status.running else '' }}" id="status-card">
    <span class="dot"></span>
    <span id="status-text">
      {% if status.running %}
        Streaming kontinu berjalan sejak {{ status.started_at }} — {{ status.sent_count }} record terkirim.
      {% else %}
        Tidak ada streaming kontinu yang berjalan.
      {% endif %}
    </span>
    {% if status.error %}<div class="flash error">Error di background thread: {{ status.error }}</div>{% endif %}
  </div>

  <form method="post" id="main-form">
    <div class="card">
      <h2>1. Rentang Sensor</h2>
      <div class="sensor-grid">
        {% for k in sensor_keys %}
          <div class="name">{{ sensor_labels[k][0] }} ({{ sensor_labels[k][1] }})</div>
          <div class="field">
            <label>Min</label>
            <input type="number" step="any" name="{{ k }}_min"
                   value="{{ form_vals.get(k ~ '_min', default_ranges[k][0]) }}">
          </div>
          <div class="field">
            <label>Max</label>
            <input type="number" step="any" name="{{ k }}_max"
                   value="{{ form_vals.get(k ~ '_max', default_ranges[k][1]) }}">
          </div>
        {% endfor %}
      </div>
      <div class="row">
        <div class="field">
          <label>Tipe fault (opsional)</label>
          <select name="fault_type">
            <option value="normal" {{ 'selected' if form_vals.get('fault_type','normal')=='normal' else '' }}>normal</option>
            {% for ft in fault_types %}
              <option value="{{ ft }}" {{ 'selected' if form_vals.get('fault_type')==ft else '' }}>{{ ft }}</option>
            {% endfor %}
          </select>
        </div>
        <div class="field">
          <label>Interval simulasi (menit/record)</label>
          <input type="number" step="any" min="0.1" name="interval_min"
                 value="{{ form_vals.get('interval_min', default_interval) }}">
        </div>
        <div class="field">
          <label>Path Database</label>
          <input type="text" name="db_path" value="{{ form_vals.get('db_path', default_db) }}">
        </div>
      </div>
    </div>

    <div class="card">
      <h2>2. Kirim jumlah tertentu</h2>
      <div class="row">
        <div class="field">
          <label>Jumlah records</label>
          <input type="number" min="1" name="num_records" value="{{ form_vals.get('num_records', 10) }}">
        </div>
        <div class="field" style="align-self:end;">
          <button type="submit" formaction="{{ url_for('send_batch') }}">Kirim</button>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>3. Mode Start/Stop (kontinu)</h2>
      <div class="row">
        <div class="field">
          <label>Jeda nyata antar record (detik)</label>
          <input type="number" step="any" min="0.1" name="tick_seconds"
                 value="{{ form_vals.get('tick_seconds', default_tick) }}">
        </div>
        <div class="field" style="align-self:end; display:flex; gap:.5rem;">
          <button type="submit" formaction="{{ url_for('start_stream') }}" {{ 'disabled' if status.running else '' }}>▶ Start</button>
          <button type="submit" formaction="{{ url_for('stop_stream') }}" class="stop" {{ '' if status.running else 'disabled' }}>⏹ Stop</button>
        </div>
      </div>
    </div>
  </form>

<script>
  // Lightweight live status refresh — plain fetch, no framework.
  async function refreshStatus() {
    try {
      const r = await fetch("/status");
      const s = await r.json();
      const card = document.getElementById("status-card");
      const text = document.getElementById("status-text");
      card.classList.toggle("running", s.running);
      text.textContent = s.running
        ? `Streaming kontinu berjalan sejak ${s.started_at} — ${s.sent_count} record terkirim.`
        : "Tidak ada streaming kontinu yang berjalan.";
    } catch (e) { /* server not reachable yet — ignore */ }
  }
  setInterval(refreshStatus, 2000);
</script>
</body>
</html>
"""


def parse_args():
    p = argparse.ArgumentParser(description="Manual random-input web form for iot_stream DB")
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--db", type=str, default=DEFAULT_DB,
                    help="Default DB path pre-filled in the form (default: ./iot_stream.db)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DEFAULT_DB = args.db
    STATE["db_path"] = args.db
    print(f"\n  IoT Random Input Form running at http://{args.host}:{args.port}/")
    print(f"  Default DB: {args.db}\n")
    app.run(host=args.host, port=args.port, threaded=True)
