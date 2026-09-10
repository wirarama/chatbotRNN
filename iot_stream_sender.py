"""
iot_stream_sender.py  —  Standalone IoT sensor-data streaming source
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A completely separate process from iot_app_v3.py. It does not import
Streamlit, torch, or anything from iot_slm_app — it only knows how to
generate (or relay) sensor readings and write them to a small SQLite
database that iot_app_v3.py polls (see iot_slm_app/live_feed.py on the
receiving end).

WHY SQLITE AS THE HAND-OFF POINT
  iot_app_v3.py is a Streamlit app: it has no listening socket of its own,
  and every user's browser tab reruns the same script independently. A
  shared SQLite file (WAL mode, so one writer + many readers is safe) is
  the simplest thing that works without extra infrastructure (no broker,
  no server process to keep alive) and without adding a network dependency
  to either side. If you later swap in MQTT/HTTP/a real device, only this
  file and iot_slm_app/live_feed.py need to change — nothing in the model,
  engine, chat parser, or chart code cares where a reading came from.

DATA REALISM
  The reading generator intentionally mirrors iot_train.py's
  simulate_data(): same diurnal temperature/humidity/light shape, the same
  weekly pressure cycle, the same weekend temperature bump, and the same
  five fault categories injected at the same ~3% rate (temp_spike,
  pressure_drop, humidity_surge, light_flicker, multi_fault). This keeps
  live readings statistically compatible with the K-Means clusters and
  Decision-Tree notifier that were trained on the historical dataset — a
  live "Hot Afternoon" window should actually look like the "Hot
  Afternoon" cluster learned at training time.
  One deliberate difference: iot_train.py's slow temperature drift is a
  ramp bounded by the fixed length of its 60-day dataset ("sea = (i/total)
  * 3.0"), which has no meaning for an open-ended live stream. Here it's
  replaced with a bounded ~14-day sinusoid of the same ~3°C amplitude, so
  the long-run distribution of temperature stays put instead of drifting
  forever.

USAGE
  # Simplest: generate a reading every 5 (real) seconds, each representing
  # 30 simulated minutes, starting from now:
  python iot_stream_sender.py

  # Instantly backfill 24 readings (one full model window, seq_len=24) so
  # iot_app_v3.py can show live chart/chat results immediately instead of
  # waiting ~2 minutes for enough history to accumulate:
  python iot_stream_sender.py --burst 24

  # Match interval/seq_len to whatever iot_train.py actually used, and
  # continue the timeline exactly where the trained dataset left off
  # (so the live feed reads as a seamless continuation of history):
  python iot_stream_sender.py --continue-model --burst 24

  # Faster demo ticking, custom DB path:
  python iot_stream_sender.py --tick-seconds 1 --db ./iot_stream.db

Then, in iot_app_v3.py's sidebar, turn on "Live Streaming" (same --db path)
to start ingesting what this script writes.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import math
import os
import random
import sqlite3
import time
from datetime import datetime, timedelta

SENSOR_KEYS = ["temperature", "pressure", "humidity", "light"]
FAULT_TYPES = ["temp_spike", "pressure_drop", "humidity_surge",
               "light_flicker", "multi_fault"]
FAULT_RATE  = 0.03
_EPOCH      = datetime(2024, 1, 1)  # arbitrary fixed reference for the slow sinusoids


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Stream simulated IoT sensor readings into a shared "
                     "SQLite DB that iot_app_v3.py polls for live data.")
    p.add_argument("--db", type=str, default="./iot_stream.db",
                    help="Path to the shared SQLite database (default: ./iot_stream.db)")
    p.add_argument("--model-dir", type=str, default="./iot_model",
                    help="Where to look for config.json when using --continue-model "
                         "or auto-detecting --interval-min (default: ./iot_model)")
    p.add_argument("--interval-min", type=float, default=None,
                    help="Simulated minutes represented by each reading. "
                         "Defaults to the trained model's interval_min if "
                         "<model-dir>/config.json exists, else 30.")
    p.add_argument("--tick-seconds", type=float, default=5.0,
                    help="Real wall-clock seconds between readings (default: 5)")
    p.add_argument("--burst", type=int, default=0,
                    help="Insert this many readings immediately (no delay) before "
                         "starting the timed loop — useful to seed enough history "
                         "for a full model window (seq_len, usually 24) right away")
    p.add_argument("--start", type=str, default="now",
                    help="ISO timestamp to start from, or 'now' (default: now)")
    p.add_argument("--continue-model", action="store_true",
                    help="Start right after <model-dir>/config.json's ref_date, "
                         "using its interval_min, so the live feed reads as a "
                         "continuation of the historical dataset instead of a "
                         "separate timeline starting at 'now'")
    p.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducible readings (default: random each run)")
    return p.parse_args()


def _load_model_cfg(model_dir):
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  READING GENERATOR — mirrors iot_train.simulate_data(), see module docstring
# ─────────────────────────────────────────────────────────────────────────────
def gen_reading(t, rng):
    """Generate one realistic sensor reading for timestamp t."""
    hr       = t.hour + t.minute / 60 + t.second / 3600
    day_frac = (t - _EPOCH).total_seconds() / 86400.0

    sea  = 1.5 * math.sin(2 * math.pi * day_frac / 14.0)          # slow multi-day drift
    temp = 24 + sea + 8 * math.sin(math.pi * (hr - 6) / 12) * (hr > 6) + rng.gauss(0, .5)
    temp += 0.6 if t.weekday() >= 5 else 0

    pres = 1013.25 + 5 * math.sin(2 * math.pi * day_frac / 7.0) + rng.gauss(0, .3)

    hum = 70 - 20 * math.sin(math.pi * (hr - 6) / 12) * (hr > 6) + rng.gauss(0, 1.5)
    hum = min(100.0, max(20.0, hum))

    if 6 <= hr <= 20:
        lux = max(0.0, 1000 * math.exp(-0.5 * ((hr - 13) / 3) ** 2) + rng.gauss(0, 30))
    else:
        lux = max(0.0, rng.gauss(5, 2))

    fault_type = "normal"
    if rng.random() < FAULT_RATE:
        fault_type = rng.choice(FAULT_TYPES)
        if fault_type == "temp_spike":
            temp += rng.uniform(8, 15)
        elif fault_type == "pressure_drop":
            pres -= rng.uniform(10, 20)
        elif fault_type == "humidity_surge":
            hum = min(100.0, hum + rng.uniform(20, 30))
        elif fault_type == "light_flicker":
            lux = rng.choice([0, 1500])
        elif fault_type == "multi_fault":
            temp += rng.uniform(5, 10)
            hum = min(100.0, hum + rng.uniform(10, 20))

    return {
        "ts":          t.isoformat(),
        "temperature": round(temp, 2),
        "pressure":    round(pres, 2),
        "humidity":    round(hum, 2),
        "light":       round(lux, 2),
        "fault_type":  fault_type,
        "is_anomaly":  int(fault_type != "normal"),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SQLITE SINK
# ─────────────────────────────────────────────────────────────────────────────
def open_db(path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL;")   # safe for one writer + many readers
    con.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            temperature REAL, pressure REAL, humidity REAL, light REAL,
            fault_type TEXT, is_anomaly INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY, value TEXT
        )
    """)
    con.commit()
    return con


def set_meta(con, **kv):
    for k, v in kv.items():
        con.execute("INSERT INTO meta(key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (k, str(v)))
    con.commit()


def insert_reading(con, r):
    con.execute(
        "INSERT INTO readings (ts, temperature, pressure, humidity, light, "
        "fault_type, is_anomaly) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (r["ts"], r["temperature"], r["pressure"], r["humidity"], r["light"],
         r["fault_type"], r["is_anomaly"]))
    con.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg  = _load_model_cfg(args.model_dir)

    interval_min = args.interval_min
    start_time   = None

    if args.continue_model:
        if not cfg:
            raise SystemExit(
                f"--continue-model was given but {args.model_dir}/config.json "
                f"was not found. Run iot_train.py first, or drop --continue-model.")
        interval_min = interval_min or cfg.get("interval_min", 30)
        ref_date     = datetime.fromisoformat(cfg["ref_date"])
        start_time   = ref_date + timedelta(minutes=interval_min)
        print(f"  Continuing from trained dataset's ref_date {cfg['ref_date']} "
              f"-> first live reading at {start_time.isoformat()}")
    else:
        if interval_min is None:
            interval_min = cfg.get("interval_min", 30) if cfg else 30
        start_time = datetime.now() if args.start == "now" else datetime.fromisoformat(args.start)

    rng = random.Random(args.seed)
    con = open_db(args.db)
    set_meta(con, interval_min=interval_min, started_at=datetime.now().isoformat(),
              sender_seed=args.seed if args.seed is not None else "random")

    print("\n" + "═" * 60)
    print("  IoT Stream Sender")
    print("═" * 60)
    print(f"  DB              : {os.path.abspath(args.db)}")
    print(f"  Interval (sim)  : {interval_min} min/reading")
    print(f"  Tick (wall time): {args.tick_seconds} s/reading")
    print(f"  Start           : {start_time.isoformat()}")
    if args.burst:
        print(f"  Burst           : {args.burst} readings inserted immediately")
    print("  Press Ctrl+C to stop.")
    print("─" * 60)

    t = start_time
    n = 0
    try:
        for _ in range(args.burst):
            r = gen_reading(t, rng)
            insert_reading(con, r)
            n += 1
            flag = f"  ⚠ {r['fault_type']}" if r["is_anomaly"] else ""
            print(f"  [{n:5d}] {r['ts']}  T={r['temperature']:6.2f}°C  "
                  f"P={r['pressure']:7.2f}hPa  H={r['humidity']:5.1f}%  "
                  f"L={r['light']:6.1f}lux{flag}")
            t += timedelta(minutes=interval_min)

        while True:
            r = gen_reading(t, rng)
            insert_reading(con, r)
            n += 1
            flag = f"  ⚠ {r['fault_type']}" if r["is_anomaly"] else ""
            print(f"  [{n:5d}] {r['ts']}  T={r['temperature']:6.2f}°C  "
                  f"P={r['pressure']:7.2f}hPa  H={r['humidity']:5.1f}%  "
                  f"L={r['light']:6.1f}lux{flag}")
            t += timedelta(minutes=interval_min)
            time.sleep(args.tick_seconds)
    except KeyboardInterrupt:
        print(f"\n  Stopped after {n} readings. DB left at {os.path.abspath(args.db)}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
