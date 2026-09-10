"""
rules.py — Chat-defined notification rules (create in chat, manage in the
Rules tab, persisted permanently as JSON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS FEATURE DOES
  The user types a rule directly into the chat box, e.g.:

      buat rule jika suhu diatas 32C dan humidity diatas 70% maka notif saya

  This module parses that sentence into one or more structured
  conditions, saves the rule permanently to a JSON file next to the repo
  (notification_rules.json), and on every app rerun checks each enabled
  rule against the latest available sensor reading. When a rule's
  condition newly becomes true (it was false a moment ago), a
  notification message is pushed into the chat conversation. Deleting or
  disabling a rule is done from the "Rules" tab in the UI — NOT from
  chat, by design (see is_delete_request() below, which just points the
  user at that tab instead of deleting anything).

  The Rules tab also has a "Run" button per rule (find_matching_indices()
  / run_rule()): instead of only checking "is this rule true right now",
  it scans the ENTIRE loaded dataset and returns every past index where
  the condition would have fired, packaged into the exact same bundle
  Engine.query() builds for a normal date-range chat query — so
  iot_app_v3.py can hand it straight to narrate()/make_charts()/
  stat_cards()/notif_badge() and it renders exactly like any other chat
  result (chart, stat cards, notification badge, hourly table).

RULE GRAMMAR (Indonesian + English, case-insensitive)
  creation phrase : buat|tambah|bikin|set|create|add|new  +  rule|notifikasi|notification
  condition       : [perubahan] <sensor> <operator> <angka>[<satuan>]
  perubahan (optional, switches the condition to a rate-of-change check):
      peningkatan | kenaikan | naiknya | meningkat  -> "increase"
      penurunan   | turunnya | menurun              -> "decrease"
      (absent)                                      -> "level" (the raw reading)
  operator:
      diatas | di atas | lebih dari | melebihi | above | greater than | >
                                                     -> "above"
      dibawah | di bawah | kurang dari | below | less than | <
                                                     -> "below"
      antara <a> dan <b>                            -> "between"
  multiple conditions are joined with "dan"/"and" (default -> ALL must be
  true) or "atau"/"or" (ANY may be true). Note "antara X dan Y" keeps its
  internal "dan" — only a "dan"/"atau" OUTSIDE a matched condition span
  is treated as the rule's logic operator, so "antara 30 dan 35" is not
  mistaken for two separate conditions.

  "Peningkatan"/"penurunan" conditions compare the latest reading against
  the reading from ~60 minutes earlier (see _lookback_steps) — i.e. "naik
  lebih dari 2C" means "this sensor has risen by more than 2C over the
  last hour", not an instantaneous value.

DATA SOURCE FOR "STREAMING"
  By default iot_app_v3.py replays a fixed historical dataset
  (iot_train.py's output). check_rules() always evaluates against the
  LAST sample in `engine.data[sensor]`, i.e. "whatever is currently the
  newest row" — this was always the intended hook for a real
  streaming/polling ingestion job to update.
  That job now exists: see iot_slm_app/live_feed.py (polls a shared
  SQLite DB written by the separate iot_stream_sender.py process) and
  Engine.append_raw()/append_window() in engine.py (what live_feed.py
  calls to actually append new samples/windows). Turn on "Live
  Streaming" in iot_app_v3.py's sidebar to wire it up — no changes were
  needed in this file to support it, exactly as this docstring always
  promised: once new readings are appended to engine.data, rule
  evaluation starts firing on genuinely new data with no further code
  changes here.

PERSISTENCE
  Rules live in <repo_root>/notification_rules.json as a plain JSON
  array, written atomically (write to .tmp then os.replace). Delete the
  file (or clear it via the Rules tab) to reset all rules.

When to edit this file:
  • Recognising new phrasing -> _CREATE_PAT / CLAUSE_RE / _OR_PAT.
  • Changing the "how far back is peningkatan/penurunan measured" window
    -> _DEFAULT_WINDOW_MIN.
  • Changing what happens on trigger -> notification_text().
  • Changing how a single condition is evaluated (for BOTH the live check
    and the historical "Run" scan) -> _condition_value_at() / _compare().
"""

import json
import os
import re
import uuid
from datetime import datetime

from .config import SENSOR_UNITS

# ─────────────────────────────────────────────────────────────────────────────
#  PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_RULES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notification_rules.json",
)


def load_rules(path=DEFAULT_RULES_PATH):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_rules(rules, path=DEFAULT_RULES_PATH):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def delete_rule(rules, rule_id):
    return [r for r in rules if r["id"] != rule_id]


# ─────────────────────────────────────────────────────────────────────────────
#  PARSING
# ─────────────────────────────────────────────────────────────────────────────
_CREATE_PAT = re.compile(
    r"\b(buat|tambah|bikin|set|create|add|new)\s+(rule|notifikasi|notification)\b",
    re.I)

_DELETE_HINT_PAT = re.compile(
    r"\b(hapus|delete|remove|hilangkan)\b.{0,20}\b(rule|notifikasi|notification)\b",
    re.I)

_OR_PAT = re.compile(r"\batau\b|\bor\b", re.I)

# One clause = optional change-word + sensor word + (between | above | below) + number(s) + optional unit
CLAUSE_RE = re.compile(
    r"(?P<change>peningkatan|kenaikan|naiknya|meningkat|penurunan|turunnya|menurun)?\s*"
    r"(?P<sensor>temp(?:erature)?|suhu|panas|humid(?:ity)?|kelembab(?:an)?|lembab|rh|"
    r"pressure|tekanan|baro(?:meter)?|hpa|light|cahaya|illu(?:minance)?|lux)\s+"
    r"(?:"
    r"antara\s+(?P<v1>-?\d+(?:[.,]\d+)?)\s*(?:°?\s*c|%|hpa|lux)?\s*(?:dan|and|sampai|to|-)\s+(?P<v2>-?\d+(?:[.,]\d+)?)"
    r"|(?P<op_above>di\s*atas|lebih\s+dari|melebihi|above|greater\s+than|>=|>)\s+(?P<v3>-?\d+(?:[.,]\d+)?)"
    r"|(?P<op_below>di\s*bawah|kurang\s+dari|below|less\s+than|<=|<)\s+(?P<v4>-?\d+(?:[.,]\d+)?)"
    r")\s*(?:°?\s*c|celsius|derajat|%|persen|hpa|lux)?",
    re.I,
)

_DEFAULT_WINDOW_MIN = 60  # "peningkatan/penurunan" look-back window, in minutes


def is_rule_creation(text):
    return bool(_CREATE_PAT.search(text))


def is_delete_request(text):
    """True if the message looks like a chat attempt to delete a rule —
    used only to redirect the user to the Rules tab, never to delete."""
    return bool(_DELETE_HINT_PAT.search(text)) and not is_rule_creation(text)


def _to_float(s):
    return float(s.replace(",", "."))


def _canon_sensor(word):
    word = word.lower()
    if word.startswith("temp") or word in ("suhu", "panas"):
        return "temperature"
    if word.startswith("humid") or word in ("kelembaban", "kelembab", "lembab", "rh"):
        return "humidity"
    if word.startswith("pressure") or word in ("tekanan", "baro", "barometer", "hpa"):
        return "pressure"
    if word.startswith("light") or word in ("cahaya", "illuminance", "illu", "lux"):
        return "light"
    return None


def _parse_clause_match(m):
    change = (m.group("change") or "").lower()
    sensor = _canon_sensor(m.group("sensor"))
    if m.group("v1") and m.group("v2"):
        op = "between"
        value = sorted([_to_float(m.group("v1")), _to_float(m.group("v2"))])
    elif m.group("v3"):
        op = "above"
        value = _to_float(m.group("v3"))
    elif m.group("v4"):
        op = "below"
        value = _to_float(m.group("v4"))
    else:
        return None

    if change in ("peningkatan", "kenaikan", "naiknya", "meningkat"):
        mode = "increase"
    elif change in ("penurunan", "turunnya", "menurun"):
        mode = "decrease"
    else:
        mode = "level"

    return {"sensor": sensor, "mode": mode, "op": op, "value": value}


def parse_rule_text(text):
    """
    Parse a natural-language rule-creation message into a list of
    conditions plus a combining logic ("and"/"or").
    Returns (conditions, logic), or (None, None) if nothing parseable.
    """
    conditions, spans = [], []
    for m in CLAUSE_RE.finditer(text):
        cond = _parse_clause_match(m)
        if cond and cond["sensor"]:
            conditions.append(cond)
            spans.append((m.start(), m.end()))
    if not conditions:
        return None, None

    # A standalone "atau"/"or" OUTSIDE any matched clause span makes this
    # an OR rule; the "dan" inside "antara X dan Y" is inside a span and
    # is correctly ignored here.
    logic = "and"
    for om in _OR_PAT.finditer(text):
        if not any(s <= om.start() < e for s, e in spans):
            logic = "or"
            break
    return conditions, logic


# ─────────────────────────────────────────────────────────────────────────────
#  DESCRIPTION / DISPLAY TEXT
# ─────────────────────────────────────────────────────────────────────────────
def describe_condition(c):
    sensor_label = c["sensor"].capitalize()
    unit = SENSOR_UNITS.get(c["sensor"], "")
    if c["mode"] == "level":
        if c["op"] == "above":
            return f"{sensor_label} > {c['value']:g}{unit}"
        if c["op"] == "below":
            return f"{sensor_label} < {c['value']:g}{unit}"
        if c["op"] == "between":
            lo, hi = c["value"]
            return f"{sensor_label} between {lo:g}{unit} and {hi:g}{unit}"
    else:
        verb = "rises" if c["mode"] == "increase" else "drops"
        cmp_word = "more than" if c["op"] == "above" else "less than"
        window = c.get("window_min", _DEFAULT_WINDOW_MIN)
        return f"{sensor_label} {verb} {cmp_word} {c['value']:g}{unit} within {window} min"
    return "?"


def describe_rule(rule):
    joiner = " AND " if rule.get("logic", "and") == "and" else " OR "
    return joiner.join(describe_condition(c) for c in rule["conditions"])


def add_rule(rules, raw_text, conditions, logic):
    rule = {
        "id":                uuid.uuid4().hex[:8],
        "raw_text":          raw_text,
        "conditions":        conditions,
        "logic":             logic,
        "enabled":           True,
        "created_at":        datetime.now().isoformat(timespec="seconds"),
        "last_state":        False,
        "last_triggered_at": None,
        "trigger_count":     0,
    }
    rule["description"] = describe_rule(rule)
    rules.append(rule)
    return rule


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION  ("streaming" = the latest sample currently in engine.data)
# ─────────────────────────────────────────────────────────────────────────────
def _lookback_steps(cfg, minutes=_DEFAULT_WINDOW_MIN):
    interval = cfg.get("interval_min") or 30
    try:
        interval = float(interval)
    except (TypeError, ValueError):
        interval = 30.0
    if interval <= 0:
        interval = 30.0
    return max(1, int(round(minutes / interval)))


def _compare(op, val, threshold):
    if val is None:
        return False
    if op == "above":
        return val > threshold
    if op == "below":
        return val < threshold
    if op == "between":
        lo, hi = threshold
        return lo <= val <= hi
    return False


def _condition_value_at(cond, engine, cfg, i):
    """
    The observed value of one condition AT dataset index i — the same
    quantity evaluate_condition() computes for "the latest reading", just
    generalised to an arbitrary index so find_matching_indices() can scan
    the whole timeline with it. Returns None where there isn't enough
    history yet (e.g. a rate-of-change condition near the start of the
    dataset).
    """
    sensor = cond["sensor"]
    arr = engine.data.get(sensor)
    if arr is None or i >= len(arr):
        return None
    if cond["mode"] == "level":
        return float(arr[i])
    steps = _lookback_steps(cfg, cond.get("window_min", _DEFAULT_WINDOW_MIN))
    if i < steps:
        return None
    d = float(arr[i] - arr[i - steps])
    return d if cond["mode"] == "increase" else -d


def evaluate_condition(cond, engine, cfg):
    """Returns (is_true, observed_value) for the LATEST reading only —
    used by check_rules() for the live/streaming notification check."""
    n = len(engine.data.get(cond["sensor"], []))
    if n == 0:
        return False, None
    val = _condition_value_at(cond, engine, cfg, n - 1)
    return _compare(cond["op"], val, cond["value"]), val


def evaluate_rule(rule, engine, cfg):
    results = [evaluate_condition(c, engine, cfg) for c in rule["conditions"]]
    oks = [r[0] for r in results]
    active = all(oks) if rule.get("logic", "and") == "and" else any(oks)
    values = [r[1] for r in results]
    return active, values


def check_rules(rules, engine, cfg):
    """
    Evaluate all enabled rules against the latest streamed reading.
    Edge-triggered: a rule notifies only on the False -> True transition,
    so a condition that stays true does not spam the chat every rerun.

    Returns (newly_triggered, changed):
      newly_triggered : list of (rule, values) that just became true.
      changed         : True if any rule's persisted state changed
                        (caller should save_rules() when True).
    Mutates `rules` in place.
    """
    newly_triggered = []
    changed = False
    now = datetime.now().isoformat(timespec="seconds")
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        active, values = evaluate_rule(rule, engine, cfg)
        was_active = rule.get("last_state", False)
        if active and not was_active:
            rule["last_triggered_at"] = now
            rule["trigger_count"] = rule.get("trigger_count", 0) + 1
            newly_triggered.append((rule, values))
            changed = True
        if active != was_active:
            rule["last_state"] = active
            changed = True
    return newly_triggered, changed


def notification_text(rule, values):
    bits = []
    for cond, val in zip(rule["conditions"], values):
        unit = SENSOR_UNITS.get(cond["sensor"], "")
        bits.append(f"{cond['sensor'].capitalize()}: no data" if val is None
                     else f"{cond['sensor'].capitalize()} = {val:.1f}{unit}")
    return (f"🔔 **Rule triggered:** {rule['description']}\n\n"
            f"Current reading — " + " | ".join(bits) +
            f"\n\n_Rule: \"{rule['raw_text']}\"_")


# ─────────────────────────────────────────────────────────────────────────────
#  RUN — scan the WHOLE dataset for every past match ("Rules" tab button)
# ─────────────────────────────────────────────────────────────────────────────
def find_matching_indices(rule, engine, cfg):
    """
    Scan every index in the currently loaded dataset and return the ones
    where this rule's condition holds — not just the latest reading.
    Used by the "Run" button in the Rules tab to answer "show me every
    time this rule would have fired historically."

    Level conditions are checked sample-by-sample; rate-of-change
    conditions (peningkatan/penurunan) compare each sample to the one
    ~1 hour earlier (same look-back window as the live check in
    evaluate_condition/check_rules), so the first `steps` samples can
    never match a rate condition — there's nothing before them to
    compare against.

    IMPORTANT — two different index spaces: `_condition_value_at()` reads
    straight from engine.data[sensor], the RAW per-sample arrays, so `i`
    here is a RAW index. But the caller (run_rule) hands the result to
    Engine.query_indices(), which indexes self.X/self.metas/self.cl — the
    WINDOW arrays, which are always exactly `engine.seq` samples shorter
    than the raw arrays (raw index i == window index i - engine.seq; see
    engine.py's query_indices()'s own `max(idx) + self.seq` forecast
    lookup, which relies on that same offset). The first `engine.seq` raw
    samples have no window at all, and — once live streaming pushes the
    raw arrays far past where they started — a match near the newest
    reading used to come out as a raw index >= len(self.X), crashing
    query_indices() with an IndexError. So every match is converted to
    its window index before being returned.
    """
    any_sensor = rule["conditions"][0]["sensor"] if rule["conditions"] else None
    arr = engine.data.get(any_sensor) if any_sensor else None
    n = len(arr) if arr is not None else 0
    n_windows = len(engine.X)

    matched = []
    for i in range(engine.seq, n):
        oks = [
            _compare(c["op"], _condition_value_at(c, engine, cfg, i), c["value"])
            for c in rule["conditions"]
        ]
        active = all(oks) if rule.get("logic", "and") == "and" else any(oks)
        if active:
            j = i - engine.seq  # raw index -> window/meta index
            if 0 <= j < n_windows:
                matched.append(j)
    return matched


def run_rule(engine, cfg, rule):
    """
    Build the same result bundle Engine.query() returns for a date range,
    but for every historical index matching `rule` — so it can be fed
    straight into the exact same narrate()/make_charts()/stat_cards()/
    notif_badge()/hourly_table() pipeline used for a normal chat query.

    Returns None if nothing in the currently loaded dataset matches.
    """
    matched = find_matching_indices(rule, engine, cfg)
    if not matched:
        return None
    result = engine.query_indices(matched, rule["description"])
    if result is None:
        return None
    # `matched` is window/meta-space (see find_matching_indices), so dates
    # come from engine.metas, not engine.data["timestamps"] (raw-space).
    dates = sorted({engine.metas[j]["timestamp"].date() for j in matched})
    result["start"], result["end"] = dates[0], dates[-1]
    result["rule_id"] = rule["id"]
    return result
