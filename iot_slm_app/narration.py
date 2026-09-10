"""
narration.py — Turns one Engine.query() result into a plain-English reply
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What lives here:
  • narrate(result, comp=None, comp_label=""): the "default" chat reply —
    used whenever the user asks a plain temporal question ("today",
    "this week", "compare today and yesterday") rather than a ranking /
    rate-of-change / cluster question (those have their own narrators in
    nlp_queries.py).
  • _td/_hd/_tw: small word-choice helpers (temperature descriptor,
    humidity descriptor, trend arrow + word) shared by narrate().

When to edit this file:
  • Changing the wording/tone of the standard chat reply -> narrate().
  • Adding a new descriptive threshold (e.g. a 5th temperature band) ->
    _td()/_hd().
"""

from .config import SENSOR_UNITS, CLUSTER_COND, NOTIF_MSG
from .model import TORCH_OK


def _td(v):
    if v >= 35: return "very hot"
    if v >= 30: return "hot"
    if v >= 26: return "warm"
    if v >= 22: return "comfortable"
    if v >= 18: return "cool"
    return "cold"


def _hd(v):
    if v >= 85: return "very humid"
    if v >= 70: return "humid"
    if v >= 50: return "moderate"
    return "dry"


def _tw(s):
    if abs(s) < 0.001: return "→ stable"
    if s > 0.05:  return "↑ rising sharply"
    if s > 0.01:  return "↑ gradually rising"
    if s < -0.05: return "↓ dropping sharply"
    if s < -0.01: return "↓ gradually falling"
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
                   f"({na / max(nt, 1) * 100:.1f}% of {nt} windows).")
    parts.append(anom_s + " " + NOTIF_MSG.get(result["notification"], ""))

    fore = result["forecast"]
    if fore is not None:
        ft = fore[:, 0].mean(); fh = fore[:, 2].mean(); fl = fore[:, 3].mean()
        parts.append(f"**3-hour forecast:** {ft:.1f}°C ({_td(ft)}), "
                      f"humidity {fh:.1f}% ({_hd(fh)}), "
                      f"light {fl:.0f} lux.")
    elif not TORCH_OK:
        parts.append("*Forecast unavailable: torch not installed.*")

    return "\n\n".join(parts)
