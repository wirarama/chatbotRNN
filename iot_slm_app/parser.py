"""
parser.py — Temporal query parser ("today", "last week", "bulan lalu", ...)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What lives here:
  • Parser: given a reference date (the dataset's last day), resolves
    English/Indonesian time-period phrases ("today"/"hari ini",
    "last week"/"minggu lalu", etc.) into (start_date, end_date, label)
    tuples, and exposes thin pass-through helpers to the three NLP query
    engines in nlp_queries.py (parse_ranking / parse_cluster / parse_rate)
    so main() only has to hold one Parser instance.

When to edit this file:
  • Adding a new relative time phrase ("2 weeks ago", "next month") ->
    add a regex to Parser.PATS and a branch to Parser.resolve().
"""

import re
from datetime import timedelta

from .nlp_queries import detect_ranking_query, detect_cluster_query, detect_rate_query


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
        if key == "yesterday":  d = r - timedelta(1); return d, d, "Yesterday"
        if key == "this_week":  s = r - timedelta(r.weekday()); return s, r, "This Week"
        if key == "last_week":
            e = r - timedelta(r.weekday() + 1); return e - timedelta(6), e, "Last Week"
        if key == "this_month": return r.replace(day=1), r, "This Month"
        if key == "last_month":
            e = (r.replace(day=1) - timedelta(1))
            return e.replace(day=1), e, "Last Month"
        if key == "2days_ago":  d = r - timedelta(2); return d, d, "2 Days Ago"
        if key == "3days_ago":  d = r - timedelta(3); return d, d, "3 Days Ago"
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
