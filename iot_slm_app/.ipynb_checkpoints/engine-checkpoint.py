"""
engine.py — Analysis engine over the loaded dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What lives here:
  • Engine: wraps the raw sensor arrays, engineered features, K-Means
    cluster labels, and Decision-Tree notifier, and exposes:
      - indices(start, end)       -> window indices covering a date range
      - query(start, end, label)  -> full stats/cluster/notification/
                                      forecast bundle for that range
                                      (this is what powers every chat reply)
      - query_indices(idx, label) -> same bundle, but for an arbitrary
                                      (not necessarily contiguous) list of
                                      window indices — query() is just a
                                      thin wrapper around this that first
                                      resolves a date range to indices.
                                      rules.py's "Run" button uses this
                                      directly with whatever indices match
                                      a rule, scattered across the whole
                                      dataset.
      - series(idx, sensor)       -> (timestamps, values) for charting
      - cluster_series(idx)       -> cluster name per window, for charting
      - append_raw(...)           -> append one new live sensor sample
      - append_window(...)        -> append one new live feature window
                                      (feature vector + meta), predicting
                                      its cluster with the already-fitted
                                      K-Means model
    append_raw()/append_window() are the live-streaming hook: see
    iot_slm_app/live_feed.py, which calls them as new readings arrive from
    iot_stream_sender.py. Everything above them (query/query_indices/
    series/cluster_series) then "just sees" the new data with no changes
    of its own, because it only ever reads self.data/self.X/self.metas/
    self.cl — the same arrays these two methods append to.

This is the "business logic" layer between the raw model/data (model.py,
loader.py) and everything user-facing (nlp_queries.py, narration.py,
charts.py, rules.py all call into an Engine instance, never the raw
pickles directly).

When to edit this file:
  • Adding a new derived statistic to every query result -> Engine.query().
  • Changing which Decision-Tree class wins when several fire in the same
    window -> the priority list in Engine.query() (`for ci in [8,7,4,5,...]`).
  • Changing what a live reading/window looks like -> append_raw() /
    append_window() (and iot_slm_app/live_feed.py, which builds their
    arguments from the shared streaming DB).
"""

from collections import defaultdict, Counter
from datetime import timedelta

import numpy as np

from .config import SENSOR_KEYS, NOTIF_CLASSES
from .model import forecast_window


class Engine:
    def __init__(self, data, X_feat, metas, cl, cn,
                 slm, slm_sc, km, km_sc, dt, dt_sc, seq=24):
        self.data   = data;  self.X     = X_feat; self.metas = metas
        self.cl     = cl;    self.cn    = cn
        self.slm    = slm;   self.slm_sc = slm_sc
        self.km     = km;    self.km_sc = km_sc
        self.dt     = dt;    self.dt_sc = dt_sc
        self.seq    = seq
        self._idx   = defaultdict(list)
        for i, m in enumerate(metas):
            self._idx[m["timestamp"].date()].append(i)

    # ─────────────────────────────────────────────────────────────────────
    #  LIVE STREAMING HOOK — see iot_slm_app/live_feed.py
    # ─────────────────────────────────────────────────────────────────────
    def append_raw(self, ts, values, fault_type="normal", is_anomaly=False):
        """
        Append one new raw sensor sample (NOT yet a full model window).
        `values` is a dict with one float per SENSOR_KEYS entry.

        This alone is enough for rules.py's live notification check —
        check_rules()/evaluate_condition() always read
        self.data[sensor][-1], so a rule can fire on this sample the very
        next rerun. It is NOT enough for chat queries ("today", "highest
        temperature this week", ...) or charts, which read self.metas/
        self.X/self.cl instead — call append_window() too once there's
        enough history (see live_feed.ingest()).
        """
        for k in SENSOR_KEYS:
            self.data[k] = np.append(self.data[k], float(values[k]))
        self.data.setdefault("timestamps", []).append(ts)
        self.data.setdefault("fault_type", []).append(fault_type)
        self.data["is_anomaly"] = np.append(
            self.data.get("is_anomaly", np.array([], dtype=bool)), bool(is_anomaly))

    def append_window(self, feat_row, meta):
        """
        Append one new fully-formed window — a 34-feature row in the exact
        shape iot_train.extract_features() produces, plus its meta dict
        (timestamp/fault_type/is_anomaly/raw sensor values) — so it
        participates in every future query()/query_indices() call exactly
        like a historical window: predicts its cluster with the
        already-fitted K-Means model, appends to self.X/self.metas/self.cl,
        and indexes it by date for indices()/query(). Returns the new
        window's index.

        Does NOT retrain or refit anything (K-Means/Decision-Tree/GRU stay
        exactly as iot_train.py trained them) — it only predicts with them,
        same as query_indices() already does for historical windows.
        """
        self.X = np.vstack([self.X, feat_row.reshape(1, -1)])
        self.metas.append(meta)
        cluster_id = int(self.km.predict(self.km_sc.transform(feat_row.reshape(1, -1)))[0])
        self.cl = np.append(self.cl, cluster_id)
        new_idx = len(self.metas) - 1
        self._idx[meta["timestamp"].date()].append(new_idx)
        return new_idx

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
        result = self.query_indices(idx, lbl)
        if result is not None:
            result["start"], result["end"] = s, e
        return result

    def query_indices(self, idx, lbl="period"):
        """Build the same stats/cluster/notification/forecast bundle as
        query(), but for an arbitrary list of window indices (need not be
        contiguous or in date order — the caller should still hand them
        in ascending order for the forecast's "latest index" logic to be
        meaningful). Returns None for an empty index list. Unlike query(),
        this does not set "start"/"end" — the caller decides what date
        range, if any, best describes the given indices."""
        if not idx:
            return None

        def sv(k):
            vals = []
            for i in idx:
                if 0 <= i < len(self.metas):
                    vals.append(self.metas[i].get(k, np.nan))
                else:
                    vals.append(np.nan)
            return np.array(vals)

        stats = {}
        for k in SENSOR_KEYS:
            v = sv(k); t = np.arange(len(v))
            slope = float(np.polyfit(t, v, 1)[0]) if len(v) > 1 else 0.0
            stats[k] = {
                "mean": float(v.mean()), "std":  float(v.std()),
                "min":  float(v.min()),  "max":  float(v.max()),
                "median": float(np.median(v)), "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
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
        for ci in [8, 7, 4, 5, 6, 3, 1, 2, 0]:
            if nc[ci] > 0: notif = NOTIF_CLASSES[ci]; break

        # Forecast
        data_idx = min(max(idx) + self.seq, len(self.data["temperature"]))
        fore, ascore = forecast_window(self.slm, self.slm_sc,
                                        self.data, data_idx, self.seq)

        n_anom = sum(1 for i in idx if self.metas[i]["is_anomaly"])
        return {
            "label":        lbl,
            "start":        None, "end": None,
            "indices":      idx,
            "n_windows":    len(idx),
            "stats":        stats,
            "cluster_id":   did,
            "cluster_name": self.cn.get(did, "Unknown"),
            "notification": notif,
            "n_anomaly":    n_anom,
            "anomaly_rate": n_anom / max(len(idx), 1),
            "forecast":     fore,
            "anomaly_score": ascore,
        }

    def series(self, idx, sensor):
        ts = [self.metas[i]["timestamp"] for i in idx]
        vs = np.array([self.metas[i][sensor] for i in idx])
        return ts, vs

    def cluster_series(self, idx):
        return [self.cn.get(self.cl[i] if i < len(self.cl) else 0, "?")
                for i in idx]
