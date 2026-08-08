"""
IoT Sensor Notification System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sensors  : BMP280 (temperature, pressure)
           AHT20  (humidity)
           BH1750 (light)
Pipeline :
  1. Simulate 60-day IoT time-series data
  2. Feature extraction from sliding windows
  3. K-Means clustering  → assign condition labels
  4. GRU-RNN SLM        → multi-step forecast + anomaly score
  5. Decision Tree       → rule-based notification classifier
  6. Notification engine → generate human-readable alerts
  7. Visualisation       → 6-panel evaluation dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── stdlib / third-party ──────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_squared_error, mean_absolute_error)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings, random, os, json, textwrap
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42); torch.manual_seed(42)

OUTPUT_DIR = os.getcwd()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── sensor metadata ───────────────────────────────────────────────────────────
SENSOR_KEYS  = ["temperature", "pressure", "humidity", "light"]
SENSOR_UNITS = {"temperature": "°C", "pressure": "hPa",
                "humidity": "% RH", "light": "lux"}
SENSOR_SHORT = ["Temp", "Press", "Humid", "Light"]

# ─────────────────────────────────────────────────────────────────────────────
#  1.  DATA SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
def simulate_iot_data(days=60, interval_minutes=30):
    """Simulate realistic multi-sensor IoT time-series with injected faults."""
    start  = datetime(2024, 1, 1)
    spd    = (24 * 60) // interval_minutes      # steps per day
    total  = days * spd

    ts, T, P, H, L, fault_type = [], [], [], [], [], []
    for i in range(total):
        t   = start + timedelta(minutes=i * interval_minutes)
        hr  = t.hour + t.minute / 60
        dow = t.weekday()
        seasonal = (i / total) * 3.0

        temp  = 24 + seasonal + 8 * np.sin(np.pi * (hr - 6) / 12) * (hr > 6)
        temp += 0.6 if dow >= 5 else 0
        temp += np.random.normal(0, 0.5)

        pres  = 1013.25 + 5 * np.sin(2 * np.pi * i / (spd * 7))
        pres += np.random.normal(0, 0.3)

        hum   = np.clip(70 - 20 * np.sin(np.pi * (hr - 6) / 12) * (hr > 6)
                        + np.random.normal(0, 1.5), 20, 100)

        if 6 <= hr <= 20:
            lux = max(0, 1000 * np.exp(-0.5 * ((hr - 13) / 3) ** 2)
                      + np.random.normal(0, 30))
        else:
            lux = max(0, np.random.normal(5, 2))

        # Inject faults (~3 %)
        ft = "normal"
        if random.random() < 0.03:
            ft = random.choice(["temp_spike", "pressure_drop",
                                "humidity_surge", "light_flicker",
                                "multi_fault"])
            if ft == "temp_spike":       temp += random.uniform(8, 15)
            elif ft == "pressure_drop":  pres -= random.uniform(10, 20)
            elif ft == "humidity_surge": hum   = min(100, hum + random.uniform(20, 30))
            elif ft == "light_flicker":  lux   = random.choice([0, 1500])
            elif ft == "multi_fault":
                temp += random.uniform(5, 10)
                hum   = min(100, hum + random.uniform(10, 20))

        ts.append(t); T.append(round(temp, 2)); P.append(round(pres, 2))
        H.append(round(hum,  2)); L.append(round(lux, 2)); fault_type.append(ft)

    return {
        "timestamps":   ts,
        "temperature":  np.array(T),
        "pressure":     np.array(P),
        "humidity":     np.array(H),
        "light":        np.array(L),
        "fault_type":   fault_type,
        "is_anomaly":   np.array([f != "normal" for f in fault_type]),
    }

# ─────────────────────────────────────────────────────────────────────────────
#  2.  FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(data, window=12):
    """
    Sliding-window statistical features for clustering / DT input.
    Returns X_feat (N, F) and meta arrays.
    """
    keys  = SENSOR_KEYS
    n     = len(data["temperature"])
    rows, metas = [], []

    for i in range(window, n):
        seg = {k: data[k][i - window:i] for k in keys}
        row = []
        for k in keys:
            s = seg[k]
            row += [s.mean(), s.std(), s.min(), s.max(),
                    s.max() - s.min(),                   # range
                    np.percentile(s, 75) - np.percentile(s, 25),  # IQR
                    np.diff(s).mean(),                   # mean delta
                    np.abs(np.diff(s)).mean()]            # mean abs delta
        # Cross-sensor features
        corr_th = np.corrcoef(seg["temperature"], seg["humidity"])[0, 1]
        corr_tl = np.corrcoef(seg["temperature"], seg["light"])[0, 1]
        row += [corr_th, corr_tl]
        rows.append(row)
        metas.append({
            "timestamp":  data["timestamps"][i],
            "is_anomaly": data["is_anomaly"][i],
            "fault_type": data["fault_type"][i],
            **{k: data[k][i] for k in keys},
        })

    return np.array(rows, dtype=np.float32), metas

# ─────────────────────────────────────────────────────────────────────────────
#  3.  K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
# Cluster labels (human-assigned after inspection of centroids)
CLUSTER_LABELS = {
    0: "NORMAL_DAY",
    1: "NORMAL_NIGHT",
    2: "HIGH_TEMP_ALERT",
    3: "HIGH_HUMIDITY_ALERT",
    4: "PRESSURE_ANOMALY",
    5: "LIGHT_ANOMALY",
    6: "MULTI_FAULT",
    7: "BORDERLINE",
}

CLUSTER_SEVERITY = {
    "NORMAL_DAY":          "INFO",
    "NORMAL_NIGHT":        "INFO",
    "HIGH_TEMP_ALERT":     "WARNING",
    "HIGH_HUMIDITY_ALERT": "WARNING",
    "PRESSURE_ANOMALY":    "CRITICAL",
    "LIGHT_ANOMALY":       "WARNING",
    "MULTI_FAULT":         "CRITICAL",
    "BORDERLINE":          "WATCH",
}

def fit_kmeans(X_feat, n_clusters=8):
    scaler  = StandardScaler()
    X_s     = scaler.fit_transform(X_feat)
    km      = KMeans(n_clusters=n_clusters, random_state=42, n_init=15, max_iter=500)
    labels  = km.fit_predict(X_s)
    return km, scaler, labels

def assign_cluster_names(km, X_feat_scaled, metas):
    """
    Heuristically name each K-Means cluster based on its centroid
    profile and the most common fault type among its members.
    Returns a dict {cluster_id -> label_string}.
    """
    centroids   = km.cluster_centers_          # (K, F) in scaled space
    labels_pred = km.labels_
    K           = km.n_clusters

    # Feature index mapping (8 stats per sensor, 4 sensors, then 2 cross-corr)
    # mean index for each sensor: temperature=0, pressure=8, humidity=16, light=24
    T_MEAN, P_MEAN, H_MEAN, L_MEAN = 0, 8, 16, 24
    T_STD,  H_STD                  = 1, 17
    CORR_TH                        = 34   # temp-humidity correlation

    assignment = {}
    used_names = set()

    for k in range(K):
        members = [i for i, l in enumerate(labels_pred) if l == k]
        fault_counts = defaultdict(int)
        for m in members:
            fault_counts[metas[m]["fault_type"]] += 1

        dominant_fault = max(fault_counts, key=fault_counts.get)
        c = centroids[k]

        # Rule-based name assignment
        if dominant_fault == "temp_spike"       or c[T_MEAN] > 1.5:
            name = "HIGH_TEMP_ALERT"
        elif dominant_fault == "humidity_surge" or c[H_MEAN] > 1.5:
            name = "HIGH_HUMIDITY_ALERT"
        elif dominant_fault == "pressure_drop"  or c[P_MEAN] < -1.5:
            name = "PRESSURE_ANOMALY"
        elif dominant_fault == "light_flicker"  or c[L_MEAN] > 2.0:
            name = "LIGHT_ANOMALY"
        elif dominant_fault == "multi_fault":
            name = "MULTI_FAULT"
        elif c[T_MEAN] > 0.3 and c[H_MEAN] < -0.2:  # typical warm day
            name = "NORMAL_DAY"
        elif c[T_MEAN] < -0.2:                        # cool/night
            name = "NORMAL_NIGHT"
        else:
            name = "BORDERLINE"

        # De-duplicate
        orig = name
        suffix = 1
        while name in used_names:
            name = f"{orig}_{suffix}"
            suffix += 1
        used_names.add(name)
        assignment[k] = name

    return assignment

# ─────────────────────────────────────────────────────────────────────────────
#  4.  RNN-SLM (GRU + Attention)
# ─────────────────────────────────────────────────────────────────────────────
class IoTSLM(nn.Module):
    """Stacked GRU with temporal self-attention for multi-step forecasting."""
    def __init__(self, input_size=4, hidden=128, n_layers=2,
                 pred_len=6, dropout=0.2):
        super().__init__()
        self.pred_len    = pred_len
        self.input_size  = input_size
        self.proj        = nn.Linear(input_size, hidden)
        self.gru         = nn.GRU(hidden, hidden, n_layers, batch_first=True,
                                  dropout=dropout if n_layers > 1 else 0.0)
        self.attn_q      = nn.Linear(hidden, hidden)
        self.attn_k      = nn.Linear(hidden, hidden)
        self.attn_v      = nn.Linear(hidden, hidden)
        self.scale       = hidden ** 0.5
        self.decoder     = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, pred_len * input_size),
        )
        self.anom_head   = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        h    = self.proj(x)
        enc, _  = self.gru(h)
        Q, K, V = self.attn_q(enc), self.attn_k(enc), self.attn_v(enc)
        w    = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        ctx  = torch.bmm(w, V).mean(dim=1)
        pred = self.decoder(ctx).view(-1, self.pred_len, self.input_size)
        return pred, self.anom_head(ctx), w


class SensorDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):   return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def build_sequences(data, seq_len=24, pred_len=6):
    feats  = np.stack([data[k] for k in SENSOR_KEYS], axis=1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feats)
    X, y   = [], []
    for i in range(len(scaled) - seq_len - pred_len + 1):
        X.append(scaled[i:i + seq_len])
        y.append(scaled[i + seq_len:i + seq_len + pred_len])
    return np.array(X), np.array(y), scaler


def train_slm(data, seq_len=24, pred_len=6, epochs=60, lr=1e-3, batch=64):
    X, y, scaler = build_sequences(data, seq_len, pred_len)
    split = int(0.8 * len(X))
    tr_ld = DataLoader(SensorDS(X[:split], y[:split]), batch_size=batch, shuffle=True)
    va_ld = DataLoader(SensorDS(X[split:], y[split:]), batch_size=batch)

    model = IoTSLM(pred_len=pred_len)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse   = nn.MSELoss()
    bce   = nn.BCELoss()
    hist  = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        model.train(); tl = 0.0
        for xb, yb in tr_ld:
            opt.zero_grad()
            p, a, _ = model(xb)
            lp  = mse(p, yb)
            re  = ((p - yb) ** 2).mean(dim=(1, 2))
            lbl = (re > re.mean() + re.std()).float().unsqueeze(1)
            loss = lp + 0.1 * bce(a, lbl)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()
        sch.step()
        model.eval()
        with torch.no_grad():
            vl = sum(mse(model(xb)[0], yb).item() for xb, yb in va_ld) / len(va_ld)
        hist["train"].append(tl / len(tr_ld))
        hist["val"].append(vl)
        if ep % 10 == 0:
            print(f"    Epoch {ep:3d}/{epochs}  train={tl/len(tr_ld):.5f}  val={vl:.5f}")

    return model, scaler, X, y, hist


def get_slm_scores(model, X_seq, batch=128):
    """Return anomaly scores for all windows (N,)."""
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X_seq), batch):
            xb = torch.tensor(X_seq[i:i + batch], dtype=torch.float32)
            _, a, _ = model(xb)
            scores.append(a.squeeze(1).numpy())
    return np.concatenate(scores)

# ─────────────────────────────────────────────────────────────────────────────
#  5.  DECISION TREE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
# Notification classes (DT output)
NOTIF_CLASSES = [
    "NO_ACTION",        # 0 – everything normal
    "INFO_DAY",         # 1 – normal day cycle, info only
    "INFO_NIGHT",       # 2 – normal night cycle, info only
    "WATCH_BORDERLINE", # 3 – borderline, keep watching
    "WARN_TEMP",        # 4 – temperature warning
    "WARN_HUMIDITY",    # 5 – humidity warning
    "WARN_LIGHT",       # 6 – light anomaly
    "CRIT_PRESSURE",    # 7 – critical pressure drop
    "CRIT_MULTI",       # 8 – multiple faults simultaneously
]

NOTIF_SEVERITY = {
    "NO_ACTION":        "NOMINAL",
    "INFO_DAY":         "INFO",
    "INFO_NIGHT":       "INFO",
    "WATCH_BORDERLINE": "WATCH",
    "WARN_TEMP":        "WARNING",
    "WARN_HUMIDITY":    "WARNING",
    "WARN_LIGHT":       "WARNING",
    "CRIT_PRESSURE":    "CRITICAL",
    "CRIT_MULTI":       "CRITICAL",
}

NOTIF_MESSAGES = {
    "NO_ACTION":
        "All sensor readings are within normal operating ranges. No action required.",
    "INFO_DAY":
        "Daytime environmental conditions detected. Temperature and light levels are within expected daytime ranges.",
    "INFO_NIGHT":
        "Nighttime environmental conditions confirmed. Low light and temperature within expected overnight range.",
    "WATCH_BORDERLINE":
        "Sensor values are approaching threshold limits. Increased monitoring recommended over the next cycle.",
    "WARN_TEMP":
        "Temperature reading exceeds the normal operational upper bound. Check HVAC, ventilation, or nearby heat sources.",
    "WARN_HUMIDITY":
        "Relative humidity is elevated above safe operating levels. Inspect for water ingress, condensation, or seal failure.",
    "WARN_LIGHT":
        "Illuminance reading is outside the expected range. Possible sensor occlusion, flicker, or unexpected lighting change.",
    "CRIT_PRESSURE":
        "CRITICAL: Atmospheric pressure has dropped sharply below the 2.5σ lower threshold. Possible weather event or sensor fault. Immediate inspection required.",
    "CRIT_MULTI":
        "CRITICAL: Multiple sensor variables are simultaneously outside normal bounds. Possible compound environmental fault or sensor array failure. Immediate intervention required.",
}


def build_dt_labels(X_feat, metas, cluster_names, cluster_labels,
                    slm_scores, seq_len):
    """
    Combine cluster assignment + SLM anomaly score + raw sensor values
    to generate ground-truth notification labels for Decision Tree training.
    """
    n       = len(X_feat)
    y_notif = np.zeros(n, dtype=int)

    for i in range(n):
        meta   = metas[i]
        cname  = cluster_names.get(cluster_labels[i], "BORDERLINE")
        score  = slm_scores[i] if i < len(slm_scores) else 0.0
        ft     = meta["fault_type"]
        temp   = meta["temperature"]
        hum    = meta["humidity"]
        pres   = meta["pressure"]
        lux    = meta["light"]

        # Rule hierarchy (highest priority first)
        if ft == "multi_fault" or (score > 0.70 and ft != "normal"):
            y_notif[i] = 8   # CRIT_MULTI
        elif ft == "pressure_drop" or (pres < 995):
            y_notif[i] = 7   # CRIT_PRESSURE
        elif ft == "temp_spike"    or (temp > 38):
            y_notif[i] = 4   # WARN_TEMP
        elif ft == "humidity_surge"or (hum  > 92):
            y_notif[i] = 5   # WARN_HUMIDITY
        elif ft == "light_flicker" or (lux > 1200 or (lux < 5 and 8 < meta["timestamp"].hour < 18)):
            y_notif[i] = 6   # WARN_LIGHT
        elif "BORDERLINE" in cname or (0.35 < score <= 0.55):
            y_notif[i] = 3   # WATCH_BORDERLINE
        elif "NIGHT" in cname:
            y_notif[i] = 2   # INFO_NIGHT
        elif "DAY" in cname:
            y_notif[i] = 1   # INFO_DAY
        else:
            y_notif[i] = 0   # NO_ACTION

    return y_notif


def train_decision_tree(X_feat, y_labels, max_depth=12):
    scaler  = StandardScaler()
    X_s     = scaler.fit_transform(X_feat)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y_labels, test_size=0.2, random_state=42, stratify=y_labels)

    # Oversample minority classes (WARN/CRIT) to at least 80 samples each
    from collections import Counter
    counts = Counter(y_tr)
    target = max(80, counts.most_common()[-1][1] * 3)
    X_over, y_over = list(X_tr), list(y_tr)
    for cls, cnt in counts.items():
        if cnt < target:
            idx   = np.where(y_tr == cls)[0]
            extra = np.random.choice(idx, target - cnt, replace=True)
            noise = np.random.normal(0, 0.05, (len(extra), X_tr.shape[1]))
            X_over.extend(X_tr[extra] + noise)
            y_over.extend([cls] * len(extra))
    X_over = np.array(X_over); y_over = np.array(y_over)

    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=3,
        min_samples_split=6,
        class_weight="balanced",
        random_state=42,
    )
    dt.fit(X_over, y_over)
    y_pred = dt.predict(X_te)
    return dt, scaler, X_te, y_te, y_pred

# ─────────────────────────────────────────────────────────────────────────────
#  6.  NOTIFICATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class NotificationEngine:
    """Generates structured notifications from DT predictions."""

    def __init__(self, dt_model, dt_scaler, km_model, km_scaler,
                 slm_model, slm_scaler, seq_len=24):
        self.dt       = dt_model
        self.dt_sc    = dt_scaler
        self.km       = km_model
        self.km_sc    = km_scaler
        self.slm      = slm_model
        self.slm_sc   = slm_scaler
        self.seq_len  = seq_len

    def process_window(self, window_data, feat_vec, timestamp):
        """
        Process one sensor window and return a notification dict.
        window_data : dict with numpy arrays (T,) for each sensor
        feat_vec    : (F,) feature vector (pre-extracted)
        """
        # SLM anomaly score
        seq = np.stack([window_data[k] for k in SENSOR_KEYS], axis=1)
        seq_scaled = self.slm_sc.transform(seq)
        xt  = torch.tensor(seq_scaled[None], dtype=torch.float32)
        with torch.no_grad():
            pred, anom, _ = self.slm(xt)
        anom_score = float(anom.squeeze())

        # K-Means cluster
        fs  = self.km_sc.transform(feat_vec[None])
        cid = int(self.km.predict(fs)[0])

        # DT notification class
        fs_dt = self.dt_sc.transform(feat_vec[None])
        nid   = int(self.dt.predict(fs_dt)[0])
        nname = NOTIF_CLASSES[nid]

        # Current sensor readings (last value in window)
        readings = {k: float(window_data[k][-1]) for k in SENSOR_KEYS}

        return {
            "timestamp":     timestamp.strftime("%Y-%m-%d %H:%M"),
            "notification":  nname,
            "severity":      NOTIF_SEVERITY[nname],
            "message":       NOTIF_MESSAGES[nname],
            "anomaly_score": round(anom_score, 4),
            "cluster_id":    cid,
            "readings":      readings,
        }

    def batch_process(self, data, X_feat, metas, cluster_names):
        """Run notification engine over all windows."""
        notifications = []
        seq = self.seq_len
        n   = len(X_feat)
        for i in range(n):
            window = {k: data[k][i:i + seq] for k in SENSOR_KEYS}
            notif  = self.process_window(window, X_feat[i], metas[i]["timestamp"])
            notif["cluster_name"] = cluster_names.get(
                int(self.km.predict(self.km_sc.transform(X_feat[i][None]))[0]),
                "UNKNOWN")
            notifications.append(notif)
        return notifications

    def summary_report(self, notifications):
        """Print a summary table of notification distribution."""
        counts = defaultdict(int)
        for n in notifications:
            counts[n["notification"]] += 1
        total = len(notifications)

        lines = [
            "=" * 64,
            "  NOTIFICATION SUMMARY REPORT",
            f"  Total windows processed : {total}",
            "=" * 64,
        ]
        for name in NOTIF_CLASSES:
            cnt  = counts[name]
            pct  = cnt / total * 100
            bar  = "[]" * int(pct / 2)
            sev  = NOTIF_SEVERITY[name]
            lines.append(f"  {sev:<16s}  {name:<22s}  {cnt:5d} ({pct:5.1f}%)  {bar}")
        lines.append("=" * 64)
        return "\n".join(lines)

    def recent_alerts(self, notifications, n=10):
        """Return the n most recent non-nominal notifications."""
        alerts = [x for x in notifications
                  if x["notification"] not in ("NO_ACTION", "INFO_DAY", "INFO_NIGHT")]
        return alerts[-n:]

# ─────────────────────────────────────────────────────────────────────────────
#  7.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (light-print-friendly)
C = {
    "navy":    "#0E2A5C",
    "blue":    "#1A5276",
    "teal":    "#0D7A6E",
    "green":   "#1D8348",
    "gold":    "#B7770D",
    "red":     "#B03A2E",
    "purple":  "#6C3483",
    "mid":     "#5D6D7E",
    "light":   "#EAF0FB",
    "bg":      "#FAFBFF",
}

SEVERITY_COLORS = {
    "NO_ACTION":        C["teal"],
    "INFO_DAY":         C["blue"],
    "INFO_NIGHT":       C["navy"],
    "WATCH_BORDERLINE": C["gold"],
    "WARN_TEMP":        "#E67E22",
    "WARN_HUMIDITY":    C["purple"],
    "WARN_LIGHT":       "#D4AC0D",
    "CRIT_PRESSURE":    C["red"],
    "CRIT_MULTI":       "#922B21",
}


def plot_dashboard(data, notifications, hist, dt_model, dt_scaler,
                   X_feat, y_true, y_pred_dt, cluster_labels, cluster_names):

    plt.rcParams.update({
        "figure.facecolor": C["bg"],
        "axes.facecolor":   C["light"],
        "axes.edgecolor":   "#C0C8D8",
        "grid.color":       "#C0C8D8",
        "text.color":       C["navy"],
        "axes.labelcolor":  C["navy"],
        "xtick.color":      C["mid"],
        "ytick.color":      C["mid"],
        "axes.titlecolor":  C["navy"],
        "font.family":      "monospace",
        "axes.spines.top":  False,
        "axes.spines.right":False,
    })

    fig = plt.figure(figsize=(22, 26), dpi=110)
    fig.patch.set_facecolor(C["bg"])

    fig.text(0.5, 0.992, "IoT-SLM + Decision Tree — Sensor Notification Dashboard",
             ha="center", va="top", fontsize=17, fontweight="bold", color=C["navy"])
    fig.text(0.5, 0.983, "BMP280 · AHT20 · BH1750  |  K-Means Clustering · GRU-RNN · Decision Tree",
             ha="center", va="top", fontsize=10, color=C["mid"], style="italic")

    outer = gridspec.GridSpec(4, 1, figure=fig,
                              top=0.978, bottom=0.022,
                              hspace=0.46, left=0.06, right=0.97)

    # ── ROW 0 : Training loss  |  Cluster distribution  |  Notification dist ──
    row0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.35)

    # Training loss
    ax = fig.add_subplot(row0[0])
    ep = range(1, len(hist["train"]) + 1)
    ax.plot(ep, hist["train"], color=C["blue"],   lw=2, label="Train")
    ax.plot(ep, hist["val"],   color=C["red"],    lw=2, label="Val", ls="--")
    ax.set_title("RNN-SLM Training Loss (MSE)", fontsize=10, pad=6)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

    # Cluster distribution
    ax = fig.add_subplot(row0[1])
    cnames_list = [cluster_names.get(k, f"C{k}") for k in sorted(cluster_names)]
    # shorten long names
    short = [n.replace("_", "\n") for n in cnames_list]
    counts_cl = [np.sum(cluster_labels == k) for k in sorted(cluster_names)]
    colors_cl  = [C["teal"] if "NORMAL" in n else
                  C["gold"] if "BORDER" in n else
                  C["red"]  for n in cnames_list]
    ax.barh(short, counts_cl, color=colors_cl, alpha=0.85)
    ax.set_title("K-Means Cluster Distribution", fontsize=10, pad=6)
    ax.set_xlabel("Window count"); ax.grid(True, alpha=0.3, axis="x")

    # Notification distribution
    ax = fig.add_subplot(row0[2])
    n_counts = defaultdict(int)
    for n in notifications:
        n_counts[n["notification"]] += 1
    n_labels = [nc for nc in NOTIF_CLASSES if nc in n_counts]
    n_vals   = [n_counts[nc] for nc in n_labels]
    n_colors = [SEVERITY_COLORS[nc] for nc in n_labels]
    bars = ax.barh([l.replace("_", "\n") for l in n_labels], n_vals,
                   color=n_colors, alpha=0.88)
    ax.set_title("Notification Class Distribution", fontsize=10, pad=6)
    ax.set_xlabel("Count"); ax.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, n_vals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8)

    # ── ROW 1 : Time-series with notification overlay ─────────────────────────
    row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.30)

    n_plot = min(2000, len(data["timestamps"]))
    ts_num = np.arange(n_plot)

    # Temperature with critical alerts
    ax = fig.add_subplot(row1[0])
    ax.plot(ts_num, data["temperature"][:n_plot],
            color=C["blue"], lw=0.8, alpha=0.85, label="Temperature")
    # Overlay critical notifications
    notif_ts_all = notifications[:n_plot]
    crit_idx = [i for i, n in enumerate(notif_ts_all)
                if "CRIT" in n["notification"]]
    warn_idx = [i for i, n in enumerate(notif_ts_all)
                if "WARN" in n["notification"]]
    if crit_idx:
        ax.scatter(crit_idx, data["temperature"][:n_plot][crit_idx],
                   color=C["red"], s=22, zorder=5, label="CRITICAL")
    if warn_idx:
        ax.scatter(warn_idx, data["temperature"][:n_plot][warn_idx],
                   color="#E67E22", s=14, zorder=4, marker="^", label="WARNING")
    ax.set_title("Temperature (°C) — Notification Overlay", fontsize=10, pad=6)
    ax.set_xlabel("Step"); ax.set_ylabel("°C")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Anomaly score over time
    ax = fig.add_subplot(row1[1])
    anom_scores = [n["anomaly_score"] for n in notif_ts_all]
    ax.fill_between(range(len(anom_scores)), anom_scores,
                    color=C["teal"], alpha=0.3)
    ax.plot(anom_scores, color=C["teal"], lw=0.9)
    ax.axhline(0.5, color=C["red"], lw=1.2, ls="--", label="Anomaly threshold (0.5)")
    ax.axhline(0.35, color=C["gold"], lw=1.0, ls=":", label="Watch threshold (0.35)")
    ax.set_title("RNN-SLM Anomaly Score Over Time", fontsize=10, pad=6)
    ax.set_xlabel("Step"); ax.set_ylabel("Score [0-1]")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── ROW 2 : Decision Tree performance  |  Confusion matrix  |  DT rules ──
    row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.38)

    # Per-class precision / recall
    ax = fig.add_subplot(row2[0])
    from sklearn.metrics import precision_score, recall_score, f1_score
    present = sorted(set(y_true))
    prec  = precision_score(y_true, y_pred_dt, labels=present, average=None, zero_division=0)
    rec   = recall_score(y_true, y_pred_dt, labels=present, average=None, zero_division=0)
    f1s   = f1_score(y_true, y_pred_dt, labels=present, average=None, zero_division=0)
    x_pos = np.arange(len(present))
    ax.bar(x_pos - 0.25, prec, 0.23, label="Precision", color=C["blue"],   alpha=0.85)
    ax.bar(x_pos,         rec,  0.23, label="Recall",    color=C["teal"],   alpha=0.85)
    ax.bar(x_pos + 0.25, f1s,  0.23, label="F1",        color=C["purple"], alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([NOTIF_CLASSES[p][:9] for p in present], fontsize=7, rotation=45)
    ax.set_title("Decision Tree — Precision / Recall / F1", fontsize=10, pad=6)
    ax.set_ylim(0, 1.1); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # Confusion matrix
    ax = fig.add_subplot(row2[1])
    cm  = confusion_matrix(y_true, y_pred_dt, labels=present)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    im  = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(present)))
    ax.set_yticks(range(len(present)))
    ax.set_xticklabels([NOTIF_CLASSES[p][:6] for p in present], fontsize=7, rotation=45)
    ax.set_yticklabels([NOTIF_CLASSES[p][:6] for p in present], fontsize=7)
    ax.set_title("Confusion Matrix (normalised)", fontsize=10, pad=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    for ri in range(len(present)):
        for ci in range(len(present)):
            v = cm_norm[ri, ci]
            ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if v > 0.5 else C["navy"])

    # Feature importance (top 15)
    ax = fig.add_subplot(row2[2])
    fi   = dt_model.feature_importances_
    top  = np.argsort(fi)[-15:]
    feat_names = []
    for k in SENSOR_KEYS:
        feat_names += [f"{k[:4]}_{s}" for s in
                       ["mean","std","min","max","range","iqr","Δmean","Δabs"]]
    feat_names += ["corr_TH", "corr_TL"]
    ax.barh([feat_names[t] for t in top], fi[top],
            color=C["teal"], alpha=0.85)
    ax.set_title("Decision Tree Feature Importance (Top 15)", fontsize=10, pad=6)
    ax.set_xlabel("Importance"); ax.grid(True, alpha=0.3, axis="x")

    # ── ROW 3 : Sensor heatmap  |  Recent alerts text ─────────────────────────
    row3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3], wspace=0.32)

    # Sensor heatmap (24h × 60 days, temperature)
    ax = fig.add_subplot(row3[0])
    spd   = 48   # steps per day (30-min interval)
    days  = len(data["temperature"]) // spd
    mat   = data["temperature"][:days * spd].reshape(days, spd)
    hmap  = ax.imshow(mat.T, aspect="auto", cmap="RdYlBu_r",
                      origin="lower", extent=[0, days, 0, 24])
    plt.colorbar(hmap, ax=ax, fraction=0.04, pad=0.02, label="°C")
    ax.set_title("Temperature Heatmap (Day × Hour)", fontsize=10, pad=6)
    ax.set_xlabel("Day"); ax.set_ylabel("Hour of day")
    # Overlay critical days
    crit_days = set()
    for i, n in enumerate(notifications):
        if "CRIT" in n["notification"] and i < days * spd:
            crit_days.add(i // spd)
    for d in crit_days:
        ax.axvline(d, color=C["red"], lw=0.8, alpha=0.7)

    # Recent critical/warning alerts as text panel
    ax = fig.add_subplot(row3[1])
    ax.set_facecolor("#F0F4FF")
    ax.axis("off")
    alerts = [n for n in notifications
              if n["notification"] not in ("NO_ACTION", "INFO_DAY", "INFO_NIGHT")][-12:]
    lines  = ["RECENT NOTIFICATIONS (non-nominal)\n" + "─" * 44]
    for a in alerts:
        sev = a["severity"]
        ts_ = a["timestamp"]
        msg = textwrap.shorten(a["message"], 55)
        r   = a["readings"]
        lines.append(
            f"{ts_}  {sev}\n"
            f"  {a['notification']}\n"
            f"  T={r['temperature']:.1f}°C  H={r['humidity']:.1f}%  "
            f"P={r['pressure']:.1f}hPa  L={r['light']:.0f}lux\n"
            f"  Anomaly score: {a['anomaly_score']:.3f}\n"
            f"  {msg}\n"
        )
    ax.text(0.02, 0.98, "\n".join(lines),
            transform=ax.transAxes, fontsize=7.5, va="top",
            family="monospace", color=C["navy"],
            bbox=dict(facecolor="#F0F4FF", edgecolor="#C0C8D8",
                      boxstyle="round,pad=0.5"))
    ax.set_title("Recent Non-Nominal Notifications", fontsize=10, pad=6)

    out_path = os.path.join(OUTPUT_DIR, "iot_slm_notification_dashboard.png")
    plt.savefig(out_path, dpi=110, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"  Dashboard saved → {out_path}")
    return out_path

# ─────────────────────────────────────────────────────────────────────────────
#  8.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    from sys import stdout, stderr
    stdout.reconfigure(encoding="utf-8")
    stderr.reconfigure(encoding="utf-8")
    print("  IoT-SLM  Notification System")
    print("  RNN-GRU · K-Means · Decision Tree")
    print("  BMP280 · AHT20 · BH1750")

    # ── Step 1: simulate data ─────────────────────────────────────────────────
    print("\n[1] Simulating 60-day IoT sensor data ...")
    data = simulate_iot_data(days=60, interval_minutes=30)
    n_fault = sum(1 for f in data["fault_type"] if f != "normal")
    print(f"    Total readings : {len(data['timestamps'])}")
    print(f"    Fault events   : {n_fault} ({n_fault/len(data['timestamps'])*100:.1f}%)")

    # ── Step 2: feature extraction ────────────────────────────────────────────
    print("\n[2] Extracting sliding-window features (window=12 steps = 6h) ...")
    SEQ_LEN = 24
    X_feat, metas = extract_features(data, window=SEQ_LEN)
    print(f"    Feature matrix : {X_feat.shape}  ({X_feat.shape[1]} features per window)")

    # ── Step 3: clustering ────────────────────────────────────────────────────
    print("\n[3] Fitting K-Means (k=8) ...")
    km, km_scaler, cluster_labels = fit_kmeans(X_feat, n_clusters=8)
    cluster_names = assign_cluster_names(km, km_scaler.transform(X_feat), metas)
    print("    Cluster assignments:")
    for k, name in sorted(cluster_names.items()):
        cnt = np.sum(cluster_labels == k)
        print(f"      Cluster {k}: {name:<25s}  ({cnt} windows)")

    # ── Step 4: train RNN-SLM ────────────────────────────────────────────────
    print("\n[4] Training IoT-SLM (GRU-RNN, 60 epochs) ...")
    slm, slm_scaler, X_seq, y_seq, hist = train_slm(
        data, seq_len=SEQ_LEN, pred_len=6, epochs=60)
    print(f"    Final train loss : {hist['train'][-1]:.5f}")
    print(f"    Final val loss   : {hist['val'][-1]:.5f}")

    # SLM anomaly scores for all feat windows (align lengths)
    print("\n[5] Computing SLM anomaly scores ...")
    slm_scores_all = get_slm_scores(slm, X_seq)
    # X_feat and X_seq may differ in length; pad/trim to align
    min_len = min(len(X_feat), len(slm_scores_all))
    X_feat_aligned  = X_feat[:min_len]
    metas_aligned   = metas[:min_len]
    cluster_labels_aligned = cluster_labels[:min_len]
    print(f"    Scored {len(slm_scores_all)} windows  "
          f"(anomaly rate >{0.5:.0%}: "
          f"{(slm_scores_all>0.5).mean()*100:.1f}%)")

    # ── Step 5: Decision Tree ─────────────────────────────────────────────────
    print("\n[6] Building DT notification labels ...")
    y_notif = build_dt_labels(
        X_feat_aligned, metas_aligned, cluster_names,
        cluster_labels_aligned, slm_scores_all, SEQ_LEN)

    label_counts = defaultdict(int)
    for lbl in y_notif:
        label_counts[NOTIF_CLASSES[lbl]] += 1
    for name, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {name:<24s} : {cnt:5d} ({cnt/len(y_notif)*100:.1f}%)")

    print("\n[7] Training Decision Tree classifier ...")
    dt, dt_scaler, X_te, y_te, y_pred = train_decision_tree(
        X_feat_aligned, y_notif, max_depth=12)
    acc = (y_pred == y_te).mean()
    print(f"    Test accuracy   : {acc*100:.2f}%")
    print(f"    Tree depth      : {dt.get_depth()}")
    print(f"    Leaf nodes      : {dt.get_n_leaves()}")

    # Print classification report
    present = sorted(set(y_te))
    print("\n    Classification Report:")
    print(classification_report(
        y_te, y_pred,
        labels=present,
        target_names=[NOTIF_CLASSES[i] for i in present],
        zero_division=0))

    # Print top DT rules
    print("\n    Top Decision Tree Rules (depth <= 3):")
    dt_shallow = DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                        random_state=42)
    dt_shallow.fit(dt_scaler.transform(X_feat_aligned), y_notif)
    feat_names = []
    for k in SENSOR_KEYS:
        feat_names += [f"{k[:4]}_{s}" for s in
                       ["mean","std","min","max","range","iqr","Δmean","Δabs"]]
    feat_names += ["corr_TH", "corr_TL"]
    rules = export_text(dt_shallow, feature_names=feat_names, max_depth=3)
    for line in rules.split("\n")[:30]:
        print("  ", line)

    # ── Step 6: Notification Engine ───────────────────────────────────────────
    print("\n[8] Running Notification Engine over all windows ...")
    engine = NotificationEngine(
        dt, dt_scaler, km, km_scaler, slm, slm_scaler, SEQ_LEN)
    notifications = engine.batch_process(
        data, X_feat_aligned, metas_aligned, cluster_names)
    print(f"    Generated {len(notifications)} notifications")

    # Summary report
    report = engine.summary_report(notifications)
    print("\n" + report)

    # Recent alerts
    alerts = engine.recent_alerts(notifications, n=5)
    print("\n  ── Recent Non-Nominal Alerts ──────────────────────────────")
    for a in alerts:
        print(f"  [{a['timestamp']}] {a['severity']}  {a['notification']}")
        print(f"    Anomaly score: {a['anomaly_score']:.4f}  "
              f"Cluster: {a['cluster_name']}")
        r = a["readings"]
        print(f"    T={r['temperature']:.1f}°C  H={r['humidity']:.1f}%RH  "
              f"P={r['pressure']:.1f}hPa  L={r['light']:.0f}lux")
        print(f"    → {a['message'][:80]}")
        print()

    # Save notifications to JSON
    json_path = os.path.join(OUTPUT_DIR, "iot_notifications.json")
    with open(json_path, "w") as f:
        json.dump(notifications[-100:], f, indent=2, default=str)
    print(f"  Last 100 notifications saved → {json_path}")

    # ── Step 7: Visualisation ─────────────────────────────────────────────────
    print("\n[9] Generating evaluation dashboard ...")
    plot_dashboard(
        data, notifications, hist, dt, dt_scaler,
        X_feat_aligned, y_te, y_pred,
        cluster_labels_aligned, cluster_names)

    print("\n" + "=" * 64)
    print("  Pipeline complete.")
    print("=" * 64 + "\n")

if __name__ == "__main__":
    main()
