"""
iot_train.py  —  IoT-SLM Model Training & Saving
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run ONCE to train all models and save them to ./iot_model/

Output files:
  iot_model/
  ├── config.json          ← dataset & model hyperparameters
  ├── data.pkl             ← simulated sensor data (numpy arrays)
  ├── features.pkl         ← X_feat (2856×34), metas list
  ├── slm_weights.pt       ← GRU-RNN state dict
  ├── slm_scaler.pkl       ← MinMaxScaler
  ├── slm_history.pkl      ← training loss history
  ├── kmeans.pkl           ← KMeans model
  ├── kmeans_scaler.pkl    ← StandardScaler for KMeans
  ├── cluster_labels.pkl   ← per-window cluster labels
  ├── cluster_names.json   ← {cluster_id: label_string}
  ├── dt.pkl               ← DecisionTree classifier
  ├── dt_scaler.pkl        ← StandardScaler for DT
  └── eval_metrics.json    ← R², MAE, RMSE per sensor

Usage:
  python iot_train.py                 # default 60 days, 60 epochs
  python iot_train.py --days 30       # shorter dataset
  python iot_train.py --epochs 30     # faster training
  python iot_train.py --out ./models  # custom output directory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse, os, json, pickle, warnings, random
from datetime import datetime, timedelta
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="IoT-SLM Training Script")
    p.add_argument("--days",    type=int, default=60,  help="Simulation days (default 60)")
    p.add_argument("--interval",type=int, default=30,  help="Sensor interval in minutes (default 30)")
    p.add_argument("--epochs",  type=int, default=60,  help="GRU training epochs (default 60)")
    p.add_argument("--seq_len", type=int, default=24,  help="RNN input sequence length (default 24)")
    p.add_argument("--pred_len",type=int, default=6,   help="RNN forecast steps (default 6)")
    p.add_argument("--hidden",  type=int, default=128, help="GRU hidden size (default 128)")
    p.add_argument("--clusters",type=int, default=8,   help="K-Means k (default 8)")
    p.add_argument("--seed",    type=int, default=42,  help="Random seed (default 42)")
    p.add_argument("--out",     type=str, default="./iot_model", help="Output directory")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SENSOR_KEYS  = ["temperature", "pressure", "humidity", "light"]
SENSOR_UNITS = {"temperature":"°C","pressure":"hPa","humidity":"% RH","light":"lux"}
NOTIF_CLASSES = [
    "NO_ACTION","INFO_DAY","INFO_NIGHT","WATCH_BORDERLINE",
    "WARN_TEMP","WARN_HUMIDITY","WARN_LIGHT","CRIT_PRESSURE","CRIT_MULTI"
]

# ─────────────────────────────────────────────────────────────────────────────
#  1. DATA SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
def simulate_data(days=60, interval_minutes=30, seed=42):
    random.seed(seed); np.random.seed(seed)
    start = datetime(2024, 1, 1)
    spd   = (24 * 60) // interval_minutes
    total = days * spd
    ts, T, P, H, L, FT = [], [], [], [], [], []

    for i in range(total):
        t   = start + timedelta(minutes=i * interval_minutes)
        hr  = t.hour + t.minute / 60
        sea = (i / total) * 3.0

        temp = 24 + sea + 8*np.sin(np.pi*(hr-6)/12)*(hr>6) + np.random.normal(0,.5)
        temp += 0.6 if t.weekday() >= 5 else 0
        pres = 1013.25 + 5*np.sin(2*np.pi*i/(spd*7)) + np.random.normal(0,.3)
        hum  = np.clip(70 - 20*np.sin(np.pi*(hr-6)/12)*(hr>6)
                       + np.random.normal(0, 1.5), 20, 100)
        lux  = (max(0, 1000*np.exp(-0.5*((hr-13)/3)**2) + np.random.normal(0, 30))
                if 6 <= hr <= 20 else max(0, np.random.normal(5, 2)))

        ft = "normal"
        if random.random() < 0.03:
            ft = random.choice(["temp_spike","pressure_drop","humidity_surge",
                                "light_flicker","multi_fault"])
            if ft == "temp_spike":       temp += random.uniform(8, 15)
            elif ft == "pressure_drop":  pres -= random.uniform(10, 20)
            elif ft == "humidity_surge": hum   = min(100, hum + random.uniform(20, 30))
            elif ft == "light_flicker":  lux   = random.choice([0, 1500])
            elif ft == "multi_fault":
                temp += random.uniform(5, 10)
                hum   = min(100, hum + random.uniform(10, 20))

        ts.append(t); T.append(round(temp,2)); P.append(round(pres,2))
        H.append(round(hum,2)); L.append(round(lux,2)); FT.append(ft)

    return {
        "timestamps":  ts,
        "temperature": np.array(T),
        "pressure":    np.array(P),
        "humidity":    np.array(H),
        "light":       np.array(L),
        "fault_type":  FT,
        "is_anomaly":  np.array([f != "normal" for f in FT]),
        "spd":         spd,
        "days":        days,
        "interval_minutes": interval_minutes,
    }

# ─────────────────────────────────────────────────────────────────────────────
#  2. FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(data, window=24):
    n = len(data["temperature"]); rows, meta = [], []
    for i in range(window, n):
        seg = {k: data[k][i-window:i] for k in SENSOR_KEYS}
        row = []
        for k in SENSOR_KEYS:
            s = seg[k]
            row += [s.mean(), s.std(), s.min(), s.max(),
                    s.max()-s.min(),
                    float(np.percentile(s,75)-np.percentile(s,25)),
                    np.diff(s).mean(), np.abs(np.diff(s)).mean()]
        row += [float(np.corrcoef(seg["temperature"],seg["humidity"])[0,1]),
                float(np.corrcoef(seg["temperature"],seg["light"])[0,1])]
        rows.append(row)
        meta.append({
            "timestamp":   data["timestamps"][i],
            "fault_type":  data["fault_type"][i],
            "is_anomaly":  bool(data["is_anomaly"][i]),
            **{k: float(data[k][i]) for k in SENSOR_KEYS}
        })
    return np.array(rows, dtype=np.float32), meta

# ─────────────────────────────────────────────────────────────────────────────
#  3. GRU-RNN SLM
# ─────────────────────────────────────────────────────────────────────────────
class IoTSLM(nn.Module):
    def __init__(self, input_size=4, hidden=128, n_layers=2,
                 pred_len=6, dropout=0.2):
        super().__init__()
        self.pred_len    = pred_len
        self.input_size  = input_size
        self.hidden_size = hidden
        self.n_layers    = n_layers
        self.proj        = nn.Linear(input_size, hidden)
        self.gru         = nn.GRU(hidden, hidden, n_layers, batch_first=True,
                                  dropout=dropout if n_layers > 1 else 0.0)
        self.attn_q = nn.Linear(hidden, hidden)
        self.attn_k = nn.Linear(hidden, hidden)
        self.attn_v = nn.Linear(hidden, hidden)
        self.scale   = hidden ** 0.5
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, pred_len * input_size),
        )
        self.anom_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        h      = self.proj(x)
        enc, _ = self.gru(h)
        Q, K, V = self.attn_q(enc), self.attn_k(enc), self.attn_v(enc)
        w      = torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
        ctx    = torch.bmm(w, V).mean(dim=1)
        pred   = self.decoder(ctx).view(-1, self.pred_len, self.input_size)
        return pred, self.anom_head(ctx), w


class SensorDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def build_sequences(data, seq_len=24, pred_len=6):
    feats  = np.stack([data[k] for k in SENSOR_KEYS], axis=1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feats)
    X, y   = [], []
    for i in range(len(scaled) - seq_len - pred_len + 1):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y), scaler


def train_slm(data, seq_len=24, pred_len=6, hidden=128,
              epochs=60, lr=1e-3, batch=64, seed=42):
    torch.manual_seed(seed)
    X, y, scaler = build_sequences(data, seq_len, pred_len)
    split  = int(0.8 * len(X))
    tr_ld  = DataLoader(SensorDS(X[:split], y[:split]), batch_size=batch, shuffle=True)
    va_ld  = DataLoader(SensorDS(X[split:], y[split:]), batch_size=batch)

    model  = IoTSLM(input_size=4, hidden=hidden, pred_len=pred_len)
    opt    = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch    = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse    = nn.MSELoss()
    bce    = nn.BCELoss()
    hist   = {"train": [], "val": []}

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for ep in range(1, epochs+1):
        model.train(); tl = 0.0
        for xb, yb in tr_ld:
            opt.zero_grad()
            p, a, _ = model(xb)
            re = ((p - yb)**2).mean(dim=(1, 2))
            lbl = (re > re.mean() + re.std()).float().unsqueeze(1)
            loss = mse(p, yb) + 0.1 * bce(a, lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        sch.step()
        model.eval()
        with torch.no_grad():
            vl = sum(mse(model(xb)[0], yb).item()
                     for xb, yb in va_ld) / len(va_ld)
        hist["train"].append(tl / len(tr_ld))
        hist["val"].append(vl)
        if ep % 10 == 0:
            print(f"    Epoch {ep:3d}/{epochs}  "
                  f"train={tl/len(tr_ld):.5f}  val={vl:.5f}")

    return model, scaler, X, y, hist


def evaluate_slm(model, scaler, X, y):
    """Compute per-sensor MAE, RMSE, R² on the validation split."""
    model.eval()
    split   = int(0.8 * len(X))
    X_val   = X[split:]
    y_val   = y[split:]
    preds   = []
    with torch.no_grad():
        for i in range(0, len(X_val), 128):
            xb    = torch.tensor(X_val[i:i+128], dtype=torch.float32)
            p, _, _ = model(xb)
            preds.append(p.numpy())
    preds = np.concatenate(preds)   # (N, pred_len, 4)

    metrics = {}
    for si, sensor in enumerate(SENSOR_KEYS):
        true_s = y_val[:, 0, si]
        pred_s = preds[:, 0, si]
        mae    = float(mean_absolute_error(true_s, pred_s))
        rmse   = float(mean_squared_error(true_s, pred_s)**0.5)
        ss_res = ((true_s - pred_s)**2).sum()
        ss_tot = ((true_s - true_s.mean())**2).sum()
        r2     = float(1 - ss_res / (ss_tot + 1e-9))
        metrics[sensor] = {"mae": round(mae,4), "rmse": round(rmse,4), "r2": round(r2,4)}
        print(f"    {sensor:<12s} MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return metrics

# ─────────────────────────────────────────────────────────────────────────────
#  4. K-MEANS
# ─────────────────────────────────────────────────────────────────────────────
def fit_kmeans(X_feat, n_clusters=8, seed=42):
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X_feat)
    km  = KMeans(n_clusters=n_clusters, random_state=seed,
                 n_init=15, max_iter=500)
    lbl = km.fit_predict(Xs)
    return km, sc, lbl


def name_clusters(km, km_sc, X_feat, metas, n=8):
    lbl   = km.predict(km_sc.transform(X_feat))
    names = {}
    used  = set()
    for k in range(n):
        members = np.where(lbl == k)[0]
        if not len(members):
            names[k] = f"Cluster {k}"
            continue
        t_m  = np.mean([metas[i]["temperature"] for i in members])
        h_m  = np.mean([metas[i]["humidity"]    for i in members])
        hr_m = np.mean([metas[i]["timestamp"].hour for i in members])
        l_m  = np.mean([metas[i]["light"]       for i in members])
        faults = defaultdict(int)
        for i in members: faults[metas[i]["fault_type"]] += 1
        df = max(faults, key=faults.get)

        if df != "normal":
            nm = {"temp_spike":"Hot Afternoon","pressure_drop":"Unstable",
                  "humidity_surge":"Very Humid","light_flicker":"Light Fault",
                  "multi_fault":"Unstable"}.get(df,"Borderline")
        elif hr_m < 6 or hr_m >= 20:
            nm = "Cool Night" if t_m < 24 else "Warm Night"
        elif t_m >= 30: nm = "Hot Afternoon"
        elif l_m > 400: nm = "Sunny Day"
        elif h_m > 72:  nm = "Humid Day"
        else:           nm = "Warm Day" if t_m >= 25 else "Borderline"

        orig, suf = nm, 1
        while nm in used: nm = f"{orig} {suf}"; suf += 1
        used.add(nm)
        names[k] = nm
    return names

# ─────────────────────────────────────────────────────────────────────────────
#  5. DECISION TREE
# ─────────────────────────────────────────────────────────────────────────────
def get_anomaly_scores(model, X_seq, batch=128):
    model.eval(); scores = []
    with torch.no_grad():
        for i in range(0, len(X_seq), batch):
            xb = torch.tensor(X_seq[i:i+batch], dtype=torch.float32)
            _, a, _ = model(xb)
            scores.append(a.squeeze(1).numpy())
    return np.concatenate(scores)


def build_dt_labels(X_feat, metas, cluster_labels, slm_scores, seq_len):
    y = np.zeros(len(X_feat), dtype=int)
    for i in range(len(X_feat)):
        m  = metas[i]
        sc = slm_scores[i] if i < len(slm_scores) else 0.0
        ft = m["fault_type"]
        if ft == "multi_fault" or (sc > 0.70 and ft != "normal"): y[i] = 8
        elif ft == "pressure_drop" or m["pressure"] < 995:         y[i] = 7
        elif ft == "temp_spike"    or m["temperature"] > 38:       y[i] = 4
        elif ft == "humidity_surge"or m["humidity"] > 92:          y[i] = 5
        elif ft == "light_flicker" or m["light"] > 1200:           y[i] = 6
        elif sc > 0.35:                                            y[i] = 3
        elif m["timestamp"].hour < 6 or m["timestamp"].hour >= 20: y[i] = 2
        elif m["timestamp"].hour >= 6: y[i] = 1
    return y


def train_dt(X_feat, y_labels, seed=42):
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X_feat)
    cnt = Counter(y_labels)
    Xo, yo = list(Xs), list(y_labels)
    np.random.seed(seed)
    for cls, c in cnt.items():
        if c < 80:
            idx = np.where(y_labels == cls)[0]
            ext = np.random.choice(idx, 80 - c, replace=True)
            noise = np.random.normal(0, 0.05, (len(ext), Xs.shape[1]))
            Xo.extend(Xs[ext] + noise)
            yo.extend([cls] * len(ext))
    dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=3,
                                class_weight="balanced", random_state=seed)
    dt.fit(np.array(Xo), np.array(yo))
    y_pred = dt.predict(Xs)
    acc    = (y_pred == y_labels).mean()
    print(f"    DT train accuracy: {acc*100:.1f}%  "
          f"depth={dt.get_depth()}  leaves={dt.get_n_leaves()}")
    return dt, sc

# ─────────────────────────────────────────────────────────────────────────────
#  6. SAVE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    kb = os.path.getsize(path) // 1024
    print(f"    ✓ {os.path.basename(path):<30s}  {kb} KB")

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    kb = os.path.getsize(path) // 1024
    print(f"    ✓ {os.path.basename(path):<30s}  {kb} KB")

def save_torch(model, path):
    torch.save(model.state_dict(), path)
    kb = os.path.getsize(path) // 1024
    print(f"    ✓ {os.path.basename(path):<30s}  {kb} KB")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("\n" + "═"*60)
    print("  IoT-SLM  ·  Training & Saving Pipeline")
    print("═"*60)
    print(f"  Output dir : {args.out}")
    print(f"  Days       : {args.days}")
    print(f"  Interval   : {args.interval} min")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Seq len    : {args.seq_len}")
    print(f"  Pred len   : {args.pred_len}")
    print(f"  Hidden     : {args.hidden}")
    print(f"  Clusters   : {args.clusters}")
    print(f"  Seed       : {args.seed}")
    print("─"*60)

    # ── Step 1: Simulate data ─────────────────────────────────────────────
    print("\n[1/7] Simulating IoT sensor data ...")
    data = simulate_data(args.days, args.interval, args.seed)
    print(f"  {len(data['timestamps']):,} readings  |  "
          f"anomalies: {data['is_anomaly'].sum()} "
          f"({data['is_anomaly'].mean()*100:.1f}%)")

    # ── Step 2: Extract features ──────────────────────────────────────────
    print("\n[2/7] Extracting sliding-window features ...")
    X_feat, metas = extract_features(data, window=args.seq_len)
    print(f"  Feature matrix: {X_feat.shape}  (34 features per window)")

    # ── Step 3: Train GRU-SLM ────────────────────────────────────────────
    print("\n[3/7] Training GRU-RNN SLM ...")
    model, slm_scaler, X_seq, y_seq, hist = train_slm(
        data, seq_len=args.seq_len, pred_len=args.pred_len,
        hidden=args.hidden, epochs=args.epochs, seed=args.seed)

    # ── Step 4: Evaluate ─────────────────────────────────────────────────
    print("\n[4/7] Evaluating forecasting performance ...")
    eval_metrics = evaluate_slm(model, slm_scaler, X_seq, y_seq)

    # ── Step 5: K-Means ───────────────────────────────────────────────────
    print("\n[5/7] Fitting K-Means clustering ...")
    km, km_sc, cluster_labels = fit_kmeans(X_feat, args.clusters, args.seed)
    cluster_names = name_clusters(km, km_sc, X_feat, metas, args.clusters)
    for k, v in sorted(cluster_names.items()):
        n = int(np.sum(cluster_labels == k))
        print(f"  Cluster {k}: {v:<22s} ({n} windows)")

    # ── Step 6: Decision Tree ─────────────────────────────────────────────
    print("\n[6/7] Training Decision Tree ...")
    slm_scores = get_anomaly_scores(model, X_seq)
    ml = min(len(X_feat), len(slm_scores))
    X_feat_aligned     = X_feat[:ml]
    metas_aligned      = metas[:ml]
    cluster_labels_al  = cluster_labels[:ml]
    y_labels = build_dt_labels(
        X_feat_aligned, metas_aligned, cluster_labels_al,
        slm_scores, args.seq_len)
    dt, dt_sc = train_dt(X_feat_aligned, y_labels, args.seed)

    # ── Step 7: Save everything ───────────────────────────────────────────
    print("\n[7/7] Saving all artifacts ...")

    # Config
    config = {
        "days":           args.days,
        "interval_min":   args.interval,
        "seq_len":        args.seq_len,
        "pred_len":       args.pred_len,
        "hidden":         args.hidden,
        "n_layers":       2,
        "dropout":        0.2,
        "n_clusters":     args.clusters,
        "n_features":     int(X_feat.shape[1]),
        "n_windows":      int(X_feat_aligned.shape[0]),
        "n_sensors":      4,
        "sensor_keys":    SENSOR_KEYS,
        "sensor_units":   SENSOR_UNITS,
        "notif_classes":  NOTIF_CLASSES,
        "cluster_names":  cluster_names,
        "ref_date":       str(data["timestamps"][-1].date()),
        "seed":           args.seed,
        "trained_at":     datetime.now().isoformat(timespec="seconds"),
    }
    save_json(config,                    os.path.join(args.out, "config.json"))

    # Data (timestamps serialised as strings for JSON-safety)
    data_save = {k: v for k, v in data.items() if k != "timestamps"}
    data_save["timestamps_str"] = [t.isoformat() for t in data["timestamps"]]
    save_pickle(data_save,               os.path.join(args.out, "data.pkl"))

    # Features and metas (timestamps → strings for pickle safety)
    for m in metas_aligned:
        m["timestamp_str"] = m["timestamp"].isoformat()
        m["timestamp"]     = m["timestamp"]   # keep datetime object too
    save_pickle((X_feat_aligned, metas_aligned),
                                         os.path.join(args.out, "features.pkl"))

    # SLM
    save_torch(model,                    os.path.join(args.out, "slm_weights.pt"))
    save_pickle(slm_scaler,              os.path.join(args.out, "slm_scaler.pkl"))
    save_pickle(hist,                    os.path.join(args.out, "slm_history.pkl"))

    # K-Means
    save_pickle(km,                      os.path.join(args.out, "kmeans.pkl"))
    save_pickle(km_sc,                   os.path.join(args.out, "kmeans_scaler.pkl"))
    save_pickle(cluster_labels_al,       os.path.join(args.out, "cluster_labels.pkl"))
    save_json({str(k): v for k,v in cluster_names.items()},
                                         os.path.join(args.out, "cluster_names.json"))

    # Decision Tree
    save_pickle(dt,                      os.path.join(args.out, "dt.pkl"))
    save_pickle(dt_sc,                   os.path.join(args.out, "dt_scaler.pkl"))
    save_pickle(y_labels,                os.path.join(args.out, "dt_labels.pkl"))

    # Eval metrics
    save_json(eval_metrics,              os.path.join(args.out, "eval_metrics.json"))

    # Summary
    total_size = sum(
        os.path.getsize(os.path.join(args.out, f))
        for f in os.listdir(args.out)) // 1024
    print(f"\n  Total saved: {total_size} KB in {args.out}/")
    print("\n" + "═"*60)
    print("  Training complete. Run the Streamlit app with:")
    print(f"  streamlit run iot_app.py -- --model {args.out}")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
