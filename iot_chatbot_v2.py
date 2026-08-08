"""
IoT SLM Chatbot  —  Offline Text-Based Assistant  (Enhanced v2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every query produces:
  1. Narasi kondisi sensor berbasis cluster + time-series trend
  2. Tabel ringkasan statistik di terminal (ASCII table)
  3. LINE CHART  – time-series suhu (& opsional sensor lain)
                  dua garis jika ada periode perbandingan
  4. SCATTER PLOT – Light vs Temperature per window
                   dua warna jika ada periode perbandingan
  5. File PNG disimpan ke  ./iot_charts/  lalu path-nya dicetak

Query yang dipahami:
  today / hari ini         yesterday / kemarin
  this week / minggu ini   last week / minggu lalu
  this month / bulan ini   last month / bulan lalu
  2 days ago / 2 hari lalu   3 days ago / 3 hari lalu
  Kombinasi (compare): "compare today and yesterday"

Run:  python3 iot_chatbot_v2.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import sys, re, textwrap, random, warnings, os
from datetime import datetime, timedelta
from collections import defaultdict

# ── numerical / visual ────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── ML ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42); torch.manual_seed(42)

CHART_DIR = "./iot_charts"
os.makedirs(CHART_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SENSOR_KEYS   = ["temperature", "pressure", "humidity", "light"]
SENSOR_UNITS  = {"temperature":"°C","pressure":"hPa","humidity":"% RH","light":"lux"}
SENSOR_SHORT  = {"temperature":"Temp","pressure":"Press","humidity":"Humid","light":"Light"}

NOTIF_CLASSES = [
    "NO_ACTION","INFO_DAY","INFO_NIGHT","WATCH_BORDERLINE",
    "WARN_TEMP","WARN_HUMIDITY","WARN_LIGHT","CRIT_PRESSURE","CRIT_MULTI",
]

# Visual palette (light-print-friendly)
PAL = {
    "A":  "#1A5276",   # deep blue  – period A line/dots
    "B":  "#C0392B",   # deep red   – period B line/dots
    "A2": "#AED6F1",   # pale blue  – period A fill
    "B2": "#F1948A",   # pale red   – period B fill
    "grid":"#D5D8DC",
    "bg":  "#FAFBFF",
    "panel":"#EEF3FA",
    "navy": "#0E2A5C",
    "teal": "#0D7A6E",
}

CLUSTER_COND = {
    "Cool Night":    "comfortable nighttime conditions",
    "Warm Day":      "typical warm daytime conditions",
    "Borderline":    "borderline conditions requiring attention",
    "Cold Night":    "cool nighttime conditions",
    "Hot Afternoon": "hot afternoon conditions with elevated temperature",
    "Warm Night":    "mild nighttime conditions",
    "Unstable":      "unstable conditions with high variability",
    "Sunny Day":     "sunny and hot daytime conditions",
    "Humid Day":     "humid daytime conditions",
    "Very Humid":    "very high humidity conditions",
    "Light Fault":   "light sensor fault conditions",
}

# ─────────────────────────────────────────────────────────────────────────────
#  1. DATA SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
def simulate_data(days=60, interval_minutes=30):
    start = datetime(2024, 1, 1)
    spd   = (24 * 60) // interval_minutes
    total = days * spd
    ts, T, P, H, L, FT = [], [], [], [], [], []
    for i in range(total):
        t   = start + timedelta(minutes=i * interval_minutes)
        hr  = t.hour + t.minute / 60
        sea = (i / total) * 3.0
        temp = 24 + sea + 8*np.sin(np.pi*(hr-6)/12)*(hr>6) + np.random.normal(0,.5)
        temp += 0.6 if t.weekday()>=5 else 0
        pres = 1013.25 + 5*np.sin(2*np.pi*i/(spd*7)) + np.random.normal(0,.3)
        hum  = np.clip(70-20*np.sin(np.pi*(hr-6)/12)*(hr>6)+np.random.normal(0,1.5),20,100)
        lux  = (max(0,1000*np.exp(-0.5*((hr-13)/3)**2)+np.random.normal(0,30))
                if 6<=hr<=20 else max(0,np.random.normal(5,2)))
        ft = "normal"
        if random.random()<0.03:
            ft = random.choice(["temp_spike","pressure_drop","humidity_surge",
                                "light_flicker","multi_fault"])
            if ft=="temp_spike":      temp += random.uniform(8,15)
            elif ft=="pressure_drop": pres -= random.uniform(10,20)
            elif ft=="humidity_surge":hum   = min(100,hum+random.uniform(20,30))
            elif ft=="light_flicker": lux   = random.choice([0,1500])
            elif ft=="multi_fault":   temp += random.uniform(5,10); hum=min(100,hum+random.uniform(10,20))
        ts.append(t); T.append(round(temp,2)); P.append(round(pres,2))
        H.append(round(hum,2)); L.append(round(lux,2)); FT.append(ft)
    return {"timestamps":ts,"temperature":np.array(T),"pressure":np.array(P),
            "humidity":np.array(H),"light":np.array(L),"fault_type":FT,
            "is_anomaly":np.array([f!="normal" for f in FT]),"spd":spd}

# ─────────────────────────────────────────────────────────────────────────────
#  2. FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(data, window=24):
    n = len(data["temperature"]); rows, meta = [], []
    for i in range(window,n):
        seg = {k:data[k][i-window:i] for k in SENSOR_KEYS}
        row = []
        for k in SENSOR_KEYS:
            s=seg[k]
            row+=[s.mean(),s.std(),s.min(),s.max(),s.max()-s.min(),
                  np.percentile(s,75)-np.percentile(s,25),
                  np.diff(s).mean(),np.abs(np.diff(s)).mean()]
        row+=[np.corrcoef(seg["temperature"],seg["humidity"])[0,1],
              np.corrcoef(seg["temperature"],seg["light"])[0,1]]
        rows.append(row)
        meta.append({"timestamp":data["timestamps"][i],"fault_type":data["fault_type"][i],
                     "is_anomaly":data["is_anomaly"][i],
                     **{k:float(data[k][i]) for k in SENSOR_KEYS}})
    return np.array(rows,dtype=np.float32), meta

# ─────────────────────────────────────────────────────────────────────────────
#  3. GRU-RNN SLM
# ─────────────────────────────────────────────────────────────────────────────
class IoTSLM(nn.Module):
    def __init__(self,input_size=4,hidden=128,n_layers=2,pred_len=6,dropout=0.2):
        super().__init__()
        self.pred_len=pred_len; self.input_size=input_size
        self.proj=nn.Linear(input_size,hidden)
        self.gru=nn.GRU(hidden,hidden,n_layers,batch_first=True,
                        dropout=dropout if n_layers>1 else 0.)
        self.attn_q=nn.Linear(hidden,hidden)
        self.attn_k=nn.Linear(hidden,hidden)
        self.attn_v=nn.Linear(hidden,hidden)
        self.scale=hidden**0.5
        self.decoder=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden//2),
                                   nn.GELU(),nn.Dropout(dropout),
                                   nn.Linear(hidden//2,pred_len*input_size))
        self.anom_head=nn.Sequential(nn.Linear(hidden,32),nn.ReLU(),
                                     nn.Linear(32,1),nn.Sigmoid())
    def forward(self,x):
        h=self.proj(x); enc,_=self.gru(h)
        Q,K,V=self.attn_q(enc),self.attn_k(enc),self.attn_v(enc)
        w=torch.softmax(torch.bmm(Q,K.transpose(1,2))/self.scale,dim=-1)
        ctx=torch.bmm(w,V).mean(dim=1)
        pred=self.decoder(ctx).view(-1,self.pred_len,self.input_size)
        return pred,self.anom_head(ctx),w

class SensorDS(Dataset):
    def __init__(self,X,y): self.X=torch.tensor(X,dtype=torch.float32); self.y=torch.tensor(y,dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.y[i]

def build_sequences(data,seq_len=24,pred_len=6):
    feats=np.stack([data[k] for k in SENSOR_KEYS],axis=1)
    scaler=MinMaxScaler(); scaled=scaler.fit_transform(feats)
    X,y=[],[]
    for i in range(len(scaled)-seq_len-pred_len+1):
        X.append(scaled[i:i+seq_len]); y.append(scaled[i+seq_len:i+seq_len+pred_len])
    return np.array(X),np.array(y),scaler

def train_slm(data,seq_len=24,pred_len=6,epochs=60,lr=1e-3,batch=64):
    X,y,scaler=build_sequences(data,seq_len,pred_len)
    split=int(0.8*len(X))
    tr=DataLoader(SensorDS(X[:split],y[:split]),batch_size=batch,shuffle=True)
    va=DataLoader(SensorDS(X[split:],y[split:]),batch_size=batch)
    model=IoTSLM(pred_len=pred_len)
    opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    mse=nn.MSELoss(); bce=nn.BCELoss()
    for ep in range(1,epochs+1):
        model.train()
        for xb,yb in tr:
            opt.zero_grad(); p,a,_=model(xb)
            re=((p-yb)**2).mean(dim=(1,2))
            lbl=(re>re.mean()+re.std()).float().unsqueeze(1)
            loss=mse(p,yb)+0.1*bce(a,lbl)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        sch.step()
        if ep%10==0:
            model.eval()
            with torch.no_grad():
                vl=sum(mse(model(xb)[0],yb).item() for xb,yb in va)/len(va)
            print(f"    Epoch {ep:3d}/{epochs}  val_mse={vl:.5f}")
    return model,scaler,X,y

def get_anomaly_scores(model,X_seq,batch=128):
    model.eval(); scores=[]
    with torch.no_grad():
        for i in range(0,len(X_seq),batch):
            xb=torch.tensor(X_seq[i:i+batch],dtype=torch.float32)
            _,a,_=model(xb); scores.append(a.squeeze(1).numpy())
    return np.concatenate(scores)

def forecast_window(model,scaler,window_arr):
    scaled=scaler.transform(window_arr)
    xt=torch.tensor(scaled[None],dtype=torch.float32)
    model.eval()
    with torch.no_grad(): pred,anom,_=model(xt)
    return scaler.inverse_transform(pred.squeeze(0).numpy()), float(anom.squeeze())

# ─────────────────────────────────────────────────────────────────────────────
#  4. K-MEANS + DT
# ─────────────────────────────────────────────────────────────────────────────
def fit_kmeans(X_feat,n_clusters=8):
    sc=StandardScaler(); Xs=sc.fit_transform(X_feat)
    km=KMeans(n_clusters=n_clusters,random_state=42,n_init=15,max_iter=500)
    return km,sc,km.fit_predict(Xs)

def build_dt_labels(X_feat,metas,cluster_labels,slm_scores,seq_len):
    y=np.zeros(len(X_feat),dtype=int)
    for i in range(len(X_feat)):
        m=metas[i]; sc=slm_scores[i] if i<len(slm_scores) else 0.; ft=m["fault_type"]
        if ft=="multi_fault" or (sc>0.70 and ft!="normal"): y[i]=8
        elif ft=="pressure_drop" or m["pressure"]<995:       y[i]=7
        elif ft=="temp_spike"    or m["temperature"]>38:     y[i]=4
        elif ft=="humidity_surge"or m["humidity"]>92:        y[i]=5
        elif ft=="light_flicker" or m["light"]>1200:         y[i]=6
        elif sc>0.35:                                        y[i]=3
        elif m["timestamp"].hour<6 or m["timestamp"].hour>=20: y[i]=2
        elif m["timestamp"].hour>=6: y[i]=1
    return y

def train_dt(X_feat,y_labels):
    sc=StandardScaler(); Xs=sc.fit_transform(X_feat)
    from collections import Counter
    counts=Counter(y_labels); Xo,yo=list(Xs),list(y_labels)
    for cls,cnt in counts.items():
        if cnt<80:
            idx=np.where(y_labels==cls)[0]
            ext=np.random.choice(idx,80-cnt,replace=True)
            Xo.extend(Xs[ext]+np.random.normal(0,.05,(len(ext),Xs.shape[1]))); yo.extend([cls]*len(ext))
    dt=DecisionTreeClassifier(max_depth=12,min_samples_leaf=3,class_weight="balanced",random_state=42)
    dt.fit(np.array(Xo),np.array(yo)); return dt,sc

# ─────────────────────────────────────────────────────────────────────────────
#  5. ANALYSIS ENGINE  (extended with raw series access)
# ─────────────────────────────────────────────────────────────────────────────
class AnalysisEngine:
    def __init__(self,data,X_feat,metas,cluster_labels,cluster_names,
                 slm_model,slm_scaler,X_seq,km,km_sc,dt,dt_sc,seq_len=24):
        self.data=data; self.X_feat=X_feat; self.metas=metas
        self.cluster_labels=cluster_labels; self.cluster_names=cluster_names
        self.slm=slm_model; self.slm_sc=slm_scaler; self.X_seq=X_seq
        self.km=km; self.km_sc=km_sc; self.dt=dt; self.dt_sc=dt_sc; self.seq_len=seq_len
        self._by_date=defaultdict(list)
        for i,m in enumerate(metas): self._by_date[m["timestamp"].date()].append(i)

    def _indices(self,s,e):
        idx=[]; d=s
        while d<=e:
            idx.extend(self._by_date.get(d,[])); d+=timedelta(days=1)
        return sorted(idx)

    def _stats(self,indices):
        if not indices: return None
        stats={}
        for k in SENSOR_KEYS:
            v=np.array([self.metas[i][k] for i in indices])
            t=np.arange(len(v))
            slope=np.polyfit(t,v,1)[0] if len(v)>1 else 0.
            stats[k]={"mean":float(v.mean()),"min":float(v.min()),"max":float(v.max()),
                      "last":float(v[-1]),"first":float(v[0]),"trend":float(slope),"std":float(v.std())}
        return stats

    def _dom_cluster(self,indices):
        if not indices: return None,None
        lbs=[self.cluster_labels[i] for i in indices if i<len(self.cluster_labels)]
        if not lbs: return None,None
        c=defaultdict(int)
        for l in lbs: c[l]+=1
        did=max(c,key=c.get)
        return did,self.cluster_names.get(did,f"Cluster {did}")

    def _dom_notif(self,indices):
        if not indices: return "NO_ACTION",0
        fs=self.dt_sc.transform(self.X_feat[indices])
        preds=self.dt.predict(fs)
        c=defaultdict(int)
        for p in preds: c[p]+=1
        for ci in [8,7,4,5,6,3,1,2,0]:
            if c[ci]>0: return NOTIF_CLASSES[ci],c[ci]
        return "NO_ACTION",0

    def _forecast_end(self,indices):
        if not indices: return None,0.
        data_idx=max(indices)+self.seq_len
        if data_idx<self.seq_len or data_idx>len(self.data["temperature"]): return None,0.
        win=np.stack([self.data[k][data_idx-self.seq_len:data_idx] for k in SENSOR_KEYS],axis=1)
        if win.shape[0]<self.seq_len: return None,0.
        return forecast_window(self.slm,self.slm_sc,win)

    def raw_series(self,indices,sensor="temperature"):
        """Return (timestamps_list, values_array) for the given indices and sensor."""
        ts=[self.metas[i]["timestamp"] for i in indices]
        vs=np.array([self.metas[i][sensor] for i in indices])
        return ts,vs

    def cluster_series(self,indices):
        """Return per-window cluster name list aligned to indices."""
        return [self.cluster_names.get(
                    self.cluster_labels[i] if i<len(self.cluster_labels) else 0,
                    "Unknown") for i in indices]

    def query(self,start_date,end_date,period_label="period"):
        idx=self._indices(start_date,end_date)
        if not idx: return {"error":"No data found for the requested period."}
        stats=self._stats(idx)
        did,dname=self._dom_cluster(idx)
        notif,_=self._dom_notif(idx)
        n_anom=sum(1 for i in idx if self.metas[i]["is_anomaly"])
        fore,ascore=self._forecast_end(idx)
        return {"period_label":period_label,"start":start_date,"end":end_date,
                "indices":idx,"n_windows":len(idx),
                "sensor_stats":stats,"cluster_id":did,"cluster_name":dname,
                "notification":notif,"n_anomalies":n_anom,
                "anomaly_rate":n_anom/max(len(idx),1),
                "forecast":fore,"anomaly_score":ascore}

# ─────────────────────────────────────────────────────────────────────────────
#  6. ASCII TABLE
# ─────────────────────────────────────────────────────────────────────────────
def ascii_table(result):
    """Print a compact sensor statistics table in the terminal."""
    ss = result["sensor_stats"]
    if not ss:
        return "  (no data)"
    lbl = result["period_label"].upper()
    cname = result["cluster_name"] or "—"
    notif = result["notification"]

    COL = 14
    SEP = "─" * (7 + 5 * COL)
    header = f"{'Sensor':<14}{'Mean':>{COL}}{'Std':>{COL}}{'Min':>{COL}}{'Max':>{COL}}{'Trend':>{COL}}"

    lines = [
        f"\n  ┌{'─'*(6+5*COL)}┐",
        f"  │  SENSOR SUMMARY — {lbl:<{4+5*COL-22}}│",
        f"  │  Cluster: {cname:<{4+5*COL-12}}│",
        f"  │  Notification: {notif:<{4+5*COL-18}}│",
        f"  ├{'─'*(6+5*COL)}┤",
        f"  │  {header}  │",
        f"  ├{'─'*(6+5*COL)}┤",
    ]

    trend_arrow = {True:"↑ rising", False:"↓ falling"}
    for k in SENSOR_KEYS:
        s = ss[k]; u = SENSOR_UNITS[k]
        slope = s["trend"]
        if abs(slope) < 0.001: tarrow = "→ stable"
        elif slope > 0: tarrow = "↑ rising"
        else: tarrow = "↓ falling"
        row = (f"{'  │  '}{k.capitalize():<14}"
               f"{s['mean']:>{COL}.2f}"
               f"{s['std']:>{COL}.2f}"
               f"{s['min']:>{COL}.2f}"
               f"{s['max']:>{COL}.2f}"
               f"{tarrow:>{COL}}"
               f"  │")
        lines.append(row)

    lines += [
        f"  ├{'─'*(6+5*COL)}┤",
        f"  │  Anomalies detected : {result['n_anomalies']} / {result['n_windows']} windows "
        f"({result['anomaly_rate']*100:.1f}%){' '*(3+5*COL-55)}│",
        f"  └{'─'*(6+5*COL)}┘",
    ]
    return "\n".join(lines)

def ascii_hourly_table(engine, result, sensor="temperature"):
    """Print hourly averages for the period."""
    idx = result["indices"]
    if not idx:
        return ""
    ts, vs = engine.raw_series(idx, sensor)
    by_hour = defaultdict(list)
    for t, v in zip(ts, vs):
        by_hour[t.hour].append(v)
    u = SENSOR_UNITS[sensor]
    lbl = result["period_label"].upper()
    lines = [f"\n  Hourly Average — {sensor.capitalize()} — {lbl}",
             f"  {'Hour':<8}{'Avg':>8}{'Min':>8}{'Max':>8}{'Count':>8}"]
    lines.append("  " + "─"*40)
    for h in sorted(by_hour):
        arr = np.array(by_hour[h])
        lines.append(f"  {h:02d}:00  {arr.mean():>7.2f}  {arr.min():>7.2f}  {arr.max():>7.2f}  {len(arr):>7}")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
#  7. VISUALISATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": PAL["bg"],
    "axes.facecolor":   PAL["panel"],
    "axes.edgecolor":   PAL["grid"],
    "grid.color":       PAL["grid"],
    "text.color":       PAL["navy"],
    "axes.labelcolor":  PAL["navy"],
    "xtick.color":      "#5D6D7E",
    "ytick.color":      "#5D6D7E",
    "axes.titlecolor":  PAL["navy"],
    "font.family":      "monospace",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


def _subsample(ts_list, vs, max_pts=300):
    """Down-sample for legible plots."""
    n = len(vs)
    if n <= max_pts:
        return ts_list, vs
    step = n // max_pts
    idx  = list(range(0, n, step))
    return [ts_list[i] for i in idx], vs[idx]


def make_charts(engine, results, query_id="query"):
    """
    results: list of 1 or 2 dicts from AnalysisEngine.query()
    Saves one PNG with:
      Top row : Line chart — Temperature time-series (+ forecast tail)
              + Line chart — Humidity time-series
      Mid row : Scatter — Light vs Temperature (colour = cluster)
              + Bar chart — Hourly mean temperature per period
      Bot row : Line chart — Pressure time-series
              + Line chart — Anomaly score over time (if available)
    Returns saved file path.
    """
    n_periods = len(results)
    fig = plt.figure(figsize=(18, 14), dpi=110)
    fig.patch.set_facecolor(PAL["bg"])

    period_labels = [r["period_label"] for r in results]
    colors_line   = [PAL["A"], PAL["B"]]
    colors_fill   = [PAL["A2"], PAL["B2"]]
    colors_scat   = [PAL["A"], PAL["B"]]

    main_title = " vs ".join(f'"{l}"' for l in period_labels)
    fig.suptitle(f"IoT-SLM  ·  Sensor Analysis  ·  {main_title}",
                 fontsize=14, fontweight="bold", color=PAL["navy"], y=0.99)

    outer = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.32,
                              left=0.07, right=0.97, top=0.95, bottom=0.06)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _get_series(r, sensor):
        idx = r.get("indices", [])
        if not idx:
            return [], np.array([])
        ts, vs = engine.raw_series(idx, sensor)
        return _subsample(ts, vs)

    def _step_labels(ts_list, max_ticks=8):
        """Generate readable x-axis tick positions/labels."""
        n = len(ts_list)
        if n == 0:
            return [], []
        step = max(1, n // max_ticks)
        positions = list(range(0, n, step))
        labels    = [ts_list[p].strftime("%m-%d\n%H:%M") for p in positions]
        return positions, labels

    # ── ROW 0, COL 0 : Temperature line chart ────────────────────────────────
    ax1 = fig.add_subplot(outer[0, 0])
    all_ts_flat, all_vs_flat = [], []
    handles = []
    for ri, r in enumerate(results):
        ts_s, vs_s = _get_series(r, "temperature")
        if len(vs_s) == 0:
            continue
        x = np.arange(len(vs_s))
        ax1.plot(x, vs_s, color=colors_line[ri], lw=1.6, alpha=0.9,
                 label=r["period_label"])
        ax1.fill_between(x, vs_s, vs_s.min(), color=colors_fill[ri], alpha=0.18)
        # Forecast tail (dashed)
        if r.get("forecast") is not None:
            fore = r["forecast"][:, 0]
            x_f  = np.arange(len(vs_s), len(vs_s) + len(fore))
            ax1.plot(x_f, fore, color=colors_line[ri], lw=1.4, ls="--", alpha=0.7)
            ax1.axvline(len(vs_s), color=colors_line[ri], lw=0.8, ls=":", alpha=0.5)
        all_ts_flat += ts_s; all_vs_flat.append(vs_s)
        handles.append(Line2D([0],[0], color=colors_line[ri], lw=2, label=r["period_label"]))

    # Add forecast legend entry
    handles.append(Line2D([0],[0], color="#888888", lw=1.4, ls="--", label="Forecast (3h)"))
    ax1.set_title("Temperature (°C) — Time Series", fontsize=10, pad=6)
    ax1.set_ylabel("°C"); ax1.grid(True, alpha=0.4)
    ax1.legend(handles=handles, fontsize=8, loc="upper left")
    if all_ts_flat:
        pos, lbl = _step_labels(all_ts_flat[:len(results[0].get("indices",[]) or [1])])
        # Use numeric x-axis; just show period label on bottom
        ax1.set_xlabel("Time →")

    # ── ROW 0, COL 1 : Humidity line chart ───────────────────────────────────
    ax2 = fig.add_subplot(outer[0, 1])
    for ri, r in enumerate(results):
        ts_s, vs_s = _get_series(r, "humidity")
        if len(vs_s) == 0: continue
        x = np.arange(len(vs_s))
        ax2.plot(x, vs_s, color=colors_line[ri], lw=1.6, alpha=0.9,
                 label=r["period_label"])
        ax2.fill_between(x, vs_s, vs_s.min(), color=colors_fill[ri], alpha=0.18)
        if r.get("forecast") is not None:
            fore=r["forecast"][:,2]; x_f=np.arange(len(vs_s),len(vs_s)+len(fore))
            ax2.plot(x_f, fore, color=colors_line[ri], lw=1.4, ls="--", alpha=0.7)
    ax2.set_title("Humidity (% RH) — Time Series", fontsize=10, pad=6)
    ax2.set_ylabel("% RH"); ax2.grid(True, alpha=0.4)
    ax2.legend(fontsize=8)
    ax2.set_xlabel("Time →")

    # ── ROW 1, COL 0 : Scatter — Light vs Temperature ─────────────────────────
    ax3 = fig.add_subplot(outer[1, 0])
    for ri, r in enumerate(results):
        idx = r.get("indices", [])
        if not idx: continue
        T_v = np.array([engine.metas[i]["temperature"] for i in idx])
        L_v = np.array([engine.metas[i]["light"]       for i in idx])
        # subsample
        step = max(1, len(T_v)//300)
        T_v, L_v = T_v[::step], L_v[::step]
        ax3.scatter(T_v, L_v, color=colors_scat[ri], alpha=0.35, s=14,
                    edgecolors="none", label=r["period_label"])
        # Trend line
        if len(T_v) > 2:
            m, b = np.polyfit(T_v, L_v, 1)
            xr = np.array([T_v.min(), T_v.max()])
            ax3.plot(xr, m*xr+b, color=colors_line[ri], lw=1.8, ls="--", alpha=0.7)

    ax3.set_title("Scatter: Temperature vs Light (with trend)", fontsize=10, pad=6)
    ax3.set_xlabel("Temperature (°C)"); ax3.set_ylabel("Light (lux)")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.4)

    # ── ROW 1, COL 1 : Scatter — Temperature vs Humidity coloured by cluster ──
    ax4 = fig.add_subplot(outer[1, 1])
    for ri, r in enumerate(results):
        idx = r.get("indices", [])
        if not idx: continue
        T_v  = np.array([engine.metas[i]["temperature"] for i in idx])
        H_v  = np.array([engine.metas[i]["humidity"]    for i in idx])
        C_lbl= engine.cluster_series(idx)
        unique_clusters = sorted(set(C_lbl))
        step = max(1, len(T_v)//200)
        T_s, H_s, C_s = T_v[::step], H_v[::step], C_lbl[::step]
        # Use a small colour map for clusters; differentiate by marker
        markers = ["o","s"]
        cmap    = plt.cm.get_cmap("tab10", max(len(unique_clusters), 2))
        for ci, cname in enumerate(unique_clusters):
            mask = np.array([c==cname for c in C_s])
            if not mask.any(): continue
            ax4.scatter(T_s[mask], H_s[mask], color=cmap(ci), alpha=0.5,
                        s=16, marker=markers[ri], edgecolors="none",
                        label=f"{r['period_label']} · {cname}" if ri==0 else f"_{cname}")
    ax4.set_title("Scatter: Temperature vs Humidity (by cluster)", fontsize=10, pad=6)
    ax4.set_xlabel("Temperature (°C)"); ax4.set_ylabel("Humidity (% RH)")
    # Keep legend manageable
    handles_4, labels_4 = ax4.get_legend_handles_labels()
    visible = [(h,l) for h,l in zip(handles_4,labels_4) if not l.startswith("_")][:10]
    if visible:
        ax4.legend(*zip(*visible), fontsize=7, ncol=2, loc="upper right")
    ax4.grid(True, alpha=0.4)

    # ── ROW 2, COL 0 : Pressure time series ──────────────────────────────────
    ax5 = fig.add_subplot(outer[2, 0])
    for ri, r in enumerate(results):
        ts_s, vs_s = _get_series(r, "pressure")
        if len(vs_s)==0: continue
        x=np.arange(len(vs_s))
        ax5.plot(x, vs_s, color=colors_line[ri], lw=1.5, alpha=0.9,
                 label=r["period_label"])
        if r.get("forecast") is not None:
            fore=r["forecast"][:,1]; x_f=np.arange(len(vs_s),len(vs_s)+len(fore))
            ax5.plot(x_f, fore, color=colors_line[ri], lw=1.3, ls="--", alpha=0.7)
    ax5.axhline(1013.25, color="#888888", lw=0.8, ls=":", alpha=0.7, label="ISA std (1013 hPa)")
    ax5.set_title("Pressure (hPa) — Time Series", fontsize=10, pad=6)
    ax5.set_ylabel("hPa"); ax5.grid(True, alpha=0.4)
    ax5.legend(fontsize=8); ax5.set_xlabel("Time →")

    # ── ROW 2, COL 1 : Hourly temperature bar chart ───────────────────────────
    ax6 = fig.add_subplot(outer[2, 1])
    bar_w  = 0.8 / max(n_periods, 1)
    offsets= np.linspace(-0.4+bar_w/2, 0.4-bar_w/2, n_periods)
    hours  = list(range(24))
    for ri, r in enumerate(results):
        idx = r.get("indices", [])
        if not idx: continue
        by_hour = defaultdict(list)
        for i in idx:
            by_hour[engine.metas[i]["timestamp"].hour].append(engine.metas[i]["temperature"])
        means = [np.mean(by_hour[h]) if by_hour[h] else float("nan") for h in hours]
        ax6.bar([h + offsets[ri] for h in hours], means, bar_w,
                color=colors_line[ri], alpha=0.8, label=r["period_label"])
    ax6.set_title("Hourly Mean Temperature (°C)", fontsize=10, pad=6)
    ax6.set_xlabel("Hour of day"); ax6.set_ylabel("°C")
    ax6.set_xticks(range(0, 24, 2))
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.4, axis="y")

    # ── Save ─────────────────────────────────────────────────────────────────
    fname = f"{query_id}.png"
    fpath = os.path.join(CHART_DIR, fname)
    plt.savefig(fpath, dpi=110, bbox_inches="tight", facecolor=PAL["bg"])
    plt.close(fig)
    return fpath

# ─────────────────────────────────────────────────────────────────────────────
#  8. NARRATION
# ─────────────────────────────────────────────────────────────────────────────
def _tdesc(v):
    if v>=35: return "very hot"
    if v>=30: return "hot"
    if v>=26: return "warm"
    if v>=22: return "comfortable"
    if v>=18: return "cool"
    return "cold"
def _hdesc(v):
    if v>=85: return "very humid"
    if v>=70: return "humid"
    if v>=50: return "moderate"
    if v>=35: return "dry"
    return "very dry"
def _pdesc(v):
    if v<1000: return "low"
    if v>1020: return "high"
    return "normal"
def _ldesc(v):
    if v>=800: return "very bright"
    if v>=400: return "bright"
    if v>=100: return "moderately lit"
    if v>=20:  return "dim"
    return "dark"
def _tword(slope):
    if abs(slope)<0.001: return "stable"
    if slope>0.05:  return "rising sharply"
    if slope>0.01:  return "gradually rising"
    if slope<-0.05: return "dropping sharply"
    if slope<-0.01: return "gradually falling"
    return "relatively stable"
def _compare(v1,v2,sensor,lbl1,lbl2):
    diff=v1-v2; u=SENSOR_UNITS[sensor]
    if abs(diff)<0.3: return f"{sensor.capitalize()} is about the same as {lbl2} ({v1:.1f}{u})."
    dir_="higher" if diff>0 else "lower"
    mag ="significantly" if abs(diff)>3 else "slightly"
    return f"{sensor.capitalize()} is {mag} {dir_} than {lbl2} ({v1:.1f}{u} vs {v2:.1f}{u}, Δ{diff:+.1f}{u})."

NOTIF_MSG={
    "NO_ACTION":        "No immediate action required.",
    "INFO_DAY":         "Conditions are within expected daytime parameters.",
    "INFO_NIGHT":       "Conditions are within expected nighttime parameters.",
    "WATCH_BORDERLINE": "⚠ Values approaching threshold limits — increased monitoring advised.",
    "WARN_TEMP":        "⚠ WARNING: Temperature exceeded normal upper bound. Check ventilation.",
    "WARN_HUMIDITY":    "⚠ WARNING: Humidity elevated. Inspect for condensation or seal failure.",
    "WARN_LIGHT":       "⚠ WARNING: Illuminance outside expected range. Possible sensor fault.",
    "CRIT_PRESSURE":    "🚨 CRITICAL: Pressure dropped sharply. Immediate inspection required.",
    "CRIT_MULTI":       "🚨 CRITICAL: Multiple sensors simultaneously abnormal. Immediate action required.",
}

def narrate(result, comparison=None, comp_label="yesterday"):
    if "error" in result: return result["error"]
    pl=result["period_label"]; ss=result["sensor_stats"]
    T=ss["temperature"]; H=ss["humidity"]; P=ss["pressure"]; L=ss["light"]
    cname=result["cluster_name"] or "Unknown"
    cond =CLUSTER_COND.get(cname,"mixed conditions")
    fore =result["forecast"]; ascore=result["anomaly_score"]

    p1=(f"During {pl}, the sensor array recorded {cond} (cluster: {cname}). "
        f"Temperature averaged {T['mean']:.1f}°C ({_tdesc(T['mean'])}), "
        f"ranging {T['min']:.1f}–{T['max']:.1f}°C. "
        f"Humidity was {_hdesc(H['mean'])} at {H['mean']:.1f}% RH. "
        f"Pressure was {_pdesc(P['mean'])} ({P['mean']:.1f} hPa). "
        f"Illuminance averaged {L['mean']:.0f} lux ({_ldesc(L['mean'])}).")

    p2=(f"Time-series trend: temperature was {_tword(T['trend'])} "
        f"({T['first']:.1f}°C → {T['last']:.1f}°C). "
        f"Humidity was {_tword(H['trend'])}. "
        f"Pressure was {_tword(P['trend'])}. "
        f"Light was {_tword(L['trend'])}.")

    p3=""
    if comparison and comparison.get("sensor_stats"):
        cs=comparison["sensor_stats"]
        lines=[_compare(T["mean"],cs["temperature"]["mean"],"temperature",pl,comp_label),
               _compare(H["mean"],cs["humidity"]["mean"],   "humidity",   pl,comp_label),
               _compare(P["mean"],cs["pressure"]["mean"],   "pressure",   pl,comp_label),
               _compare(L["mean"],cs["light"]["mean"],      "light",      pl,comp_label)]
        p3="Comparison with "+comp_label+": "+" ".join(lines)

    na=result["n_anomalies"]; nt=result["n_windows"]
    anom_s=("No anomalous readings detected." if na==0
            else f"{na} anomalous readings ({na/max(nt,1)*100:.1f}% of {nt} windows).")
    p4=anom_s+" "+NOTIF_MSG.get(result["notification"],"")

    if fore is not None:
        ft=fore[:,0].mean(); fh=fore[:,2].mean(); fl=fore[:,3].mean()
        td=fore[-1,0]-fore[0,0]
        p5=(f"3-hour forecast: temperature {ft:.1f}°C ({_tdesc(ft)}), "
            f"humidity {fh:.1f}% ({_hdesc(fh)}), light {fl:.0f} lux ({_ldesc(fl)}). "
            f"Temperature expected to {'rise' if td>0.3 else 'fall' if td<-0.3 else 'remain stable'}.")
        if ascore>0.5: p5+=" (Elevated anomaly probability.)"
    else:
        p5="Forecast unavailable."

    parts=[p1,p2]+([p3] if p3 else [])+[p4,p5]
    return "\n\n".join(textwrap.fill(p,90) for p in parts)

# ─────────────────────────────────────────────────────────────────────────────
#  9. QUERY PARSER
# ─────────────────────────────────────────────────────────────────────────────
class QueryParser:
    PATS={
        "today":      re.compile(r"\b(today|this day|hari ini)\b",re.I),
        "yesterday":  re.compile(r"\b(yesterday|kemarin)\b",re.I),
        "this_week":  re.compile(r"\b(this week|minggu ini)\b",re.I),
        "last_week":  re.compile(r"\b(last week|minggu lalu|minggu kemarin)\b",re.I),
        "this_month": re.compile(r"\b(this month|bulan ini)\b",re.I),
        "last_month": re.compile(r"\b(last month|bulan lalu|bulan kemarin)\b",re.I),
        "2days_ago":  re.compile(r"\b(2 days? ago|2 hari lalu|dua hari lalu)\b",re.I),
        "3days_ago":  re.compile(r"\b(3 days? ago|3 hari lalu|tiga hari lalu)\b",re.I),
    }
    def __init__(self,ref): self.ref=ref
    def _resolve(self,key):
        r=self.ref
        if key=="today":      return r,r,"today"
        if key=="yesterday":  d=r-timedelta(1);return d,d,"yesterday"
        if key=="this_week":  s=r-timedelta(days=r.weekday());return s,r,"this week"
        if key=="last_week":  e=r-timedelta(days=r.weekday()+1);s=e-timedelta(6);return s,e,"last week"
        if key=="this_month": return r.replace(day=1),r,"this month"
        if key=="last_month":
            e=(r.replace(day=1)-timedelta(1));return e.replace(day=1),e,"last month"
        if key=="2days_ago":  d=r-timedelta(2);return d,d,"two days ago"
        if key=="3days_ago":  d=r-timedelta(3);return d,d,"three days ago"
        return None,None,None
    def parse(self,text):
        found=[]
        for key,pat in self.PATS.items():
            if pat.search(text):
                s,e,lbl=self._resolve(key)
                if s: found.append((s,e,lbl))
                if len(found)==2: break
        return found

# ─────────────────────────────────────────────────────────────────────────────
#  10. CHATBOT
# ─────────────────────────────────────────────────────────────────────────────
HELP_TEXT="""
┌─ Available queries ──────────────────────────────────────────────────────────┐
│  today / hari ini            — today's sensor summary + charts               │
│  yesterday / kemarin         — yesterday's summary + charts                  │
│  this week / minggu ini      — this week's summary + charts                  │
│  last week / minggu lalu     — last week's summary + charts                  │
│  this month / bulan ini      — this month's summary + charts                 │
│  last month / bulan lalu     — last month's summary + charts                 │
│  2 days ago / 2 hari lalu    — two days ago                                  │
│  compare today and yesterday — 2-period comparison with overlay charts       │
│  compare this week and last week                                             │
│  compare this month and last month                                           │
│                                                                              │
│  Commands: help  ·  status  ·  quit / exit / keluar                          │
│                                                                              │
│  Charts saved to:  ./iot_charts/                                             │
└──────────────────────────────────────────────────────────────────────────────┘
"""

class IoTChatbot:
    def __init__(self,engine,ref_date,n_days,n_readings):
        self.engine=engine; self.parser=QueryParser(ref_date)
        self.ref_date=ref_date; self.n_days=n_days; self.n_readings=n_readings
        self.q_count=0

    def _banner(self):
        print("\n"+"═"*74)
        print("  IoT-SLM Chatbot  v2  —  Sensor Intelligence with Charts & Tables")
        print("  GRU-RNN · K-Means Clustering · Decision Tree")
        print("═"*74)
        print(f"  Dataset: {self.n_days} days ending {self.ref_date}  |  type 'help' for query list")
        print("═"*74+"\n")

    def _respond(self,text):
        tl=text.strip().lower()
        if tl in ("help","?","bantuan"): return HELP_TEXT,None
        if tl in ("status","info"):
            return (f"\n  Dataset  : {self.n_days} days  ({self.n_readings:,} readings)\n"
                    f"  Windows  : {len(self.engine.metas):,}\n"
                    f"  Ref date : {self.ref_date}  (='today')\n"
                    f"  Charts   : {CHART_DIR}/\n"), None
        if tl in ("quit","exit","keluar","bye","q"): return "__EXIT__",None

        periods=self.parser.parse(text)
        if not periods:
            return ("I didn't recognise a time period in your query.\n"
                    "Try: 'today', 'yesterday', 'this week', 'this month'.\n"
                    "Type 'help' for the full list."), None

        self.q_count+=1
        qid=f"q{self.q_count:03d}_{'_'.join(l.replace(' ','') for _,_,l in periods)}"

        if len(periods)==1:
            s,e,lbl=periods[0]
            r=self.engine.query(s,e,lbl)
            table=ascii_table(r)
            hourly=ascii_hourly_table(self.engine,r,"temperature")
            narr=narrate(r)
            chart_path=make_charts(self.engine,[r],qid)
            return f"{table}\n{hourly}\n\n{narr}",chart_path

        # comparison
        s1,e1,l1=periods[0]; s2,e2,l2=periods[1]
        r1=self.engine.query(s1,e1,l1); r2=self.engine.query(s2,e2,l2)
        sep="─"*72
        out=(f"{sep}\n  PERIOD 1:  {l1.upper()}  ({s1} → {e1})\n{sep}\n"
             +ascii_table(r1)+"\n"
             +narrate(r1,comparison=r2,comp_label=l2)
             +f"\n\n{sep}\n  PERIOD 2:  {l2.upper()}  ({s2} → {e2})\n{sep}\n"
             +ascii_table(r2)+"\n"
             +narrate(r2,comparison=r1,comp_label=l1))
        chart_path=make_charts(self.engine,[r1,r2],qid)
        return out,chart_path

    def run(self):
        self._banner()
        while True:
            try:
                user=input("You › ").strip()
            except (EOFError,KeyboardInterrupt):
                print("\nGoodbye!"); break
            if not user: continue
            resp,chart=self._respond(user)
            if resp=="__EXIT__":
                print("\nBot › Goodbye! Stay safe. 👋\n"); break
            print(f"\nBot ›\n{resp}\n")
            if chart:
                print(f"  📊 Chart saved → {chart}\n")

# ─────────────────────────────────────────────────────────────────────────────
#  11. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n"+"━"*64)
    print("  IoT-SLM Chatbot v2  —  Initialising Pipeline ...")
    print("━"*64)

    print("[1/6] Simulating 60-day sensor data ...")
    data=simulate_data(days=60,interval_minutes=30)
    n_readings=len(data["timestamps"]); ref_date=data["timestamps"][-1].date()
    print(f"      {n_readings:,} readings  |  ref date: {ref_date}")

    print("[2/6] Extracting sliding-window features ...")
    SEQ=24; X_feat,metas=extract_features(data,window=SEQ)
    print(f"      {X_feat.shape[0]:,} windows × {X_feat.shape[1]} features")

    print("[3/6] Fitting K-Means (k=8) ...")
    km,km_sc,cluster_labels=fit_kmeans(X_feat,8)
    cluster_names={}; used=set()
    for k in range(8):
        members=np.where(cluster_labels==k)[0]
        if len(members)==0: cluster_names[k]=f"Cluster {k}"; continue
        t_m=np.mean([metas[i]["temperature"] for i in members])
        h_m=np.mean([metas[i]["humidity"]    for i in members])
        hr_m=np.mean([metas[i]["timestamp"].hour for i in members])
        l_m=np.mean([metas[i]["light"]       for i in members])
        faults=defaultdict(int)
        for i in members: faults[metas[i]["fault_type"]]+=1
        df=max(faults,key=faults.get)
        if df!="normal":
            name={"temp_spike":"Hot Afternoon","pressure_drop":"Unstable",
                  "humidity_surge":"Very Humid","light_flicker":"Light Fault",
                  "multi_fault":"Unstable"}.get(df,"Borderline")
        elif hr_m<6 or hr_m>=20:
            name="Cool Night" if t_m<24 else "Warm Night"
        elif t_m>=30: name="Hot Afternoon"
        elif l_m>400: name="Sunny Day"
        elif h_m>72:  name="Humid Day"
        else:         name="Warm Day" if t_m>=25 else "Borderline"
        orig,suf=name,1
        while name in used: name=f"{orig} {suf}"; suf+=1
        used.add(name); cluster_names[k]=name
    for k,v in sorted(cluster_names.items()):
        n=np.sum(cluster_labels==k); print(f"      Cluster {k}: {v:<22s} ({n} windows)")

    print("[4/6] Training GRU-RNN SLM (60 epochs) ...")
    slm,slm_sc,X_seq,y_seq=train_slm(data,seq_len=SEQ,pred_len=6,epochs=60)

    print("[5/6] Computing SLM anomaly scores ...")
    slm_scores=get_anomaly_scores(slm,X_seq)
    ml=min(len(X_feat),len(slm_scores))
    X_feat,metas,cluster_labels=X_feat[:ml],metas[:ml],cluster_labels[:ml]
    print(f"      Scored {len(slm_scores)} windows")

    print("[6/6] Training Decision Tree ...")
    y_lbl=build_dt_labels(X_feat,metas,cluster_labels,slm_scores,SEQ)
    dt,dt_sc=train_dt(X_feat,y_lbl)
    acc=(dt.predict(dt_sc.transform(X_feat))==y_lbl).mean()
    print(f"      Train accuracy: {acc*100:.1f}%  |  Depth: {dt.get_depth()}")

    engine=AnalysisEngine(data,X_feat,metas,cluster_labels,cluster_names,
                          slm,slm_sc,X_seq,km,km_sc,dt,dt_sc,seq_len=SEQ)
    bot=IoTChatbot(engine,ref_date,n_days=60,n_readings=n_readings)
    bot.run()

if __name__=="__main__":
    main()
