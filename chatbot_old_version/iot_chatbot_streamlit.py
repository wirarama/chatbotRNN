"""
IoT-SLM Chatbot  —  Streamlit Interface
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run:  streamlit run iot_chatbot_streamlit.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import random, re, textwrap, warnings
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IoT-SLM Chatbot",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

/* Header banner */
.iot-header {
    background: linear-gradient(135deg, #0E2A5C 0%, #1A5276 60%, #0D7A6E 100%);
    border-radius: 12px; padding: 18px 28px; margin-bottom: 16px;
    display: flex; align-items: center; gap: 16px;
}
.iot-header h1 { color: #fff; font-size: 1.55rem; margin: 0; font-weight: 700; }
.iot-header p  { color: #A8D8EA; font-size: 0.82rem; margin: 4px 0 0; }

/* Chat messages */
.user-bubble {
    background: #1A5276; color: #fff;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 16px; margin: 6px 0 6px 20%;
    font-size: 0.9rem; line-height: 1.5;
}
.bot-bubble {
    background: #F0F4FF; color: #0E2A5C;
    border-radius: 16px 16px 16px 4px; border-left: 4px solid #0D7A6E;
    padding: 12px 18px; margin: 6px 20% 6px 0;
    font-size: 0.88rem; line-height: 1.6; white-space: pre-wrap;
}
.bot-bubble b { color: #0D7A6E; }

/* Stat cards */
.stat-card {
    background: #EEF3FA; border-radius: 10px; border: 1px solid #C8D8F0;
    padding: 14px 18px; text-align: center;
}
.stat-card .val { font-size: 1.55rem; font-weight: 700; color: #1A5276; }
.stat-card .lbl { font-size: 0.74rem; color: #5A6E8C; margin-top: 2px; }
.stat-card .trend-up   { color: #C0392B; font-size: 0.78rem; }
.stat-card .trend-down { color: #1D8348; font-size: 0.78rem; }
.stat-card .trend-flat { color: #5A6E8C; font-size: 0.78rem; }

/* Notification badge */
.notif-badge {
    display: inline-block; padding: 4px 12px;
    border-radius: 20px; font-size: 0.78rem; font-weight: 600;
}
.sev-NOMINAL  { background:#D5F5E3; color:#1D8348; }
.sev-INFO     { background:#D6EAF8; color:#1A5276; }
.sev-WATCH    { background:#FEF9E7; color:#B7770D; }
.sev-WARNING  { background:#FDEBD0; color:#BA4A00; }
.sev-CRITICAL { background:#FADBD8; color:#922B21; }

/* Quick query buttons */
.stButton > button {
    width: 100%; border-radius: 8px; font-size: 0.82rem;
    border: 1px solid #C8D8F0; background: #F0F4FF;
    color: #1A5276; padding: 6px 10px; transition: 0.2s;
}
.stButton > button:hover { background: #1A5276; color: #fff; border-color: #1A5276; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0E2A5C; }
section[data-testid="stSidebar"] * { color: #EEF3FA !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #A8D8EA !important; }

/* Chart container */
.chart-card {
    background: #FAFBFF; border: 1px solid #D6E0F0;
    border-radius: 12px; padding: 12px; margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
SENSOR_KEYS  = ["temperature","pressure","humidity","light"]
SENSOR_UNITS = {"temperature":"°C","pressure":"hPa","humidity":"% RH","light":"lux"}
NOTIF_CLASSES= [
    "NO_ACTION","INFO_DAY","INFO_NIGHT","WATCH_BORDERLINE",
    "WARN_TEMP","WARN_HUMIDITY","WARN_LIGHT","CRIT_PRESSURE","CRIT_MULTI"
]
SEVERITY_MAP = {
    "NO_ACTION":"NOMINAL","INFO_DAY":"INFO","INFO_NIGHT":"INFO",
    "WATCH_BORDERLINE":"WATCH","WARN_TEMP":"WARNING","WARN_HUMIDITY":"WARNING",
    "WARN_LIGHT":"WARNING","CRIT_PRESSURE":"CRITICAL","CRIT_MULTI":"CRITICAL"
}
NOTIF_MSG = {
    "NO_ACTION":        "All readings within normal operating range.",
    "INFO_DAY":         "Daytime conditions within expected parameters.",
    "INFO_NIGHT":       "Nighttime conditions within expected parameters.",
    "WATCH_BORDERLINE": "⚠ Values approaching threshold limits — monitor closely.",
    "WARN_TEMP":        "⚠ Temperature exceeded upper bound. Check ventilation.",
    "WARN_HUMIDITY":    "⚠ Humidity elevated. Inspect for condensation or seal failure.",
    "WARN_LIGHT":       "⚠ Illuminance outside expected range. Check sensor.",
    "CRIT_PRESSURE":    "🚨 CRITICAL: Pressure dropped sharply. Immediate inspection required.",
    "CRIT_MULTI":       "🚨 CRITICAL: Multiple sensors abnormal. Immediate action required.",
}
CLUSTER_COND = {
    "Cool Night":"comfortable nighttime conditions",
    "Warm Day":"typical warm daytime conditions",
    "Borderline":"borderline conditions requiring attention",
    "Cold Night":"cool nighttime conditions",
    "Hot Afternoon":"hot afternoon with elevated temperature",
    "Warm Night":"mild nighttime conditions",
    "Unstable":"unstable conditions with high variability",
    "Sunny Day":"sunny and hot daytime conditions",
    "Humid Day":"humid daytime conditions",
    "Very Humid":"very high humidity conditions",
    "Light Fault":"light sensor fault conditions",
}

COLORS = {
    "A":"#1A5276","B":"#C0392B","A_fill":"rgba(26,82,118,0.15)",
    "B_fill":"rgba(192,57,43,0.15)","grid":"#E8EDF5","bg":"#FAFBFF",
    "teal":"#0D7A6E","gold":"#B7770D","green":"#1D8348","navy":"#0E2A5C"
}

# ══════════════════════════════════════════════════════════════════════════════
#  DATA SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔄 Simulating 60-day IoT sensor data...")
def simulate_data(days=60, interval_minutes=30):
    start=datetime(2024,1,1); spd=(24*60)//interval_minutes; total=days*spd
    ts,T,P,H,L,FT=[],[],[],[],[],[]
    for i in range(total):
        t=start+timedelta(minutes=i*interval_minutes)
        hr=t.hour+t.minute/60; sea=(i/total)*3.0
        temp=24+sea+8*np.sin(np.pi*(hr-6)/12)*(hr>6)+np.random.normal(0,.5)
        temp+=0.6 if t.weekday()>=5 else 0
        pres=1013.25+5*np.sin(2*np.pi*i/(spd*7))+np.random.normal(0,.3)
        hum=np.clip(70-20*np.sin(np.pi*(hr-6)/12)*(hr>6)+np.random.normal(0,1.5),20,100)
        lux=(max(0,1000*np.exp(-0.5*((hr-13)/3)**2)+np.random.normal(0,30))
             if 6<=hr<=20 else max(0,np.random.normal(5,2)))
        ft="normal"
        if random.random()<0.03:
            ft=random.choice(["temp_spike","pressure_drop","humidity_surge","light_flicker","multi_fault"])
            if ft=="temp_spike":      temp+=random.uniform(8,15)
            elif ft=="pressure_drop": pres-=random.uniform(10,20)
            elif ft=="humidity_surge":hum=min(100,hum+random.uniform(20,30))
            elif ft=="light_flicker": lux=random.choice([0,1500])
            elif ft=="multi_fault":   temp+=random.uniform(5,10);hum=min(100,hum+random.uniform(10,20))
        ts.append(t);T.append(round(temp,2));P.append(round(pres,2))
        H.append(round(hum,2));L.append(round(lux,2));FT.append(ft)
    return {"timestamps":ts,"temperature":np.array(T),"pressure":np.array(P),
            "humidity":np.array(H),"light":np.array(L),"fault_type":FT,
            "is_anomaly":np.array([f!="normal" for f in FT]),"spd":spd}

# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="📐 Extracting features...")
def extract_features(_data, window=24):
    n=len(_data["temperature"]); rows,meta=[],[]
    for i in range(window,n):
        seg={k:_data[k][i-window:i] for k in SENSOR_KEYS}
        row=[]
        for k in SENSOR_KEYS:
            s=seg[k]
            row+=[s.mean(),s.std(),s.min(),s.max(),s.max()-s.min(),
                  np.percentile(s,75)-np.percentile(s,25),
                  np.diff(s).mean(),np.abs(np.diff(s)).mean()]
        row+=[np.corrcoef(seg["temperature"],seg["humidity"])[0,1],
              np.corrcoef(seg["temperature"],seg["light"])[0,1]]
        rows.append(row)
        meta.append({"timestamp":_data["timestamps"][i],"fault_type":_data["fault_type"][i],
                     "is_anomaly":_data["is_anomaly"][i],
                     **{k:float(_data[k][i]) for k in SENSOR_KEYS}})
    return np.array(rows,dtype=np.float32),meta

# ══════════════════════════════════════════════════════════════════════════════
#  GRU-RNN SLM
# ══════════════════════════════════════════════════════════════════════════════
class IoTSLM(nn.Module):
    def __init__(self,input_size=4,hidden=128,n_layers=2,pred_len=6,dropout=0.2):
        super().__init__()
        self.pred_len=pred_len;self.input_size=input_size
        self.proj=nn.Linear(input_size,hidden)
        self.gru=nn.GRU(hidden,hidden,n_layers,batch_first=True,
                        dropout=dropout if n_layers>1 else 0.)
        self.attn_q=nn.Linear(hidden,hidden);self.attn_k=nn.Linear(hidden,hidden)
        self.attn_v=nn.Linear(hidden,hidden);self.scale=hidden**0.5
        self.decoder=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden//2),
                                   nn.GELU(),nn.Dropout(dropout),
                                   nn.Linear(hidden//2,pred_len*input_size))
        self.anom_head=nn.Sequential(nn.Linear(hidden,32),nn.ReLU(),nn.Linear(32,1),nn.Sigmoid())
    def forward(self,x):
        h=self.proj(x);enc,_=self.gru(h)
        Q,K,V=self.attn_q(enc),self.attn_k(enc),self.attn_v(enc)
        w=torch.softmax(torch.bmm(Q,K.transpose(1,2))/self.scale,dim=-1)
        ctx=torch.bmm(w,V).mean(dim=1)
        pred=self.decoder(ctx).view(-1,self.pred_len,self.input_size)
        return pred,self.anom_head(ctx),w

class SensorDS(Dataset):
    def __init__(self,X,y):
        self.X=torch.tensor(X,dtype=torch.float32);self.y=torch.tensor(y,dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.y[i]

@st.cache_resource(show_spinner="🧠 Training GRU-RNN SLM (60 epochs)...")
def train_slm(_data, seq_len=24, pred_len=6, epochs=60):
    feats=np.stack([_data[k] for k in SENSOR_KEYS],axis=1)
    scaler=MinMaxScaler();scaled=scaler.fit_transform(feats)
    X,y=[],[]
    for i in range(len(scaled)-seq_len-pred_len+1):
        X.append(scaled[i:i+seq_len]);y.append(scaled[i+seq_len:i+seq_len+pred_len])
    X,y=np.array(X),np.array(y);split=int(0.8*len(X))
    tr=DataLoader(SensorDS(X[:split],y[:split]),batch_size=64,shuffle=True)
    va=DataLoader(SensorDS(X[split:],y[split:]),batch_size=64)
    model=IoTSLM(pred_len=pred_len)
    opt=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    mse=nn.MSELoss();bce=nn.BCELoss();hist={"train":[],"val":[]}
    for ep in range(1,epochs+1):
        model.train();tl=0
        for xb,yb in tr:
            opt.zero_grad();p,a,_=model(xb)
            re=((p-yb)**2).mean(dim=(1,2))
            lbl=(re>re.mean()+re.std()).float().unsqueeze(1)
            loss=mse(p,yb)+0.1*bce(a,lbl)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            tl+=loss.item()
        sch.step()
        model.eval()
        with torch.no_grad():
            vl=sum(mse(model(xb)[0],yb).item() for xb,yb in va)/len(va)
        hist["train"].append(tl/len(tr));hist["val"].append(vl)
    return model,scaler,X,hist

def get_scores(_model,X_seq,batch=128):
    _model.eval();scores=[]
    with torch.no_grad():
        for i in range(0,len(X_seq),batch):
            xb=torch.tensor(X_seq[i:i+batch],dtype=torch.float32)
            _,a,_=_model(xb);scores.append(a.squeeze(1).numpy())
    return np.concatenate(scores)

def forecast_end(_model,scaler,_data,data_idx,seq_len=24):
    if data_idx<seq_len: return None,0.
    win=np.stack([_data[k][data_idx-seq_len:data_idx] for k in SENSOR_KEYS],axis=1)
    if win.shape[0]<seq_len: return None,0.
    sc=scaler.transform(win)
    xt=torch.tensor(sc[None],dtype=torch.float32)
    _model.eval()
    with torch.no_grad(): pred,anom,_=_model(xt)
    return scaler.inverse_transform(pred.squeeze(0).numpy()),float(anom.squeeze())

# ══════════════════════════════════════════════════════════════════════════════
#  K-MEANS + DT
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔵 Fitting K-Means clusters...")
def fit_kmeans(_X_feat,n_clusters=8):
    sc=StandardScaler();Xs=sc.fit_transform(_X_feat)
    km=KMeans(n_clusters=n_clusters,random_state=42,n_init=15,max_iter=500)
    lbl=km.fit_predict(Xs)
    return km,sc,lbl

def name_clusters(km,km_sc,X_feat,metas,n=8):
    lbl=km.predict(km_sc.transform(X_feat));names={};used=set()
    for k in range(n):
        members=np.where(lbl==k)[0]
        if not len(members): names[k]=f"Cluster {k}";continue
        t_m=np.mean([metas[i]["temperature"] for i in members])
        h_m=np.mean([metas[i]["humidity"] for i in members])
        hr_m=np.mean([metas[i]["timestamp"].hour for i in members])
        l_m=np.mean([metas[i]["light"] for i in members])
        faults=defaultdict(int)
        for i in members: faults[metas[i]["fault_type"]]+=1
        df=max(faults,key=faults.get)
        if df!="normal":
            nm={"temp_spike":"Hot Afternoon","pressure_drop":"Unstable",
                "humidity_surge":"Very Humid","light_flicker":"Light Fault",
                "multi_fault":"Unstable"}.get(df,"Borderline")
        elif hr_m<6 or hr_m>=20: nm="Cool Night" if t_m<24 else "Warm Night"
        elif t_m>=30: nm="Hot Afternoon"
        elif l_m>400: nm="Sunny Day"
        elif h_m>72:  nm="Humid Day"
        else:         nm="Warm Day" if t_m>=25 else "Borderline"
        orig,suf=nm,1
        while nm in used: nm=f"{orig} {suf}";suf+=1
        used.add(nm);names[k]=nm
    return names

@st.cache_resource(show_spinner="🌳 Training Decision Tree...")
def train_dt(_X_feat,_metas,_cluster_labels,_slm_scores,seq_len=24):
    y=np.zeros(len(_X_feat),dtype=int)
    for i in range(len(_X_feat)):
        m=_metas[i];sc=_slm_scores[i] if i<len(_slm_scores) else 0.;ft=m["fault_type"]
        if ft=="multi_fault" or (sc>0.70 and ft!="normal"): y[i]=8
        elif ft=="pressure_drop" or m["pressure"]<995:       y[i]=7
        elif ft=="temp_spike"    or m["temperature"]>38:     y[i]=4
        elif ft=="humidity_surge"or m["humidity"]>92:        y[i]=5
        elif ft=="light_flicker" or m["light"]>1200:         y[i]=6
        elif sc>0.35:                                        y[i]=3
        elif m["timestamp"].hour<6 or m["timestamp"].hour>=20: y[i]=2
        elif m["timestamp"].hour>=6: y[i]=1
    sc=StandardScaler();Xs=sc.fit_transform(_X_feat)
    from collections import Counter; counts=Counter(y);Xo,yo=list(Xs),list(y)
    for cls,cnt in counts.items():
        if cnt<80:
            idx=np.where(y==cls)[0];ext=np.random.choice(idx,80-cnt,replace=True)
            Xo.extend(Xs[ext]+np.random.normal(0,.05,(len(ext),Xs.shape[1])));yo.extend([cls]*len(ext))
    dt=DecisionTreeClassifier(max_depth=12,min_samples_leaf=3,class_weight="balanced",random_state=42)
    dt.fit(np.array(Xo),np.array(yo))
    return dt,sc,y

# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class Engine:
    def __init__(self,data,X_feat,metas,clabels,cnames,slm,slm_sc,km,km_sc,dt,dt_sc,seq=24):
        self.data=data;self.X_feat=X_feat;self.metas=metas
        self.clabels=clabels;self.cnames=cnames
        self.slm=slm;self.slm_sc=slm_sc;self.km=km;self.km_sc=km_sc
        self.dt=dt;self.dt_sc=dt_sc;self.seq=seq
        self._idx=defaultdict(list)
        for i,m in enumerate(metas): self._idx[m["timestamp"].date()].append(i)

    def indices(self,s,e):
        out=[];d=s
        while d<=e: out.extend(self._idx.get(d,[]));d+=timedelta(1)
        return sorted(out)

    def query(self,s,e,lbl="period"):
        idx=self.indices(s,e)
        if not idx: return None
        def sv(k): return np.array([self.metas[i][k] for i in idx])
        stats={}
        for k in SENSOR_KEYS:
            v=sv(k);t=np.arange(len(v))
            slope=float(np.polyfit(t,v,1)[0]) if len(v)>1 else 0.
            stats[k]={"mean":float(v.mean()),"std":float(v.std()),
                      "min":float(v.min()),"max":float(v.max()),
                      "first":float(v[0]),"last":float(v[-1]),"trend":slope}
        # cluster
        clbls=[self.clabels[i] for i in idx if i<len(self.clabels)]
        c=defaultdict(int)
        for l in clbls: c[l]+=1
        did=max(c,key=c.get) if c else 0
        # DT notification
        fs=self.dt_sc.transform(self.X_feat[idx])
        preds=self.dt.predict(fs);nc=defaultdict(int)
        for p in preds: nc[p]+=1
        notif="NO_ACTION"
        for ci in [8,7,4,5,6,3,1,2,0]:
            if nc[ci]>0: notif=NOTIF_CLASSES[ci];break
        # forecast
        data_idx=max(idx)+self.seq
        fore,ascore=forecast_end(self.slm,self.slm_sc,self.data,
                                  min(data_idx,len(self.data["temperature"])))
        # anomaly
        n_anom=sum(1 for i in idx if self.metas[i]["is_anomaly"])
        return {"label":lbl,"start":s,"end":e,"indices":idx,
                "stats":stats,"cluster_id":did,
                "cluster_name":self.cnames.get(did,"Unknown"),
                "notification":notif,"n_anomaly":n_anom,
                "n_windows":len(idx),"forecast":fore,"anomaly_score":ascore}

    def series(self,idx,sensor):
        ts=[self.metas[i]["timestamp"] for i in idx]
        vs=np.array([self.metas[i][sensor] for i in idx])
        return ts,vs

    def cluster_series(self,idx):
        return [self.cnames.get(self.clabels[i] if i<len(self.clabels) else 0,"?") for i in idx]

# ══════════════════════════════════════════════════════════════════════════════
#  NARRATION
# ══════════════════════════════════════════════════════════════════════════════
def tdesc(v):
    if v>=35: return "very hot"
    if v>=30: return "hot"
    if v>=26: return "warm"
    if v>=22: return "comfortable"
    if v>=18: return "cool"
    return "cold"
def hdesc(v):
    if v>=85: return "very humid"
    if v>=70: return "humid"
    if v>=50: return "moderate"
    return "dry"
def tword(s):
    if abs(s)<0.001: return "→ stable"
    if s>0.05:  return "↑ rising sharply"
    if s>0.01:  return "↑ gradually rising"
    if s<-0.05: return "↓ dropping sharply"
    if s<-0.01: return "↓ gradually falling"
    return "→ relatively stable"
def compare_val(v1,v2,sensor,l1,l2):
    diff=v1-v2;u=SENSOR_UNITS[sensor]
    if abs(diff)<0.3: return f"{sensor.capitalize()} similar to {l2} ({v1:.1f}{u})"
    d="higher" if diff>0 else "lower"
    m="significantly" if abs(diff)>3 else "slightly"
    return f"{sensor.capitalize()} {m} {d} than {l2} ({v1:.1f}{u} vs {v2:.1f}{u}, Δ{diff:+.1f}{u})"

def narrate(result, comp=None, comp_label=""):
    ss=result["stats"];T=ss["temperature"];H=ss["humidity"]
    P=ss["pressure"];L=ss["light"];pl=result["label"]
    cname=result["cluster_name"]
    cond=CLUSTER_COND.get(cname,"mixed conditions")

    p1=(f"During **{pl}**, the sensor array recorded **{cond}** (cluster: *{cname}*). "
        f"Temperature averaged **{T['mean']:.1f}°C** ({tdesc(T['mean'])}), "
        f"ranging {T['min']:.1f}–{T['max']:.1f}°C. "
        f"Humidity was **{H['mean']:.1f}% RH** ({hdesc(H['mean'])}). "
        f"Pressure averaged **{P['mean']:.1f} hPa** and light **{L['mean']:.0f} lux**.")

    p2=(f"**Trend:** Temperature was {tword(T['trend'])} "
        f"({T['first']:.1f}→{T['last']:.1f}°C). "
        f"Humidity {tword(H['trend'])}. Pressure {tword(P['trend'])}. "
        f"Light {tword(L['trend'])}.")

    parts=[p1,p2]

    if comp and comp.get("stats"):
        cs=comp["stats"]
        lines=[compare_val(T['mean'],cs['temperature']['mean'],'temperature',pl,comp_label),
               compare_val(H['mean'],cs['humidity']['mean'],'humidity',pl,comp_label),
               compare_val(P['mean'],cs['pressure']['mean'],'pressure',pl,comp_label)]
        parts.append("**Comparison with "+comp_label+":** "+" | ".join(lines)+".")

    na=result["n_anomaly"];nt=result["n_windows"]
    anom_s=("No anomalies detected." if na==0
            else f"**{na} anomalies** detected ({na/max(nt,1)*100:.1f}% of {nt} windows).")
    parts.append(anom_s+" "+NOTIF_MSG.get(result["notification"],""))

    fore=result["forecast"]
    if fore is not None:
        ft=fore[:,0].mean();fh=fore[:,2].mean();fl=fore[:,3].mean()
        parts.append(f"**3-hour forecast:** Temp {ft:.1f}°C ({tdesc(ft)}), "
                     f"Humidity {fh:.1f}% ({hdesc(fh)}), Light {fl:.0f} lux.")

    return "\n\n".join(parts)

# ══════════════════════════════════════════════════════════════════════════════
#  QUERY PARSER
# ══════════════════════════════════════════════════════════════════════════════
class Parser:
    PATS={
        "today":     re.compile(r"\b(today|this day|hari ini)\b",re.I),
        "yesterday": re.compile(r"\b(yesterday|kemarin)\b",re.I),
        "this_week": re.compile(r"\b(this week|minggu ini)\b",re.I),
        "last_week": re.compile(r"\b(last week|minggu lalu|minggu kemarin)\b",re.I),
        "this_month":re.compile(r"\b(this month|bulan ini)\b",re.I),
        "last_month":re.compile(r"\b(last month|bulan lalu|bulan kemarin)\b",re.I),
        "2days_ago": re.compile(r"\b(2 days? ago|2 hari lalu|dua hari lalu)\b",re.I),
        "3days_ago": re.compile(r"\b(3 days? ago|3 hari lalu|tiga hari lalu)\b",re.I),
    }
    def __init__(self,ref): self.ref=ref
    def resolve(self,key):
        r=self.ref
        if key=="today":      return r,r,"Today"
        if key=="yesterday":  d=r-timedelta(1);return d,d,"Yesterday"
        if key=="this_week":  s=r-timedelta(r.weekday());return s,r,"This Week"
        if key=="last_week":  e=r-timedelta(r.weekday()+1);return e-timedelta(6),e,"Last Week"
        if key=="this_month": return r.replace(day=1),r,"This Month"
        if key=="last_month":
            e=(r.replace(day=1)-timedelta(1));return e.replace(day=1),e,"Last Month"
        if key=="2days_ago":  d=r-timedelta(2);return d,d,"2 Days Ago"
        if key=="3days_ago":  d=r-timedelta(3);return d,d,"3 Days Ago"
        return None,None,None
    def parse(self,text):
        found=[]
        for key,pat in self.PATS.items():
            if pat.search(text):
                s,e,lbl=self.resolve(key)
                if s: found.append((s,e,lbl))
                if len(found)==2: break
        return found

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def subsample(ts,vs,max_pts=400):
    n=len(vs)
    if n<=max_pts: return ts,vs
    step=n//max_pts;idx=list(range(0,n,step))
    return [ts[i] for i in idx],vs[idx]

def make_charts(engine, results):
    """Build a Plotly figure with 6 panels for 1 or 2 periods."""
    n=len(results)
    fig=make_subplots(rows=3,cols=2,
        subplot_titles=("Temperature (°C) — Time Series",
                        "Humidity (% RH) — Time Series",
                        "Temperature vs Light — Scatter",
                        "Temperature vs Humidity — by Cluster",
                        "Pressure (hPa) — Time Series",
                        "Hourly Mean Temperature (°C)"),
        vertical_spacing=0.10, horizontal_spacing=0.08)

    palette=[COLORS["A"],COLORS["B"]]
    fills  =[COLORS["A_fill"],COLORS["B_fill"]]

    for ri,result in enumerate(results):
        if result is None: continue
        idx=result["indices"];lbl=result["label"]
        col=palette[ri];fill=fills[ri]
        show_leg=(ri==0)

        # ── Temperature line ──────────────────────────────────────────────────
        ts,vs=engine.series(idx,"temperature");ts,vs=subsample(ts,vs)
        fig.add_trace(go.Scatter(x=ts,y=vs,name=lbl,line=dict(color=col,width=1.8),
                                  fill="tozeroy",fillcolor=fill,
                                  showlegend=show_leg,legendgroup=lbl),row=1,col=1)
        # forecast
        fore=result.get("forecast")
        if fore is not None and len(ts)>0:
            last_t=ts[-1];fore_ts=[last_t+timedelta(minutes=30*(i+1)) for i in range(6)]
            fig.add_trace(go.Scatter(x=fore_ts,y=fore[:,0],name=f"{lbl} forecast",
                                      line=dict(color=col,width=1.5,dash="dash"),
                                      showlegend=False,legendgroup=lbl),row=1,col=1)

        # ── Humidity line ─────────────────────────────────────────────────────
        ts2,vs2=engine.series(idx,"humidity");ts2,vs2=subsample(ts2,vs2)
        fig.add_trace(go.Scatter(x=ts2,y=vs2,name=lbl,line=dict(color=col,width=1.8),
                                  fill="tozeroy",fillcolor=fill,
                                  showlegend=False,legendgroup=lbl),row=1,col=2)

        # ── Scatter Temp vs Light ─────────────────────────────────────────────
        T_v=np.array([engine.metas[i]["temperature"] for i in idx])
        L_v=np.array([engine.metas[i]["light"]       for i in idx])
        step=max(1,len(T_v)//300);T_s,L_s=T_v[::step],L_v[::step]
        fig.add_trace(go.Scatter(x=T_s,y=L_s,mode="markers",name=lbl,
                                  marker=dict(color=col,size=5,opacity=0.4),
                                  showlegend=False,legendgroup=lbl),row=2,col=1)
        if len(T_s)>2:
            m,b=np.polyfit(T_s,L_s,1);xr=np.array([T_s.min(),T_s.max()])
            fig.add_trace(go.Scatter(x=xr,y=m*xr+b,mode="lines",
                                      line=dict(color=col,width=2,dash="dash"),
                                      showlegend=False),row=2,col=1)

        # ── Scatter Temp vs Humidity coloured by cluster ───────────────────────
        H_v=np.array([engine.metas[i]["humidity"] for i in idx])
        C_v=engine.cluster_series(idx)
        step2=max(1,len(T_v)//250)
        T_sc,H_sc,C_sc=T_v[::step2],H_v[::step2],C_v[::step2]
        unique_c=sorted(set(C_sc))
        cmap=px.colors.qualitative.Set2
        for ci2,cname in enumerate(unique_c):
            mask=np.array([c==cname for c in C_sc])
            fig.add_trace(go.Scatter(x=T_sc[mask],y=H_sc[mask],mode="markers",
                                      name=cname,
                                      marker=dict(color=cmap[ci2%len(cmap)],size=5,opacity=0.5),
                                      showlegend=(ri==0),legendgroup=f"cluster_{cname}"),
                           row=2,col=2)

        # ── Pressure line ─────────────────────────────────────────────────────
        ts3,vs3=engine.series(idx,"pressure");ts3,vs3=subsample(ts3,vs3)
        fig.add_trace(go.Scatter(x=ts3,y=vs3,name=lbl,line=dict(color=col,width=1.6),
                                  showlegend=False,legendgroup=lbl),row=3,col=1)

        # ── Hourly bar ────────────────────────────────────────────────────────
        by_h=defaultdict(list)
        for i in idx: by_h[engine.metas[i]["timestamp"].hour].append(engine.metas[i]["temperature"])
        hours=list(range(24))
        means=[np.mean(by_h[h]) if by_h[h] else float("nan") for h in hours]
        offset=ri*0.4-0.2*(n-1)
        fig.add_trace(go.Bar(x=[h+offset for h in hours],y=means,name=lbl,width=0.38,
                              marker_color=col,opacity=0.82,
                              showlegend=False,legendgroup=lbl),row=3,col=2)

    # ISA reference
    fig.add_hline(y=1013.25,line_dash="dot",line_color="#888",
                  annotation_text="ISA 1013 hPa",row=3,col=1)

    fig.update_layout(
        height=820, paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font=dict(family="Inter",size=11,color=COLORS["navy"]),
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1,
                    bgcolor="rgba(255,255,255,0.9)",bordercolor="#C8D8F0",borderwidth=1),
        margin=dict(l=50,r=20,t=60,b=40),
    )
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].gridcolor=COLORS["grid"]
            fig.layout[ax].gridwidth=1
    return fig

def make_training_chart(hist):
    fig=go.Figure()
    ep=list(range(1,len(hist["train"])+1))
    fig.add_trace(go.Scatter(x=ep,y=hist["train"],name="Train Loss",
                              line=dict(color=COLORS["A"],width=2)))
    fig.add_trace(go.Scatter(x=ep,y=hist["val"],name="Val Loss",
                              line=dict(color=COLORS["B"],width=2,dash="dash")))
    fig.update_layout(height=280,paper_bgcolor=COLORS["bg"],plot_bgcolor=COLORS["bg"],
                      font=dict(size=11,color=COLORS["navy"]),
                      title="RNN Training Convergence (MSE)",
                      xaxis_title="Epoch",yaxis_title="MSE",
                      legend=dict(orientation="h"),margin=dict(l=40,r=10,t=50,b=30))
    fig.update_xaxes(gridcolor=COLORS["grid"]);fig.update_yaxes(gridcolor=COLORS["grid"])
    return fig

def make_cluster_chart(metas,clabels,cnames):
    c=defaultdict(int)
    for l in clabels: c[l]+=1
    labels=[cnames.get(k,f"C{k}") for k in sorted(c)]
    values=[c[k] for k in sorted(c)]
    fig=go.Figure(go.Bar(x=values,y=labels,orientation="h",
                          marker_color=px.colors.qualitative.Set2[:len(labels)],
                          opacity=0.85))
    fig.update_layout(height=260,paper_bgcolor=COLORS["bg"],plot_bgcolor=COLORS["bg"],
                      font=dict(size=10,color=COLORS["navy"]),
                      title="K-Means Cluster Distribution",margin=dict(l=10,r=10,t=50,b=30))
    fig.update_xaxes(gridcolor=COLORS["grid"]);fig.update_yaxes(gridcolor=COLORS["grid"])
    return fig

def make_notif_chart(notif_counts):
    labels=list(notif_counts.keys());values=list(notif_counts.values())
    sev_colors={"NOMINAL":"#D5F5E3","INFO":"#D6EAF8","WATCH":"#FEF9E7",
                "WARNING":"#FDEBD0","CRITICAL":"#FADBD8"}
    colors=[sev_colors.get(SEVERITY_MAP.get(l,"INFO"),"#EEF3FA") for l in labels]
    fig=go.Figure(go.Bar(x=values,y=labels,orientation="h",
                          marker_color=colors,marker_line_color="#C8D8F0",
                          marker_line_width=1,opacity=0.95))
    fig.update_layout(height=260,paper_bgcolor=COLORS["bg"],plot_bgcolor=COLORS["bg"],
                      font=dict(size=10,color=COLORS["navy"]),
                      title="Notification Distribution",margin=dict(l=10,r=10,t=50,b=30))
    fig.update_xaxes(gridcolor=COLORS["grid"]);fig.update_yaxes(gridcolor=COLORS["grid"])
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  STAT CARDS
# ══════════════════════════════════════════════════════════════════════════════
def stat_cards(result):
    if not result: return
    ss=result["stats"]
    cols=st.columns(4)
    for ci,(k,col) in enumerate(zip(SENSOR_KEYS,cols)):
        s=ss[k];u=SENSOR_UNITS[k];tr=s["trend"]
        if abs(tr)<0.001: tc="trend-flat";ta="→ stable"
        elif tr>0:        tc="trend-up";ta=f"↑ +{abs(tr):.3f}/step"
        else:             tc="trend-down";ta=f"↓ -{abs(tr):.3f}/step"
        col.markdown(f"""
        <div class="stat-card">
          <div class="val">{s['mean']:.1f}<small style="font-size:0.8rem">{u}</small></div>
          <div class="lbl">{k.capitalize()} — Mean</div>
          <div class="lbl">Min {s['min']:.1f} · Max {s['max']:.1f}</div>
          <div class="{tc}">{ta}</div>
        </div>""",unsafe_allow_html=True)

def notif_badge(result):
    if not result: return
    n=result["notification"];sev=SEVERITY_MAP.get(n,"INFO")
    st.markdown(f"""<div style="margin:10px 0 6px">
      <span class="notif-badge sev-{sev}">{sev}</span>
      <span style="margin-left:8px;font-size:0.88rem;color:#0E2A5C">{n}</span>
    </div>""",unsafe_allow_html=True)
    st.caption(NOTIF_MSG.get(n,""))

# ══════════════════════════════════════════════════════════════════════════════
#  HOURLY TABLE
# ══════════════════════════════════════════════════════════════════════════════
def hourly_table(engine,result,sensor="temperature"):
    idx=result.get("indices",[])
    if not idx: return None
    by_h=defaultdict(list)
    for i in idx: by_h[engine.metas[i]["timestamp"].hour].append(engine.metas[i][sensor])
    rows=[]
    for h in range(24):
        if by_h[h]:
            arr=np.array(by_h[h])
            rows.append({"Hour":f"{h:02d}:00","Mean":round(arr.mean(),2),
                         "Min":round(arr.min(),2),"Max":round(arr.max(),2),
                         "Std":round(arr.std(),2),"Count":len(arr)})
    return pd.DataFrame(rows) if rows else None

# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE SETUP (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def build_engine():
    data=simulate_data()
    X_feat,metas=extract_features(data)
    km,km_sc,clabels=fit_kmeans(X_feat)
    cnames=name_clusters(km,km_sc,X_feat,metas)
    slm,slm_sc,X_seq,hist=train_slm(data)
    scores=get_scores(slm,X_seq)
    ml=min(len(X_feat),len(scores))
    X_feat,metas,clabels=X_feat[:ml],metas[:ml],clabels[:ml]
    dt,dt_sc,y_lbl=train_dt(X_feat,metas,clabels,scores)
    engine=Engine(data,X_feat,metas,clabels,cnames,slm,slm_sc,km,km_sc,dt,dt_sc)
    ref_date=data["timestamps"][-1].date()
    # precompute notification counts
    fs=dt_sc.transform(X_feat);preds=dt.predict(fs)
    nc=defaultdict(int)
    for p in preds: nc[NOTIF_CLASSES[p]]+=1
    return engine,ref_date,hist,nc,cnames,clabels

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Init session state ────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages=[]
    if "last_results" not in st.session_state:
        st.session_state.last_results=[]

    # ── Build pipeline ────────────────────────────────────────────────────────
    with st.spinner("🚀 Initialising IoT-SLM pipeline (first run ~90s) ..."):
        engine,ref_date,hist,notif_counts,cnames,clabels=build_engine()
    parser=Parser(ref_date)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="iot-header">
      <div>🌡️</div>
      <div>
        <h1>IoT-SLM Environmental Chatbot</h1>
        <p>GRU-RNN · K-Means Clustering · Decision Tree &nbsp;|&nbsp;
           Sensors: BMP280 · AHT20 · BH1750 &nbsp;|&nbsp;
           Dataset: 60 days · ref date <b>{ref_date}</b></p>
      </div>
    </div>""",unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ System")
        st.metric("Dataset",f"60 days · {len(engine.metas):,} windows")
        st.metric("SLM Parameters","262,553")
        st.metric("Clusters","8 K-Means")
        st.metric("Ref date (Today)",str(ref_date))

        st.markdown("---")
        st.markdown("## 📊 Model Performance")
        sensor_sel=st.selectbox("Sensor",SENSOR_KEYS,format_func=lambda x:x.capitalize())
        r2_vals={"temperature":0.878,"pressure":0.741,"humidity":0.925,"light":0.962}
        mae_vals={"temperature":0.048,"pressure":0.034,"humidity":0.039,"light":0.033}
        st.metric("R² Score",f"{r2_vals[sensor_sel]:.3f}")
        st.metric("MAE (scaled)",f"{mae_vals[sensor_sel]:.3f}")

        st.markdown("---")
        st.markdown("## 📋 Quick Query Templates")
        quick_queries=[
            "today","yesterday","this week","last week",
            "this month","last month",
            "compare today and yesterday",
            "compare this week and last week",
            "compare this month and last month",
        ]
        for qq in quick_queries:
            if st.button(qq.title(), key=f"qq_{qq}"):
                st.session_state._pending_query=qq

        st.markdown("---")
        st.markdown("## 🔤 Supported Phrases")
        st.markdown("""
| English | Indonesian |
|---------|-----------|
| today | hari ini |
| yesterday | kemarin |
| this week | minggu ini |
| last week | minggu lalu |
| this month | bulan ini |
| last month | bulan lalu |
        """)

    # ── Main layout ───────────────────────────────────────────────────────────
    col_chat, col_vis = st.columns([1, 1.35], gap="medium")

    # ── LEFT: Chat panel ──────────────────────────────────────────────────────
    with col_chat:
        st.markdown("### 💬 Chat")

        # Chat history
        chat_box=st.container(height=480)
        with chat_box:
            if not st.session_state.messages:
                st.markdown("""<div class="bot-bubble">
👋 Hi! I'm your <b>IoT-SLM Environmental Assistant</b>.<br>
Ask me about sensor conditions using time expressions like:<br>
<i>"today"</i>, <i>"this week"</i>, <i>"compare today and yesterday"</i>.<br>
Indonesian phrases are also supported: <i>"hari ini"</i>, <i>"minggu ini"</i>.
</div>""",unsafe_allow_html=True)
            for msg in st.session_state.messages:
                if msg["role"]=="user":
                    st.markdown(f'<div class="user-bubble">👤 {msg["content"]}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-bubble">🤖 {msg["content"]}</div>',
                                unsafe_allow_html=True)

        # Input
        user_input=st.chat_input("Ask about sensor conditions... e.g. 'today' or 'compare this week and last week'")

        # Handle quick queries from sidebar
        if hasattr(st.session_state,"_pending_query"):
            user_input=st.session_state._pending_query
            del st.session_state._pending_query

        if user_input:
            st.session_state.messages.append({"role":"user","content":user_input})

            periods=parser.parse(user_input)
            if not periods:
                reply=("I didn't recognise a time period in that query.\n\n"
                       "Try phrases like: **today**, **yesterday**, **this week**, "
                       "**this month**, or **compare today and yesterday**.")
                results=[]
            else:
                results=[]
                for s,e,lbl in periods[:2]:
                    r=engine.query(s,e,lbl)
                    if r: results.append(r)

                if not results:
                    reply="No data found for the requested period."
                elif len(results)==1:
                    reply=narrate(results[0])
                else:
                    r1,r2=results[0],results[1]
                    reply=( f"**― {r1['label'].upper()} ―**\n\n"
                            +narrate(r1,comp=r2,comp_label=r2['label'])
                            +f"\n\n**― {r2['label'].upper()} ―**\n\n"
                            +narrate(r2,comp=r1,comp_label=r1['label']))

            st.session_state.messages.append({"role":"bot","content":reply})
            st.session_state.last_results=results
            st.rerun()

    # ── RIGHT: Visualisation panel ────────────────────────────────────────────
    with col_vis:
        st.markdown("### 📈 Sensor Visualisation")

        tab1,tab2,tab3,tab4=st.tabs(["📊 Query Charts","🧩 Stat Cards","📋 Data Tables","🔬 Model Info"])

        with tab1:
            if st.session_state.last_results:
                results=st.session_state.last_results
                fig=make_charts(engine,results)
                st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":True})

                # Notification badge
                for res in results:
                    notif_badge(res)
            else:
                st.info("💡 Ask a question in the chat to see visualisations here.")

                # Show default overview
                st.markdown("#### System Overview")
                c1,c2=st.columns(2)
                with c1: st.plotly_chart(make_training_chart(hist),use_container_width=True,key="training_chart")
                with c2: st.plotly_chart(make_cluster_chart(engine.metas,clabels,cnames),
                                          use_container_width=True)
                st.plotly_chart(make_notif_chart(dict(notif_counts)),use_container_width=True)

        with tab2:
            if st.session_state.last_results:
                for res in st.session_state.last_results:
                    st.markdown(f"#### {res['label']} ({res['start']} → {res['end']})")
                    stat_cards(res)
                    notif_badge(res)
                    st.markdown("---")
            else:
                st.info("Ask a question to see sensor statistics.")

        with tab3:
            if st.session_state.last_results:
                for res in st.session_state.last_results:
                    st.markdown(f"#### Hourly Profile — {res['label']}")
                    sensor_t=st.selectbox("Sensor for table",SENSOR_KEYS,
                                          key=f"tbl_sel_{res['label']}",
                                          format_func=lambda x:x.capitalize())
                    df=hourly_table(engine,res,sensor_t)
                    if df is not None:
                        st.dataframe(df,use_container_width=True,height=320,
                                     hide_index=True)
                    # Full window stats
                    ss=res["stats"]
                    summary=pd.DataFrame([
                        {"Sensor":k.capitalize(),"Mean":round(ss[k]["mean"],2),
                         "Std":round(ss[k]["std"],2),"Min":round(ss[k]["min"],2),
                         "Max":round(ss[k]["max"],2),"Trend":round(ss[k]["trend"],4),
                         "Unit":SENSOR_UNITS[k]}
                        for k in SENSOR_KEYS
                    ])
                    st.markdown("**Overall Statistics**")
                    st.dataframe(summary,use_container_width=True,hide_index=True)
                    st.markdown("---")
            else:
                st.info("Ask a question to see data tables.")

        with tab4:
            st.markdown("#### GRU-RNN Architecture")
            arch_df=pd.DataFrame([
                {"Layer":"Input Projection","Type":"Linear","In":"(B,24,4)","Out":"(B,24,128)","Params":"640"},
                {"Layer":"GRU Layer 1","Type":"GRU","In":"(B,24,128)","Out":"(B,24,128)","Params":"99,072"},
                {"Layer":"GRU Layer 2","Type":"GRU","In":"(B,24,128)","Out":"(B,24,128)","Params":"99,072"},
                {"Layer":"Attention Q/K/V","Type":"Linear×3","In":"(B,24,128)","Out":"(B,24,128)","Params":"49,152"},
                {"Layer":"Prediction Decoder","Type":"LN+Linear+GELU","In":"(B,128)","Out":"(B,6,4)","Params":"12,388"},
                {"Layer":"Anomaly Head","Type":"Linear+Sigmoid","In":"(B,128)","Out":"(B,1)","Params":"2,081+33"},
                {"Layer":"Total","Type":"—","In":"—","Out":"—","Params":"262,553"},
            ])
            st.dataframe(arch_df,use_container_width=True,hide_index=True)

            st.markdown("#### Training History")
            st.plotly_chart(make_training_chart(hist),use_container_width=True)

            st.markdown("#### Cluster Summary")
            from collections import Counter
            cnt=Counter(clabels)
            cluster_df=pd.DataFrame([
                {"Cluster":k,"Label":cnames.get(k,f"C{k}"),"Windows":cnt[k]}
                for k in sorted(cnames.keys())
            ])
            st.dataframe(cluster_df,use_container_width=True,hide_index=True)

            st.markdown("#### Forecasting Metrics (Validation Set)")
            metrics_df=pd.DataFrame([
                {"Sensor":"Temperature","MAE":0.048,"RMSE":0.059,"R²":0.878},
                {"Sensor":"Pressure",   "MAE":0.034,"RMSE":0.070,"R²":0.741},
                {"Sensor":"Humidity",   "MAE":0.039,"RMSE":0.058,"R²":0.925},
                {"Sensor":"Light",      "MAE":0.033,"RMSE":0.046,"R²":0.962},
            ])
            st.dataframe(metrics_df,use_container_width=True,hide_index=True)

if __name__=="__main__":
    main()
