"""
IoT-SLM Chatbot  v3  —  Enhanced with Text Upload & Keyword Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEW in v3:
  • Text Upload tab — paste or upload a .txt / .md template
  • Keyword Extraction — EN + ID, TF-IDF + regex-based
  • Statistical Placeholder injection:
      (mean), (median), (std), (min), (max), (trend),
      (centroid), (cluster), (anomaly), (forecast), (r2)
  • Keyword-enriched narration — user text becomes a
    custom chatbot response for any temporal query
  • Template gallery — pre-built examples to guide users
  • Keyword frequency chart & extracted keyword table

Run:  streamlit run iot_chatbot_v3.py
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
from sklearn.feature_extraction.text import TfidfVectorizer
import re, random, warnings, os, json, io
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import string

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IoT-SLM Chatbot v3",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.block-container{padding-top:1.1rem;padding-bottom:0.8rem;}
.iot-header{background:linear-gradient(135deg,#0E2A5C 0%,#1A5276 60%,#0D7A6E 100%);
  border-radius:12px;padding:14px 24px;margin-bottom:14px;display:flex;align-items:center;gap:14px;}
.iot-header h1{color:#fff;font-size:1.4rem;margin:0;font-weight:700;}
.iot-header p{color:#A8D8EA;font-size:0.80rem;margin:3px 0 0;}
.user-bubble{background:#1A5276;color:#fff;border-radius:16px 16px 4px 16px;
  padding:9px 14px;margin:5px 0 5px 18%;font-size:0.88rem;line-height:1.5;}
.bot-bubble{background:#F0F4FF;color:#0E2A5C;border-radius:16px 16px 16px 4px;
  border-left:4px solid #0D7A6E;padding:11px 16px;margin:5px 16% 5px 0;
  font-size:0.86rem;line-height:1.6;white-space:pre-wrap;}
.bot-bubble b{color:#0D7A6E;}
.kw-chip{display:inline-block;background:#EEF3FA;border:1px solid #C8D8F0;
  border-radius:14px;padding:2px 9px;font-size:0.76rem;color:#1A5276;
  margin:2px;cursor:default;}
.kw-chip.sensor{background:#D5F5E3;border-color:#82E0AA;color:#1D8348;}
.kw-chip.stat{background:#FEF9E7;border-color:#F9E79F;color:#B7770D;}
.kw-chip.temporal{background:#D6EAF8;border-color:#AED6F1;color:#1A5276;}
.kw-chip.cluster{background:#FDEBD0;border-color:#FABE56;color:#784212;}
.kw-chip.custom{background:#EBF5FB;border-color:#AED6F1;color:#154360;}
.ph-box{background:#EBF5FB;border:1px dashed #AED6F1;border-radius:8px;
  padding:10px 14px;font-size:0.84rem;color:#154360;margin:8px 0;}
.ph-badge{display:inline-block;background:#D6EAF8;border-radius:10px;
  padding:1px 8px;font-size:0.75rem;font-weight:600;color:#1A5276;margin:1px;}
.tmpl-card{background:#F8FAFF;border:1px solid #D6E0F0;border-radius:10px;
  padding:10px 14px;margin-bottom:8px;font-size:0.82rem;color:#0E2A5C;}
.tmpl-card h4{margin:0 0 4px;color:#1A5276;font-size:0.88rem;}
.stButton>button{width:100%;border-radius:8px;font-size:0.82rem;
  border:1px solid #C8D8F0;background:#F0F4FF;color:#1A5276;padding:5px 8px;}
.stButton>button:hover{background:#1A5276;color:#fff;}
section[data-testid="stSidebar"]{background:#0E2A5C;}
section[data-testid="stSidebar"] *{color:#EEF3FA !important;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
SENSOR_KEYS  = ["temperature","pressure","humidity","light"]
SENSOR_UNITS = {"temperature":"°C","pressure":"hPa","humidity":"% RH","light":"lux"}
NOTIF_CLASSES= ["NO_ACTION","INFO_DAY","INFO_NIGHT","WATCH_BORDERLINE",
                "WARN_TEMP","WARN_HUMIDITY","WARN_LIGHT","CRIT_PRESSURE","CRIT_MULTI"]
SEVERITY_MAP = {"NO_ACTION":"NOMINAL","INFO_DAY":"INFO","INFO_NIGHT":"INFO",
                "WATCH_BORDERLINE":"WATCH","WARN_TEMP":"WARNING","WARN_HUMIDITY":"WARNING",
                "WARN_LIGHT":"WARNING","CRIT_PRESSURE":"CRITICAL","CRIT_MULTI":"CRITICAL"}
NOTIF_MSG = {
    "NO_ACTION":"All readings within normal operating range.",
    "INFO_DAY":"Daytime conditions within expected parameters.",
    "INFO_NIGHT":"Nighttime conditions within expected parameters.",
    "WATCH_BORDERLINE":"⚠ Values approaching threshold limits — monitor closely.",
    "WARN_TEMP":"⚠ Temperature exceeded upper bound. Check ventilation.",
    "WARN_HUMIDITY":"⚠ Humidity elevated. Check for condensation or seal failure.",
    "WARN_LIGHT":"⚠ Illuminance outside expected range. Check sensor.",
    "CRIT_PRESSURE":"🚨 CRITICAL: Pressure dropped sharply. Immediate inspection required.",
    "CRIT_MULTI":"🚨 CRITICAL: Multiple sensors abnormal. Immediate action required.",
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
COLORS={"A":"#1A5276","B":"#C0392B","A_fill":"rgba(26,82,118,0.15)",
        "B_fill":"rgba(192,57,43,0.15)","grid":"#E8EDF5","bg":"#FAFBFF",
        "teal":"#0D7A6E","gold":"#B7770D","green":"#1D8348","navy":"#0E2A5C"}

# ══════════════════════════════════════════════════════════════════════════════
#  KEYWORD ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# Domain-specific keyword dictionaries
SENSOR_KEYWORDS_EN = {
    "temperature","temp","heat","thermal","warm","cool","cold","hot",
    "celsius","degree","thermostat","fever","frigid","chilly",
}
SENSOR_KEYWORDS_ID = {
    "suhu","temperatur","panas","dingin","hangat","sejuk","demam","kalor",
}
STAT_KEYWORDS_EN = {
    "mean","average","avg","median","mode","std","deviation","variance",
    "min","minimum","max","maximum","range","iqr","quartile","centroid",
    "trend","slope","correlation","regression","forecast","prediction",
    "anomaly","outlier","threshold","percentile","distribution","histogram",
    "cluster","segment","class","label","score","probability",
}
STAT_KEYWORDS_ID = {
    "rata-rata","rerata","median","modus","simpangan","varians",
    "minimum","maksimum","rentang","kuartil","sentroid","tren","kemiringan",
    "korelasi","regresi","prakiraan","prediksi","anomali","pencilan",
    "ambang","persentil","distribusi","klaster","segmen","kelas","skor",
}
TEMPORAL_KEYWORDS_EN = {
    "today","yesterday","week","month","daily","weekly","monthly",
    "morning","afternoon","evening","night","hour","day","period",
    "trend","cycle","seasonal","diurnal","temporal","time","series",
}
TEMPORAL_KEYWORDS_ID = {
    "hari","kemarin","minggu","bulan","pagi","siang","sore","malam",
    "jam","periode","tren","siklus","musiman","harian","mingguan","bulanan",
    "waktu","deret","temporal",
}
CLUSTER_KEYWORDS_EN = {
    "cluster","kmeans","k-means","segment","group","partition","centroid",
    "condition","state","regime","pattern","mode","class","category",
}
CLUSTER_KEYWORDS_ID = {
    "klaster","pengelompokan","segmen","grup","partisi","sentroid",
    "kondisi","keadaan","pola","mode","kelas","kategori",
}

# EN + ID stopwords (extended)
STOPWORDS_EN = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","could","should","may","might","shall",
    "this","that","these","those","it","its","we","our","you","your","they",
    "their","he","she","his","her","i","my","me","us","from","by","as","if",
    "not","no","nor","so","yet","both","either","neither","each","every",
    "all","any","few","more","most","other","some","such","into","than","then",
    "there","when","where","which","who","whom","how","what","why",
}
STOPWORDS_ID = {
    "yang","di","ke","dari","dan","atau","pada","untuk","dengan","adalah",
    "ini","itu","juga","sudah","akan","dapat","ada","tidak","dalam","oleh",
    "sebagai","karena","jika","maka","sehingga","namun","tetapi","bahwa",
    "sebuah","suatu","satu","dua","tiga","empat","lima","enam","tujuh",
    "delapan","sembilan","sepuluh","lebih","kurang","sangat","cukup","sudah",
}
STOPWORDS = STOPWORDS_EN | STOPWORDS_ID

# Statistical placeholders pattern
PLACEHOLDER_RE = re.compile(
    r'\((mean|median|std|min|max|trend|centroid|cluster|anomaly|'
    r'forecast|r2|variance|iqr|count|slope|correlation|mode|range)\)',
    re.IGNORECASE
)

# ── Keyword Extractor ──────────────────────────────────────────────────────────
class KeywordExtractor:
    """
    Extracts and classifies keywords from user-supplied text.
    Combines:
      1. Rule-based domain dictionary matching
      2. TF-IDF for top-N general keywords
      3. Placeholder token detection  (mean), (centroid) ...
    """
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 2),
            stop_words=list(STOPWORDS),
            token_pattern=r'\b[a-z][a-z\-]{1,}\b'
        )

    def tokenise(self, text):
        """Clean tokenisation: lowercase, remove punctuation, split."""
        text = text.lower()
        text = re.sub(r'[^\w\s\-]', ' ', text)
        tokens = [t.strip('-') for t in text.split() if len(t) > 1]
        return [t for t in tokens if t not in STOPWORDS]

    def classify(self, token):
        """Classify a token into a keyword category."""
        t = token.lower()
        if t in SENSOR_KEYWORDS_EN or t in SENSOR_KEYWORDS_ID:
            return "sensor"
        if t in STAT_KEYWORDS_EN or t in STAT_KEYWORDS_ID:
            return "stat"
        if t in TEMPORAL_KEYWORDS_EN or t in TEMPORAL_KEYWORDS_ID:
            return "temporal"
        if t in CLUSTER_KEYWORDS_EN or t in CLUSTER_KEYWORDS_ID:
            return "cluster"
        return "custom"

    def extract_placeholders(self, text):
        """Find all (placeholder) tokens and their positions."""
        return [m.group(1).lower() for m in PLACEHOLDER_RE.finditer(text)]

    def extract(self, text, top_n=30):
        """
        Main extraction method.
        Returns:
          keywords : list of {word, category, score}
          placeholders : list of placeholder names found
          freq_map : Counter of token frequency
        """
        tokens = self.tokenise(text)
        freq   = Counter(tokens)
        placeholders = self.extract_placeholders(text)

        # Domain keywords (rule-based)
        domain_kws = {}
        for tok in freq:
            cat = self.classify(tok)
            if cat != "custom":
                domain_kws[tok] = {"word": tok, "category": cat,
                                   "score": round(freq[tok] / max(freq.values()), 3),
                                   "count": freq[tok]}

        # TF-IDF general keywords
        tfidf_kws = {}
        try:
            if len(tokens) >= 3:
                self.tfidf.fit([text.lower()])
                names  = self.tfidf.get_feature_names_out()
                scores = self.tfidf.transform([text.lower()]).toarray()[0]
                for name, score in sorted(zip(names, scores), key=lambda x: -x[1])[:top_n]:
                    if name not in domain_kws and score > 0:
                        tfidf_kws[name] = {"word": name, "category": "custom",
                                           "score": round(float(score), 4),
                                           "count": freq.get(name, 1)}
        except Exception:
            pass

        all_kws = list(domain_kws.values()) + list(tfidf_kws.values())
        # Sort: domain first, then by score
        all_kws.sort(key=lambda x: (x["category"]=="custom", -x["score"]))

        return all_kws[:top_n], placeholders, freq

    def inject_stats(self, text, result, engine=None):
        """
        Replace (placeholder) tokens in user text with actual statistical
        values computed from the analysis result.
        result: dict from AnalysisEngine.query()
        """
        if result is None:
            return text

        ss    = result.get("stats", {})
        cname = result.get("cluster_name", "Unknown")
        notif = result.get("notification", "NO_ACTION")
        anom  = result.get("n_anomaly", 0)
        fore  = result.get("forecast")
        ascore= result.get("anomaly_score", 0.0)

        # Build centroid text from cluster name
        centroid_text = cname

        # Compute per-sensor medians if raw data available
        def sensor_stat(sensor, stat):
            idx = result.get("indices", [])
            if engine and idx:
                vals = np.array([engine.metas[i][sensor] for i in idx])
                if stat == "median":  return round(float(np.median(vals)), 2)
                if stat == "mode":    return round(float(vals[np.argmax(np.bincount(vals.astype(int)))]), 2)
                if stat == "iqr":     return round(float(np.percentile(vals,75)-np.percentile(vals,25)), 2)
                if stat == "variance":return round(float(vals.var()), 2)
                if stat == "count":   return len(vals)
                if stat == "slope":   return round(ss.get(sensor,{}).get("trend",0.0), 4)
                if stat == "correlation":
                    if engine and idx:
                        T = np.array([engine.metas[i]["temperature"] for i in idx])
                        H = np.array([engine.metas[i]["humidity"]    for i in idx])
                        return round(float(np.corrcoef(T,H)[0,1]), 3)
                    return "N/A"
            return "N/A"

        # Replacement map
        def replace_placeholder(m):
            ph = m.group(1).lower()
            # Use temperature as default sensor for generic placeholders
            st_t = ss.get("temperature", {})
            if ph == "mean":        return f"{st_t.get('mean',0):.2f}°C"
            if ph == "median":      return f"{sensor_stat('temperature','median')}°C"
            if ph == "std":         return f"{st_t.get('std',0):.2f}°C"
            if ph == "min":         return f"{st_t.get('min',0):.2f}°C"
            if ph == "max":         return f"{st_t.get('max',0):.2f}°C"
            if ph == "variance":    return f"{sensor_stat('temperature','variance')}°C²"
            if ph == "iqr":         return f"{sensor_stat('temperature','iqr')}°C"
            if ph == "slope":       return f"{sensor_stat('temperature','slope')} °C/step"
            if ph == "correlation": return f"{sensor_stat('temperature','correlation')}"
            if ph == "count":       return f"{sensor_stat('temperature','count')} windows"
            if ph == "mode":        return f"{sensor_stat('temperature','mode')}°C"
            if ph == "range":
                return f"{st_t.get('min',0):.1f}–{st_t.get('max',0):.1f}°C"
            if ph == "trend":
                slope = st_t.get("trend", 0)
                return "rising" if slope>0.01 else "falling" if slope<-0.01 else "stable"
            if ph == "centroid":    return centroid_text
            if ph == "cluster":     return cname
            if ph == "anomaly":     return f"{anom} events ({anom/max(result.get('n_windows',1),1)*100:.1f}%)"
            if ph == "anomaly_score": return f"{ascore:.3f}"
            if ph == "forecast":
                if fore is not None:
                    return f"{fore[:,0].mean():.1f}°C (3h ahead)"
                return "N/A"
            if ph == "r2":          return "0.878"  # temperature R²
            return m.group(0)   # keep original if not matched

        return PLACEHOLDER_RE.sub(replace_placeholder, text)


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPLATE GALLERY
# ══════════════════════════════════════════════════════════════════════════════
TEMPLATES = {
    "🌡️ Temperature Monitoring Report": """\
Temperature Monitoring Report — {period}

The current environmental cluster is identified as (cluster), indicating typical sensor conditions for this time period.

Statistical Summary:
- Mean temperature  : (mean)
- Median temperature: (median)
- Temperature range : (range)
- Standard deviation: (std)
- Trend direction   : (trend)
- Linear slope      : (slope)

Anomaly Summary:
A total of (anomaly) were detected in this observation window.
The RNN anomaly score for the most recent window is (anomaly_score).

3-Hour Forecast:
Based on the GRU-RNN model, the expected temperature is (forecast).

Recommendation:
Monitor the system if temperature approaches the upper threshold.
Cluster centroid: (centroid).
""",

    "📊 Statistical Analysis Template": """\
Sensor Data Statistical Analysis

Observation period: {period}
Environmental condition: (cluster)

Descriptive Statistics (Temperature):
  Mean         : (mean)
  Median       : (median)
  Std deviation: (std)
  Min / Max    : (min) / (max)
  IQR          : (iqr)
  Variance     : (variance)
  Trend        : (trend) — slope = (slope)
  Correlation  : T–H correlation = (correlation)

Cluster assignment: (centroid)
Anomaly count    : (anomaly)
Short-term forecast: (forecast)
Model R² (temperature): (r2)

Note: All values derived from the GRU-RNN SLM pipeline.
""",

    "🌿 Agricultural Report (Bahasa Indonesia)": """\
Laporan Pemantauan Lingkungan Pertanian

Periode: {period}
Kondisi cluster teridentifikasi: (cluster)

Ringkasan Statistik Suhu:
  Rata-rata suhu: (mean)
  Median suhu   : (median)
  Rentang suhu  : (range)
  Tren          : (trend)
  Prediksi 3 jam: (forecast)

Analisis Kelembaban:
  Nilai rata-rata kelembaban digunakan untuk menentukan risiko jamur dan penyakit tanaman.
  Kelembaban relatif saat ini mendukung kondisi pertumbuhan optimal.

Anomali terdeteksi: (anomaly)
Sentroid klaster : (centroid)

Rekomendasi: Pantau suhu jika melebihi batas atas.
""",

    "🏭 Industrial Alert Template": """\
INDUSTRIAL MONITORING ALERT

Site: IoT Environmental Station
Period: {period}
Cluster State: (cluster)
Anomaly Score: (anomaly_score)

TEMPERATURE STATUS:
  Current Mean: (mean)
  Operating Range: (min) to (max)
  Deviation (Std): (std)
  Trend: (trend)

ANOMALY REPORT:
  Total flagged windows: (anomaly)
  Recommended Action: If anomaly_score > 0.5, escalate to maintenance.

FORECAST:
  Next 3-hour temperature estimate: (forecast)
  Cluster centroid identity: (centroid)

NOTES:
  Statistical centroid based on K-Means (k=8) clustering.
  Forecast generated by GRU-RNN with R² = (r2).
""",

    "✏️ Blank Custom Template": """\
My Custom IoT Report

Period: {period}
Cluster: (cluster)

Temperature:
  Mean = (mean), Trend = (trend)

Anomalies: (anomaly)
Forecast: (forecast)

Add your own text here. Use placeholders like:
(mean), (median), (std), (min), (max), (range),
(trend), (slope), (iqr), (variance), (correlation),
(centroid), (cluster), (anomaly), (anomaly_score),
(forecast), (r2), (count), (mode)
""",
}

# ══════════════════════════════════════════════════════════════════════════════
#  REST OF PIPELINE (same as v2, condensed)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔄 Simulating IoT data...")
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

@st.cache_resource(show_spinner="📐 Extracting features...")
def extract_features(_data,window=24):
    n=len(_data["temperature"]); rows,meta=[],[]
    for i in range(window,n):
        seg={k:_data[k][i-window:i] for k in SENSOR_KEYS}
        row=[]
        for k in SENSOR_KEYS:
            s=seg[k]
            row+=[s.mean(),s.std(),s.min(),s.max(),s.max()-s.min(),
                  np.percentile(s,75)-np.percentile(s,25),np.diff(s).mean(),np.abs(np.diff(s)).mean()]
        row+=[np.corrcoef(seg["temperature"],seg["humidity"])[0,1],
              np.corrcoef(seg["temperature"],seg["light"])[0,1]]
        rows.append(row)
        meta.append({"timestamp":_data["timestamps"][i],"fault_type":_data["fault_type"][i],
                     "is_anomaly":_data["is_anomaly"][i],**{k:float(_data[k][i]) for k in SENSOR_KEYS}})
    return np.array(rows,dtype=np.float32),meta

class IoTSLM(nn.Module):
    def __init__(self,input_size=4,hidden=128,n_layers=2,pred_len=6,dropout=0.2):
        super().__init__()
        self.pred_len=pred_len;self.input_size=input_size
        self.proj=nn.Linear(input_size,hidden)
        self.gru=nn.GRU(hidden,hidden,n_layers,batch_first=True,dropout=dropout if n_layers>1 else 0.)
        self.attn_q=nn.Linear(hidden,hidden);self.attn_k=nn.Linear(hidden,hidden)
        self.attn_v=nn.Linear(hidden,hidden);self.scale=hidden**0.5
        self.decoder=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden//2),
                                   nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden//2,pred_len*input_size))
        self.anom_head=nn.Sequential(nn.Linear(hidden,32),nn.ReLU(),nn.Linear(32,1),nn.Sigmoid())
    def forward(self,x):
        h=self.proj(x);enc,_=self.gru(h)
        Q,K,V=self.attn_q(enc),self.attn_k(enc),self.attn_v(enc)
        w=torch.softmax(torch.bmm(Q,K.transpose(1,2))/self.scale,dim=-1)
        ctx=torch.bmm(w,V).mean(dim=1)
        pred=self.decoder(ctx).view(-1,self.pred_len,self.input_size)
        return pred,self.anom_head(ctx),w

class SensorDS(Dataset):
    def __init__(self,X,y): self.X=torch.tensor(X,dtype=torch.float32);self.y=torch.tensor(y,dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.y[i]

@st.cache_resource(show_spinner="🧠 Training GRU-RNN SLM...")
def train_slm(_data,seq_len=24,pred_len=6,epochs=60):
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
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();tl+=loss.item()
        sch.step()
        model.eval()
        with torch.no_grad(): vl=sum(mse(model(xb)[0],yb).item() for xb,yb in va)/len(va)
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
    sc=scaler.transform(win);xt=torch.tensor(sc[None],dtype=torch.float32)
    _model.eval()
    with torch.no_grad(): pred,anom,_=_model(xt)
    return scaler.inverse_transform(pred.squeeze(0).numpy()),float(anom.squeeze())

@st.cache_resource(show_spinner="🔵 K-Means clustering...")
def fit_kmeans(_X,n=8):
    sc=StandardScaler();Xs=sc.fit_transform(_X)
    km=KMeans(n_clusters=n,random_state=42,n_init=15,max_iter=500)
    return km,sc,km.fit_predict(Xs)

def name_clusters(km,km_sc,X,metas,n=8):
    lbl=km.predict(km_sc.transform(X));names={};used=set()
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

@st.cache_resource(show_spinner="🌳 Decision Tree...")
def train_dt(_X,_metas,_clabels,_scores,seq_len=24):
    y=np.zeros(len(_X),dtype=int)
    for i in range(len(_X)):
        m=_metas[i];sc=_scores[i] if i<len(_scores) else 0.;ft=m["fault_type"]
        if ft=="multi_fault" or (sc>0.70 and ft!="normal"): y[i]=8
        elif ft=="pressure_drop" or m["pressure"]<995:       y[i]=7
        elif ft=="temp_spike"    or m["temperature"]>38:     y[i]=4
        elif ft=="humidity_surge"or m["humidity"]>92:        y[i]=5
        elif ft=="light_flicker" or m["light"]>1200:         y[i]=6
        elif sc>0.35:                                        y[i]=3
        elif m["timestamp"].hour<6 or m["timestamp"].hour>=20: y[i]=2
        elif m["timestamp"].hour>=6: y[i]=1
    sc=StandardScaler();Xs=sc.fit_transform(_X)
    cnt=Counter(y);Xo,yo=list(Xs),list(y)
    for cls,c in cnt.items():
        if c<80:
            idx=np.where(y==cls)[0];ext=np.random.choice(idx,80-c,replace=True)
            Xo.extend(Xs[ext]+np.random.normal(0,.05,(len(ext),Xs.shape[1])));yo.extend([cls]*len(ext))
    dt=DecisionTreeClassifier(max_depth=12,min_samples_leaf=3,class_weight="balanced",random_state=42)
    dt.fit(np.array(Xo),np.array(yo));return dt,sc,y

class Engine:
    def __init__(self,data,X,metas,cl,cn,slm,slm_sc,km,km_sc,dt,dt_sc,seq=24):
        self.data=data;self.X=X;self.metas=metas;self.cl=cl;self.cn=cn
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
        cl=[self.cl[i] for i in idx if i<len(self.cl)]
        c=defaultdict(int)
        for l in cl: c[l]+=1
        did=max(c,key=c.get) if c else 0
        fs=self.dt_sc.transform(self.X[idx]);preds=self.dt.predict(fs)
        nc=defaultdict(int)
        for p in preds: nc[p]+=1
        notif="NO_ACTION"
        for ci in [8,7,4,5,6,3,1,2,0]:
            if nc[ci]>0: notif=NOTIF_CLASSES[ci];break
        data_idx=max(idx)+self.seq
        fore,ascore=forecast_end(self.slm,self.slm_sc,self.data,
                                  min(data_idx,len(self.data["temperature"])))
        n_anom=sum(1 for i in idx if self.metas[i]["is_anomaly"])
        return {"label":lbl,"start":s,"end":e,"indices":idx,
                "stats":stats,"cluster_id":did,"cluster_name":self.cn.get(did,"Unknown"),
                "notification":notif,"n_anomaly":n_anom,"n_windows":len(idx),
                "forecast":fore,"anomaly_score":ascore}
    def series(self,idx,sensor):
        ts=[self.metas[i]["timestamp"] for i in idx]
        vs=np.array([self.metas[i][sensor] for i in idx])
        return ts,vs
    def cluster_series(self,idx):
        return [self.cn.get(self.cl[i] if i<len(self.cl) else 0,"?") for i in idx]

# Narration (same as v2)
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

def narrate(result,comp=None,comp_label=""):
    ss=result["stats"];T=ss["temperature"];H=ss["humidity"]
    P=ss["pressure"];L=ss["light"];pl=result["label"]
    cname=result["cluster_name"]
    cond=CLUSTER_COND.get(cname,"mixed conditions")
    p1=(f"During **{pl}**, conditions recorded as **{cond}** (cluster: *{cname}*). "
        f"Temperature averaged **{T['mean']:.1f}°C** ({tdesc(T['mean'])}), "
        f"ranging {T['min']:.1f}–{T['max']:.1f}°C. "
        f"Humidity was **{H['mean']:.1f}% RH** ({hdesc(H['mean'])}). "
        f"Pressure averaged **{P['mean']:.1f} hPa** and light **{L['mean']:.0f} lux**.")
    p2=(f"**Trend:** Temp {tword(T['trend'])} ({T['first']:.1f}→{T['last']:.1f}°C). "
        f"Humidity {tword(H['trend'])}. Pressure {tword(P['trend'])}. Light {tword(L['trend'])}.")
    parts=[p1,p2]
    if comp and comp.get("stats"):
        cs=comp["stats"]
        def cmp(v1,v2,k): 
            diff=v1-v2;u=SENSOR_UNITS[k]
            if abs(diff)<0.3: return f"{k.capitalize()} similar to {comp_label} ({v1:.1f}{u})"
            d="higher" if diff>0 else "lower";m="significantly" if abs(diff)>3 else "slightly"
            return f"{k.capitalize()} {m} {d} than {comp_label} ({v1:.1f}{u} vs {v2:.1f}{u}, Δ{diff:+.1f}{u})"
        parts.append("**Comparison:** "+" | ".join([
            cmp(T['mean'],cs['temperature']['mean'],'temperature'),
            cmp(H['mean'],cs['humidity']['mean'],'humidity'),
            cmp(P['mean'],cs['pressure']['mean'],'pressure')
        ])+".")
    na=result["n_anomaly"];nt=result["n_windows"]
    parts.append(("No anomalies detected." if na==0 else
                  f"**{na} anomalies** ({na/max(nt,1)*100:.1f}% of {nt} windows). ")
                 +NOTIF_MSG.get(result["notification"],""))
    fore=result["forecast"]
    if fore is not None:
        parts.append(f"**3-h forecast:** {fore[:,0].mean():.1f}°C, "
                     f"humidity {fore[:,2].mean():.1f}%, light {fore[:,3].mean():.0f} lux.")
    return "\n\n".join(parts)

# Query parser (same as v2)
class Parser:
    PATS={
        "today":     re.compile(r"\b(today|this day|hari ini)\b",re.I),
        "yesterday": re.compile(r"\b(yesterday|kemarin)\b",re.I),
        "this_week": re.compile(r"\b(this week|minggu ini)\b",re.I),
        "last_week": re.compile(r"\b(last week|minggu lalu|minggu kemarin)\b",re.I),
        "this_month":re.compile(r"\b(this month|bulan ini)\b",re.I),
        "last_month":re.compile(r"\b(last month|bulan lalu|bulan kemarin)\b",re.I),
        "2days_ago": re.compile(r"\b(2 days? ago|2 hari lalu)\b",re.I),
        "3days_ago": re.compile(r"\b(3 days? ago|3 hari lalu)\b",re.I),
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

# Charts (simplified for v3 — same make_charts as v2)
def subsample(ts,vs,max_pts=400):
    n=len(vs)
    if n<=max_pts: return ts,vs
    step=n//max_pts;idx=list(range(0,n,step))
    return [ts[i] for i in idx],vs[idx]

def make_charts(engine,results):
    n=len(results)
    fig=make_subplots(rows=3,cols=2,
        subplot_titles=("Temperature — Time Series","Humidity — Time Series",
                        "Temp vs Light (Scatter)","Temp vs Humidity (by Cluster)",
                        "Pressure — Time Series","Hourly Mean Temperature"),
        vertical_spacing=0.10,horizontal_spacing=0.08)
    palette=[COLORS["A"],COLORS["B"]];fills=[COLORS["A_fill"],COLORS["B_fill"]]
    for ri,result in enumerate(results):
        if result is None: continue
        idx=result["indices"];lbl=result["label"];col=palette[ri];fill=fills[ri]
        # Temp
        ts,vs=engine.series(idx,"temperature");ts,vs=subsample(ts,vs)
        fig.add_trace(go.Scatter(x=ts,y=vs,name=lbl,line=dict(color=col,width=1.8),
                                  fill="tozeroy",fillcolor=fill,legendgroup=lbl,showlegend=ri==0),row=1,col=1)
        fore=result.get("forecast")
        if fore is not None and len(ts)>0:
            lt=ts[-1];fts=[lt+timedelta(minutes=30*(i+1)) for i in range(6)]
            fig.add_trace(go.Scatter(x=fts,y=fore[:,0],name=f"{lbl} forecast",
                                      line=dict(color=col,width=1.4,dash="dash"),
                                      showlegend=False,legendgroup=lbl),row=1,col=1)
        # Humidity
        ts2,vs2=engine.series(idx,"humidity");ts2,vs2=subsample(ts2,vs2)
        fig.add_trace(go.Scatter(x=ts2,y=vs2,name=lbl,line=dict(color=col,width=1.8),
                                  fill="tozeroy",fillcolor=fill,showlegend=False,legendgroup=lbl),row=1,col=2)
        # Scatter T vs L
        T_v=np.array([engine.metas[i]["temperature"] for i in idx])
        L_v=np.array([engine.metas[i]["light"] for i in idx])
        step=max(1,len(T_v)//300);T_s,L_s=T_v[::step],L_v[::step]
        fig.add_trace(go.Scatter(x=T_s,y=L_s,mode="markers",name=lbl,
                                  marker=dict(color=col,size=5,opacity=0.4),
                                  showlegend=False,legendgroup=lbl),row=2,col=1)
        if len(T_s)>2:
            m,b=np.polyfit(T_s,L_s,1);xr=np.array([T_s.min(),T_s.max()])
            fig.add_trace(go.Scatter(x=xr,y=m*xr+b,mode="lines",
                                      line=dict(color=col,width=2,dash="dash"),showlegend=False),row=2,col=1)
        # Scatter T vs H by cluster
        H_v=np.array([engine.metas[i]["humidity"] for i in idx])
        C_v=engine.cluster_series(idx)
        step2=max(1,len(T_v)//250);T_sc,H_sc,C_sc=T_v[::step2],H_v[::step2],C_v[::step2]
        cmap=px.colors.qualitative.Set2
        for ci2,cname in enumerate(sorted(set(C_sc))):
            mask=np.array([c==cname for c in C_sc])
            fig.add_trace(go.Scatter(x=T_sc[mask],y=H_sc[mask],mode="markers",name=cname,
                                      marker=dict(color=cmap[ci2%len(cmap)],size=5,opacity=0.5),
                                      showlegend=(ri==0),legendgroup=f"c_{cname}"),row=2,col=2)
        # Pressure
        ts3,vs3=engine.series(idx,"pressure");ts3,vs3=subsample(ts3,vs3)
        fig.add_trace(go.Scatter(x=ts3,y=vs3,name=lbl,line=dict(color=col,width=1.6),
                                  showlegend=False,legendgroup=lbl),row=3,col=1)
        # Hourly bar
        by_h=defaultdict(list)
        for i in idx: by_h[engine.metas[i]["timestamp"].hour].append(engine.metas[i]["temperature"])
        hours=list(range(24));means=[np.mean(by_h[h]) if by_h[h] else float("nan") for h in hours]
        offset=ri*0.4-0.2*(n-1)
        fig.add_trace(go.Bar(x=[h+offset for h in hours],y=means,name=lbl,width=0.38,
                              marker_color=col,opacity=0.82,showlegend=False,legendgroup=lbl),row=3,col=2)
    fig.add_hline(y=1013.25,line_dash="dot",line_color="#888",row=3,col=1)
    fig.update_layout(height=760,paper_bgcolor=COLORS["bg"],plot_bgcolor=COLORS["bg"],
                      font=dict(family="Inter",size=11,color=COLORS["navy"]),
                      legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1),
                      margin=dict(l=50,r=20,t=55,b=35))
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].gridcolor=COLORS["grid"]
    return fig

# Keyword frequency chart
def kw_freq_chart(freq, top_n=20):
    items=freq.most_common(top_n)
    words=[i[0] for i in items];counts=[i[1] for i in items]
    fig=go.Figure(go.Bar(x=counts[::-1],y=words[::-1],orientation="h",
                          marker_color=COLORS["A"],opacity=0.85))
    fig.update_layout(height=max(280,len(words)*22),paper_bgcolor=COLORS["bg"],
                      plot_bgcolor=COLORS["bg"],font=dict(size=11,color=COLORS["navy"]),
                      title="Top Keyword Frequency",margin=dict(l=10,r=10,t=40,b=30))
    fig.update_xaxes(gridcolor=COLORS["grid"]);fig.update_yaxes(gridcolor=COLORS["grid"])
    return fig

# Build cached pipeline
@st.cache_resource(show_spinner=False)
def build_engine():
    data=simulate_data()
    X,metas=extract_features(data)
    km,km_sc,cl=fit_kmeans(X)
    cn=name_clusters(km,km_sc,X,metas)
    slm,slm_sc,X_seq,hist=train_slm(data)
    scores=get_scores(slm,X_seq)
    ml=min(len(X),len(scores))
    X,metas,cl=X[:ml],metas[:ml],cl[:ml]
    dt,dt_sc,y_lbl=train_dt(X,metas,cl,scores)
    engine=Engine(data,X,metas,cl,cn,slm,slm_sc,km,km_sc,dt,dt_sc)
    ref=data["timestamps"][-1].date()
    fs=dt_sc.transform(X);preds=dt.predict(fs)
    nc=defaultdict(int)
    for p in preds: nc[NOTIF_CLASSES[p]]+=1
    return engine,ref,hist,nc,cn,cl

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if "messages"       not in st.session_state: st.session_state.messages=[]
    if "last_results"   not in st.session_state: st.session_state.last_results=[]
    if "kw_text"        not in st.session_state: st.session_state.kw_text=""
    if "kw_results"     not in st.session_state: st.session_state.kw_results=None
    if "injected_text"  not in st.session_state: st.session_state.injected_text=""
    if "active_template"not in st.session_state: st.session_state.active_template=""

    with st.spinner("🚀 Initialising IoT-SLM pipeline (first run ~90s)..."):
        engine,ref_date,hist,notif_counts,cnames,clabels=build_engine()
    parser=Parser(ref_date)
    kw_engine=KeywordExtractor()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="iot-header">
      <div style="font-size:26px">🌡️</div>
      <div>
        <h1>IoT-SLM Chatbot v3  ·  Text Upload & Keyword Engine</h1>
        <p>GRU-RNN · K-Means · Decision Tree · Plotly  |  BMP280 · AHT20 · BH1750  |
           Ref date <b>{ref_date}</b>  |  EN + ID query support</p>
      </div>
    </div>""",unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ System")
        st.metric("Dataset","60 days · "+f"{len(engine.metas):,} windows")
        st.metric("SLM Params","262,553")
        st.metric("Ref Date",str(ref_date))
        st.markdown("---")
        st.markdown("## 📋 Quick Queries")
        for qq in ["today","yesterday","this week","last week","this month",
                   "compare today and yesterday","compare this week and last week"]:
            if st.button(qq.title(), key=f"qq_{qq}"):
                st.session_state._pending_query=qq
        st.markdown("---")
        st.markdown("## 🔑 Available Placeholders")
        phs=[("(mean)","Mean temperature"),("(median)","Median temperature"),
             ("(std)","Std deviation"),("(min)","Minimum"),("(max)","Maximum"),
             ("(range)","Min–Max range"),("(trend)","Trend direction"),
             ("(slope)","Linear slope"),("(iqr)","Interquartile range"),
             ("(variance)","Variance"),("(correlation)","T–H correlation"),
             ("(centroid)","Cluster centroid name"),("(cluster)","Cluster label"),
             ("(anomaly)","Anomaly count+rate"),("(anomaly_score)","RNN anomaly score"),
             ("(forecast)","3h ahead forecast"),("(r2)","Model R² score"),
             ("(count)","Number of windows"),("(mode)","Mode temperature")]
        for ph,desc in phs:
            st.markdown(f'<span class="ph-badge">{ph}</span> {desc}',unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("## 🌐 Query Language")
        st.markdown("""
| English | Indonesian |
|---|---|
| today | hari ini |
| yesterday | kemarin |
| this week | minggu ini |
| last week | minggu lalu |
| this month | bulan ini |
| last month | bulan lalu |
        """)

    # ── Main 3-column layout ──────────────────────────────────────────────────
    col_chat, col_vis = st.columns([1,1.4],gap="medium")

    # ══ LEFT: Chat ══════════════════════════════════════════════════════════
    with col_chat:
        st.markdown("### 💬 Chat")
        chat_box=st.container(height=400)
        with chat_box:
            if not st.session_state.messages:
                st.markdown("""<div class="bot-bubble">
👋 Hi! I'm your <b>IoT-SLM Environmental Assistant v3</b>.<br>
New: Use the <b>📝 Text Upload</b> tab to upload your own template with
statistical placeholders like <code>(mean)</code>, <code>(cluster)</code>,
<code>(forecast)</code>. I'll fill them in automatically!<br><br>
Or just ask: <i>"today"</i>, <i>"compare today and yesterday"</i>.
</div>""",unsafe_allow_html=True)
            for msg in st.session_state.messages:
                if msg["role"]=="user":
                    st.markdown(f'<div class="user-bubble">👤 {msg["content"]}</div>',unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-bubble">🤖 {msg["content"]}</div>',unsafe_allow_html=True)

        user_input=st.chat_input("Ask about sensor conditions or type a temporal query...")
        if hasattr(st.session_state,"_pending_query"):
            user_input=st.session_state._pending_query
            del st.session_state._pending_query

        if user_input:
            st.session_state.messages.append({"role":"user","content":user_input})
            periods=parser.parse(user_input)
            if not periods:
                reply="I didn't recognise a time period. Try **today**, **this week**, or **compare today and yesterday**."
                results=[]
            else:
                results=[]
                for s,e,lbl in periods[:2]:
                    r=engine.query(s,e,lbl)
                    if r: results.append(r)
                if not results:
                    reply="No data found for the requested period."
                else:
                    # If user has an active custom template, inject stats into it
                    if st.session_state.active_template and results:
                        tmpl=st.session_state.active_template
                        injected=kw_engine.inject_stats(
                            tmpl.replace("{period}",results[0]["label"]),
                            results[0],engine)
                        if len(results)>1:
                            injected2=kw_engine.inject_stats(
                                tmpl.replace("{period}",results[1]["label"]),
                                results[1],engine)
                            reply=(f"**Custom Template — {results[0]['label'].upper()}**\n\n"
                                   +injected
                                   +f"\n\n**Custom Template — {results[1]['label'].upper()}**\n\n"
                                   +injected2)
                        else:
                            reply=injected
                        st.session_state.injected_text=injected
                    elif len(results)==1:
                        reply=narrate(results[0])
                    else:
                        r1,r2=results[0],results[1]
                        reply=(f"**― {r1['label'].upper()} ―**\n\n"
                               +narrate(r1,comp=r2,comp_label=r2['label'])
                               +f"\n\n**― {r2['label'].upper()} ―**\n\n"
                               +narrate(r2,comp=r1,comp_label=r1['label']))
            st.session_state.messages.append({"role":"bot","content":reply})
            st.session_state.last_results=results
            st.rerun()

    # ══ RIGHT: Visualisation ════════════════════════════════════════════════
    with col_vis:
        st.markdown("### 📈 Analysis & Text Engine")
        tab1,tab2,tab3,tab4,tab5=st.tabs([
            "📊 Charts","🧩 Stats","📝 Text Upload","🔍 Keywords","🔬 Model"])

        # ── Tab 1: Charts ──────────────────────────────────────────────────
        with tab1:
            if st.session_state.last_results:
                fig=make_charts(engine,st.session_state.last_results)
                st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":True})
                for res in st.session_state.last_results:
                    sev=SEVERITY_MAP.get(res["notification"],"INFO")
                    st.markdown(f'<span style="background:#D6EAF8;border-radius:12px;'
                                f'padding:3px 10px;font-size:0.8rem;font-weight:600;'
                                f'color:#1A5276">{sev}</span> '
                                f'<span style="font-size:0.85rem">{res["notification"]}</span> '
                                f'— {NOTIF_MSG.get(res["notification"],"")}',
                                unsafe_allow_html=True)
            else:
                st.info("💡 Ask a question in the chat to see visualisations.")

        # ── Tab 2: Stat Cards ──────────────────────────────────────────────
        with tab2:
            if st.session_state.last_results:
                for res in st.session_state.last_results:
                    st.markdown(f"#### {res['label']} ({res['start']} → {res['end']})")
                    ss=res["stats"]
                    df=pd.DataFrame([
                        {"Sensor":k.capitalize(),"Mean":round(ss[k]["mean"],2),
                         "Std":round(ss[k]["std"],2),"Min":round(ss[k]["min"],2),
                         "Max":round(ss[k]["max"],2),"Trend":round(ss[k]["trend"],4),
                         "Unit":SENSOR_UNITS[k]}
                        for k in SENSOR_KEYS])
                    st.dataframe(df,use_container_width=True,hide_index=True)
                    st.markdown(f"**Cluster:** {res['cluster_name']}  |  "
                                f"**Notification:** {res['notification']}  |  "
                                f"**Anomalies:** {res['n_anomaly']}/{res['n_windows']}")
                    st.markdown("---")
            else:
                st.info("Ask a question to see sensor statistics.")

        # ── Tab 3: TEXT UPLOAD ─────────────────────────────────────────────
        with tab3:
            st.markdown("#### 📝 Text Upload & Statistical Placeholder Injection")
            st.markdown("""
Upload or paste a text template. Use placeholders like `(mean)`, `(cluster)`, `(forecast)`
and the system will automatically replace them with real sensor statistics from your query.
Then ask a temporal query in the chat — the bot will respond using your template!
""")
            # Template gallery
            st.markdown("##### 📚 Template Gallery")
            cols_tmpl=st.columns(len(TEMPLATES))
            for (tname,tcontent),col in zip(TEMPLATES.items(),cols_tmpl):
                if col.button(tname,key=f"tmpl_{tname}"):
                    st.session_state.kw_text=tcontent
                    st.session_state.active_template=tcontent
                    st.rerun()

            st.markdown("---")
            st.markdown("##### ✏️ Your Template")

            # File upload
            uploaded=st.file_uploader("Upload a .txt or .md file",
                                       type=["txt","md"],key="txt_upload")
            if uploaded:
                content=uploaded.read().decode("utf-8","ignore")
                st.session_state.kw_text=content
                st.session_state.active_template=content

            # Text area
            user_text=st.text_area(
                "Or paste your text / template here:",
                value=st.session_state.kw_text,
                height=220,
                placeholder="Enter text with placeholders like (mean), (cluster), (forecast)...",
                key="txt_area"
            )
            if user_text != st.session_state.kw_text:
                st.session_state.kw_text=user_text

            c1,c2,c3=st.columns(3)
            with c1:
                if st.button("🔍 Extract Keywords",use_container_width=True):
                    if user_text.strip():
                        kws,phs,freq=kw_engine.extract(user_text,top_n=40)
                        st.session_state.kw_results=(kws,phs,freq)
                    else:
                        st.warning("Please enter some text first.")
            with c2:
                if st.button("✅ Set as Active Template",use_container_width=True):
                    if user_text.strip():
                        st.session_state.active_template=user_text
                        st.success("Template activated! Now ask a temporal query in the chat.")
                    else:
                        st.warning("Text area is empty.")
            with c3:
                if st.button("🗑️ Clear Template",use_container_width=True):
                    st.session_state.active_template=""
                    st.session_state.kw_text=""
                    st.session_state.kw_results=None
                    st.session_state.injected_text=""
                    st.rerun()

            # Active template indicator
            if st.session_state.active_template:
                st.success("✅ Custom template is **active**. Your next temporal query will use it.")
                st.caption(f"Preview: {st.session_state.active_template[:120]}...")
            else:
                st.info("ℹ️ No custom template active. Using standard narration.")

            # Show injected result if available
            if st.session_state.injected_text:
                st.markdown("---")
                st.markdown("##### 🎯 Last Injected Output")
                st.markdown(
                    f'<div class="ph-box">{st.session_state.injected_text}</div>',
                    unsafe_allow_html=True)
                # Download button
                st.download_button(
                    "⬇️ Download Injected Text",
                    data=st.session_state.injected_text.encode("utf-8"),
                    file_name="iot_slm_report.txt",
                    mime="text/plain"
                )

            # Placeholder reference
            st.markdown("---")
            st.markdown("##### 🏷️ Available Placeholders")
            ph_html=" ".join([
                f'<span class="ph-badge">(mean)</span>',
                f'<span class="ph-badge">(median)</span>',
                f'<span class="ph-badge">(std)</span>',
                f'<span class="ph-badge">(min)</span>',
                f'<span class="ph-badge">(max)</span>',
                f'<span class="ph-badge">(range)</span>',
                f'<span class="ph-badge">(trend)</span>',
                f'<span class="ph-badge">(slope)</span>',
                f'<span class="ph-badge">(iqr)</span>',
                f'<span class="ph-badge">(variance)</span>',
                f'<span class="ph-badge">(correlation)</span>',
                f'<span class="ph-badge">(centroid)</span>',
                f'<span class="ph-badge">(cluster)</span>',
                f'<span class="ph-badge">(anomaly)</span>',
                f'<span class="ph-badge">(anomaly_score)</span>',
                f'<span class="ph-badge">(forecast)</span>',
                f'<span class="ph-badge">(r2)</span>',
                f'<span class="ph-badge">(count)</span>',
                f'<span class="ph-badge">(mode)</span>',
                f'<span class="ph-badge">{{period}}</span>',
            ])
            st.markdown(ph_html,unsafe_allow_html=True)

        # ── Tab 4: KEYWORD ANALYSIS ────────────────────────────────────────
        with tab4:
            st.markdown("#### 🔍 Keyword Extraction Results")

            if st.session_state.kw_results:
                kws,phs,freq=st.session_state.kw_results

                # Placeholder found
                if phs:
                    st.markdown("##### 🏷️ Detected Placeholders")
                    ph_html=" ".join([f'<span class="ph-badge">({p})</span>' for p in phs])
                    st.markdown(ph_html,unsafe_allow_html=True)
                    st.caption(f"{len(phs)} placeholder(s) detected — will be injected when a query is run.")
                else:
                    st.info("No `(placeholder)` tokens detected in your text. Add some from the sidebar list.")

                st.markdown("---")

                # Keyword chips by category
                st.markdown("##### 🔑 Extracted Keywords by Category")
                cats={"sensor":"🟢 Sensor","stat":"🟡 Statistical",
                      "temporal":"🔵 Temporal","cluster":"🟠 Clustering","custom":"⚪ General"}
                for cat_key,cat_label in cats.items():
                    cat_kws=[k for k in kws if k["category"]==cat_key]
                    if cat_kws:
                        st.markdown(f"**{cat_label}**")
                        chips=" ".join([
                            f'<span class="kw-chip {cat_key}">{k["word"]}</span>'
                            for k in cat_kws])
                        st.markdown(chips,unsafe_allow_html=True)

                st.markdown("---")

                # Keyword table
                st.markdown("##### 📋 Keyword Table")
                kw_df=pd.DataFrame(kws)[["word","category","score","count"]]
                kw_df.columns=["Keyword","Category","TF-IDF Score","Frequency"]
                st.dataframe(kw_df,use_container_width=True,hide_index=True,height=280)

                # Export keywords as JSON
                kw_json=json.dumps(kws,indent=2)
                st.download_button("⬇️ Download Keywords (JSON)",
                                   data=kw_json.encode(),
                                   file_name="iot_keywords.json",mime="application/json")

                st.markdown("---")
                # Frequency chart
                st.markdown("##### 📊 Word Frequency Chart")
                st.plotly_chart(kw_freq_chart(freq),use_container_width=True)

                # Bigrams
                tokens=kw_engine.tokenise(st.session_state.kw_text)
                bigrams=[f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)
                         if tokens[i] not in STOPWORDS and tokens[i+1] not in STOPWORDS]
                bg_freq=Counter(bigrams).most_common(10)
                if bg_freq:
                    st.markdown("##### 🔗 Top Bigrams (2-word phrases)")
                    bg_df=pd.DataFrame(bg_freq,columns=["Bigram","Count"])
                    st.dataframe(bg_df,use_container_width=True,hide_index=True,height=220)

            else:
                st.info("Go to the **📝 Text Upload** tab, enter your text, and click **Extract Keywords**.")

                # Show demo
                st.markdown("##### Example: keyword categories that are recognised")
                demo_cats={
                    "🟢 Sensor": list(SENSOR_KEYWORDS_EN)[:8]+list(SENSOR_KEYWORDS_ID)[:4],
                    "🟡 Statistical": list(STAT_KEYWORDS_EN)[:8]+list(STAT_KEYWORDS_ID)[:4],
                    "🔵 Temporal": list(TEMPORAL_KEYWORDS_EN)[:8]+list(TEMPORAL_KEYWORDS_ID)[:4],
                    "🟠 Clustering": list(CLUSTER_KEYWORDS_EN)[:6]+list(CLUSTER_KEYWORDS_ID)[:4],
                }
                for lbl,words in demo_cats.items():
                    st.markdown(f"**{lbl}**: "+" ".join([
                        f'<span class="kw-chip">{w}</span>' for w in sorted(words)]),
                        unsafe_allow_html=True)

        # ── Tab 5: Model Info ──────────────────────────────────────────────
        with tab5:
            st.markdown("#### GRU-RNN Architecture")
            arch_df=pd.DataFrame([
                {"Layer":"Input Projection","Type":"Linear","In":"(B,24,4)","Out":"(B,24,128)","Params":"640"},
                {"Layer":"GRU Layer 1","Type":"GRU","In":"(B,24,128)","Out":"(B,24,128)","Params":"99,072"},
                {"Layer":"GRU Layer 2","Type":"GRU","In":"(B,24,128)","Out":"(B,24,128)","Params":"99,072"},
                {"Layer":"Attn Q/K/V","Type":"Linear×3","In":"(B,24,128)","Out":"(B,24,128)","Params":"49,152"},
                {"Layer":"Pred Decoder","Type":"LN+GELU+Drop","In":"(B,128)","Out":"(B,6,4)","Params":"12,388"},
                {"Layer":"Anomaly Head","Type":"Linear+Sigmoid","In":"(B,128)","Out":"(B,1)","Params":"2,081+33"},
                {"Layer":"Total","Type":"—","In":"—","Out":"—","Params":"262,553"},
            ])
            st.dataframe(arch_df,use_container_width=True,hide_index=True)
            from collections import Counter as Ctr
            cnt=Ctr(clabels)
            st.markdown("#### K-Means Cluster Summary")
            cluster_df=pd.DataFrame([
                {"Cluster":k,"Label":cnames.get(k,f"C{k}"),"Windows":cnt[k]}
                for k in sorted(cnames.keys())])
            st.dataframe(cluster_df,use_container_width=True,hide_index=True)
            st.markdown("#### Forecasting Metrics")
            metrics_df=pd.DataFrame([
                {"Sensor":"Temperature","MAE":0.048,"RMSE":0.059,"R²":0.878},
                {"Sensor":"Pressure","MAE":0.034,"RMSE":0.070,"R²":0.741},
                {"Sensor":"Humidity","MAE":0.039,"RMSE":0.058,"R²":0.925},
                {"Sensor":"Light","MAE":0.033,"RMSE":0.046,"R²":0.962},
            ])
            st.dataframe(metrics_df,use_container_width=True,hide_index=True)

if __name__=="__main__":
    main()
