"""
IoT-SLM  ·  Explainable AI (XAI) Extension
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
XAI Methods implemented:
  1. Gradient Saliency         — input × gradient importance
  2. Integrated Gradients      — path-integral attribution (IG)
  3. Attention Rollout         — temporal attention weight analysis
  4. SHAP-style Perturbation   — model-agnostic feature importance
  5. LIME-style Local Explain  — linear surrogate per sample
  6. Counterfactual Analysis   — minimal input change to flip anomaly
  7. Concept Attribution       — human-interpretable concept scores
  8. XAI NL Report             — natural language XAI narrative
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings, random, os
from datetime import datetime, timedelta
from collections import defaultdict
from copy import deepcopy

warnings.filterwarnings('ignore')
random.seed(42); np.random.seed(42); torch.manual_seed(42)

SENSOR_KEYS  = ['temperature', 'pressure', 'humidity', 'light']
SENSOR_SHORT = ['Temp', 'Press', 'Humid', 'Light']
SENSOR_UNITS = {'temperature':'°C','pressure':'hPa','humidity':'%RH','light':'lux'}

# 
#  RE-USE CORE FROM iot_slm_rnn  (import or inline mini-copies)
# 
def simulate_iot_data(days=60, interval_minutes=30):
    timestamps, temps, pressures, humidities, lights, anomaly_flags = [],[],[],[],[],[]
    start = datetime(2024, 1, 1)
    steps_per_day = (24*60)//interval_minutes
    total_steps   = days * steps_per_day
    for i in range(total_steps):
        ts   = start + timedelta(minutes=i*interval_minutes)
        hour = ts.hour + ts.minute/60
        seasonal = (i/total_steps)*3.0
        temp = 24+seasonal + 8*np.sin(np.pi*(hour-6)/12)*(hour>6) + np.random.normal(0,.5)
        temp += 0.75 if ts.weekday()>=5 else 0
        pressure = 1013.25+5*np.sin(2*np.pi*i/(steps_per_day*7))+np.random.normal(0,.3)
        humidity = np.clip(70-20*np.sin(np.pi*(hour-6)/12)*(hour>6)+np.random.normal(0,1.5),20,100)
        light    = max(0,1000*np.exp(-0.5*((hour-13)/3)**2)+np.random.normal(0,30)) if 6<=hour<=20 else max(0,np.random.normal(5,2))
        is_anom  = False
        if random.random()<0.02:
            k = random.choice(['temp_spike','pressure_drop','humidity_surge','light_flicker'])
            if k=='temp_spike':      temp     += random.uniform(8,15)
            elif k=='pressure_drop': pressure -= random.uniform(10,20)
            elif k=='humidity_surge':humidity  = min(100,humidity+random.uniform(20,30))
            else:                    light     = random.choice([0,1500])
            is_anom = True
        timestamps.append(ts); temps.append(round(temp,2)); pressures.append(round(pressure,2))
        humidities.append(round(humidity,2)); lights.append(round(light,2)); anomaly_flags.append(is_anom)
    return {'timestamps':timestamps,'temperature':np.array(temps),'pressure':np.array(pressures),
            'humidity':np.array(humidities),'light':np.array(lights),'anomaly':np.array(anomaly_flags,dtype=bool)}


class SensorDataset(Dataset):
    def __init__(self, X, y):
        self.X=torch.tensor(X,dtype=torch.float32); self.y=torch.tensor(y,dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.y[i]


def build_sequences(data, seq_len=24, pred_len=6):
    features = np.stack([data[k] for k in SENSOR_KEYS], axis=1)
    scaler   = MinMaxScaler()
    scaled   = scaler.fit_transform(features)
    X,y=[],[]
    for i in range(len(scaled)-seq_len-pred_len+1):
        X.append(scaled[i:i+seq_len]); y.append(scaled[i+seq_len:i+seq_len+pred_len])
    return np.array(X),np.array(y),scaler,scaled


class IoTSLM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, pred_len=6, dropout=0.2):
        super().__init__()
        self.hidden_size=hidden_size; self.pred_len=pred_len; self.input_size=input_size
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.encoder    = nn.GRU(hidden_size, hidden_size, num_layers=num_layers,
                                 batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.attn_q = nn.Linear(hidden_size,hidden_size)
        self.attn_k = nn.Linear(hidden_size,hidden_size)
        self.attn_v = nn.Linear(hidden_size,hidden_size)
        self.attn_scale = hidden_size**0.5
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size,hidden_size//2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size//2, pred_len*input_size))
        self.anomaly_head = nn.Sequential(nn.Linear(hidden_size,32),nn.ReLU(),nn.Linear(32,1),nn.Sigmoid())

    def attention(self,x):
        Q,K,V = self.attn_q(x),self.attn_k(x),self.attn_v(x)
        w = torch.softmax(torch.bmm(Q,K.transpose(1,2))/self.attn_scale, dim=-1)
        return torch.bmm(w,V), w

    def forward(self,x):
        h = self.input_proj(x)
        enc_out,_ = self.encoder(h)
        ctx,attn_w = self.attention(enc_out)
        pooled = ctx.mean(dim=1)
        pred   = self.decoder(pooled).view(-1,self.pred_len,self.input_size)
        return pred, self.anomaly_head(pooled), attn_w


def train_model(data, seq_len=24, pred_len=6, epochs=60, lr=1e-3):
    X,y,scaler,scaled = build_sequences(data,seq_len,pred_len)
    split = int(0.8*len(X))
    tr_ld = DataLoader(SensorDataset(X[:split],y[:split]), batch_size=64, shuffle=True)
    va_ld = DataLoader(SensorDataset(X[split:], y[split:]),  batch_size=64)
    model = IoTSLM(pred_len=pred_len)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse   = nn.MSELoss(); bce = nn.BCELoss()
    history = {'train':[],'val':[]}
    for ep in range(1,epochs+1):
        model.train(); tl=0
        for xb,yb in tr_ld:
            opt.zero_grad()
            p,a,_ = model(xb); loss_p=mse(p,yb)
            re = ((p-yb)**2).mean(dim=(1,2))
            lbl = (re>re.mean()+re.std()).float().unsqueeze(1)
            loss = loss_p+0.1*bce(a,lbl); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tl+=loss.item()
        sch.step()
        with torch.no_grad():
            vl=sum(mse(model(xb)[0],yb).item() for xb,yb in va_ld)/len(va_ld)
        history['train'].append(tl/len(tr_ld)); history['val'].append(vl)
        if ep%10==0: print(f"    Epoch {ep:3d}/{epochs} | Train:{tl/len(tr_ld):.5f} | Val:{vl:.5f}")
    return model, scaler, scaled, X, y, history


# 
#  XAI  MODULE
# 

class XAIExplainer:
    """
    Unified XAI explainer for IoT-SLM.
    All methods operate on a single input window tensor of shape (1, seq_len, n_features).
    """
    def __init__(self, model: IoTSLM, scaler: MinMaxScaler, feature_names=None):
        self.model = model
        self.model.eval()
        self.scaler = scaler
        self.feature_names = feature_names or SENSOR_SHORT
        self.n_features = len(self.feature_names)

    # ── helpers ───────────────────────────────────────────────
    def _pred_scalar(self, x_tensor, target_sensor=0, step=0):
        """Return scalar prediction for gradient computation."""
        pred, anom, _ = self.model(x_tensor)
        return pred[0, step, target_sensor]

    def _anomaly_score(self, x_tensor):
        _, anom, _ = self.model(x_tensor)
        return anom[0, 0]

    # ─────────────────────────────────────────────────────────
    #  1. GRADIENT SALIENCY
    # ─────────────────────────────────────────────────────────
    def gradient_saliency(self, x_np, target_sensor=0, step=0):
        """
        Saliency = |∂output/∂input|  — highlights which time-steps
        and features most directly change the prediction.
        """
        self.model.zero_grad()
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
        x.requires_grad_(True)
        # forward without no_grad
        h = self.model.input_proj(x)
        enc_out, _ = self.model.encoder(h)
        ctx, attn_w = self.model.attention(enc_out)
        pooled = ctx.mean(dim=1)
        pred = self.model.decoder(pooled).view(-1, self.model.pred_len, self.model.input_size)
        out = pred[0, step, target_sensor]
        out.backward()
        saliency = x.grad.squeeze(0).abs().detach().numpy()   # (seq_len, n_feat)
        return saliency

    # ─────────────────────────────────────────────────────────
    #  2. INTEGRATED GRADIENTS
    # ─────────────────────────────────────────────────────────
    def integrated_gradients(self, x_np, baseline=None, n_steps=50,
                              target_sensor=0, step=0):
        """
        IG = (x − baseline) × ∫₀¹ ∂f(baseline + α(x−baseline))/∂x dα
        Approximated with Riemann sum. Attribution satisfies completeness axiom.
        """
        if baseline is None:
            baseline = np.zeros_like(x_np)

        x_b   = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0)
        x_inp = torch.tensor(x_np,     dtype=torch.float32).unsqueeze(0)
        diff  = x_inp - x_b

        grad_sum = torch.zeros_like(x_inp)
        for i in range(1, n_steps+1):
            alpha  = i / n_steps
            interp = (x_b + alpha * diff).clone().detach().requires_grad_(True)
            self.model.zero_grad()
            h = self.model.input_proj(interp)
            enc_out, _ = self.model.encoder(h)
            ctx, _ = self.model.attention(enc_out)
            pooled = ctx.mean(dim=1)
            pred = self.model.decoder(pooled).view(-1, self.model.pred_len, self.model.input_size)
            out  = pred[0, step, target_sensor]
            out.backward()
            grad_sum += interp.grad.clone()

        ig = (diff * grad_sum / n_steps).squeeze(0).detach().numpy()
        return ig  # (seq_len, n_feat)

    # ─────────────────────────────────────────────────────────
    #  3. ATTENTION ROLLOUT
    # ─────────────────────────────────────────────────────────
    def attention_rollout(self, x_np):
        """
        Extract raw attention weight matrix from the self-attention layer.
        Returns per-timestep importance aggregated across the query dimension.
        Shape: (seq_len,)  — how much each past step was 'attended to'.
        """
        x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, _, attn_w = self.model(x)      # attn_w: (1, T, T)
        w = attn_w.squeeze(0).numpy()         # (T, T)
        # Add residual connection (identity) then re-normalize → rollout
        eye = np.eye(w.shape[0])
        rollout = 0.5 * w + 0.5 * eye
        rollout /= rollout.sum(axis=-1, keepdims=True)
        # Aggregate: mean query weight per key time-step
        temporal_importance = rollout.mean(axis=0)            # (T,)
        return temporal_importance, w

    # ─────────────────────────────────────────────────────────
    #  4. SHAP-STYLE PERMUTATION IMPORTANCE
    # ─────────────────────────────────────────────────────────
    def shap_permutation(self, x_np, n_repeats=30, target_sensor=0, step=0):
        """
        Model-agnostic: for each feature, randomly shuffle its values
        across the time axis and measure prediction degradation.
        Returns importance per feature: (n_features,)
        """
        x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            base_pred = self._pred_scalar(x_t, target_sensor, step).item()

        importance = np.zeros(self.n_features)
        for fi in range(self.n_features):
            deltas = []
            for _ in range(n_repeats):
                x_perm = x_np.copy()
                perm_idx = np.random.permutation(x_np.shape[0])
                x_perm[:, fi] = x_np[perm_idx, fi]
                x_pt = torch.tensor(x_perm, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    p = self._pred_scalar(x_pt, target_sensor, step).item()
                deltas.append(abs(base_pred - p))
            importance[fi] = np.mean(deltas)
        importance /= (importance.sum() + 1e-9)
        return importance                                      # (n_features,)

    # ─────────────────────────────────────────────────────────
    #  5. LIME-STYLE LOCAL SURROGATE
    # ─────────────────────────────────────────────────────────
    def lime_local(self, x_np, n_samples=200, sigma=0.15,
                   target_sensor=0, step=0):
        """
        Fit a linear Ridge surrogate in the neighbourhood of x_np.
        Returns per-feature linear coefficient as local importance.
        """
        seq_len, n_feat = x_np.shape
        perturbations   = np.random.normal(0, sigma, (n_samples, seq_len, n_feat))
        Xs  = x_np[None] + perturbations         # (N, T, F)
        dists = np.linalg.norm(perturbations.reshape(n_samples,-1), axis=1)
        weights = np.exp(-0.5 * (dists / dists.std())**2)

        ys = []
        for s in Xs:
            xt = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                ys.append(self._pred_scalar(xt, target_sensor, step).item())
        ys = np.array(ys)

        # Flatten time × feature for linear model
        Xf = Xs.reshape(n_samples, -1)
        xf = x_np.flatten()
        Xf_centered = Xf - xf[None]

        reg = Ridge(alpha=0.01)
        reg.fit(Xf_centered, ys, sample_weight=weights)
        coefs = reg.coef_.reshape(seq_len, n_feat)    # (T, F)
        feature_importance = np.abs(coefs).mean(axis=0)
        feature_importance /= (feature_importance.sum()+1e-9)
        return feature_importance, coefs              # (n_feat,), (T, F)

    # ─────────────────────────────────────────────────────────
    #  6. COUNTERFACTUAL ANALYSIS
    # ─────────────────────────────────────────────────────────
    def counterfactual(self, x_np, target_score=0.1, n_steps=200, lr=0.05):
        """
        Find the minimal perturbation δ such that anomaly_score(x+δ) ≈ target_score.
        Useful for: 'what minimal change would make this non-anomalous?'
        Returns the delta map (seq_len, n_feat) and final anomaly score.
        """
        x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
        delta = nn.Parameter(torch.zeros_like(x_t))
        opt   = optim.Adam([delta], lr=lr)
        mse   = nn.MSELoss()

        for step in range(n_steps):
            opt.zero_grad()
            x_cf  = torch.clamp(x_t + delta, 0, 1)
            score = self._anomaly_score(x_cf)
            loss  = mse(score, torch.tensor(target_score)) + 0.1*(delta**2).mean()
            loss.backward()
            opt.step()

        final_x   = torch.clamp(x_t + delta, 0, 1).detach()
        with torch.no_grad():
            final_score = self._anomaly_score(final_x).item()

        return delta.detach().squeeze(0).numpy(), final_score   # (T, F), scalar

    # ─────────────────────────────────────────────────────────
    #  7. CONCEPT ATTRIBUTION
    # ─────────────────────────────────────────────────────────
    def concept_attribution(self, x_np, scaler):
        """
        Map model input to human-interpretable concepts:
          - Thermal Stress   : high temperature deviation
          - Humidity Risk    : high humidity deviation
          - Pressure Anomaly : pressure deviation
          - Light Anomaly    : light deviation
          - Temporal Recency : how recent the anomalous steps are
          - Cross-sensor Corr: temp vs humidity anti-correlation break
        Returns dict {concept: score ∈ [0,1]}
        """
        # Inverse-scale for physical interpretation
        x_phys = scaler.inverse_transform(x_np)      # (T, 4)
        T_vals = x_phys[:, 0]   # temperature
        P_vals = x_phys[:, 1]   # pressure
        H_vals = x_phys[:, 2]   # humidity
        L_vals = x_phys[:, 3]   # light

        def zscore_max(arr):
            mu,s = arr.mean(), arr.std()+1e-6
            return float(np.clip(np.abs((arr-mu)/s).max()/3, 0, 1))

        # Cross-sensor: temp should be anti-correlated with humidity
        corr = np.corrcoef(T_vals, H_vals)[0,1]
        cross_break = float(np.clip((corr + 1)/2, 0, 1))  # ~0 when normal anti-corr

        # Temporal recency: weight anomaly by time position
        saliency = self.gradient_saliency(x_np, target_sensor=0)
        sal_per_step = saliency.mean(axis=1)
        weights    = np.linspace(0.5, 1.0, len(sal_per_step))
        recency    = float(np.clip((sal_per_step * weights).sum() / (sal_per_step.sum()+1e-6), 0, 1))

        return {
            'Thermal Stress'    : zscore_max(T_vals),
            'Humidity Risk'     : zscore_max(H_vals),
            'Pressure Anomaly'  : zscore_max(P_vals),
            'Light Anomaly'     : zscore_max(L_vals),
            'Temporal Recency'  : recency,
            'Cross-sensor Break': cross_break,
        }

    # ─────────────────────────────────────────────────────────
    #  8. XAI NARRATIVE (NL report section)
    # ─────────────────────────────────────────────────────────
    def xai_narrative(self, sample_idx, x_np, scaler,
                      ig_attr, shap_imp, lime_imp, concepts,
                      attn_importance, cf_score):
        """Generate a human-readable XAI explanation for one sample."""
        top_ig   = SENSOR_SHORT[np.argmax(np.abs(ig_attr).mean(axis=0))]
        top_shap = SENSOR_SHORT[np.argmax(shap_imp)]
        top_lime = SENSOR_SHORT[np.argmax(lime_imp)]
        peak_t   = int(np.argmax(attn_importance))
        top_conc = max(concepts, key=concepts.get)

        x_phys = scaler.inverse_transform(x_np)
        cur_temp = x_phys[-1, 0]; cur_hum = x_phys[-1, 2]

        lines = [
            f"┌─ XAI Explanation  [Sample #{sample_idx}] ─────────────────────────────",
            f"│  Current window  : Temp={cur_temp:.1f}°C  Humidity={cur_hum:.1f}%RH",
            f"│",
            f"│  [Integrated Gradients]  → '{top_ig}' has highest path-integral attribution.",
            f"│     The model's prediction is most sensitive to changes in {top_ig} values.",
            f"│",
            f"│  [Permutation Importance] → '{top_shap}' causes largest prediction shift",
            f"│     when randomly permuted, confirming its causal role in this window.",
            f"│",
            f"│  [LIME Surrogate]        → Local linear model ranks '{top_lime}' as",
            f"│     the dominant feature in the immediate neighbourhood of this sample.",
            f"│",
            f"│  [Attention Rollout]     → Timestep t-{len(attn_importance)-peak_t-1} (peak weight)",
            f"│     carries the highest temporal attention; the model 'focuses' on this",
            f"│     historical moment when generating its forecast.",
            f"│",
            f"│  [Concept Attribution]   → Dominant concept: '{top_conc}'",
            f"│     Score: {concepts[top_conc]:.3f}  (1=maximum concern, 0=nominal)",
            f"│",
            f"│  [Counterfactual]        → Anomaly score would drop to {cf_score:.3f}",
            f"│     with minimal input perturbation, indicating the decision boundary",
            f"│     is {'close' if cf_score<0.3 else 'far'} — anomaly is {'borderline' if cf_score<0.3 else 'clear'}.",
            f"└─────────────────────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


# 
#  XAI VISUALIZATION DASHBOARD
# 

def plot_xai_dashboard(explainer, x_samples, scaler, data, anomaly_indices):
    """
    Produce a comprehensive 6-row XAI dashboard.
    x_samples: list of (seq_len, n_feat) numpy arrays
    """
    DARK   = '#0F1117'
    PANEL  = '#1A1D2E'
    PANEL2 = '#1E2235'
    LIGHT  = '#E8EAF6'
    ACCENT = '#7C83FD'
    COLORS = ['#FF6B6B','#4ECDC4','#FFE66D','#95E1D3']
    HEAT_P = LinearSegmentedColormap.from_list('xai',['#0F1117','#2B2F5E','#7C83FD','#FF6B6B'])
    HEAT_N = LinearSegmentedColormap.from_list('xain',['#FF6B6B','#1A1D2E','#4ECDC4'])

    plt.rcParams.update({
        'figure.facecolor':DARK,'axes.facecolor':PANEL,'axes.edgecolor':'#2E3250',
        'grid.color':'#2E3250','text.color':LIGHT,'axes.labelcolor':LIGHT,
        'xtick.color':LIGHT,'ytick.color':LIGHT,'axes.titlecolor':LIGHT,
        'font.family':'monospace','axes.spines.top':False,'axes.spines.right':False
    })

    # ── pick one normal + one anomalous sample for detailed explanation
    x_normal = x_samples[10]      # representative normal window
    x_anom   = x_samples[anomaly_indices[0]] if len(anomaly_indices) else x_samples[50]
    print("  Computing XAI explanations (this may take ~30 sec) ...")

    # ── compute for normal sample
    print("    [1/7] Gradient saliency ...")
    sal_n = explainer.gradient_saliency(x_normal, target_sensor=0)
    sal_a = explainer.gradient_saliency(x_anom,   target_sensor=0)

    print("    [2/7] Integrated Gradients ...")
    ig_n  = explainer.integrated_gradients(x_normal, target_sensor=0)
    ig_a  = explainer.integrated_gradients(x_anom,   target_sensor=0)

    print("    [3/7] Attention rollout ...")
    attn_n, attn_mat_n = explainer.attention_rollout(x_normal)
    attn_a, attn_mat_a = explainer.attention_rollout(x_anom)

    print("    [4/7] SHAP permutation ...")
    shap_n = explainer.shap_permutation(x_normal, n_repeats=20, target_sensor=0)
    shap_a = explainer.shap_permutation(x_anom,   n_repeats=20, target_sensor=0)

    print("    [5/7] LIME local surrogate ...")
    lime_n, lime_coef_n = explainer.lime_local(x_normal, n_samples=150, target_sensor=0)
    lime_a, lime_coef_a = explainer.lime_local(x_anom,   n_samples=150, target_sensor=0)

    print("    [6/7] Counterfactual ...")
    cf_delta_a, cf_score_a = explainer.counterfactual(x_anom, target_score=0.1)

    print("    [7/7] Concept attribution ...")
    conc_n = explainer.concept_attribution(x_normal, scaler)
    conc_a = explainer.concept_attribution(x_anom,   scaler)

    # ── aggregate across many samples for global feature importance
    print("    [+] Global importance over 80 samples ...")
    global_shap = np.zeros(4)
    global_ig   = np.zeros(4)
    for xi in x_samples[:80]:
        try:
            gs = explainer.shap_permutation(xi, n_repeats=10, target_sensor=0)
            gi = np.abs(explainer.integrated_gradients(xi, target_sensor=0)).mean(axis=0)
            global_shap += gs
            global_ig   += gi / (gi.sum()+1e-9)
        except Exception:
            pass
    global_shap /= 80; global_ig /= 80

    #  FIGURE 
    fig = plt.figure(figsize=(24, 32), dpi=100)
    fig.patch.set_facecolor(DARK)

    fig.text(0.5, 0.993, 'IoT-SLM  ·  Explainable AI Dashboard',
             ha='center', va='top', fontsize=20, fontweight='bold', color=LIGHT)
    fig.text(0.5, 0.983, 'Gradient Saliency · Integrated Gradients · Attention Rollout · SHAP · LIME · Counterfactual · Concepts',
             ha='center', va='top', fontsize=9, color='#6870B0')

    outer = gridspec.GridSpec(6,1, figure=fig, top=0.978, bottom=0.022,
                              hspace=0.52, left=0.055, right=0.97)

    # ── ROW 0: Global importance comparison ──────────────────
    row0 = gridspec.GridSpecFromSubplotSpec(1,3, subplot_spec=outer[0], wspace=0.38)

    ax = fig.add_subplot(row0[0])
    x_pos = np.arange(4)
    ax.bar(x_pos-0.2, global_shap, 0.38, label='SHAP-Perm', color=COLORS[0], alpha=0.85)
    ax.bar(x_pos+0.2, global_ig,   0.38, label='Int.Grad',  color=COLORS[1], alpha=0.85)
    ax.set_xticks(x_pos); ax.set_xticklabels(SENSOR_SHORT, fontsize=9)
    ax.set_title('Global Feature Importance\n(avg over 80 samples)', fontsize=10, pad=5)
    ax.legend(fontsize=8); ax.grid(True,alpha=0.25,axis='y')
    ax.set_ylabel('Relative importance')

    ax2 = fig.add_subplot(row0[1])
    # Radar / spider via bar-in-polar
    angles  = np.linspace(0, 2*np.pi, 4, endpoint=False)
    w_n = [conc_n[k] for k in conc_n][:4]   # take first 4 concepts
    w_a = [conc_a[k] for k in conc_a][:4]
    cnames = list(conc_n.keys())[:4]
    ax2.bar(np.arange(4)-0.2, w_n, 0.38, color=COLORS[1], alpha=0.8, label='Normal')
    ax2.bar(np.arange(4)+0.2, w_a, 0.38, color=COLORS[0], alpha=0.8, label='Anomaly')
    ax2.set_xticks(np.arange(4)); ax2.set_xticklabels([c.replace(' ','\n') for c in cnames], fontsize=7)
    ax2.set_title('Concept Scores\nNormal vs Anomaly window', fontsize=10, pad=5)
    ax2.legend(fontsize=8); ax2.set_ylim(0,1.1); ax2.grid(True,alpha=0.25,axis='y')
    ax2.set_ylabel('Score [0–1]')

    ax3 = fig.add_subplot(row0[2])
    methods  = ['SHAP', 'IG', 'LIME']
    vals_n   = [shap_n, global_ig, lime_n]
    for mi,(method,vals) in enumerate(zip(methods,vals_n)):
        ax3.plot(SENSOR_SHORT, vals, 'o-', color=COLORS[mi], lw=1.8, ms=6, label=method)
    ax3.set_title('Method Agreement\n(Normal window)', fontsize=10, pad=5)
    ax3.legend(fontsize=8); ax3.grid(True,alpha=0.25); ax3.set_ylabel('Importance')

    # ── ROW 1: Gradient saliency heatmaps ────────────────────
    row1 = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=outer[1], wspace=0.25)

    for ax_i, (sal, title) in enumerate([(sal_n,'Normal Window'),(sal_a,'Anomaly Window')]):
        ax = fig.add_subplot(row1[ax_i])
        im = ax.imshow(sal.T, aspect='auto', cmap=HEAT_P,
                       vmin=0, vmax=sal.max())
        ax.set_yticks(range(4)); ax.set_yticklabels(SENSOR_SHORT, fontsize=9)
        ax.set_xlabel('Timestep (past →  now)'); ax.set_title(f'Gradient Saliency\n{title}', fontsize=10, pad=5)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='|∂output/∂input|')
        # Mark peak
        pk = np.unravel_index(sal.argmax(), sal.shape)
        ax.plot(pk[0], pk[1], 'w*', ms=12, zorder=5)

    # ── ROW 2: Integrated Gradients ──────────────────────────
    row2 = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=outer[2], wspace=0.25)

    for ax_i,(ig,title) in enumerate([(ig_n,'Normal'),(ig_a,'Anomaly')]):
        ax = fig.add_subplot(row2[ax_i])
        im = ax.imshow(ig.T, aspect='auto', cmap=HEAT_N,
                       vmin=-np.abs(ig).max(), vmax=np.abs(ig).max())
        ax.set_yticks(range(4)); ax.set_yticklabels(SENSOR_SHORT, fontsize=9)
        ax.set_xlabel('Timestep'); ax.set_title(f'Integrated Gradients\n{title} (blue=negative, red=positive)', fontsize=10, pad=5)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='Attribution')

    # ── ROW 3: Attention + LIME coef ─────────────────────────
    row3 = gridspec.GridSpecFromSubplotSpec(1,3, subplot_spec=outer[3], wspace=0.32)

    # Attention rollout - temporal curve
    ax = fig.add_subplot(row3[0])
    t_axis = np.arange(len(attn_n))
    ax.fill_between(t_axis, attn_n, color=COLORS[1], alpha=0.35)
    ax.plot(t_axis, attn_n, color=COLORS[1], lw=2, label='Normal')
    ax.fill_between(t_axis, attn_a, color=COLORS[0], alpha=0.35)
    ax.plot(t_axis, attn_a, color=COLORS[0], lw=2, label='Anomaly')
    ax.set_title('Attention Rollout\nTemporal importance', fontsize=10, pad=5)
    ax.set_xlabel('Timestep (0=oldest)'); ax.set_ylabel('Attention weight')
    ax.legend(fontsize=8); ax.grid(True,alpha=0.25)

    # Attention matrix heatmap (anomaly)
    ax = fig.add_subplot(row3[1])
    im = ax.imshow(attn_mat_a, aspect='auto', cmap='inferno', vmin=0)
    ax.set_title('Attention Matrix\n(Anomaly window)', fontsize=10, pad=5)
    ax.set_xlabel('Key timestep'); ax.set_ylabel('Query timestep')
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # LIME coef (anomaly)
    ax = fig.add_subplot(row3[2])
    lime_mean = lime_coef_a.mean(axis=0)   # (n_feat,)
    cols_lime = [COLORS[0] if v>0 else COLORS[1] for v in lime_mean]
    ax.barh(SENSOR_SHORT, lime_mean, color=cols_lime, alpha=0.85)
    ax.axvline(0, color='white', lw=0.8)
    ax.set_title('LIME Surrogate Coefficients\n(Anomaly window)', fontsize=10, pad=5)
    ax.set_xlabel('Avg coefficient (time avg)'); ax.grid(True,alpha=0.25,axis='x')

    # ── ROW 4: SHAP comparison + Counterfactual ───────────────
    row4 = gridspec.GridSpecFromSubplotSpec(1,3, subplot_spec=outer[4], wspace=0.35)

    ax = fig.add_subplot(row4[0])
    x_pos = np.arange(4)
    ax.bar(x_pos-0.2, shap_n, 0.38, color=COLORS[1], alpha=0.85, label='Normal')
    ax.bar(x_pos+0.2, shap_a, 0.38, color=COLORS[0], alpha=0.85, label='Anomaly')
    ax.set_xticks(x_pos); ax.set_xticklabels(SENSOR_SHORT)
    ax.set_title('SHAP Permutation Importance\nNormal vs Anomaly', fontsize=10, pad=5)
    ax.legend(fontsize=8); ax.set_ylabel('Importance (norm.)'); ax.grid(True,alpha=0.25,axis='y')

    # Counterfactual delta heatmap
    ax = fig.add_subplot(row4[1])
    im = ax.imshow(cf_delta_a.T, aspect='auto', cmap=HEAT_N,
                   vmin=-np.abs(cf_delta_a).max(), vmax=np.abs(cf_delta_a).max())
    ax.set_yticks(range(4)); ax.set_yticklabels(SENSOR_SHORT, fontsize=9)
    ax.set_xlabel('Timestep')
    ax.set_title(f'Counterfactual Δ\n(anomaly→normal, final score={cf_score_a:.3f})', fontsize=10, pad=5)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='Required Δ')

    # All concept scores side by side
    ax = fig.add_subplot(row4[2])
    conc_keys  = list(conc_n.keys())
    conc_vals_n = [conc_n[k] for k in conc_keys]
    conc_vals_a = [conc_a[k] for k in conc_keys]
    y_pos = np.arange(len(conc_keys))
    ax.barh(y_pos-0.2, conc_vals_n, 0.38, color=COLORS[1], alpha=0.8, label='Normal')
    ax.barh(y_pos+0.2, conc_vals_a, 0.38, color=COLORS[0], alpha=0.8, label='Anomaly')
    ax.set_yticks(y_pos); ax.set_yticklabels([c.replace(' ','\n') for c in conc_keys], fontsize=8)
    ax.set_title('All Concept Scores\nNormal vs Anomaly', fontsize=10, pad=5)
    ax.set_xlabel('Score [0–1]'); ax.legend(fontsize=8); ax.grid(True,alpha=0.25,axis='x')
    ax.set_xlim(0,1.15)

    # ── ROW 5: XAI narrative text box ────────────────────────
    row5 = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=outer[5])
    ax_txt = fig.add_subplot(row5[0])
    ax_txt.set_facecolor(PANEL2)
    ax_txt.axis('off')

    narrative = explainer.xai_narrative(
        sample_idx = anomaly_indices[0] if len(anomaly_indices) else 50,
        x_np       = x_anom,
        scaler     = scaler,
        ig_attr    = ig_a,
        shap_imp   = shap_a,
        lime_imp   = lime_a,
        concepts   = conc_a,
        attn_importance = attn_a,
        cf_score   = cf_score_a
    )
    ax_txt.text(0.01, 0.98, narrative, transform=ax_txt.transAxes,
                fontsize=8.5, va='top', ha='left', family='monospace',
                color='#C8CEF8', linespacing=1.55,
                bbox=dict(facecolor=PANEL2, edgecolor='#3A3F70', boxstyle='round,pad=0.5'))
    ax_txt.set_title('XAI Natural Language Explanation (Anomaly Sample)', fontsize=10, pad=8)

    out_path = 'iot_slm_xai_dashboard.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"  XAI dashboard saved → {out_path}")
    return out_path


#  XAI REPORT (text)

def generate_xai_report(explainer, x_samples, scaler, anomaly_indices):
    """Full XAI report covering all methods with global stats."""
    lines = []
    lines.append("=" * 72)
    lines.append("  IoT-SLM  ·  EXPLAINABLE AI (XAI) FULL REPORT")
    lines.append("  Methods: Grad-Saliency · IG · Attn-Rollout · SHAP · LIME · CF · Concepts")
    lines.append("=" * 72)

    lines.append("\n[XAI-1] GLOBAL FEATURE IMPORTANCE (avg over 50 samples)")
    lines.append("-" * 50)
    global_shap = np.zeros(4)
    for xi in x_samples[:50]:
        try:
            global_shap += explainer.shap_permutation(xi,n_repeats=8,target_sensor=0)
        except: pass
    global_shap /= 50
    for i,s in enumerate(SENSOR_SHORT):
        bar = '█' * int(global_shap[i]*40)
        lines.append(f"  {s:6s}  {bar:<40s}  {global_shap[i]:.4f}")

    lines.append("\n[XAI-2] ATTENTION ANALYSIS")
    lines.append("-" * 50)
    peak_steps = []
    for xi in x_samples[:30]:
        try:
            attn_i, _ = explainer.attention_rollout(xi)
            peak_steps.append(int(np.argmax(attn_i)))
        except: pass
    if peak_steps:
        avg_peak = np.mean(peak_steps)
        seq_len  = x_samples[0].shape[0]
        lines.append(f"  Average peak attention at timestep : {avg_peak:.1f} / {seq_len}")
        lines.append(f"  This corresponds to ~{(seq_len-avg_peak)*0.5:.1f}h before the forecast point.")
        lines.append(f"  Interpretation: model relies most on data from {(seq_len-avg_peak)*0.5:.1f}h ago.")

    lines.append("\n[XAI-3] INTEGRATED GRADIENTS — TOP CONTRIBUTORS")
    lines.append("-" * 50)
    for xi in x_samples[:5]:
        ig = explainer.integrated_gradients(xi, target_sensor=0)
        top = SENSOR_SHORT[np.argmax(np.abs(ig).mean(axis=0))]
        lines.append(f"  Sample → top IG feature: {top}")

    lines.append("\n[XAI-4] LIME LOCAL SURROGATE — NORMAL vs ANOMALY")
    lines.append("-" * 50)
    if len(x_samples)>10:
        ln,_ = explainer.lime_local(x_samples[10], n_samples=100, target_sensor=0)
        lines.append(f"  Normal  window top LIME feature: {SENSOR_SHORT[np.argmax(ln)]}")
    if len(anomaly_indices)>0:
        la,_ = explainer.lime_local(x_samples[anomaly_indices[0]], n_samples=100, target_sensor=0)
        lines.append(f"  Anomaly window top LIME feature: {SENSOR_SHORT[np.argmax(la)]}")

    lines.append("\n[XAI-5] COUNTERFACTUAL ANALYSIS")
    lines.append("-" * 50)
    if len(anomaly_indices)>0:
        _,cf_s = explainer.counterfactual(x_samples[anomaly_indices[0]], target_score=0.1)
        lines.append(f"  Anomaly sample counterfactual score  : {cf_s:.4f}")
        lines.append(f"  Interpretation: a minimal perturbation reduces anomaly score to {cf_s:.4f}.")
        lines.append(f"  The anomaly is {'borderline' if cf_s<0.25 else 'well-established'} — "
                     f"{'close to decision boundary' if cf_s<0.25 else 'requires significant change to explain away'}.")

    lines.append("\n[XAI-6] CONCEPT ATTRIBUTION SUMMARY")
    lines.append("-" * 50)
    for xi, label in [(x_samples[10],'Normal'), (x_samples[anomaly_indices[0]] if anomaly_indices else x_samples[50],'Anomaly')]:
        concs = explainer.concept_attribution(xi, scaler)
        lines.append(f"  {label} window:")
        for k,v in concs.items():
            bar = '▓' * int(v*20)
            lines.append(f"    {k:22s} {bar:<20s} {v:.3f}")

    lines.append("\n[XAI-7] MODEL TRANSPARENCY SUMMARY")
    lines.append("-" * 50)
    lines.append("  Architecture     : GRU (2-layer) + Temporal Self-Attention + Decoder")
    lines.append("  Explainability   : Gradient-based (Saliency, IG) + Perturbation (SHAP, LIME)")
    lines.append("  Local methods    : LIME surrogate, Counterfactual optimization")
    lines.append("  Global methods   : SHAP permutation aggregated, IG path-integral average")
    lines.append("  Attention        : Extractable from model forward pass (no hooks needed)")
    lines.append("  Concepts         : Physics-grounded, scaler-inverted interpretation")
    lines.append("  All XAI methods are post-hoc and model-agnostic compatible.")

    lines.append("\n" + "=" * 72)
    lines.append("  END OF XAI REPORT  |  Generated by IoT-SLM XAI Module v1.0")
    lines.append("=" * 72)
    return "\n".join(lines)

#  MAIN

def main():
    from sys import stdout, stderr
    stdout.reconfigure(encoding="utf-8")
    stderr.reconfigure(encoding="utf-8")
    print("\n" + "="*62)
    print("  IoT-SLM  ·  XAI Extension Pipeline")
    print("  BMP280 | AHT20 | BH1750")
    print("="*62)

    print("\n[1] Simulating IoT sensor data ...")
    data = simulate_iot_data(days=60, interval_minutes=30)
    print(f"    Readings: {len(data['timestamps'])}  |  True anomalies: {data['anomaly'].sum()}")

    print("\n[2] Training IoT-SLM (RNN-GRU) ...")
    SEQ_LEN, PRED_LEN = 24, 6
    model, scaler, scaled, X, y, history = train_model(
        data, seq_len=SEQ_LEN, pred_len=PRED_LEN, epochs=60)

    print("\n[3] Identifying anomalous windows in validation set ...")
    split = int(0.8 * len(X))
    X_val = X[split:]
    # For XAI we work on scaled windows
    all_windows = X           # (N, T, F)

    # Find which windows contain known anomaly points
    val_offset = split + SEQ_LEN
    anomaly_indices_val = []
    for wi in range(len(X_val)):
        global_idx = val_offset + wi
        if global_idx < len(data['anomaly']) and data['anomaly'][global_idx]:
            anomaly_indices_val.append(wi)

    # Use absolute indices into all_windows for XAI
    anom_abs = [split + i for i in anomaly_indices_val[:5]]
    print(f"    Found {len(anom_abs)} anomaly windows for XAI analysis")

    print("\n[4] Building XAI explainer ...")
    explainer = XAIExplainer(model, scaler)

    print("\n[5] Generating XAI dashboard ...")
    plot_xai_dashboard(explainer, list(all_windows), scaler, data, anom_abs)

    print("\n[6] Generating XAI text report ...")
    report = generate_xai_report(explainer, list(all_windows), scaler, anom_abs)
    print("\n" + report)

    report_path = 'iot_slm_xai_report.txt'
    with open(report_path,'w', encoding='utf-8') as f: f.write(report)
    print(f"\n  XAI report saved → {report_path}")

    print("\n" + "="*62)
    print("  XAI Pipeline complete.")
    print("="*62+"\n")


if __name__=='__main__':
    main()
