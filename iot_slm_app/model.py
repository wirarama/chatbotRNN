"""
model.py — The SLM (Small Language Model) formed by the GRU-RNN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THIS IS THE FILE TO EDIT WHEN CHANGING THE SLM / RNN ITSELF.

What lives here:
  • TORCH_OK: whether torch is importable (the app still runs without it,
    just with forecasting disabled).
  • IoTSLM: the GRU-RNN network — projection -> stacked GRU -> self
    attention -> decoder head (multi-step forecast) + anomaly head.
    Its architecture MUST stay identical to whatever iot_train.py builds,
    because we load a trained state_dict into it (no training happens in
    the app — see the module docstring in the original iot_train.py).
  • forecast_window(): the single forward pass used at inference time —
    takes the last `seq_len` readings, scales them, runs the model, and
    inverse-transforms the prediction back to physical units.

What does NOT live here:
  • Loading pickles/JSON from disk (that's loader.py — it constructs an
    IoTSLM and hands it to the Engine).
  • Anything about chat, dates, or narration.

To change the RNN itself (add a layer, swap GRU for LSTM, change hidden
size, add a new prediction head, etc.), edit the IoTSLM class below —
just remember iot_train.py's model definition has to match, or the saved
weights (slm_weights.pt) won't load.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

from .config import SENSOR_KEYS


# ─────────────────────────────────────────────────────────────────────────────
#  GRU MODEL CLASS  (must match iot_train.py exactly for state_dict loading)
# ─────────────────────────────────────────────────────────────────────────────
if TORCH_OK:
    class IoTSLM(nn.Module):
        def __init__(self, input_size=4, hidden=128, n_layers=2,
                     pred_len=6, dropout=0.2):
            super().__init__()
            self.pred_len    = pred_len
            self.input_size  = input_size
            self.proj        = nn.Linear(input_size, hidden)
            self.gru         = nn.GRU(hidden, hidden, n_layers, batch_first=True,
                                      dropout=dropout if n_layers > 1 else 0.0)
            self.attn_q = nn.Linear(hidden, hidden)
            self.attn_k = nn.Linear(hidden, hidden)
            self.attn_v = nn.Linear(hidden, hidden)
            self.scale   = hidden ** 0.5
            self.decoder = nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden//2),
                nn.GELU(), nn.Dropout(dropout),
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
            w = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / self.scale, dim=-1)
            ctx  = torch.bmm(w, V).mean(dim=1)
            pred = self.decoder(ctx).view(-1, self.pred_len, self.input_size)
            return pred, self.anom_head(ctx), w
else:
    IoTSLM = None  # torch not installed — forecasting/anomaly-head disabled


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE HELPER  (no training — pure forward pass)
# ─────────────────────────────────────────────────────────────────────────────
def forecast_window(slm, slm_sc, data, data_idx, seq_len=24):
    """Run a single forward pass to get 6-step forecast + anomaly score."""
    if slm is None or not TORCH_OK:
        return None, 0.0
    if data_idx < seq_len or data_idx > len(data["temperature"]):
        return None, 0.0
    win = np.stack([data[k][data_idx - seq_len:data_idx]
                     for k in SENSOR_KEYS], axis=1)
    if win.shape[0] < seq_len:
        return None, 0.0
    sc  = slm_sc.transform(win)
    xt  = torch.tensor(sc[None], dtype=torch.float32)
    with torch.no_grad():
        pred, anom, _ = slm(xt)
    phys  = slm_sc.inverse_transform(pred.squeeze(0).numpy())
    score = float(anom.squeeze())
    return phys, score
