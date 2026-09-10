"""
loader.py — Reads iot_train.py's output directory and builds the Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What lives here:
  • load_models(model_dir): the single @st.cache_resource entry point that
    reads every artifact iot_train.py wrote (config.json, data.pkl,
    features.pkl, kmeans*.pkl, dt*.pkl, slm_scaler.pkl, slm_weights.pt,
    slm_history.pkl, eval_metrics.json), rebuilds an IoTSLM (model.py) and
    wraps everything in an Engine (engine.py).

This file exists separately from model.py/engine.py purely to avoid a
circular import: engine.py needs the SLM's forecast_window(), and
load_models() needs both IoTSLM and Engine, so the "glue" that depends on
both lives in its own module.

When to edit this file:
  • iot_train.py starts saving a new artifact -> add an `lp(...)`/`lj(...)`
    call here and thread it into the Engine/IoTSLM constructor.
"""

import json
import os
import pickle
from datetime import datetime

import streamlit as st

from .model import TORCH_OK, IoTSLM
from .engine import Engine

if TORCH_OK:
    import torch


@st.cache_resource(show_spinner="Loading pre-trained models from disk...")
def load_models(model_dir):
    """
    Load all saved artifacts from iot_train.py output directory.
    Returns (engine, config, hist, eval_metrics, load_report).
    """
    load_report = {}

    def lp(fname):
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found — run iot_train.py first")
        with open(path, "rb") as f:
            return pickle.load(f)

    def lj(fname):
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        with open(path) as f:
            return json.load(f)

    # Config
    cfg = lj("config.json")
    load_report["config"] = "✓"

    # Data
    data_raw = lp("data.pkl")
    # Restore datetime timestamps
    data = dict(data_raw)
    data["timestamps"] = [datetime.fromisoformat(s)
                           for s in data_raw["timestamps_str"]]
    load_report["data"] = "✓"

    # Features + metas
    X_feat, metas = lp("features.pkl")
    load_report["features"] = "✓"

    # K-Means
    km     = lp("kmeans.pkl")
    km_sc  = lp("kmeans_scaler.pkl")
    cl     = lp("cluster_labels.pkl")
    cn_raw = lj("cluster_names.json")
    cn     = {int(k): v for k, v in cn_raw.items()}
    load_report["kmeans"] = "✓"

    # Decision Tree
    dt    = lp("dt.pkl")
    dt_sc = lp("dt_scaler.pkl")
    load_report["decision_tree"] = "✓"

    # SLM scaler
    slm_sc = lp("slm_scaler.pkl")

    # GRU model
    slm = None
    if TORCH_OK:
        weights_path = os.path.join(model_dir, "slm_weights.pt")
        if os.path.exists(weights_path):
            slm = IoTSLM(
                input_size=cfg.get("n_sensors", 4),
                hidden    =cfg.get("hidden", 128),
                n_layers  =cfg.get("n_layers", 2),
                pred_len  =cfg.get("pred_len", 6),
                dropout   =cfg.get("dropout", 0.2),
            )
            slm.load_state_dict(
                torch.load(weights_path, map_location="cpu",
                           weights_only=True))
            slm.eval()
            load_report["slm_gru"] = "✓"
        else:
            load_report["slm_gru"] = "weights file not found"
    else:
        load_report["slm_gru"] = "⚠ torch not available — forecast disabled"

    # Training history
    hist = lp("slm_history.pkl")
    load_report["history"] = "✓"

    # Eval metrics
    eval_metrics = lj("eval_metrics.json")
    load_report["eval_metrics"] = "✓"

    # Build engine
    seq_len = cfg.get("seq_len", 24)
    engine  = Engine(data, X_feat, metas, cl, cn, slm, slm_sc,
                      km, km_sc, dt, dt_sc, seq_len)

    return engine, cfg, hist, eval_metrics, load_report
