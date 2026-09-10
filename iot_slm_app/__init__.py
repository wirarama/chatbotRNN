"""
iot_slm_app — modular package backing iot_app_v3.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This package used to be a single ~1800-line file (iot_app_v3.py). It has
been split by responsibility so that each concern — the SLM/GRU model, the
analysis engine, the chat-query NLP, the charts, the notification-rule
engine — can be edited independently without scrolling through unrelated
code. See iot_slm_app/README.md for a one-paragraph description of every
file and where to make common changes (especially: modifying the SLM
architecture trained by iot_train.py lives entirely in model.py).
"""
