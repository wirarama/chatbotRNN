"""
iot_app_v3.py  —  IoT-SLM Streamlit App  (model-loading only, entry point)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Requires pre-trained model files from iot_train.py.
Does NOT train anything — pure inference + visualisation.

This file is now a thin orchestrator: page layout + the chat-routing
"if/elif" chain. Everything else lives in the iot_slm_app/ package next
to this file — see iot_slm_app/README.md for what each module does and
where to make common changes (in particular: the GRU/SLM architecture
itself lives entirely in iot_slm_app/model.py, isolated from everything
else, so you can edit it without touching chat parsing, charts, etc).

Usage:
  streamlit run iot_app_v3.py                        # looks for ./iot_model/
  streamlit run iot_app_v3.py -- --model /path/to/    # custom model dir

Dependencies (much lighter than training):
  pip install streamlit plotly numpy pandas scikit-learn torch
  (torch is only used for loading weights + forward pass, no training)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import warnings
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

from iot_slm_app.config import (
    configure_page, MODEL_DIR, SENSOR_KEYS, SENSOR_UNITS,
    DEFAULT_STREAM_DB, DEFAULT_POLL_SECONDS, DEFAULT_LIVE_CHART_POINTS,
)
from iot_slm_app.loader import load_models
from iot_slm_app.parser import Parser
from iot_slm_app.nlp_queries import (
    cluster_list_query, cluster_detail_query, rate_of_change_query, ranking_query,
)
from iot_slm_app.narration import narrate
from iot_slm_app.charts import (
    make_charts, training_chart, live_stream_chart, rate_of_change_chart,
)
from iot_slm_app.ui_components import stat_cards, notif_badge, hourly_table
from iot_slm_app import rules as rules_mod
from iot_slm_app import live_feed

# st.set_page_config(...) must be the very first Streamlit call.
configure_page()


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE STREAMING — one polling cycle + whatever it triggers
# ─────────────────────────────────────────────────────────────────────────────
def _run_live_ingest(engine, cfg, db_path):
    """
    Run one live_feed.poll_and_ingest() cycle. If anything new arrived,
    also re-check notification rules against it (same rules_mod.check_rules
    the offline-replay path already calls — edge-triggered, so calling it
    twice in one rerun never double-notifies) and flag that the rest of
    the page (chat panel, charts — everything outside the streaming
    fragment) should be refreshed with a full st.rerun().

    Returns (status_dict, should_rerun).
    """
    status = live_feed.poll_and_ingest(engine, db_path)
    should_rerun = False
    if status.get("connected") and status.get("n_new_raw", 0) > 0:
        newly_triggered, rules_changed = rules_mod.check_rules(
            st.session_state.rules, engine, cfg)
        if newly_triggered:
            for rule, values in newly_triggered:
                st.session_state.messages.append({
                    "role": "notif",
                    "content": rules_mod.notification_text(rule, values),
                })
        if rules_changed:
            rules_mod.save_rules(st.session_state.rules)
        should_rerun = True
    return status, should_rerun


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Session state
    for key, default in [("messages", []), ("last_results", [])]:
        if key not in st.session_state:
            st.session_state[key] = default
    if "rules" not in st.session_state:
        st.session_state.rules = rules_mod.load_rules()

    # ── Load models ───────────────────────────────────────────────────────
    load_ok = True
    load_err = ""
    try:
        engine, cfg, hist, eval_metrics, load_report = load_models(MODEL_DIR)
    except Exception as e:
        load_ok  = False
        load_err = str(e)

    if not load_ok:
        st.error(f"**Cannot load models from `{MODEL_DIR}`**\n\n"
                 f"Error: `{load_err}`\n\n"
                 "Run the training script first:\n"
                 "```\npython iot_train.py\n```")
        st.stop()

    ref_date = datetime.fromisoformat(cfg["ref_date"]).date()
    parser   = Parser(ref_date)

    # ── Live streaming (see iot_stream_sender.py + iot_slm_app/live_feed.py)
    #    Runs BEFORE the notification-rule check below so a reading that
    #    just arrived is visible to it in the same rerun. Disabled by
    #    default; toggled from the sidebar "Live Streaming" section further
    #    down — widget state is read here via session_state defaults since
    #    the widgets themselves are declared later in the script. ────────
    live_enabled  = st.session_state.get("live_enabled", False)
    live_db_path  = st.session_state.get("live_db_path", DEFAULT_STREAM_DB)
    live_poll_sec = st.session_state.get("live_poll_seconds", DEFAULT_POLL_SECONDS)
    live_status   = st.session_state.get("_live_status", {"connected": False})

    if live_enabled:
        if hasattr(st, "fragment"):
            try:
                @st.fragment(run_every=live_poll_sec)
                def _live_fragment():
                    status, should_rerun = _run_live_ingest(engine, cfg, live_db_path)
                    st.session_state["_live_status"] = status
                    if should_rerun:
                        st.rerun()
                _live_fragment()
                live_status = st.session_state.get("_live_status", live_status)
            except TypeError:
                # Installed Streamlit's st.fragment doesn't accept run_every
                # (older version) — fall back to a plain one-shot poll.
                live_status, _ = _run_live_ingest(engine, cfg, live_db_path)
                st.session_state["_live_status"] = live_status
        else:
            # No st.fragment at all (Streamlit < 1.33) — poll once per
            # normal rerun; use the sidebar's "Poll now" button to refresh.
            live_status, _ = _run_live_ingest(engine, cfg, live_db_path)
            st.session_state["_live_status"] = live_status

    # ── Notification-rule check (runs every rerun against the latest
    #    available reading — this is the same check the live-streaming
    #    block above already ran when it ingested new data; edge-triggered
    #    state means running it again here is harmless and keeps the
    #    offline-replay path — no streaming enabled — working exactly as
    #    before) ──────────────────────────────────────────────────────────
    newly_triggered, rules_changed = rules_mod.check_rules(
        st.session_state.rules, engine, cfg)
    if newly_triggered:
        for rule, values in newly_triggered:
            st.session_state.messages.append({
                "role": "notif",
                "content": rules_mod.notification_text(rule, values),
            })
        try:
            st.toast(f"{len(newly_triggered)} notification rule(s) triggered", icon="🔔")
        except Exception:
            pass  # st.toast requires a newer Streamlit; safe to skip
    if rules_changed:
        rules_mod.save_rules(st.session_state.rules)

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="iot-navbar">
      <span class="brand">IoT-SLM Environmental Chatbot</span>
      <span class="meta">
        Models: {MODEL_DIR} &nbsp;|&nbsp;
        GRU-RNN · K-Means · Decision Tree &nbsp;|&nbsp;
        Ref date: {ref_date} &nbsp;|&nbsp;
        {cfg.get('n_windows', 0):,} windows
      </span>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("#### Model Status")
        for k, v in load_report.items():
            css_cls = "status-ok" if v == "✓" else "status-err"
            label   = "OK" if v == "✓" else v
            st.markdown(
                f'<div class="{css_cls}">{k}: {label}</div>',
                unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Validation Metrics")
        for sensor, m in eval_metrics.items():
            st.markdown(f"**{sensor.capitalize()}** &nbsp; "
                        f"R² {m['r2']} &nbsp; MAE {m['mae']}",
                        unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Quick Queries")
        quick = [
            "today", "yesterday", "this week", "last week",
            "this month", "last month",
            "compare today and yesterday",
            "compare this week and last week",
            "clusters",
        ]
        for qq in quick:
            if st.button(qq.title(), key=f"qq_{qq}"):
                st.session_state._pending = qq

        st.markdown("---")
        st.markdown("#### Query Language")
        st.markdown("""
| English | Indonesian |
|---|---|
| today | hari ini |
| yesterday | kemarin |
| this week | minggu ini |
| last week | minggu lalu |
| this month | bulan ini |
| last month | bulan lalu |
| clusters | cluster apa saja |
| Hot Afternoon | (cluster name) |
        """)

        with st.expander("🔔 Notification rules — how to phrase them"):
            st.markdown("""
Create a rule straight from the chat box:

`buat rule jika suhu diatas 32C dan humidity diatas 70% maka notif saya`

- **diatas** / **above** — value must exceed a threshold
- **dibawah** / **below** — value must be under a threshold
- **antara X dan Y** / **between X and Y** — value must fall in a range
- **peningkatan** / **penurunan** before a sensor — a *rate of change*
  (rise/drop) over the last hour, e.g. *peningkatan suhu diatas 2C*
- join multiple conditions with **dan** (AND) or **atau** (OR)

Manage or delete rules in the **Rules** tab — deleting is not done
through chat.
            """)

        st.markdown("---")
        st.markdown("#### 📡 Live Streaming")
        st.checkbox(
            "Enable live streaming", key="live_enabled",
            value=st.session_state.get("live_enabled", False),
            help="Poll a shared SQLite DB written by a separate running "
                 "process (iot_stream_sender.py) and fold new readings "
                 "into this app as they arrive.")
        st.text_input(
            "Stream DB path", key="live_db_path",
            value=st.session_state.get("live_db_path", DEFAULT_STREAM_DB))
        st.number_input(
            "Poll interval (s)", key="live_poll_seconds", min_value=1, max_value=300,
            value=st.session_state.get("live_poll_seconds", DEFAULT_POLL_SECONDS))

        if st.button("🔄 Poll now", key="live_poll_now_btn"):
            _status, _ = _run_live_ingest(
                engine, cfg, st.session_state.get("live_db_path", DEFAULT_STREAM_DB))
            st.session_state["_live_status"] = _status
            st.rerun()

        _ls = st.session_state.get("_live_status", {"connected": False})
        if not _ls.get("connected"):
            st.caption("⚪ Not connected — start `python iot_stream_sender.py` "
                       "and enable streaming above.")
        else:
            last_ts = _ls.get("last_ts")
            st.markdown(
                f'<div class="status-ok">🟢 Connected — '
                f'{_ls.get("total_rows_seen", 0):,} readings ingested'
                f'{" · last: " + last_ts.strftime("%Y-%m-%d %H:%M") if last_ts else ""}'
                f'</div>', unsafe_allow_html=True)
            sender_interval = _ls.get("sender_interval_min")
            model_interval  = cfg.get("interval_min")
            if sender_interval is not None and model_interval is not None \
                    and abs(float(sender_interval) - float(model_interval)) > 1e-6:
                st.markdown(
                    f'<div class="status-err">⚠ Sender interval '
                    f'({sender_interval} min) doesn\'t match the trained '
                    f'model\'s ({model_interval} min) — rate-of-change '
                    f'rules will be slightly off. Restart the sender with '
                    f'--continue-model or matching --interval-min.</div>',
                    unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"**Model directory**  \n`{MODEL_DIR}`")
        st.markdown(f"Trained: {cfg.get('trained_at', '?')[:19]}")
        st.markdown(f"Days: {cfg.get('days')} &nbsp; Interval: {cfg.get('interval_min')} min",
                    unsafe_allow_html=True)

    # ── Layout ────────────────────────────────────────────────────────────
    col_chat, col_vis = st.columns([1, 1.4], gap="medium")

    # ══ Chat ════════════════════════════════════════════════════════════
    with col_chat:
        st.markdown("#### Conversation")
        chat_box = st.container(height=440)
        with chat_box:
            if not st.session_state.messages:
                st.markdown("""<div class="msg-bot">
<b>IoT-SLM Assistant</b><br>
Models loaded from disk. Ready for queries.<br><br>
Supported query types:<br>
<b>Temporal:</b> today, this week, compare today and yesterday<br>
<b>Ranking:</b> highest temperature this week<br>
<b>Rate:</b> fastest temperature increase this month<br>
<b>Clusters:</b> clusters, Hot Afternoon, Warm Day<br>
<b>Rules:</b> buat rule jika suhu diatas 32C dan humidity diatas 70% maka notif saya
</div>""", unsafe_allow_html=True)
            for msg in st.session_state.messages:
                role = msg["role"]
                css = {"user": "msg-user", "bot": "msg-bot", "notif": "msg-notif"}.get(role, "msg-bot")
                prefix = "You: " if role == "user" else ""
                st.markdown(
                    f'<div class="{css}">{prefix}{msg["content"]}</div>',
                    unsafe_allow_html=True)

        # Auto-scroll the chat box to the newest message whenever the
        # message list actually grows (new user input or bot reply) —
        # tracked via session_state so unrelated reruns (e.g. toggling a
        # sidebar option) don't keep fighting a manual scroll-up.
        _msg_count = len(st.session_state.messages)
        if st.session_state.get("_chat_msg_count") != _msg_count:
            st.session_state._chat_msg_count = _msg_count
            components.html(
                """
                <script>
                function findChatScrollable() {
                    const doc = window.parent.document;
                    const marker = doc.querySelector('.msg-user, .msg-bot, .msg-notif');
                    if (!marker) return null;
                    let el = marker.parentElement;
                    while (el) {
                        const style = window.getComputedStyle(el);
                        if ((style.overflowY === 'auto' || style.overflowY === 'scroll')
                                && el.scrollHeight > el.clientHeight) {
                            return el;
                        }
                        el = el.parentElement;
                    }
                    return null;
                }
                function scrollChatToBottom() {
                    const el = findChatScrollable();
                    if (el) { el.scrollTop = el.scrollHeight; }
                }
                setTimeout(scrollChatToBottom, 50);
                setTimeout(scrollChatToBottom, 250);
                setTimeout(scrollChatToBottom, 600);
                </script>
                """,
                height=0,
            )

        # Input
        user_input = st.chat_input("Enter a query, e.g.: today / highest temperature this week / buat rule ...")
        if hasattr(st.session_state, "_pending"):
            user_input = st.session_state._pending
            del st.session_state._pending

        if user_input:
            st.session_state.messages.append(
                {"role": "user", "content": user_input})

            # ── 0. Notification-rule creation / delete redirect ───────
            if rules_mod.is_rule_creation(user_input):
                conditions, logic = rules_mod.parse_rule_text(user_input)
                if not conditions:
                    reply = (
                        "I couldn't understand that rule. Try phrasing it like:\n\n"
                        "**buat rule jika suhu diatas 32C dan humidity diatas 70% maka notif saya**\n\n"
                        "Supported words: **diatas** / **dibawah** / **antara ... dan ...**, "
                        "and for rate-of-change: **peningkatan** / **penurunan** "
                        "(e.g. *peningkatan suhu diatas 2C*).")
                    results = []
                else:
                    rule = rules_mod.add_rule(
                        st.session_state.rules, user_input, conditions, logic)
                    rules_mod.save_rules(st.session_state.rules)
                    reply = (f"✅ Rule saved: **{rule['description']}**\n\n"
                             f"I'll post a notification in this chat whenever this "
                             f"condition becomes true. Manage or delete it in the "
                             f"**Rules** tab.")
                    results = []

            elif rules_mod.is_delete_request(user_input):
                reply = ("To remove a rule, open the **Rules** tab and click "
                          "**Delete** next to it — deletion isn't done from chat.")
                results = []

            else:
                # ── 1. Cluster query (list all or detail one) ─────────────
                cl_qtype, cl_name = parser.parse_cluster(user_input, engine.cn)
                if cl_qtype == "list":
                    reply   = cluster_list_query(engine)
                    results = []

                elif cl_qtype == "detail" and cl_name:
                    reply   = cluster_detail_query(engine, cl_name)
                    results = []

                # ── 2. Rate-of-change query ───────────────────────────────
                elif (lambda s, c, r: s is not None)(
                        *parser.parse_rate(user_input)):
                    r_sensor, r_change, r_rate = parser.parse_rate(user_input)
                    r_periods2 = parser.parse(user_input)
                    if r_periods2:
                        s2, e2, lbl2 = r_periods2[0]
                        base2 = engine.query(s2, e2, lbl2)
                        if base2:
                            reply = rate_of_change_query(
                                engine, base2["indices"],
                                r_sensor, r_change, r_rate, lbl2, top_n=5)
                            results = [base2]
                        else:
                            reply = "No data found for the requested period."; results = []
                    else:
                        reply = ("Please specify a time period. Example: "
                                 "**fastest temperature increase this week**, "
                                 "**slowest humidity decrease this month**."); results = []

                # ── 3. Ranking query (highest/lowest value) ───────────────
                elif (lambda s, d, p: s is not None)(
                        *parser.parse_ranking(user_input)):
                    sensor, direction, r_periods = parser.parse_ranking(user_input)
                    if r_periods:
                        s, e, lbl = r_periods[0]
                        base = engine.query(s, e, lbl)
                        if base:
                            rk = ranking_query(engine, base["indices"],
                                               sensor, direction, lbl, top_n=5)
                            reply   = rk["narrative"] if rk else "No data found."
                            results = [base]
                        else:
                            reply   = "No data found for the requested period."
                            results = []
                    else:
                        reply = ("Please specify a time period. Example: "
                                 "**highest temperature this week**, "
                                 "**lowest humidity last month**."); results = []

                else:
                    # ── 4. Standard temporal query ────────────────────────
                    periods = parser.parse(user_input)
                    if not periods:
                        reply   = (
                            "I didn't recognise the query. Here are examples:\n\n"
                            "**Temporal:** today · this week · compare today and yesterday\n"
                            "**Ranking:** highest temperature this week · suhu tertinggi hari ini\n"
                            "**Rate:** fastest temperature increase this week · "
                            "slowest humidity decrease this month\n"
                            "**Clusters:** clusters · what clusters exist · "
                            "Hot Afternoon · Warm Day\n"
                            "**Rules:** buat rule jika suhu diatas 32C dan humidity diatas 70% maka notif saya")
                        results = []
                    else:
                        results = []
                        for s, e, lbl in periods[:2]:
                            r = engine.query(s, e, lbl)
                            if r: results.append(r)

                        if not results:
                            reply = "No data found for the requested period."
                        elif len(results) == 1:
                            reply = narrate(results[0])
                        else:
                            r1, r2 = results[0], results[1]
                            reply = (f"**― {r1['label'].upper()} ―**\n\n"
                                     + narrate(r1, comp=r2, comp_label=r2["label"])
                                     + f"\n\n**― {r2['label'].upper()} ―**\n\n"
                                     + narrate(r2, comp=r1, comp_label=r1["label"]))

            st.session_state.messages.append(
                {"role": "bot", "content": reply})
            st.session_state.last_results = results
            st.rerun()

    # ══ Visualisation ════════════════════════════════════════════════════
    with col_vis:
        st.markdown("#### Sensor Analysis")
        t1, t_live, t2, t3, t4, t5 = st.tabs(
            ["Charts", "📡 Live Stream", "Statistics", "Data Tables", "Model Info", "Rules"])

        # Charts
        with t1:
            if st.session_state.last_results:
                fig = make_charts(engine, st.session_state.last_results)
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": True},
                                key="chart_main")
                for res in st.session_state.last_results:
                    notif_badge(res)
            else:
                st.info("Submit a query in the chat panel to generate charts.")
                st.plotly_chart(training_chart(hist),
                                use_container_width=True,
                                key="training_chart_tab1")

        # Live Stream — per-sensor LINE chart over only the live-streamed
        # data (see iot_slm_app/live_feed.py's engine._live_base_len),
        # with a horizontal threshold line per "level" condition of every
        # enabled notification rule touching that sensor. Rate-of-change
        # ("peningkatan"/"penurunan") conditions get their own separate
        # BAR chart below — a rate threshold isn't a level on the raw
        # value, so it never belongs on this line chart (see charts.py's
        # module docstring).
        with t_live:
            st.markdown("##### Pembacaan Sensor")
            fig_live = live_stream_chart(
                engine, st.session_state.rules, max_points=DEFAULT_LIVE_CHART_POINTS)
            if fig_live is None:
                st.info(
                    "Belum ada data live. Nyalakan **Live Streaming** di sidebar, lalu "
                    "jalankan `iot_stream_sender.py` atau `iot_stream_form.py` di "
                    "terminal terpisah untuk mulai mengirim pembacaan.")
            else:
                st.plotly_chart(fig_live, use_container_width=True,
                                config={"displayModeBar": True},
                                key="live_stream_chart")
                st.caption(
                    f"Garis putus-putus/titik-titik = ambang **level** tiap rule "
                    f"notifikasi yang **enabled**. × merah = pembacaan bertanda "
                    f"anomali. Menampilkan hingga {DEFAULT_LIVE_CHART_POINTS} "
                    f"pembacaan live terbaru per sensor.")

                st.markdown("##### Laju Perubahan (Rate of Change)")
                fig_rate = rate_of_change_chart(
                    engine, st.session_state.rules, cfg,
                    max_points=DEFAULT_LIVE_CHART_POINTS)
                if fig_rate is None:
                    st.info(
                        "Belum cukup riwayat live untuk menghitung laju perubahan "
                        "(perlu data live sepanjang minimal satu jendela lookback "
                        "rule rate-of-change, biasanya 60 menit).")
                else:
                    st.plotly_chart(fig_rate, use_container_width=True,
                                    config={"displayModeBar": True},
                                    key="rate_of_change_chart")
                    st.caption(
                        "Batang = perubahan nilai dibanding ~60 menit sebelumnya "
                        "(naik = warna panas, turun = warna dingin). Garis putus-putus "
                        "= ambang tiap rule **peningkatan/penurunan** yang **enabled**.")

        # Stat Cards
        with t2:
            if st.session_state.last_results:
                for res in st.session_state.last_results:
                    st.markdown(f"**{res['label']}** &nbsp; {res['start']} to {res['end']}", unsafe_allow_html=True)
                    stat_cards(res)
                    notif_badge(res)
                    st.markdown("---")
            else:
                st.info("Submit a query in the chat panel to see statistics.")

        # Data Tables
        with t3:
            if st.session_state.last_results:
                for res in st.session_state.last_results:
                    st.markdown(f"**{res['label']}**")
                    sensor_t = st.selectbox(
                        "Sensor for hourly table", SENSOR_KEYS,
                        key=f"sel_{res['label']}",
                        format_func=str.capitalize)
                    df = hourly_table(engine, res, sensor_t)
                    if df is not None:
                        st.dataframe(df, use_container_width=True,
                                     height=300, hide_index=True,
                                     key=f"hourly_df_{res['label']}_{sensor_t}")
                    # Full stats with median + IQR
                    ss = res["stats"]
                    sumdf = pd.DataFrame([
                        {"Sensor":  k.capitalize(),
                         "Mean":    round(ss[k]["mean"], 2),
                         "Median":  round(ss[k]["median"], 2),
                         "Std":     round(ss[k]["std"], 2),
                         "IQR":     round(ss[k]["iqr"], 2),
                         "Min":     round(ss[k]["min"], 2),
                         "Max":     round(ss[k]["max"], 2),
                         "Trend":   round(ss[k]["trend"], 4),
                         "Unit":    SENSOR_UNITS[k]}
                        for k in SENSOR_KEYS])
                    st.markdown("**Overall Statistics**")
                    st.dataframe(sumdf, use_container_width=True,
                                 hide_index=True,
                                 key=f"sumdf_{res['label']}")
                    st.markdown("---")
            else:
                st.info("Submit a query in the chat panel to see data tables.")

        # Model Info
        with t4:
            st.markdown("**Training Convergence**")
            st.plotly_chart(training_chart(hist),
                            use_container_width=True,
                            key="training_chart_tab4")

            st.markdown("**Forecasting Metrics (Validation Set)**")
            mdf = pd.DataFrame([
                {"Sensor":    sensor.capitalize(),
                 "MAE":       m["mae"],
                 "RMSE":      m["rmse"],
                 "R²":        m["r2"]}
                for sensor, m in eval_metrics.items()
            ])
            st.dataframe(mdf, use_container_width=True, hide_index=True,
                        key="mdf_metrics")

            st.markdown("**K-Means Cluster Summary**")
            from collections import Counter
            cl_arr = engine.cl
            cnt    = Counter(cl_arr.tolist())
            cldf   = pd.DataFrame([
                {"Cluster": k, "Label": engine.cn.get(k, f"C{k}"),
                 "Windows": cnt.get(k, 0)}
                for k in sorted(engine.cn.keys())
            ])
            st.dataframe(cldf, use_container_width=True, hide_index=True,
                        key="cldf_clusters")

            st.markdown("**GRU-RNN Architecture**")
            arch = pd.DataFrame([
                {"Layer":"Input Projection","In":"(B,24,4)","Out":"(B,24,128)","Params":"640"},
                {"Layer":"GRU Layer 1","In":"(B,24,128)","Out":"(B,24,128)","Params":"99,072"},
                {"Layer":"GRU Layer 2","In":"(B,24,128)","Out":"(B,24,128)","Params":"99,072"},
                {"Layer":"Attention Q/K/V","In":"(B,24,128)","Out":"(B,24,128)","Params":"49,152"},
                {"Layer":"Pred Decoder","In":"(B,128)","Out":"(B,6,4)","Params":"12,388"},
                {"Layer":"Anomaly Head","In":"(B,128)","Out":"(B,1)","Params":"2,114"},
                {"Layer":"Total","In":"—","Out":"—","Params":"262,553"},
            ])
            st.dataframe(arch, use_container_width=True, hide_index=True,
                        key="arch_table")

            st.markdown("**Model Load Report**")
            for k, v in load_report.items():
                status = "OK" if v == "✓" else v
                st.markdown(f"`{k}`: {status}")

            st.markdown(f"#### Config\n```json\n"
                        + json.dumps(cfg, indent=2, default=str)
                        + "\n```")

        # Rules
        with t5:
            st.markdown("#### Notification Rules")
            st.caption(
                "Create rules from the chat, e.g. **buat rule jika suhu diatas "
                "32°C dan humidity diatas 70% maka notif saya**. Press **Run** "
                "to pull up every historical match for a rule — charted and "
                "narrated exactly like a normal chat query.")
            if not st.session_state.rules:
                st.info("No rules yet. Try typing a rule in the chat panel.")
            else:
                for rule in list(st.session_state.rules):
                    disabled_cls = "" if rule.get("enabled", True) else " disabled"
                    meta_bits = [
                        f"Created {rule['created_at'][:16].replace('T', ' ')}",
                        f"Triggered {rule.get('trigger_count', 0)}×",
                    ]
                    if rule.get("last_triggered_at"):
                        meta_bits.append(
                            f"Last: {rule['last_triggered_at'][:16].replace('T', ' ')}")
                    st.markdown(f"""
                    <div class="rule-card{disabled_cls}">
                      <div class="rule-desc">{rule['description']}</div>
                      <div class="rule-raw">"{rule['raw_text']}"</div>
                      <div class="rule-meta">{' · '.join(meta_bits)}</div>
                    </div>""", unsafe_allow_html=True)

                    c1, c2, c3 = st.columns([1, 1, 1])
                    with c1:
                        en = st.checkbox(
                            "Enabled", value=rule.get("enabled", True),
                            key=f"en_{rule['id']}")
                        if en != rule.get("enabled", True):
                            rule["enabled"] = en
                            rules_mod.save_rules(st.session_state.rules)
                            st.rerun()
                    with c2:
                        run_clicked = st.button("▶ Run", key=f"run_{rule['id']}")
                    with c3:
                        if st.button("Delete", key=f"del_{rule['id']}"):
                            st.session_state.rules = rules_mod.delete_rule(
                                st.session_state.rules, rule["id"])
                            rules_mod.save_rules(st.session_state.rules)
                            st.rerun()

                    if run_clicked:
                        run_result = rules_mod.run_rule(engine, cfg, rule)
                        if run_result is None:
                            st.warning(
                                "No historical data currently matches this "
                                "rule's condition.")
                        else:
                            header = (
                                f"**🔎 Rule scan — {rule['description']}**\n\n"
                                f"Found **{run_result['n_windows']} matching "
                                f"windows**, from **{run_result['start']}** to "
                                f"**{run_result['end']}**.\n\n")
                            st.session_state.messages.append(
                                {"role": "bot", "content": header + narrate(run_result)})
                            st.session_state.last_results = [run_result]
                            st.rerun()
                    st.markdown("---")


if __name__ == "__main__":
    main()
