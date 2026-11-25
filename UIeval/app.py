"""Streamlit monitor for existing training CSVs. No training code is imported."""
from __future__ import annotations

import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Both `streamlit run UIeval/app.py` and the dedicated launcher are supported.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from UIeval.charts import comparison_chart, epoch_chart, metric_chart
from UIeval.data import DEFAULT_RUNS_ROOT, RunReader, discover_runs, epoch_scores, filter_frame

st.set_page_config(page_title="UIeval · Training monitor", layout="wide", initial_sidebar_state="expanded")
st.markdown('''<style>
.stApp { background: #0b1220; color: #e7edf8; }
[data-testid="stSidebar"] { background: #111c2e; }
.block-container { padding-top: 2.4rem; padding-bottom: 2rem; max-width: 1680px; }
[data-testid="stMetric"] { background: #131f33; border: 1px solid #25344c;
 border-radius: 12px; padding: 16px; min-height: 116px; }
[data-testid="stMetricLabel"] { color: #b8c7dc; }
[data-testid="stMetricLabel"] p { white-space: normal; overflow: visible; }
[data-testid="stMetricValue"] { color: #f1f6ff; font-size: 1.65rem; }
h1 { letter-spacing: -.035em; padding-bottom: .3rem; }
.eyebrow { color: #58d9c6; letter-spacing: .14em; font-size: .8rem; font-weight: 600; }
.stCaption { color: #adbed6; }
</style>''', unsafe_allow_html=True)


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def number(value, digits=4):
    return f"{float(value):,.{digits}f}" if finite(value) else "—"


def reader_for(run):
    readers = st.session_state.setdefault("run_readers", {})
    key = str(run.resolve())
    if key not in readers:
        readers[key] = RunReader()
    # Keep selection + up to three comparisons without retaining every visited log.
    while len(readers) > 6:
        oldest = next(iter(readers))
        if oldest == key:
            break
        readers.pop(oldest)
    return readers[key]


def selected_values(label, values, key):
    values = sorted(values)
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in values]
    return st.multiselect(label, values, key=key, placeholder="All")


st.sidebar.markdown("## UIeval")
st.sidebar.caption("Training runs · live monitoring")
root_text = st.sidebar.text_input("Trained-model directory", value=os.environ.get("UIEVAL_RUNS_ROOT", str(DEFAULT_RUNS_ROOT)))
root = Path(root_text).expanduser()
runs = discover_runs(root)
st.sidebar.button("Rescan runs", width="stretch")
if not runs:
    st.title("Training monitor")
    st.info("No runs found yet. Start training, then use Rescan runs. Run folders need log.csv or JSON metadata.")
    st.code(str(root))
    st.stop()
by_name = {path.name: path for path in runs}
if st.session_state.get("selected_run") not in by_name:
    st.session_state["selected_run"] = runs[0].name
selection = st.sidebar.selectbox("Training run", list(by_name), key="selected_run")
run = by_name[selection]
comparison_options = [name for name in by_name if name != selection]
if "comparison_runs" in st.session_state:
    st.session_state["comparison_runs"] = [name for name in st.session_state["comparison_runs"] if name in comparison_options]
comparisons = st.sidebar.multiselect("Compare with", comparison_options, max_selections=3, key="comparison_runs")
st.sidebar.divider()
live = st.sidebar.toggle("Live updates", value=True, key="live_updates")
interval = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 2, key="refresh_seconds")
st.sidebar.button("Refresh now", width="stretch")
window = st.sidebar.slider("Smoothing window (backward passes)", 1, 250, 50, key="smooth_window")
axis = st.sidebar.radio("Horizontal axis", ["Backward pass", "Optimizer step"], key="chart_axis")
st.sidebar.caption("Read-only dashboard. Training continues independently when you close this page.")

st.markdown('<div class="eyebrow">TAHOE · EXPERIMENT MONITOR</div>', unsafe_allow_html=True)
st.title("Training monitor")
st.caption(selection)


@st.fragment(run_every=interval if live else None)
def render_live():
    snapshot, info, config, metadata_notes = reader_for(run).poll(run)
    frame = snapshot.frame
    scores = epoch_scores(info)
    status = str(info.get("status", "unknown")).replace("_", " ").capitalize()
    age = max(0., time.time() - snapshot.modified) if snapshot.modified else None
    model_type = str(config.get("model_type", "unknown")).capitalize()
    mode = "Single file" if config.get("mode") == "single" else "All files"
    status_columns = st.columns([3, 2])
    status_columns[0].markdown(f"**{model_type} model** · {mode} · Recorded status: **{status}**")
    age_text = f"{age:.0f}s ago" if age is not None else "not yet available"
    status_columns[1].caption(f"{'Live' if live else 'Paused'} · Updated {datetime.now():%H:%M:%S} · Last log write {age_text}")
    if snapshot.pending:
        st.caption("Waiting for the final CSV record to finish; complete rows remain visible.")
    for message in snapshot.warnings:
        if "last row" not in message:
            st.caption(message)
    if frame.empty:
        st.info("Waiting for the first complete backward-pass record. Setup and data indexing may still be running.")
        with st.expander("Run configuration"):
            st.json(config)
        return
    if age is not None and age > 120 and info.get("status") in ("running", "indexing"):
        st.caption("No new records for more than two minutes. This can happen during evaluation or file loading; it does not establish that training stopped.")
    latest = frame.iloc[-1]
    completed = int(scores.epoch.max()) if not scores.empty else 0
    probe_column = next((c for c in ("reconstruction_probe", "validation", "val_loss") if c in scores), None)
    probe = scores[probe_column].dropna().iloc[-1] if probe_column and not scores[probe_column].dropna().empty else None
    cards = [*st.columns(3), *st.columns(3)]
    cards[0].metric("Latest loss", number(latest.get("loss")))
    cards[1].metric(f"Last {window} passes · mean", number(frame.loss.tail(window).mean()))
    passes = latest.get("backward_pass", len(frame))
    cards[2].metric("Backward-pass index", f"{int(passes):,}" if finite(passes) else "—")
    total_epochs = config.get("epochs", info.get("target_epochs", "?"))
    cards[3].metric("Completed epochs", f"{completed} / {total_epochs}")
    is_probe = (probe_column == "reconstruction_probe" or config.get("model_type") == "control"
                or config.get("mode") == "single")
    cards[4].metric("Latest probe" if is_probe else "Latest validation", number(probe))
    peak = frame.peak_vram_gb.max() if "peak_vram_gb" in frame else None
    cards[5].metric("Recorded peak VRAM", f"{float(peak):.2f} GB" if finite(peak) else "—")
    st.caption("Cards summarize the entire selected run. Filters below apply to per-pass charts and the log table. Probe scores update at completed epochs.")

    with st.expander("Filter logged backward passes", expanded=False):
        filters = st.columns(3)
        with filters[0]:
            epochs = selected_values("Epochs", frame.epoch.dropna().astype(int).unique() if "epoch" in frame else [], f"{selection}:epochs")
        with filters[1]:
            plates = selected_values("Plates", frame.plate.dropna().astype(int).unique() if "plate" in frame else [], f"{selection}:plates")
        with filters[2]:
            lines = selected_values("Cell lines", frame.cell_line.dropna().astype(str).unique() if "cell_line" in frame else [], f"{selection}:lines")
    view = filter_frame(frame, epochs=epochs, plates=plates, lines=lines)
    st.caption(f"{len(view):,} of {len(frame):,} complete records shown · {view['file'].nunique() if 'file' in view else '—'} files")
    if view.empty:
        st.info("No rows match these filters. Clear one or more selections.")
        return
    labels = ["Loss", "Gradients", "Performance", "Compare runs", "Run details", "Log table"]
    loss_tab, grad_tab, perf_tab, compare_tab, details_tab, logs_tab = st.tabs(labels)
    revision = str(run.resolve())
    with loss_tab:
        st.subheader("Expression loss")
        st.plotly_chart(metric_chart(view, ["loss"], axis=axis, window=window, revision=revision + ':loss',
                                    raw=True, y_title="Expression loss"), width="stretch", key="loss_plot")
        st.caption("Faint line: individual backward passes. Bright line: rolling mean across sampled experiments; this is not a fixed evaluation set.")
        st.subheader("Completed-epoch scores · entire run")
        if scores.empty:
            st.info("Epoch training/probe scores will appear after the first epoch completes.")
        else:
            st.plotly_chart(epoch_chart(scores, revision + ':epochs'), width="stretch", key="epoch_plot")
            if "reconstruction_probe" in scores:
                st.caption("Reconstruction probes use training files and fixed sampling. They measure reconstruction, not held-out cell-line generalization.")
    with grad_tab:
        st.subheader("Per-backward gradient norms")
        available = [c for c in view.columns if c.startswith("grad_")]
        if not available:
            st.info("This run has no gradient columns.")
        else:
            if not view[available].apply(lambda c: c.map(finite)).all().all():
                st.warning("Some gradient values are missing or non-finite; charts omit those values.")
            defaults = [c for c in ("grad_total", "grad_gene_embedding", "grad_conc_emb", "grad_coordi_layer") if c in available]
            module_key = f"{selection}:gradient_modules"
            if module_key in st.session_state:
                st.session_state[module_key] = [c for c in st.session_state[module_key] if c in available]
            else:
                st.session_state[module_key] = defaults or available[:4]
            modules = st.multiselect("Gradient series", available, key=module_key)
            logarithmic = st.toggle("Logarithmic gradient axis", value=True, key=f"{selection}:grad_log")
            if modules:
                st.plotly_chart(metric_chart(view, modules, axis=axis, window=window, revision=revision + ':grad',
                                            median=True, log=logarithmic, y_title="Gradient norm · rolling median"),
                                width="stretch", key="gradient_plot")
                summary = pd.DataFrame({"Latest": view[modules].iloc[-1], "Median": view[modules].median(), "Maximum": view[modules].max()})
                st.dataframe(summary, width="stretch")
            st.caption("Norms describe each backward pass before clipping. The optimizer clips accumulated gradients; these columns cannot give the actual clipping rate. Log axes omit zero values.")
    with perf_tab:
        st.subheader("Training performance")
        left, right = st.columns(2)
        with left:
            if "seconds" in view:
                st.plotly_chart(metric_chart(view, ["seconds"], axis=axis, window=window, revision=revision + ':time',
                                            median=True, raw=True, y_title="Seconds per backward pass"),
                                width="stretch", key="timing_plot")
        with right:
            if "peak_vram_gb" in view:
                st.plotly_chart(metric_chart(view, ["peak_vram_gb"], axis=axis, window=1, revision=revision + ':memory',
                                            y_title="Peak allocated GPU memory · GB"),
                                width="stretch", key="memory_plot")
        st.caption("Memory is the recorded peak allocated during each backward pass, not live GPU utilization. Timing follows the training log's measurement scope.")
    with compare_tab:
        st.subheader("Run comparison")
        if not comparisons:
            st.info("Choose up to three additional runs in the sidebar to overlay their curves.")
        else:
            metric = st.selectbox("Compare metric", ["loss", "grad_total", "seconds", "peak_vram_gb"], key="comparison_metric")
            comparison_data = [(selection, frame)]
            signatures = {(config.get("model_type"), info.get("model", {}).get("n_genes"), config.get("scale_exponent", .5), config.get("mode"))}
            for name in comparisons:
                other, other_info, other_config, _ = reader_for(by_name[name]).poll(by_name[name])
                comparison_data.append((name, other.frame))
                signatures.add((other_config.get("model_type"), other_info.get("model", {}).get("n_genes"), other_config.get("scale_exponent", .5), other_config.get("mode")))
            if metric == "loss" and len(signatures) > 1:
                st.warning("These runs differ in target type, gene count, loss scaling or training mode. Their loss values are not a direct quality ranking.")
            st.plotly_chart(comparison_chart(comparison_data, metric=metric, window=window, axis=axis, revision=revision + ':comparison'),
                            width="stretch", key="comparison_plot")
            st.caption("Comparisons use complete runs, independent of the selected run's filters. Matching gene panels, targets and evaluation data are needed for fair comparisons.")
    with details_tab:
        st.subheader("Architecture and run settings")
        model = info.get("model", {})
        settings = {"Model": config.get("model_type"), "Training mode": config.get("mode"),
                    "Heads": config.get("heads"), "Head dimension": config.get("head_dim"),
                    "Total dimension": config.get("dim"), "Bottleneck dimension": config.get("conc_dim"),
                    "Layers": config.get("layers"), "Genes": model.get("n_genes"),
                    "Parameters": model.get("parameter_total"), "Normalization": config.get("normalization", "none"),
                    "Gene statistics": config.get("gene_statistics", False), "Gradient clip": config.get("grad_clip"),
                    "Precision": config.get("amp"), "Learning rate": config.get("lr")}
        st.dataframe(pd.DataFrame({"Setting": settings.keys(), "Value": [str(v) if v is not None else "—" for v in settings.values()]}),
                     hide_index=True, width="stretch")
        if model.get("parameters"):
            st.markdown("**Parameters by module**")
            st.dataframe(pd.DataFrame(list(model["parameters"].items()), columns=["Module", "Parameters"]), hide_index=True, width="stretch")
        if not scores.empty:
            st.markdown("**Saved epoch metrics**")
            st.dataframe(scores, hide_index=True, width="stretch")
        with st.expander("Full configuration"):
            st.json(config)
        for note in metadata_notes:
            st.caption(note)
    with logs_tab:
        st.subheader("Logged backward passes")
        st.caption("Latest 1,000 matching records. This table does not edit the training log.")
        st.dataframe(view.tail(1000), hide_index=True, width="stretch", height=450)


render_live()
