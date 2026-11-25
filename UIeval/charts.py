"""Small Plotly builders shared by the live tabs and tests."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from UIeval.data import downsample, smoothing, x_values

COLORS = ["#58d9c6", "#8aaeff", "#e9b76e", "#db91c2", "#9fda8f", "#c1a2ff", "#f18d8d", "#78c9ea"]


def layout(figure, *, y_title, x_title, revision, log=False, height=360):
    figure.update_layout(template="plotly_dark", height=height, margin=dict(l=12, r=12, t=20, b=12),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(family="Arial, sans-serif", size=13, color="#dce7f5"),
                         xaxis_title=x_title, yaxis_title=y_title, uirevision=revision,
                         legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="rgba(170,190,220,.12)", type="log" if log else "linear")
    return figure


def metric_chart(frame, columns, *, axis, window, revision, median=False, log=False, raw=False, y_title="Value"):
    x, x_title = x_values(frame, axis)
    plot = pd.DataFrame({"x": x}, index=frame.index)
    for column in columns:
        plot[column] = smoothing(frame[column], window, median=median)
    if raw and len(columns) == 1:
        plot["raw"] = pd.to_numeric(frame[columns[0]], errors="coerce")
    plot = downsample(plot)
    figure = go.Figure()
    if "raw" in plot:
        figure.add_trace(go.Scattergl(x=plot.x, y=plot.raw, name="Per backward pass", mode="lines",
                                     line=dict(color="rgba(170,190,220,.30)", width=1)))
    for index, column in enumerate(columns):
        values = plot[column].where(plot[column] > 0) if log else plot[column]
        label = column.replace("grad_", "").replace("_", " ")
        if len(columns) == 1:
            label = f"{window}-pass {'median' if median else 'mean'}"
        figure.add_trace(go.Scattergl(x=plot.x, y=values, name=label, mode="lines",
                                     line=dict(color=COLORS[index % len(COLORS)], width=2)))
    return layout(figure, y_title=y_title, x_title=x_title, revision=revision, log=log)


def epoch_chart(scores, revision):
    figure = go.Figure()
    for index, (column, label) in enumerate((("train_loss", "Training mean"),
                                            ("reconstruction_probe", "Reconstruction probe"),
                                            ("validation", "Validation"), ("val_loss", "Validation"))):
        if column in scores:
            figure.add_trace(go.Scatter(x=scores.epoch, y=scores[column], mode="lines+markers", name=label,
                                       line=dict(color=COLORS[index], width=2), marker=dict(size=7)))
    return layout(figure, y_title="Expression loss", x_title="Completed epoch", revision=revision, height=300)


def comparison_chart(runs, *, metric, window, axis, revision):
    figure = go.Figure()
    x_title = axis
    for index, (name, frame) in enumerate(runs):
        if frame.empty or metric not in frame:
            continue
        x, x_title = x_values(frame, axis)
        values = smoothing(frame[metric], window)
        plot = downsample(pd.DataFrame({"x": x, "value": values}))
        figure.add_trace(go.Scattergl(x=plot.x, y=plot.value, name=name, mode="lines",
                                     line=dict(color=COLORS[index % len(COLORS)], width=2)))
    return layout(figure, y_title=metric.replace("_", " "), x_title=x_title, revision=revision)
