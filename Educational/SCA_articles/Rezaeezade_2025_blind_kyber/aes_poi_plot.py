from __future__ import annotations

import numpy as np


def plot_aes_poi_correlations(correlation_h_m, correlation_h_y, pois_h_m=None, pois_h_y=None):
    correlation_h_m = np.asarray(correlation_h_m, dtype=np.float64)
    correlation_h_y = np.asarray(correlation_h_y, dtype=np.float64)
    sample_axis = np.arange(correlation_h_m.size)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
        axes[0].plot(sample_axis, correlation_h_m, color="#1f77b4", linewidth=1.2)
        axes[0].set_title("Correlation with h_m = HW(m)")
        axes[0].set_ylabel("Pearson correlation")
        axes[0].grid(alpha=0.25)
        axes[1].plot(sample_axis, correlation_h_y, color="#ff7f0e", linewidth=1.2)
        axes[1].set_title("Correlation with h_y = HW(Sbox(m xor k))")
        axes[1].set_xlabel("Sample index")
        axes[1].set_ylabel("Pearson correlation")
        axes[1].grid(alpha=0.25)
        if pois_h_m is not None:
            pois_h_m = np.asarray(pois_h_m, dtype=np.int64)
            axes[0].scatter(pois_h_m, correlation_h_m[pois_h_m], color="#00bcd4", edgecolor="black", s=40)
        if pois_h_y is not None:
            pois_h_y = np.asarray(pois_h_y, dtype=np.int64)
            axes[1].scatter(pois_h_y, correlation_h_y[pois_h_y], color="#ffd166", edgecolor="black", s=40)
        return fig

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=("Correlation with h_m = HW(m)", "Correlation with h_y = HW(Sbox(m xor k))"),
    )
    fig.add_trace(
        go.Scatter(
            x=sample_axis,
            y=correlation_h_m,
            mode="lines",
            line=dict(color="#1f77b4", width=1.3),
            name="h_m correlation",
            hovertemplate="sample=%{x}<br>corr=%{y:.4f}<extra>h_m</extra>",
        ),
        row=1,
        col=1,
    )
    if pois_h_m is not None:
        pois_h_m = np.asarray(pois_h_m, dtype=np.int64)
        fig.add_trace(
            go.Scatter(
                x=pois_h_m,
                y=correlation_h_m[pois_h_m],
                mode="markers",
                marker=dict(color="#00bcd4", size=8, line=dict(color="#111111", width=1)),
                name="h_m PoIs",
                hovertemplate="PoI=%{x}<br>corr=%{y:.4f}<extra>h_m PoI</extra>",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=sample_axis,
            y=correlation_h_y,
            mode="lines",
            line=dict(color="#ff7f0e", width=1.3),
            name="h_y correlation",
            hovertemplate="sample=%{x}<br>corr=%{y:.4f}<extra>h_y</extra>",
        ),
        row=2,
        col=1,
    )
    if pois_h_y is not None:
        pois_h_y = np.asarray(pois_h_y, dtype=np.int64)
        fig.add_trace(
            go.Scatter(
                x=pois_h_y,
                y=correlation_h_y[pois_h_y],
                mode="markers",
                marker=dict(color="#ffd166", size=8, line=dict(color="#111111", width=1)),
                name="h_y PoIs",
                hovertemplate="PoI=%{x}<br>corr=%{y:.4f}<extra>h_y PoI</extra>",
            ),
            row=2,
            col=1,
        )
    fig.update_yaxes(title_text="Pearson correlation", row=1, col=1)
    fig.update_yaxes(title_text="Pearson correlation", row=2, col=1)
    fig.update_xaxes(title_text="Sample index", row=2, col=1)
    fig.update_layout(
        height=820,
        width=1200,
        template="plotly_white",
        hovermode="x unified",
        dragmode="zoom",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=70, r=30, t=90, b=60),
    )
    return fig
