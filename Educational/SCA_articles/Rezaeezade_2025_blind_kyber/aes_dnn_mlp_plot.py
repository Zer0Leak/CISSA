from __future__ import annotations

import numpy as np

from util import decode_joint_hw_labels


def plot_dnn_training_history(history):
    import matplotlib.pyplot as plt

    values = history.history if hasattr(history, "history") else history
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].plot(values.get("loss", []), marker="o", label="train")
    axes[0].plot(values.get("val_loss", []), marker="o", label="validation")
    axes[0].set_title("Categorical cross-entropy")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=True)
    axes[1].plot(values.get("accuracy", []), marker="o", label="train")
    axes[1].plot(values.get("val_accuracy", []), marker="o", label="validation")
    axes[1].set_title("Accuracy against pseudo-labels")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("pseudo-label accuracy")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=True)
    return fig, axes


def plot_dnn_prediction_diagnostics(true_joint_labels, predicted_joint_labels, predicted_probabilities, title_prefix="", num_bits=8):
    import matplotlib.pyplot as plt

    true_joint_labels = np.asarray(true_joint_labels, dtype=np.int64)
    predicted_joint_labels = np.asarray(predicted_joint_labels, dtype=np.int64)
    max_probability = np.max(predicted_probabilities, axis=1)
    joint_correct = predicted_joint_labels == true_joint_labels
    predicted_h_m, predicted_h_y = decode_joint_hw_labels(predicted_joint_labels, num_bits=num_bits)
    true_h_m, true_h_y = decode_joint_hw_labels(true_joint_labels, num_bits=num_bits)
    metrics = [
        np.mean(predicted_h_m == true_h_m),
        np.mean(predicted_h_y == true_h_y),
        np.mean(joint_correct),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    bars = axes[0].bar(["h_m", "h_y", "joint"], metrics, color=["#4c78a8", "#f58518", "#54a24b"])
    axes[0].bar_label(bars, fmt="%.3f", padding=2)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title(f"{title_prefix} accuracy against true labels")
    axes[0].set_ylabel("accuracy")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].hist(max_probability[joint_correct], bins=20, alpha=0.75, label="joint correct")
    axes[1].hist(max_probability[~joint_correct], bins=20, alpha=0.65, label="joint wrong")
    axes[1].set_title(f"{title_prefix} confidence split")
    axes[1].set_xlabel("maximum predicted probability")
    axes[1].set_ylabel("number of traces")
    axes[1].legend(frameon=True)
    return fig, axes


def plot_dnn_confidence_weight_correctness_interactive(pseudo_joint_labels, true_joint_labels, trace_weights, label_confidence, num_bins=12):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    pseudo_joint_labels = np.asarray(pseudo_joint_labels)
    true_joint_labels = np.asarray(true_joint_labels)
    trace_weights = np.asarray(trace_weights)
    label_confidence = np.asarray(label_confidence)
    correct = pseudo_joint_labels == true_joint_labels
    bins = np.linspace(0, 1, num_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    counts = np.zeros(num_bins, dtype=int)
    correctness = np.full(num_bins, np.nan)
    mean_weight = np.full(num_bins, np.nan)
    mean_confidence = np.full(num_bins, np.nan)
    for index in range(num_bins):
        mask = (label_confidence >= bins[index]) & (label_confidence < bins[index + 1])
        if index == num_bins - 1:
            mask |= label_confidence == bins[index + 1]
        counts[index] = int(np.sum(mask))
        if np.any(mask):
            correctness[index] = float(np.mean(correct[mask]))
            mean_weight[index] = float(np.mean(trace_weights[mask]))
            mean_confidence[index] = float(np.mean(label_confidence[mask]))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
    fig.add_bar(x=centers, y=counts, name="trace count", marker_color="#bdbdbd", row=1, col=1, secondary_y=False)
    fig.add_scatter(x=centers, y=correctness, name="observed joint-label accuracy", mode="lines+markers", row=1, col=1, secondary_y=True)
    fig.add_scatter(x=centers, y=mean_weight, name="mean sample weight", mode="lines+markers", row=2, col=1, secondary_y=False)
    fig.add_scatter(x=centers, y=mean_confidence, name="mean confidence", mode="lines+markers", row=2, col=1, secondary_y=True)
    fig.update_layout(height=760, width=1100, template="plotly_white", hovermode="x unified")
    fig.update_xaxes(title_text="confidence bin center", row=2, col=1)
    fig.update_yaxes(title_text="number of traces", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="observed joint-label accuracy", row=1, col=1, secondary_y=True, range=[0, 1])
    fig.update_yaxes(title_text="mean sample weight", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="mean confidence", row=2, col=1, secondary_y=True, range=[0, 1])
    return fig
