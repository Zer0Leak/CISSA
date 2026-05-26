from __future__ import annotations

import numpy as np


def plot_mc_feature_preparation(raw_feature_matrix, aligned_feature_matrix, num_pois_h_m, num_examples=160):
    import matplotlib.pyplot as plt

    raw_feature_matrix = np.asarray(raw_feature_matrix, dtype=np.float64)
    aligned_feature_matrix = np.asarray(aligned_feature_matrix, dtype=np.float64)
    num_examples = min(num_examples, raw_feature_matrix.shape[0])
    raw_block = raw_feature_matrix[:num_examples]
    aligned_block = aligned_feature_matrix[:num_examples]
    color_limit = np.max(np.abs(np.concatenate([raw_block, aligned_block], axis=0)))
    color_limit = 1.0 if color_limit == 0.0 else color_limit
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    for axis, block, title in [
        (axes[0], raw_block, "PoI-truncated traces before sign alignment"),
        (axes[1], aligned_block, "PoI-truncated traces after sign alignment"),
    ]:
        image = axis.imshow(block, aspect="auto", cmap="coolwarm", vmin=-color_limit, vmax=color_limit)
        axis.axvline(num_pois_h_m - 0.5, color="black", linewidth=2)
        axis.set_title(title)
        axis.set_ylabel("trace index")
    axes[1].set_xlabel("PoI index inside truncated feature vector")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.85)
    colorbar.set_label("sample value")
    return fig, axes


def plot_mc_cluster_size_distribution(
    cluster_sizes,
    expected_cluster_sizes=None,
    model_prior_cluster_sizes=None,
    title_prefix=None,
):
    import matplotlib.pyplot as plt

    cluster_sizes = np.asarray(cluster_sizes, dtype=np.int64)
    sorted_sizes = np.sort(cluster_sizes)[::-1]
    expected_sorted_sizes = None
    if expected_cluster_sizes is not None:
        expected_sorted_sizes = np.sort(np.asarray(expected_cluster_sizes, dtype=np.float64).reshape(-1))[::-1]
        if expected_sorted_sizes.shape != sorted_sizes.shape:
            raise ValueError("expected_cluster_sizes must have the same number of clusters as cluster_sizes.")
    model_prior_sorted_sizes = None
    if model_prior_cluster_sizes is not None:
        model_prior_sorted_sizes = np.sort(np.asarray(model_prior_cluster_sizes, dtype=np.float64).reshape(-1))[::-1]
        if model_prior_sorted_sizes.shape != sorted_sizes.shape:
            raise ValueError("model_prior_cluster_sizes must have the same number of clusters as cluster_sizes.")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    axes[0].bar(np.arange(cluster_sizes.size), cluster_sizes, color="#1f77b4")
    axes[0].set_title("Cluster sizes in GMM order" if title_prefix is None else f"{title_prefix}: cluster sizes in GMM order")
    axes[0].set_xlabel("cluster index")
    axes[0].set_ylabel("number of traces")
    x_positions = np.arange(sorted_sizes.size)
    axes[1].bar(x_positions, sorted_sizes, color="#ff7f0e", alpha=0.78, label="observed GMM cluster size")
    if expected_sorted_sizes is not None:
        axes[1].plot(
            x_positions,
            expected_sorted_sizes,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="expected binomial joint size",
        )
    if model_prior_sorted_sizes is not None:
        axes[1].plot(
            x_positions,
            model_prior_sorted_sizes,
            color="#2ca02c",
            linestyle="-",
            linewidth=1.8,
            label="GMM prior size",
        )
    if expected_sorted_sizes is not None or model_prior_sorted_sizes is not None:
        axes[1].legend(frameon=True)
    axes[1].set_title(
        "Cluster sizes sorted largest to smallest"
        if title_prefix is None
        else f"{title_prefix}: cluster sizes sorted largest to smallest"
    )
    axes[1].set_xlabel("ranked cluster index")
    axes[1].set_ylabel("number of traces")
    return fig, axes


def plot_mc_cluster_assignment_pca(feature_matrix, cluster_assignments, cluster_centers=None, show_cluster_centers=False, num_examples=2500):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)
    cluster_assignments = np.asarray(cluster_assignments, dtype=np.int64)
    num_examples = min(num_examples, feature_matrix.shape[0])
    rng = np.random.default_rng(0)
    shown_indices = rng.choice(feature_matrix.shape[0], size=num_examples, replace=False)
    pca = PCA(n_components=2, random_state=0)
    projected = pca.fit_transform(feature_matrix)
    explained = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    scatter = ax.scatter(
        projected[shown_indices, 0],
        projected[shown_indices, 1],
        c=cluster_assignments[shown_indices],
        cmap="tab20",
        s=8,
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )
    if show_cluster_centers and cluster_centers is not None:
        projected_centers = pca.transform(np.asarray(cluster_centers, dtype=np.float64))
        ax.scatter(projected_centers[:, 0], projected_centers[:, 1], marker="X", s=55, c="white", edgecolor="black")
    ax.set_title("GMM clusters projected to 2D with PCA")
    ax.set_xlabel(f"PC 1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC 2 ({explained[1] * 100:.1f}% variance)")
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.85)
    colorbar.set_label("cluster index")
    return fig, ax


def plot_mc_feature_heatmap(feature_matrix, predicted_h_m, predicted_h_y, num_pois_h_m, num_examples=160):
    import matplotlib.pyplot as plt

    feature_matrix = np.asarray(feature_matrix, dtype=np.float64)
    predicted_h_m = np.asarray(predicted_h_m, dtype=np.int64)
    predicted_h_y = np.asarray(predicted_h_y, dtype=np.int64)
    order = np.lexsort((predicted_h_y, predicted_h_m))
    shown = order[: min(num_examples, order.size)]
    heatmap = feature_matrix[shown]
    column_mean = heatmap.mean(axis=0, keepdims=True)
    column_std = heatmap.std(axis=0, keepdims=True)
    heatmap = np.divide(heatmap - column_mean, column_std, out=np.zeros_like(heatmap), where=column_std > 0)
    fig, ax = plt.subplots(figsize=(13.5, 5.6), constrained_layout=True)
    image = ax.imshow(heatmap, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    ax.axvline(num_pois_h_m - 0.5, color="black", linewidth=2)
    ax.set_title("PoI features sorted by predicted (h_m, h_y)")
    ax.set_xlabel("PoI index")
    ax.set_ylabel("sorted trace index")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("per-PoI z-score")
    return fig, ax


def plot_mc_label_diagnostics(true_h_m, predicted_h_m, true_h_y, predicted_h_y):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for axis, true_values, predicted_values, title in [
        (axes[0], true_h_m, predicted_h_m, "h_m confusion"),
        (axes[1], true_h_y, predicted_h_y, "h_y confusion"),
    ]:
        confusion = np.zeros((9, 9), dtype=np.int64)
        np.add.at(confusion, (true_values, predicted_values), 1)
        column_sums = confusion.sum(axis=1, keepdims=True)
        normalized = np.divide(confusion, column_sums, out=np.zeros_like(confusion, dtype=np.float64), where=column_sums > 0)
        image = axis.imshow(normalized, origin="lower", cmap="Blues", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.set_xlabel("predicted")
        axis.set_ylabel("true")
        axis.set_xticks(range(9))
        axis.set_yticks(range(9))
    colorbar = fig.colorbar(image, ax=axes, shrink=0.85)
    colorbar.set_label("row-normalized fraction")
    return fig, axes


def _accuracy_values(metrics):
    return [
        float(metrics.get("h_m", metrics.get("accuracy_h_m"))),
        float(metrics.get("h_y", metrics.get("accuracy_h_y"))),
        float(metrics.get("joint", metrics.get("joint_accuracy"))),
    ]


def plot_mc_accuracy_comparison(training_accuracy_by_variant, attack_accuracy_by_method=None):
    import matplotlib.pyplot as plt

    panels = [("MC-labeling slice", training_accuracy_by_variant)]
    if attack_accuracy_by_method:
        panels.append(("attack slice", attack_accuracy_by_method))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    metric_names = ["h_m", "h_y", "joint"]
    x_positions = np.arange(len(metric_names), dtype=np.float64)

    for axis, (title, accuracy_by_name) in zip(axes, panels, strict=False):
        names = list(accuracy_by_name)
        bar_width = min(0.36, 0.80 / max(len(names), 1))
        for index, name in enumerate(names):
            offsets = x_positions - (bar_width * (len(names) - 1) / 2.0) + index * bar_width
            bars = axis.bar(offsets, _accuracy_values(accuracy_by_name[name]), width=bar_width, label=name)
            axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        axis.set_title(title)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(metric_names)
        axis.set_ylabel("accuracy against true labels")
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=True)

    return fig, axes
