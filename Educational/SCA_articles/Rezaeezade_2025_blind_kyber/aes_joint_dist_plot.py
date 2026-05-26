from __future__ import annotations

import numpy as np


def plot_aes_joint_probabilities(joint_probabilities, key_guesses, annotate=False):
    import matplotlib.pyplot as plt

    key_guesses = list(key_guesses)
    fig, axes = plt.subplots(1, len(key_guesses), figsize=(5.2 * len(key_guesses), 4.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    vmax = max(float(joint_probabilities[key].max()) for key in key_guesses)
    for axis, key_guess in zip(axes, key_guesses, strict=False):
        probabilities = joint_probabilities[key_guess]
        image = axis.imshow(probabilities.T, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax)
        axis.set_title(f"AES theoretical joint distribution\nkey guess = 0x{key_guess:02X}")
        axis.set_xlabel("HW(m)")
        axis.set_ylabel("HW(Sbox(m xor k))")
        axis.set_xticks(range(9))
        axis.set_yticks(range(9))
        if annotate:
            for h_m in range(9):
                for h_y in range(9):
                    if probabilities[h_m, h_y] > 0:
                        axis.text(h_m, h_y, f"{probabilities[h_m, h_y]:.2f}", ha="center", va="center", fontsize=7)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
    colorbar.set_label("probability")
    return fig, axes


def plot_empirical_joint_distributions(distributions_by_method, num_bits=8):
    import matplotlib.pyplot as plt

    method_names = list(distributions_by_method.keys())
    vmax = max(float(distributions_by_method[name]["probabilities"].max()) for name in method_names)
    fig, axes = plt.subplots(1, len(method_names), figsize=(5.0 * len(method_names), 4.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, method_name in zip(axes, method_names, strict=False):
        probabilities = distributions_by_method[method_name]["probabilities"]
        image = axis.imshow(probabilities.T, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest")
        axis.set_title(method_name)
        axis.set_xlabel("assigned h_m")
        axis.set_ylabel("assigned h_y")
        axis.set_xticks(range(num_bits + 1))
        axis.set_yticks(range(num_bits + 1))
        axis.set_xticks(np.arange(-0.5, num_bits + 1, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, num_bits + 1, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.0, alpha=0.8)
        axis.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
    colorbar.set_label("empirical probability mass")
    return fig, axes


def plot_empirical_marginals(distributions_by_method, num_bits=8):
    import matplotlib.pyplot as plt

    h_axis = np.arange(num_bits + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    for method_name, distribution in distributions_by_method.items():
        probabilities = distribution["probabilities"]
        axes[0].plot(h_axis, probabilities.sum(axis=1), marker="o", linewidth=1.7, label=method_name)
        axes[1].plot(h_axis, probabilities.sum(axis=0), marker="o", linewidth=1.7, label=method_name)
    axes[0].set_title("Marginal empirical distribution of h_m")
    axes[0].set_xlabel("assigned h_m")
    axes[0].set_ylabel("probability mass")
    axes[0].set_xticks(h_axis)
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title("Marginal empirical distribution of h_y")
    axes[1].set_xlabel("assigned h_y")
    axes[1].set_ylabel("probability mass")
    axes[1].set_xticks(h_axis)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=True)
    return fig, axes


def plot_paper_style_ge_curves(results_by_method):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), constrained_layout=True)
    for method_name, result in results_by_method.items():
        trace_axis = np.arange(1, result["ge"].size + 1)
        axes[0].plot(trace_axis, result["ge"], linewidth=1.8, label=method_name)
        axes[1].plot(trace_axis, result["success_rate"], linewidth=1.8, label=method_name)
    axes[0].set_title("Guessing entropy")
    axes[0].set_xlabel("number of attack traces")
    axes[0].set_ylabel("GE, lower is better")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=True)
    axes[1].set_title("Success rate")
    axes[1].set_xlabel("number of attack traces")
    axes[1].set_ylabel("fraction of attacks with rank <= threshold")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=True)
    return fig, axes


def plot_key_candidate_scores(scores_by_method, correct_key=None, top_n=20):
    import matplotlib.pyplot as plt

    num_methods = len(scores_by_method)
    fig, axes = plt.subplots(num_methods, 1, figsize=(12, 3.4 * num_methods), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, (method_name, score_result) in zip(axes, scores_by_method.items(), strict=False):
        ranking = score_result["ranking"][:top_n]
        scores = score_result["score"][ranking]
        axis.bar(np.arange(top_n), scores, color="#2f6fbb", alpha=0.85)
        axis.set_title(method_name)
        axis.set_xticks(np.arange(top_n))
        axis.set_xticklabels([f"0x{k:02X}" for k in ranking], rotation=45, ha="right")
        axis.set_ylabel("score")
        if correct_key is not None and int(correct_key) in ranking:
            correct_index = int(np.flatnonzero(ranking == int(correct_key))[0])
            axis.bar(correct_index, scores[correct_index], color="#2ca02c")
    return fig, axes
