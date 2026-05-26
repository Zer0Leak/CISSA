from __future__ import annotations


def plot_cnn_random_search_results(search_results):
    import matplotlib.pyplot as plt

    trial_numbers = [record["trial"] for record in search_results]
    val_losses = [record["best_val_loss"] for record in search_results]
    best_index = min(range(len(search_results)), key=lambda index: val_losses[index])
    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    bars = ax.bar(trial_numbers, val_losses, color="#2f6fbb", alpha=0.85)
    bars[best_index].set_color("#2ca02c")
    ax.set_title("CNN random search: best validation loss by trial")
    ax.set_xlabel("trial")
    ax.set_ylabel("best validation loss")
    ax.set_xticks(trial_numbers)
    ax.grid(True, axis="y", alpha=0.25)
    return fig, ax
