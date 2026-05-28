import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

def calculate_fairness_index(participation):
    """
    Compute Jain's Fairness Index

    Args:
        participation: list of ints with the cumulative number
                       of times each device has participated in
                       model training
    
    Returns:
        Integer value between 0 (biased participation) and 1 
        (equal participation)
    """
    participation = np.array(participation)
    num_clients = len(participation)
    
    return (np.sum(participation) ** 2) / (num_clients * np.sum(participation ** 2))

def plot_results(metrics, local_accuracies, wo_total, wt_total, save_dir="plots"):
    """
    Plot FL metrics (accuracy, f1 score, Jain's fairness index, client device energy usage
    and wearable energy usage)

    Args:
        metrics: a dictionary containing accuracy, f1 score, precision, recall
        local_accuracies: list containing local accuracy of clients in final round
        of training
        wo_total: total energy usage of wearable one
        wt_total: total energy usage of wearable two
        save_dir: location to save plots
    """
    labels=[
        "BRUSH_TEETH",
        "DRESSING",
        "DRINK",
        "PREP_MEAL_AND_EAT",
        "KITCHEN_BIN",
        "LAY",
        "LAY_SIT",
        "MEDICINE",
        "SHOWER",
        "SIT",
        "SIT_STAND",
        "STAIRS",
        "STAND",
        "USE_TOILET",
        "WALK",
        "WASH_FACE"
    ]

    os.makedirs(save_dir, exist_ok=True)
    rounds = np.arange(1, len(metrics) + 1)
    num_classes = len(metrics[0]["per_class"])

    STYLE = {
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "black",
        "axes.labelcolor":   "black",
        "axes.labelsize":    16,
        "axes.grid":         True,
        "grid.color":        "white",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.8,
        "xtick.color":       "black",
        "ytick.color":       "black",
        "text.color":        "black",
        "legend.facecolor":  "white",
        "legend.edgecolor":  "black",
        "legend.fontsize":   14,
        "font.family":       "sans-serif",
    }

    #######################################################
    # Plot accuracy vs aggregation round
    #######################################################
    accuracy = [np.mean(m["local_accuracy"]) for m in metrics]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(rounds, accuracy, color="#6ee7f7", linewidth=2, zorder=3)
        ax.fill_between(rounds, accuracy, alpha=0.15, color="#6ee7f7", zorder=2)
        ax.scatter(
            rounds, accuracy,
            color="#6ee7f7", s=40, zorder=4,
            edgecolors="#0f1117", linewidths=0.8,
        )

        
        best_v = max(accuracy)
        ax.axhline(y=best_v, color='red', linestyle='--', linewidth=1.5, label=f'Maximum accuracy: {best_v:.2f}%')
        ax.legend()

        ax.set_xlabel("Global Aggregation Round", fontsize=12)
        ax.set_ylabel("HAR Accuracy (%)", fontsize=12)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_xlim(rounds[0] - 0.5, rounds[-1] + 0.5)
        ax.set_ylim(max(0, min(accuracy) - 5), min(100, max(accuracy) + 5))

        fig.tight_layout()
        path = os.path.join(save_dir, f"accuracy_vs_round.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    
    #######################################################
    # Plot per-class F1 vs aggregation round
    #######################################################
    cmap = plt.cm.get_cmap("tab20", num_classes)
    colors = [cmap(i) for i in range(num_classes)]

    f1_per_class = {
        i: [m["per_class"][i]["f1"] for m in metrics]
        for i in range(num_classes)
    }

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 6))

        for i in range(num_classes):
            ax.plot(rounds, f1_per_class[i], color=colors[i],
                    linewidth=1.8, label=f"{labels[i]}", zorder=3)

        ax.set_xlabel("Training Round", fontsize=12)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_title("Per-class F1 Score vs Training Round", fontsize=14, pad=14)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_xlim(rounds[0] - 0.5, rounds[-1] + 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(
            loc="lower right", ncol=max(1, num_classes // 10),
            fontsize=8, framealpha=0.9,
        )

        fig.tight_layout()
        path = os.path.join(save_dir, f"f1_per_class_vs_round.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    
    #######################################################
    # Plot energy vs aggregation round
    #######################################################
    num_clients = len(metrics[0]["energy"])
    cmap = plt.cm.get_cmap("tab20", num_clients)
    colors = [cmap(i) for i in range(num_clients)]

    energy_per_client = {
        i: [m["energy"][i] for m in metrics]
        for i in range(num_clients)
    }

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 6))

        for i in range(num_clients):
            ax.plot(rounds, energy_per_client[i], color=colors[i],
                    linewidth=1.8, label=f"Client {i + 1}", zorder=3)

        ax.set_xlabel("Training Round", fontsize=12)
        ax.set_ylabel("Total Device Energy (%)", fontsize=12)
        ax.set_title("Total Client Device Energy vs Training Round", fontsize=14, pad=14)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_xlim(rounds[0] - 0.5, rounds[-1] + 0.5)
        ax.margins(y=0.05)
        ax.legend(
            loc="lower right", ncol=max(1, num_clients // 10),
            fontsize=8, framealpha=0.9,
        )

        fig.tight_layout()
        path = os.path.join(save_dir, f"energy_vs_round.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    
    #######################################################
    # Plot total energy client devices
    #######################################################
    total_energies = [metrics[-1]["used_energy"][i] for i in range(num_clients)]
    x = [i + 1 for i in range(len(total_energies))]

    fig, ax = plt.subplots()

    ax.bar(x, total_energies, color='skyblue', edgecolor='black')

    ax.set_xlabel("Client")
    ax.set_ylabel("Total Client Device Energy Usage (Wh)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    path = os.path.join(save_dir, f"client_energy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    
    #######################################################
    # Plot Jain's fairness index across aggregation rounds
    #######################################################
    fairness_index = []

    for m in metrics:
        fairness_index.append(calculate_fairness_index(m["participation"]))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(rounds, fairness_index, color="#6ee7f7", linewidth=2, zorder=3)
        ax.fill_between(rounds, fairness_index, alpha=0.15, color="#6ee7f7", zorder=2)
        ax.scatter(
            rounds, fairness_index,
            color="#6ee7f7", s=40, zorder=4,
            edgecolors="#0f1117", linewidths=0.8,
        )

        best_v = max(fairness_index)
        ax.axhline(y=best_v, color='red', linestyle='--', linewidth=1.5, label=f'Maximum fairness: {best_v:.2f}%')
        ax.legend()

        ax.set_xlabel("Aggregation Round", fontsize=12)
        ax.set_ylabel("Jain's Fairness Index", fontsize=12)
        ax.set_title("Device Participation vs Aggregation Round", fontsize=14, pad=14)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_xlim(rounds[0] - 0.5, rounds[-1] + 0.5)

        fig.tight_layout()
        path = os.path.join(save_dir, f"fairness_index_vs_round.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    
    #######################################################
    # Plot local accuracy results
    #######################################################
    labels = [f"{i + 1}" for i in range(len(local_accuracies))]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, local_accuracies, color='skyblue', edgecolor='black')

    # Add average line
    avg = sum(local_accuracies) / len(local_accuracies)
    ax.axhline(y=avg, color='red', linestyle='--', linewidth=1.5, label=f'Average accuracy: {avg:.2f}%')
    ax.legend()

    # Add labels and title
    ax.set_xlabel("Client")
    ax.set_ylabel("Local Accuracy (%)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    path = os.path.join(save_dir, f"local_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    #######################################################
    # Plot wearable energies
    #######################################################
    if wo_total is not None and wt_total is not None:
        x = np.arange(len(wo_total))
        width = 0.35

        fig, ax = plt.subplots()

        for i in range(len(wo_total)):
            wo_total[i] = wo_total[i] / 3600000
            wt_total[i] = wt_total[i] / 3600000

        ax.bar(x - width/2, wo_total, width, color='skyblue', edgecolor='black', label="Wearable 1")
        ax.bar(x + width/2, wt_total, width, color='salmon', edgecolor='black', label="Wearable 2")

        ax.set_xticks(x)
        ax.set_xticklabels([str(i+1) for i in x])
        ax.set_xlabel("Client")
        ax.set_ylabel("Wearable Energy Usage (Wh)")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()

        fig.tight_layout()
        path = os.path.join(save_dir, f"wearable_energy.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")