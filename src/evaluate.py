"""
Evaluation metrics and visualization functions for surgical workflow analysis.

Computes frame-wise F1-score, mean average precision, segment-level edit
score, and generates publication-quality plots for the project report.

Author: Omar Morsi (40236376)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    confusion_matrix,
)


def compute_phase_f1(predictions, targets, num_classes=7):
    """Compute macro-averaged F1-score for phase recognition.

    Args:
        predictions (torch.Tensor): Predicted phase labels, shape (N,).
        targets (torch.Tensor): Ground truth phase labels, shape (N,).
        num_classes (int): Number of surgical phases. Default: 7.

    Returns:
        float: Macro-averaged F1-score across all phases.

    Example:
        >>> preds = torch.tensor([0, 1, 2, 1, 0])
        >>> targets = torch.tensor([0, 1, 1, 1, 0])
        >>> f1 = compute_phase_f1(preds, targets)
        >>> print(f"{f1:.4f}")
        0.7778
    """
    preds_np = predictions.numpy().astype(int)
    targets_np = targets.numpy().astype(int)
    return f1_score(
        targets_np, preds_np, average="macro",
        labels=list(range(num_classes)), zero_division=0
    )


def compute_per_phase_f1(predictions, targets, num_classes=7):
    """Compute per-class F1-scores for phase recognition.

    Args:
        predictions (torch.Tensor): Predicted phase labels, shape (N,).
        targets (torch.Tensor): Ground truth phase labels, shape (N,).
        num_classes (int): Number of surgical phases. Default: 7.

    Returns:
        np.ndarray: F1-score for each phase, shape (num_classes,).

    Example:
        >>> preds = torch.tensor([0, 1, 2, 1, 0])
        >>> targets = torch.tensor([0, 1, 1, 1, 0])
        >>> per_f1 = compute_per_phase_f1(preds, targets)
        >>> print(per_f1.shape)
        (7,)
    """
    preds_np = predictions.numpy().astype(int)
    targets_np = targets.numpy().astype(int)
    return f1_score(
        targets_np, preds_np, average=None,
        labels=list(range(num_classes)), zero_division=0
    )


def compute_tool_map(predictions, targets):
    """Compute mean Average Precision (mAP) for tool detection.

    Args:
        predictions (torch.Tensor): Predicted tool probabilities, shape (N, 7).
        targets (torch.Tensor): Ground truth binary tool labels, shape (N, 7).

    Returns:
        float: Mean Average Precision across all tools.

    Example:
        >>> preds = torch.rand(100, 7)
        >>> targets = torch.randint(0, 2, (100, 7)).float()
        >>> mAP = compute_tool_map(preds, targets)
        >>> print(f"{mAP:.4f}")
    """
    return float(compute_per_tool_ap(predictions, targets).mean())


def compute_per_tool_ap(predictions, targets):
    """Compute per-tool Average Precision scores.

    Args:
        predictions (torch.Tensor): Predicted tool probabilities, shape (N, 7).
        targets (torch.Tensor): Ground truth binary tool labels, shape (N, 7).

    Returns:
        np.ndarray: Average precision for each tool, shape (7,).

    Example:
        >>> preds = torch.rand(100, 7)
        >>> targets = torch.randint(0, 2, (100, 7)).float()
        >>> per_ap = compute_per_tool_ap(preds, targets)
        >>> print(per_ap.shape)
        (7,)
    """
    preds_np = predictions.numpy()
    targets_np = targets.numpy()
    aps = []
    for i in range(targets_np.shape[1]):
        if targets_np[:, i].sum() > 0:
            aps.append(average_precision_score(targets_np[:, i], preds_np[:, i]))
        else:
            aps.append(0.0)
    return np.array(aps)


def compute_edit_score(predictions, targets):
    """Compute segment-level edit distance score for temporal consistency.

    Measures how "smooth" phase predictions are by comparing the sequence
    of phase segments (run-length encoded) between prediction and ground
    truth using normalized Levenshtein distance.

    A score of 1.0 means perfect segment match; lower scores indicate
    more "flickering" in predictions.

    Args:
        predictions (torch.Tensor): Predicted phase labels, shape (T,).
        targets (torch.Tensor): Ground truth phase labels, shape (T,).

    Returns:
        float: Normalized edit score between 0.0 and 1.0.

    Example:
        >>> preds = torch.tensor([0, 0, 1, 1, 2, 2])
        >>> targets = torch.tensor([0, 0, 1, 1, 2, 2])
        >>> score = compute_edit_score(preds, targets)
        >>> print(f"{score:.2f}")
        1.00
    """
    def run_length_encode(seq):
        """Convert a sequence to its run-length encoded segments."""
        segments = []
        if len(seq) == 0:
            return segments
        current = seq[0].item()
        segments.append(current)
        for val in seq[1:]:
            if val.item() != current:
                current = val.item()
                segments.append(current)
        return segments

    pred_segments = run_length_encode(predictions)
    target_segments = run_length_encode(targets)

    # Levenshtein distance between segment sequences
    m, n = len(pred_segments), len(target_segments)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred_segments[i - 1] == target_segments[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    edit_dist = dp[m][n]
    max_len = max(m, n)
    if max_len == 0:
        return 1.0
    return 1.0 - edit_dist / max_len


def compute_phase_accuracy(predictions, targets):
    """Compute frame-wise accuracy for phase recognition.

    Args:
        predictions (torch.Tensor): Predicted phase labels, shape (N,).
        targets (torch.Tensor): Ground truth phase labels, shape (N,).

    Returns:
        float: Frame-wise accuracy.

    Example:
        >>> preds = torch.tensor([0, 1, 2, 1, 0])
        >>> targets = torch.tensor([0, 1, 1, 1, 0])
        >>> acc = compute_phase_accuracy(preds, targets)
        >>> print(f"{acc:.2f}")
        0.80
    """
    correct = (predictions == targets).sum().item()
    total = targets.shape[0]
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------


def plot_training_curves(histories, model_names, save_path=None):
    """Plot training and validation loss/F1 curves for all models.

    Creates a 2x2 grid of subplots showing total loss, phase loss,
    validation F1, and validation mAP over epochs.

    Args:
        histories (list): List of history dicts from Trainer.train().
        model_names (list): List of model name strings for the legend.
        save_path (str, optional): File path to save the figure.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Example:
        >>> fig = plot_training_curves([hist1, hist2],
        ...                            ["Baseline", "MS-TCN"])
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for hist, name in zip(histories, model_names):
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0, 0].plot(epochs, hist["train_loss"], label=f"{name} (train)")
        axes[0, 0].plot(epochs, hist["val_loss"], "--", label=f"{name} (val)")
        axes[0, 1].plot(epochs, hist["train_phase_loss"], label=f"{name} (train)")
        axes[0, 1].plot(epochs, hist["val_phase_loss"], "--", label=f"{name} (val)")
        axes[1, 0].plot(epochs, hist["val_phase_f1"], label=name)
        axes[1, 1].plot(epochs, hist["val_tool_map"], label=name)

    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].set_title("Phase Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()

    axes[1, 0].set_title("Validation Phase F1")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].legend()

    axes[1, 1].set_title("Validation Tool mAP")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("mAP")
    axes[1, 1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(predictions, targets, phase_names, save_path=None):
    """Plot a confusion matrix heatmap for phase predictions.

    Args:
        predictions (torch.Tensor): Predicted phase labels, shape (N,).
        targets (torch.Tensor): Ground truth phase labels, shape (N,).
        phase_names (list): List of phase name strings.
        save_path (str, optional): File path to save the figure.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Example:
        >>> preds = torch.randint(0, 7, (1000,))
        >>> targets = torch.randint(0, 7, (1000,))
        >>> names = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
        >>> fig = plot_confusion_matrix(preds, targets, names)
    """
    cm = confusion_matrix(
        targets.numpy(), predictions.numpy(),
        labels=list(range(len(phase_names)))
    )
    # Normalize by row (true label)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=phase_names, yticklabels=phase_names, ax=ax
    )
    ax.set_xlabel("Predicted Phase")
    ax.set_ylabel("True Phase")
    ax.set_title("Phase Recognition Confusion Matrix (Normalized)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_per_class_metrics(phase_f1s, tool_aps, phase_names, tool_names,
                           save_path=None):
    """Plot bar charts for per-phase F1 and per-tool AP.

    Args:
        phase_f1s (np.ndarray): F1-score per phase, shape (7,).
        tool_aps (np.ndarray): Average precision per tool, shape (7,).
        phase_names (list): Phase name strings.
        tool_names (list): Tool name strings.
        save_path (str, optional): File path to save the figure.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Example:
        >>> fig = plot_per_class_metrics(np.random.rand(7), np.random.rand(7),
        ...                              ["P1"]*7, ["T1"]*7)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors_phase = sns.color_palette("viridis", len(phase_names))
    ax1.barh(phase_names, phase_f1s, color=colors_phase)
    ax1.set_xlabel("F1 Score")
    ax1.set_title("Per-Phase F1 Score")
    ax1.set_xlim(0, 1)
    for i, v in enumerate(phase_f1s):
        ax1.text(v + 0.01, i, f"{v:.3f}", va="center")

    colors_tool = sns.color_palette("magma", len(tool_names))
    ax2.barh(tool_names, tool_aps, color=colors_tool)
    ax2.set_xlabel("Average Precision")
    ax2.set_title("Per-Tool Average Precision")
    ax2.set_xlim(0, 1)
    for i, v in enumerate(tool_aps):
        ax2.text(v + 0.01, i, f"{v:.3f}", va="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_timeline_ribbon(predictions, targets, phase_names, video_name="",
                         save_path=None):
    """Plot ground truth vs predicted phase timeline for a single video.

    Creates a ribbon visualization showing the sequence of phases over time,
    with ground truth on top and predictions on the bottom.

    Args:
        predictions (torch.Tensor): Predicted phase labels, shape (T,).
        targets (torch.Tensor): Ground truth phase labels, shape (T,).
        phase_names (list): Phase name strings.
        video_name (str): Video identifier for the title. Default: "".
        save_path (str, optional): File path to save the figure.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Example:
        >>> preds = torch.randint(0, 7, (500,))
        >>> targets = torch.randint(0, 7, (500,))
        >>> fig = plot_timeline_ribbon(preds, targets, ["P1"]*7, "Video 41")
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 3), sharex=True)
    cmap = plt.colormaps["tab10"]

    for ax, data, label in zip(axes, [targets, predictions],
                                ["Ground Truth", "Prediction"]):
        data_np = data.numpy().reshape(1, -1)
        ax.imshow(data_np, aspect="auto", cmap=cmap, vmin=0,
                  vmax=len(phase_names) - 1, interpolation="nearest")
        ax.set_ylabel(label, fontsize=10)
        ax.set_yticks([])

    axes[1].set_xlabel("Time (frames at 1 fps)")
    fig.suptitle(f"Phase Timeline -- {video_name}", fontsize=12)

    # Add colorbar legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=cmap(i), label=name)
               for i, name in enumerate(phase_names)]
    fig.legend(handles=patches, loc="center right", fontsize=8,
               bbox_to_anchor=(1.15, 0.5))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cooccurrence_heatmap(learned_matrix, gt_matrix, phase_names, tool_names,
                              save_path=None):
    """Plot learned vs ground truth tool-phase co-occurrence heatmaps.

    Args:
        learned_matrix (np.ndarray): Learned co-occurrence, shape (7, 7).
        gt_matrix (np.ndarray): Ground truth co-occurrence, shape (7, 7).
        phase_names (list): Phase name strings.
        tool_names (list): Tool name strings.
        save_path (str, optional): File path to save the figure.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Example:
        >>> fig = plot_cooccurrence_heatmap(np.random.rand(7, 7),
        ...                                 np.random.rand(7, 7),
        ...                                 ["P1"]*7, ["T1"]*7)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(gt_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=tool_names, yticklabels=phase_names, ax=ax1)
    ax1.set_title("Ground Truth P(tool | phase)")

    sns.heatmap(learned_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=tool_names, yticklabels=phase_names, ax=ax2)
    ax2.set_title("Model Predicted P(tool | phase)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(results_dict, save_path=None):
    """Plot ablation bar chart comparing all model variants.

    Args:
        results_dict (dict): Mapping model_name -> {"phase_f1": float,
            "tool_map": float, "edit_score": float}.
        save_path (str, optional): File path to save the figure.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Example:
        >>> results = {"Baseline": {"phase_f1": 0.65, "tool_map": 0.75,
        ...                          "edit_score": 0.55},
        ...            "MS-TCN": {"phase_f1": 0.85, "tool_map": 0.90,
        ...                        "edit_score": 0.80}}
        >>> fig = plot_model_comparison(results)
    """
    model_names = list(results_dict.keys())
    metrics = ["phase_f1", "tool_map", "edit_score"]
    metric_labels = ["Phase F1", "Tool mAP", "Edit Score"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = sns.color_palette("deep", len(model_names))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        values = [results_dict[name][metric] for name in model_names]
        bars = ax.bar(model_names, values, color=colors)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=9)
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
