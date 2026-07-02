"""
plot_training_metrics.py - DeepLabV3+ Training Metrics Plot
============================================================
Generates a 3-panel plot matching the UNet++ training_metrics_plot style:
  1. Loss (BCE + Dice)
  2. Dice Coefficient (derived from IoU: Dice = 2*IoU / (1+IoU))
  3. IoU Score

Reads from: outputs/result.txt and outputs/result_part2.txt
Saves to:   outputs/training_metrics_plot.png
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt


def parse_training_logs():
    """Parse epoch-level metrics from result.txt and result_part2.txt."""
    filepaths = ["outputs/result.txt", "outputs/result_part2.txt"]

    losses = []
    ious = []
    val_losses = []
    val_ious = []

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping.")
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if "val_loss:" in line:
                    try:
                        loss_match = re.search(r'- loss: ([\d\.]+)', line)
                        if loss_match:
                            losses.append(float(loss_match.group(1)))

                        iou_match = re.search(r'- iou_score: ([\d\.]+)', line)
                        if iou_match:
                            ious.append(float(iou_match.group(1)))

                        val_loss_match = re.search(r'- val_loss: ([\d\.]+)', line)
                        if val_loss_match:
                            val_losses.append(float(val_loss_match.group(1)))

                        val_iou_match = re.search(r'- val_iou_score: ([\d\.]+)', line)
                        if val_iou_match:
                            val_ious.append(float(val_iou_match.group(1)))
                    except Exception as e:
                        print(f"Error parsing line: {line.strip()} -> {e}")

    return losses, val_losses, ious, val_ious


def iou_to_dice(iou_values):
    """Convert IoU values to Dice coefficient using: Dice = 2*IoU / (1+IoU)."""
    iou_arr = np.array(iou_values)
    return (2.0 * iou_arr / (1.0 + iou_arr)).tolist()


def plot_training_metrics():
    losses, val_losses, ious, val_ious = parse_training_logs()

    if not losses:
        print("Error: No training data parsed. Check result.txt files.")
        return

    epochs_range = range(len(losses))

    # Derive Dice coefficient from IoU
    dice_train = iou_to_dice(ious)
    dice_val = iou_to_dice(val_ious)

    print(f"Parsed {len(losses)} total epochs from training logs.")

    # ── Plot: 3-panel matching UNet++ training_metrics_plot style ─────────
    metrics_to_plot = [
        (losses, val_losses, "Loss (BCE + Dice)", "Train Loss (BCE + Dice)", "Val Loss (BCE + Dice)"),
        (dice_train, dice_val, "Dice Coefficient", "Train Dice Coefficient", "Val Dice Coefficient"),
        (ious, val_ious, "IoU Score", "Train IoU Score", "Val IoU Score"),
    ]

    plt.figure(figsize=(15, 5))

    for i, (train_vals, val_vals, title, train_label, val_label) in enumerate(metrics_to_plot, 1):
        plt.subplot(1, 3, i)
        plt.plot(epochs_range, train_vals, label=train_label, marker='o', markersize=4)
        plt.plot(epochs_range, val_vals, label=val_label, marker='o', markersize=4)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

    plt.tight_layout()

    out_path = "outputs/training_metrics_plot.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Training metrics plot saved to -> {out_path}")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    plot_training_metrics()
