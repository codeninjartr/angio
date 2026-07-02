"""
plot_parsed_training_curves.py - UNet++ Parsed Training Curves
==============================================================
Generates a 3-panel plot matching the DeepLabV3+ parsed_training_curves style:
  1. Loss Curves (Combined)
  2. Accuracy Curves (Combined)
  3. IoU Score Curves (Combined)

Reads from: outputs/training_log.csv
Saves to:   outputs/parsed_training_curves.png
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_parsed_training_curves():
    log_path = "outputs/training_log.csv"
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    # Read CSV manually (no pandas dependency)
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    losses = [float(r['loss']) for r in rows]
    val_losses = [float(r['val_loss']) for r in rows]
    accuracies = [float(r['accuracy']) for r in rows]
    val_accuracies = [float(r['val_accuracy']) for r in rows]
    ious = [float(r['iou_score']) for r in rows]
    val_ious = [float(r['val_iou_score']) for r in rows]

    epochs_range = range(1, len(rows) + 1)

    plt.figure(figsize=(18, 5))

    # ── Panel 1: Loss Curves ─────────────────────────────────────────────────
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, losses, 'b-o', label='Training Loss', markersize=4)
    plt.plot(epochs_range, val_losses, 'r-o', label='Validation Loss', markersize=4)
    plt.title('Loss Curves (Combined)')
    plt.xlabel('Effective Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # ── Panel 2: Accuracy Curves ─────────────────────────────────────────────
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, accuracies, 'b-o', label='Training Accuracy', markersize=4)
    plt.plot(epochs_range, val_accuracies, 'r-o', label='Validation Accuracy', markersize=4)
    plt.title('Accuracy Curves (Combined)')
    plt.xlabel('Effective Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # ── Panel 3: IoU Score Curves ────────────────────────────────────────────
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, ious, 'b-o', label='Training IoU', markersize=4)
    plt.plot(epochs_range, val_ious, 'r-o', label='Validation IoU', markersize=4)
    plt.title('IoU Score Curves (Combined)')
    plt.xlabel('Effective Epochs')
    plt.ylabel('IoU Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    out_path = "outputs/parsed_training_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Parsed training curves saved to -> {out_path}")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    plot_parsed_training_curves()
