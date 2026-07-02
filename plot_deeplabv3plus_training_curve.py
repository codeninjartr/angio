"""
plot_deeplabv3plus_training_curve.py
====================================
Generates a 2x2 training curve grid for DeepLabV3+ matching the UNet++ style:
  1. Loss (BCE + Dice)
  2. Dice Coefficient (derived from IoU)
  3. IoU Score
  4. Accuracy

Reads from: deeplabv3/outputs/result.txt and result_part2.txt
Saves to:   outputs/deeplabv3plus_training_curve.png
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = r"d:\labdatanew_Seemant\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Parse training logs ──────────────────────────────────────────────────────
filepaths = [
    r"d:\labdatanew_Seemant\deeplabv3\outputs\result.txt",
    r"d:\labdatanew_Seemant\deeplabv3\outputs\result_part2.txt"
]

losses, val_losses = [], []
accuracies, val_accuracies = [], []
ious, val_ious = [], []

for filepath in filepaths:
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found, skipping.")
        continue
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if "val_loss:" in line:
                try:
                    m = re.search(r'- loss: ([\d\.]+)', line)
                    if m: losses.append(float(m.group(1)))

                    m = re.search(r'- accuracy: ([\d\.]+)', line)
                    if m: accuracies.append(float(m.group(1)))

                    m = re.search(r'- iou_score: ([\d\.]+)', line)
                    if m: ious.append(float(m.group(1)))

                    m = re.search(r'- val_loss: ([\d\.]+)', line)
                    if m: val_losses.append(float(m.group(1)))

                    m = re.search(r'- val_accuracy: ([\d\.]+)', line)
                    if m: val_accuracies.append(float(m.group(1)))

                    m = re.search(r'- val_iou_score: ([\d\.]+)', line)
                    if m: val_ious.append(float(m.group(1)))
                except Exception as e:
                    print(f"Error parsing: {e}")

print(f"Parsed {len(losses)} total epochs.")

# ── Derive Dice from IoU: Dice = 2*IoU / (1+IoU) ────────────────────────────
dice_train = (2.0 * np.array(ious) / (1.0 + np.array(ious))).tolist()
dice_val = (2.0 * np.array(val_ious) / (1.0 + np.array(val_ious))).tolist()

epochs = list(range(len(losses)))

# ── Plot 2x2 grid matching UNet++ style ──────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DeepLabV3+ (ResNet50 Backbone) Training & Validation Curves", fontsize=16, fontweight="bold")

# Plot 1: Loss
axes[0, 0].plot(epochs, losses, label="Train Loss", color="#e74c3c", linewidth=2.5)
axes[0, 0].plot(epochs, val_losses, label="Val Loss", color="#c0392b", linewidth=2.5, linestyle="--")
axes[0, 0].set_title("Model Loss (BCE + Dice Combined)", fontsize=12, fontweight="bold")
axes[0, 0].set_xlabel("Epoch", fontsize=10)
axes[0, 0].set_ylabel("Loss Value", fontsize=10)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Plot 2: Dice Coefficient
axes[0, 1].plot(epochs, dice_train, label="Train Dice", color="#2ecc71", linewidth=2.5)
axes[0, 1].plot(epochs, dice_val, label="Val Dice", color="#27ae60", linewidth=2.5, linestyle="--")
axes[0, 1].set_title("Vessel Dice Coefficient (F1-Score)", fontsize=12, fontweight="bold")
axes[0, 1].set_xlabel("Epoch", fontsize=10)
axes[0, 1].set_ylabel("Dice Score", fontsize=10)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Plot 3: IoU Score
axes[1, 0].plot(epochs, ious, label="Train IoU", color="#3498db", linewidth=2.5)
axes[1, 0].plot(epochs, val_ious, label="Val IoU", color="#2980b9", linewidth=2.5, linestyle="--")
axes[1, 0].set_title("Mean Intersection over Union (IoU / Jaccard Index)", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("Epoch", fontsize=10)
axes[1, 0].set_ylabel("IoU Score", fontsize=10)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Plot 4: Accuracy (DeepLabV3+ logged accuracy instead of precision/recall)
axes[1, 1].plot(epochs, accuracies, label="Train Accuracy", color="#9b59b6", linewidth=2.5)
axes[1, 1].plot(epochs, val_accuracies, label="Val Accuracy", color="#8e44ad", linewidth=2.5, linestyle="--")
axes[1, 1].set_title("Pixel-wise Accuracy", fontsize=12, fontweight="bold")
axes[1, 1].set_xlabel("Epoch", fontsize=10)
axes[1, 1].set_ylabel("Accuracy", fontsize=10)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plot_path = os.path.join(OUTPUT_DIR, "deeplabv3plus_training_curve.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Success! Generated training curve plot at: {plot_path}")
