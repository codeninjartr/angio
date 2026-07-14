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

# ── Plot definitions ─────────────────────────────────────────────────────────
TITLE_SIZE  = 36
LABEL_SIZE  = 28
TICK_SIZE   = 22
LEGEND_SIZE = 22
LINE_WIDTH  = 3.5

plots = [
    {
        "filename": "deeplabv3plus_loss.png",
        "title": "Model Loss (BCE + Dice Combined)",
        "xlabel": "Epoch",
        "ylabel": "Loss Value",
        "lines": [
            {"data": losses,      "label": "Train Loss", "color": "#e74c3c", "ls": "-"},
            {"data": val_losses,   "label": "Val Loss",   "color": "#c0392b", "ls": "--"},
        ],
    },
    {
        "filename": "deeplabv3plus_dice.png",
        "title": "Vessel Dice Coefficient (F1-Score)",
        "xlabel": "Epoch",
        "ylabel": "Dice Score",
        "lines": [
            {"data": dice_train, "label": "Train Dice", "color": "#2ecc71", "ls": "-"},
            {"data": dice_val,   "label": "Val Dice",   "color": "#27ae60", "ls": "--"},
        ],
    },
    {
        "filename": "deeplabv3plus_iou.png",
        "title": "Mean Intersection over Union\n(IoU / Jaccard Index)",
        "xlabel": "Epoch",
        "ylabel": "IoU Score",
        "lines": [
            {"data": ious,      "label": "Train IoU", "color": "#3498db", "ls": "-"},
            {"data": val_ious,  "label": "Val IoU",   "color": "#2980b9", "ls": "--"},
        ],
    },
    {
        "filename": "deeplabv3plus_accuracy.png",
        "title": "Pixel-wise Accuracy",
        "xlabel": "Epoch",
        "ylabel": "Accuracy",
        "lines": [
            {"data": accuracies,     "label": "Train Accuracy", "color": "#9b59b6", "ls": "-"},
            {"data": val_accuracies, "label": "Val Accuracy",   "color": "#8e44ad", "ls": "--"},
        ],
    },
]

# ── Generate 4 separate figures ──────────────────────────────────────────────
for p in plots:
    fig, ax = plt.subplots(figsize=(12, 9))

    for line in p["lines"]:
        ax.plot(epochs, line["data"], label=line["label"],
                color=line["color"], linewidth=LINE_WIDTH, linestyle=line["ls"])

    ax.set_title(p["title"], fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel(p["xlabel"], fontsize=LABEL_SIZE)
    ax.set_ylabel(p["ylabel"], fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_SIZE)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, p["filename"])
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
