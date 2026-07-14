import os
import pandas as pd
import matplotlib.pyplot as plt

# Paths
LOG_CSV = r"d:\labdatanew_Seemant\unet++\outputs\training_log.csv"
OUTPUT_DIR = r"d:\labdatanew_Seemant\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load training logs
if not os.path.exists(LOG_CSV):
    print(f"Error: Log file not found at {LOG_CSV}")
    exit(1)

df = pd.read_csv(LOG_CSV)

# ── Plot definitions ─────────────────────────────────────────────────────────
TITLE_SIZE  = 36
LABEL_SIZE  = 28
TICK_SIZE   = 22
LEGEND_SIZE = 22
LINE_WIDTH  = 3.5

plots = [
    {
        "filename": "unetpp_loss.png",
        "title": "Model Loss (BCE + Dice Combined)",
        "xlabel": "Epoch",
        "ylabel": "Loss Value",
        "lines": [
            {"x": df["epoch"], "y": df["loss"],     "label": "Train Loss", "color": "#e74c3c", "ls": "-"},
            {"x": df["epoch"], "y": df["val_loss"],  "label": "Val Loss",   "color": "#c0392b", "ls": "--"},
        ],
    },
    {
        "filename": "unetpp_dice.png",
        "title": "Vessel Dice Coefficient (F1-Score)",
        "xlabel": "Epoch",
        "ylabel": "Dice Score",
        "lines": [
            {"x": df["epoch"], "y": df["dice_coeff"],     "label": "Train Dice", "color": "#2ecc71", "ls": "-"},
            {"x": df["epoch"], "y": df["val_dice_coeff"],  "label": "Val Dice",   "color": "#27ae60", "ls": "--"},
        ],
    },
    {
        "filename": "unetpp_iou.png",
        "title": "Mean Intersection over Union\n(IoU / Jaccard Index)",
        "xlabel": "Epoch",
        "ylabel": "IoU Score",
        "lines": [
            {"x": df["epoch"], "y": df["iou_score"],     "label": "Train IoU", "color": "#3498db", "ls": "-"},
            {"x": df["epoch"], "y": df["val_iou_score"],  "label": "Val IoU",   "color": "#2980b9", "ls": "--"},
        ],
    },
    {
        "filename": "unetpp_precision_recall.png",
        "title": "Precision and Recall\n(Vessel Detection Rates)",
        "xlabel": "Epoch",
        "ylabel": "Score Value",
        "lines": [
            {"x": df["epoch"], "y": df["precision"],     "label": "Train Precision", "color": "#9b59b6", "ls": "-"},
            {"x": df["epoch"], "y": df["val_precision"],  "label": "Val Precision",   "color": "#8e44ad", "ls": "--"},
            {"x": df["epoch"], "y": df["recall"],         "label": "Train Recall",    "color": "#f1c40f", "ls": "-"},
            {"x": df["epoch"], "y": df["val_recall"],     "label": "Val Recall",      "color": "#f39c12", "ls": "--"},
        ],
    },
]

# ── Generate 4 separate figures ──────────────────────────────────────────────
for p in plots:
    fig, ax = plt.subplots(figsize=(12, 9))

    for line in p["lines"]:
        ax.plot(line["x"], line["y"], label=line["label"],
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

