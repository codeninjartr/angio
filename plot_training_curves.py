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

# Create a beautiful 2x2 grid plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("UNet++ (ResNet50 Backbone) Training & Validation Curves", fontsize=16, fontweight="bold")

# Plot 1: Loss Curve
axes[0, 0].plot(df["epoch"], df["loss"], label="Train Loss", color="#e74c3c", linewidth=2.5)
axes[0, 0].plot(df["epoch"], df["val_loss"], label="Val Loss", color="#c0392b", linewidth=2.5, linestyle="--")
axes[0, 0].set_title("Model Loss (BCE + Dice Combined)", fontsize=12, fontweight="bold")
axes[0, 0].set_xlabel("Epoch", fontsize=10)
axes[0, 0].set_ylabel("Loss Value", fontsize=10)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Plot 2: Dice Coefficient (F1-Score)
axes[0, 1].plot(df["epoch"], df["dice_coeff"], label="Train Dice", color="#2ecc71", linewidth=2.5)
axes[0, 1].plot(df["epoch"], df["val_dice_coeff"], label="Val Dice", color="#27ae60", linewidth=2.5, linestyle="--")
axes[0, 1].set_title("Vessel Dice Coefficient (F1-Score)", fontsize=12, fontweight="bold")
axes[0, 1].set_xlabel("Epoch", fontsize=10)
axes[0, 1].set_ylabel("Dice Score", fontsize=10)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Plot 3: Mean IoU Score
axes[1, 0].plot(df["epoch"], df["iou_score"], label="Train IoU", color="#3498db", linewidth=2.5)
axes[1, 0].plot(df["epoch"], df["val_iou_score"], label="Val IoU", color="#2980b9", linewidth=2.5, linestyle="--")
axes[1, 0].set_title("Mean Intersection over Union (IoU / Jaccard Index)", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("Epoch", fontsize=10)
axes[1, 0].set_ylabel("IoU Score", fontsize=10)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Plot 4: Precision & Recall (Sensitivity)
axes[1, 1].plot(df["epoch"], df["precision"], label="Train Precision", color="#9b59b6", linewidth=2)
axes[1, 1].plot(df["epoch"], df["val_precision"], label="Val Precision", color="#8e44ad", linewidth=2, linestyle="--")
axes[1, 1].plot(df["epoch"], df["recall"], label="Train Recall", color="#f1c40f", linewidth=2)
axes[1, 1].plot(df["epoch"], df["val_recall"], label="Val Recall", color="#f39c12", linewidth=2, linestyle="--")
axes[1, 1].set_title("Precision and Recall (Vessel Detection Rates)", fontsize=12, fontweight="bold")
axes[1, 1].set_xlabel("Epoch", fontsize=10)
axes[1, 1].set_ylabel("Score Value", fontsize=10)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save output
plot_path = os.path.join(OUTPUT_DIR, "training_curve.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Success! Generated training curve plot at: {plot_path}")
