"""
Replot Pearson correlation scatter graphs — ALL fonts at size 36.
Reads from pearson_skeleton_data.csv. No TensorFlow needed.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

OUTPUT_DIR = r"d:\labdatanew_Seemant\outputs"
CSV_PATH = os.path.join(OUTPUT_DIR, "pearson_skeleton_data.csv")

df = pd.read_csv(CSV_PATH)
valid_ids = df["Image_ID"].tolist()

metrics = ["Connected", "Branches", "Endpoints"]
metric_labels = {
    "Connected": "Connected Components",
    "Branches": "Branch Points",
    "Endpoints": "Endpoints"
}

# ── ALL fonts at 36 ─────────────────────────────────────────────────────────
FONT_SIZE = 36

colors_up = "#2196F3"
colors_dl = "#FF5722"

fig, axes = plt.subplots(2, 3, figsize=(54, 34))
fig.suptitle("Pearson Correlation: Ground Truth vs Predicted Skeleton Properties\n"
             "(Each point = one test image)",
             fontsize=FONT_SIZE, fontweight="bold", y=0.98)

for col, m in enumerate(metrics):
    gt_arr = df[f"GT_{m}"].values.astype(float)
    up_arr = df[f"UNetPP_{m}"].values.astype(float)
    dl_arr = df[f"DeepLabV3_{m}"].values.astype(float)

    r_up, p_up = pearsonr(gt_arr, up_arr)
    r_dl, p_dl = pearsonr(gt_arr, dl_arr)

    # ---- Row 0: UNet++ ----
    ax = axes[0, col]
    ax.scatter(gt_arr, up_arr, c=colors_up, s=120, edgecolors='k', linewidths=0.5, zorder=3)
    if np.std(gt_arr) > 0 and np.std(up_arr) > 0:
        z = np.polyfit(gt_arr, up_arr, 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(gt_arr.min(), gt_arr.max(), 100)
        ax.plot(x_range, p_line(x_range), '--', color=colors_up, alpha=0.7, linewidth=2.5)
    for i, img_id in enumerate(valid_ids):
        ax.annotate(str(img_id), (gt_arr[i], up_arr[i]),
                     fontsize=FONT_SIZE * 0.4, ha='left', va='bottom', xytext=(4, 4),
                     textcoords='offset points')
    ax.set_xlabel(f"Ground Truth {metric_labels[m]}", fontsize=FONT_SIZE)
    ax.set_ylabel(f"UNet++ Predicted {metric_labels[m]}", fontsize=FONT_SIZE)
    ax.set_title(f"UNet++ - {metric_labels[m]}\nr = {r_up:.4f}, p = {p_up:.4f}",
                 fontsize=FONT_SIZE, fontweight="bold", color=colors_up)
    ax.tick_params(axis='both', labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.3)

    # ---- Row 1: DeepLabV3+ ----
    ax = axes[1, col]
    ax.scatter(gt_arr, dl_arr, c=colors_dl, s=120, edgecolors='k', linewidths=0.5, zorder=3)
    if np.std(gt_arr) > 0 and np.std(dl_arr) > 0:
        z = np.polyfit(gt_arr, dl_arr, 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(gt_arr.min(), gt_arr.max(), 100)
        ax.plot(x_range, p_line(x_range), '--', color=colors_dl, alpha=0.7, linewidth=2.5)
    for i, img_id in enumerate(valid_ids):
        ax.annotate(str(img_id), (gt_arr[i], dl_arr[i]),
                     fontsize=FONT_SIZE * 0.4, ha='left', va='bottom', xytext=(4, 4),
                     textcoords='offset points')
    ax.set_xlabel(f"Ground Truth {metric_labels[m]}", fontsize=FONT_SIZE)
    ax.set_ylabel(f"DeepLabV3+ Predicted {metric_labels[m]}", fontsize=FONT_SIZE)
    ax.set_title(f"DeepLabV3+ - {metric_labels[m]}\nr = {r_dl:.4f}, p = {p_dl:.4f}",
                 fontsize=FONT_SIZE, fontweight="bold", color=colors_dl)
    ax.tick_params(axis='both', labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plot_path = os.path.join(OUTPUT_DIR, "pearson_correlation_plots_font36.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved -> {plot_path}")
