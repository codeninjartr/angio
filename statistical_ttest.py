"""
statistical_ttest.py  -  Paired t-test: UNet++ vs DeepLabV3+
=============================================================
Loads both models, computes per-image Dice and IoU on test images (501-637),
then performs paired t-tests and Wilcoxon signed-rank tests to determine
if UNet++ is statistically significantly better than DeepLabV3+.
Generates publication-quality result plots and tables.
"""

import os
import sys
import numpy as np
import cv2
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon
import tensorflow as tf

# ── Config ──────────────────────────────────────────────────────────────────
IMAGE_IDS   = list(range(501, 638))
IMG_SIZE    = (256, 256)
THRESHOLD   = 0.5
RGB_DIR     = r"d:\labdatanew_Seemant\137_rgb_mask\RGB"
MASK_DIR    = r"d:\labdatanew_Seemant\137_rgb_mask\MASK"
OUTPUT_DIR  = r"d:\labdatanew_Seemant\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Per-image metric functions ──────────────────────────────────────────────
def dice_score(gt, pred):
    """Compute Dice coefficient between two binary masks."""
    intersection = np.sum(gt * pred)
    if np.sum(gt) + np.sum(pred) == 0:
        return 1.0  # Both empty
    return (2.0 * intersection) / (np.sum(gt) + np.sum(pred))

def iou_score(gt, pred):
    """Compute IoU between two binary masks."""
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred) - intersection
    if union == 0:
        return 1.0  # Both empty
    return intersection / union

def precision_score(gt, pred):
    """Compute Precision."""
    tp = np.sum(gt * pred)
    fp = np.sum(pred) - tp
    if tp + fp == 0:
        return 1.0
    return tp / (tp + fp)

def recall_score(gt, pred):
    """Compute Recall (Sensitivity)."""
    tp = np.sum(gt * pred)
    fn = np.sum(gt) - tp
    if tp + fn == 0:
        return 1.0
    return tp / (tp + fn)

# ── Load models ─────────────────────────────────────────────────────────────
# UNet++
sys.path.insert(0, os.path.join("unet++", "src"))
from losses  import bce_dice_loss
from metrics import dice_coeff, iou_score as iou_metric, precision, recall

print("[1/5] Loading UNet++ model...")
unetpp_model = tf.keras.models.load_model(
    os.path.join("unet++", "models", "best_model.keras"),
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "dice_coeff"   : dice_coeff,
        "iou_score"    : iou_metric,
        "precision"    : precision,
        "recall"       : recall,
    },
    compile=False
)

# DeepLabV3+
sys.path.insert(0, "deeplabv3")
import importlib
import losses as dl_losses
importlib.reload(dl_losses)
import metrics as dl_metrics
importlib.reload(dl_metrics)

print("[2/5] Loading DeepLabV3+ model...")
deeplabv3_model = tf.keras.models.load_model(
    os.path.join("deeplabv3", "models", "best_model.keras"),
    custom_objects={
        "bce_dice_loss": dl_losses.bce_dice_loss,
        "iou_score"    : dl_metrics.iou_score,
    },
    compile=False
)

# ── Process images and compute per-image metrics ───────────────────────────
print("[3/5] Computing per-image Dice, IoU, Precision, Recall...")

unetpp_dice_list = []
unetpp_iou_list  = []
unetpp_prec_list = []
unetpp_rec_list  = []

dlv3_dice_list = []
dlv3_iou_list  = []
dlv3_prec_list = []
dlv3_rec_list  = []

valid_ids = []

for img_id in IMAGE_IDS:
    rgb_path  = os.path.join(RGB_DIR, f"{img_id}_RGB.jpg")
    mask_path = os.path.join(MASK_DIR, f"gauss_{img_id}_RGB.jpg")

    if not os.path.exists(rgb_path) or not os.path.exists(mask_path):
        continue

    # Load and preprocess
    img_bgr     = cv2.imread(rgb_path)
    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_norm    = img_resized.astype(np.float32) / 255.0

    # Ground truth mask
    gt_raw      = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_resized  = cv2.resize(gt_raw, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    gt_binary   = (gt_resized > 127).astype(np.uint8)

    # UNet++ prediction
    inp = np.expand_dims(img_norm, axis=0)
    unetpp_prob = unetpp_model.predict(inp, verbose=0)[0, :, :, 0]
    unetpp_mask = (unetpp_prob > THRESHOLD).astype(np.uint8)

    # DeepLabV3+ prediction
    deeplabv3_pred = deeplabv3_model.predict(inp, verbose=0)
    deeplabv3_mask = (deeplabv3_pred[0] > THRESHOLD).astype(np.float32).squeeze()
    deeplabv3_mask = (deeplabv3_mask > 0.5).astype(np.uint8)

    # Compute metrics
    valid_ids.append(img_id)

    unetpp_dice_list.append(dice_score(gt_binary, unetpp_mask))
    unetpp_iou_list.append(iou_score(gt_binary, unetpp_mask))
    unetpp_prec_list.append(precision_score(gt_binary, unetpp_mask))
    unetpp_rec_list.append(recall_score(gt_binary, unetpp_mask))

    dlv3_dice_list.append(dice_score(gt_binary, deeplabv3_mask))
    dlv3_iou_list.append(iou_score(gt_binary, deeplabv3_mask))
    dlv3_prec_list.append(precision_score(gt_binary, deeplabv3_mask))
    dlv3_rec_list.append(recall_score(gt_binary, deeplabv3_mask))

    if len(valid_ids) % 20 == 0:
        print(f"  Processed {len(valid_ids)} images...")

print(f"  Total images processed: {len(valid_ids)}")

# Convert to arrays
unetpp_dice = np.array(unetpp_dice_list)
unetpp_iou  = np.array(unetpp_iou_list)
unetpp_prec = np.array(unetpp_prec_list)
unetpp_rec  = np.array(unetpp_rec_list)

dlv3_dice = np.array(dlv3_dice_list)
dlv3_iou  = np.array(dlv3_iou_list)
dlv3_prec = np.array(dlv3_prec_list)
dlv3_rec  = np.array(dlv3_rec_list)

# ── Save per-image CSV ──────────────────────────────────────────────────────
print("[4/5] Saving per-image metrics CSV...")
csv_path = os.path.join(OUTPUT_DIR, "per_image_metrics.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Image_ID",
        "UNetPP_Dice", "UNetPP_IoU", "UNetPP_Precision", "UNetPP_Recall",
        "DeepLabV3_Dice", "DeepLabV3_IoU", "DeepLabV3_Precision", "DeepLabV3_Recall"
    ])
    for i, img_id in enumerate(valid_ids):
        writer.writerow([
            img_id,
            f"{unetpp_dice[i]:.6f}", f"{unetpp_iou[i]:.6f}",
            f"{unetpp_prec[i]:.6f}", f"{unetpp_rec[i]:.6f}",
            f"{dlv3_dice[i]:.6f}", f"{dlv3_iou[i]:.6f}",
            f"{dlv3_prec[i]:.6f}", f"{dlv3_rec[i]:.6f}",
        ])
print(f"  Saved -> {csv_path}")

# ── Statistical Tests ───────────────────────────────────────────────────────
print("[5/5] Running statistical tests...")

metrics_data = {
    "Dice Score": (unetpp_dice, dlv3_dice),
    "IoU Score":  (unetpp_iou, dlv3_iou),
    "Precision":  (unetpp_prec, dlv3_prec),
    "Recall":     (unetpp_rec, dlv3_rec),
}

results = {}
for name, (up_arr, dl_arr) in metrics_data.items():
    # Paired t-test
    t_stat, t_pval = ttest_rel(up_arr, dl_arr)

    # Wilcoxon signed-rank test (non-parametric alternative)
    try:
        w_stat, w_pval = wilcoxon(up_arr, dl_arr)
    except ValueError:
        w_stat, w_pval = float('nan'), float('nan')

    results[name] = {
        "unetpp_mean": np.mean(up_arr),
        "unetpp_std":  np.std(up_arr),
        "dlv3_mean":   np.mean(dl_arr),
        "dlv3_std":    np.std(dl_arr),
        "diff_mean":   np.mean(up_arr - dl_arr),
        "t_stat":      t_stat,
        "t_pval":      t_pval,
        "w_stat":      w_stat,
        "w_pval":      w_pval,
    }

    sig = "YES (p < 0.05)" if t_pval < 0.05 else "NO (p >= 0.05)"
    print(f"\n  {name}:")
    print(f"    UNet++     : {np.mean(up_arr):.4f} +/- {np.std(up_arr):.4f}")
    print(f"    DeepLabV3+ : {np.mean(dl_arr):.4f} +/- {np.std(dl_arr):.4f}")
    print(f"    Paired t-test : t = {t_stat:.4f}, p = {t_pval:.6f}")
    print(f"    Wilcoxon      : W = {w_stat:.1f}, p = {w_pval:.6f}")
    print(f"    Significant?  : {sig}")

# ── Plot 1: Box plots comparing models ──────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
fig.suptitle("Statistical Comparison: UNet++ vs DeepLabV3+\n(Per-Image Test Metrics, n = %d images)" % len(valid_ids),
             fontsize=14, fontweight="bold")

metric_names = list(metrics_data.keys())
colors = {"UNet++": "#2196F3", "DeepLabV3+": "#FF5722"}

for idx, name in enumerate(metric_names):
    ax = axes[idx]
    up_arr, dl_arr = metrics_data[name]
    r = results[name]

    bp = ax.boxplot([up_arr, dl_arr],
                    tick_labels=["UNet++", "DeepLabV3+"],
                    patch_artist=True,
                    widths=0.5,
                    medianprops=dict(color='black', linewidth=2))

    bp['boxes'][0].set_facecolor('#90CAF9')
    bp['boxes'][0].set_edgecolor('#1565C0')
    bp['boxes'][1].set_facecolor('#FFAB91')
    bp['boxes'][1].set_edgecolor('#D84315')

    # Add significance annotation
    y_max = max(up_arr.max(), dl_arr.max())
    y_min = min(up_arr.min(), dl_arr.min())
    y_range = y_max - y_min
    bar_y = y_max + y_range * 0.05

    if r["t_pval"] < 0.001:
        sig_text = "***"
    elif r["t_pval"] < 0.01:
        sig_text = "**"
    elif r["t_pval"] < 0.05:
        sig_text = "*"
    else:
        sig_text = "ns"

    # Draw bracket
    ax.plot([1, 1, 2, 2], [bar_y, bar_y + y_range*0.02, bar_y + y_range*0.02, bar_y],
            color='black', linewidth=1.5)
    ax.text(1.5, bar_y + y_range*0.03, f"p = {r['t_pval']:.4f}\n({sig_text})",
            ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add mean +/- std below
    ax.text(1, y_min - y_range*0.12,
            f"{r['unetpp_mean']:.4f}\n+/-{r['unetpp_std']:.4f}",
            ha='center', fontsize=8, color='#1565C0', fontweight='bold')
    ax.text(2, y_min - y_range*0.12,
            f"{r['dlv3_mean']:.4f}\n+/-{r['dlv3_std']:.4f}",
            ha='center', fontsize=8, color='#D84315', fontweight='bold')

    ax.set_ylim([y_min - y_range*0.25, bar_y + y_range*0.15])

plt.tight_layout(rect=[0, 0.02, 1, 0.93])
boxplot_path = os.path.join(OUTPUT_DIR, "statistical_ttest_boxplots.png")
plt.savefig(boxplot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Saved boxplots -> {boxplot_path}")

# ── Plot 2: Summary table as image ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')
ax.set_title("Statistical Test Results: UNet++ vs DeepLabV3+\n(Paired t-test & Wilcoxon Signed-Rank Test)",
             fontsize=14, fontweight="bold", pad=20)

table_data = []
for name in metric_names:
    r = results[name]
    sig_t = "***" if r["t_pval"] < 0.001 else "**" if r["t_pval"] < 0.01 else "*" if r["t_pval"] < 0.05 else "ns"
    sig_w = "***" if r["w_pval"] < 0.001 else "**" if r["w_pval"] < 0.01 else "*" if r["w_pval"] < 0.05 else "ns"

    better = "UNet++" if r["diff_mean"] > 0 else "DeepLabV3+"

    table_data.append([
        name,
        f"{r['unetpp_mean']:.4f} +/- {r['unetpp_std']:.4f}",
        f"{r['dlv3_mean']:.4f} +/- {r['dlv3_std']:.4f}",
        f"{r['t_stat']:.3f}",
        f"{r['t_pval']:.6f} ({sig_t})",
        f"{r['w_pval']:.6f} ({sig_w})",
        better,
    ])

col_labels = ["Metric", "UNet++ (mean+/-std)", "DeepLabV3+ (mean+/-std)",
              "t-statistic", "t-test p-value", "Wilcoxon p-value", "Better Model"]

table = ax.table(cellText=table_data, colLabels=col_labels,
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.8)

# Style header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#37474F')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Style data rows
for i in range(1, len(table_data) + 1):
    for j in range(len(col_labels)):
        if i % 2 == 0:
            table[i, j].set_facecolor('#ECEFF1')
        else:
            table[i, j].set_facecolor('#FFFFFF')

    # Highlight better model column
    last_col = len(col_labels) - 1
    better = table_data[i-1][last_col]
    if better == "UNet++":
        table[i, last_col].set_facecolor('#C8E6C9')
        table[i, last_col].set_text_props(fontweight='bold')
    else:
        table[i, last_col].set_facecolor('#FFCDD2')
        table[i, last_col].set_text_props(fontweight='bold')

# Add significance legend
fig.text(0.5, 0.02,
         "Significance: *** p < 0.001  |  ** p < 0.01  |  * p < 0.05  |  ns = not significant",
         ha='center', fontsize=10, style='italic', color='gray')

table_path = os.path.join(OUTPUT_DIR, "statistical_ttest_table.png")
plt.savefig(table_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved table   -> {table_path}")

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STATISTICAL TEST SUMMARY")
print("=" * 70)
print(f"  {'Metric':<15} {'UNet++':<20} {'DeepLabV3+':<20} {'p-value':<12} {'Sig?':<5}")
print("-" * 70)
for name in metric_names:
    r = results[name]
    sig = "***" if r["t_pval"] < 0.001 else "**" if r["t_pval"] < 0.01 else "*" if r["t_pval"] < 0.05 else "ns"
    print(f"  {name:<15} {r['unetpp_mean']:.4f}+/-{r['unetpp_std']:.4f}   "
          f"{r['dlv3_mean']:.4f}+/-{r['dlv3_std']:.4f}   "
          f"{r['t_pval']:<12.6f} {sig}")
print("=" * 70)
print("\n[DONE] Statistical t-test analysis complete!")
print(f"  Boxplots -> {boxplot_path}")
print(f"  Table    -> {table_path}")
print(f"  CSV      -> {csv_path}")
