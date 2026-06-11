"""
pearson_correlation.py  -  Pearson Correlation between GT and Predicted Mask
=============================================================================
Skeletonizes both Ground Truth and Predicted masks for images 501-510,
extracts topological properties (Connected Components, Branch Points, Endpoints),
and plots Pearson correlation scatter graphs for both UNet++ and DeepLabV3+.
"""

import os
import sys
import numpy as np
import cv2
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import convolve
from skimage.morphology import skeletonize
from skimage.measure import label
import tensorflow as tf

# ── Config ──────────────────────────────────────────────────────────────────
IMAGE_IDS   = [501, 502, 503, 504, 505, 506, 507, 508, 509, 510]
IMG_SIZE    = (256, 256)
THRESHOLD   = 0.5
RGB_DIR     = r"d:\labdatanew_Seemant\137_rgb_mask\RGB"
MASK_DIR    = r"d:\labdatanew_Seemant\137_rgb_mask\MASK"
OUTPUT_DIR  = r"d:\labdatanew_Seemant\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Skeletonization helper ──────────────────────────────────────────────────
def skeleton_properties(binary_mask):
    """
    Given a binary mask (uint8, 0 or 1), skeletonize it and return
    (num_connected_components, num_branch_points, num_endpoints).
    """
    skeleton = skeletonize(binary_mask.astype(bool))
    _, num_ccs = label(skeleton, return_num=True)

    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    neighbors = convolve(skeleton.astype(int), kernel, mode='constant')

    num_branches  = int(np.sum((skeleton > 0) & (neighbors > 2)))
    num_endpoints = int(np.sum((skeleton > 0) & (neighbors == 1)))

    return num_ccs, num_branches, num_endpoints


# ── Load models ─────────────────────────────────────────────────────────────
# UNet++
sys.path.insert(0, os.path.join("unet++", "src"))
from losses  import bce_dice_loss
from metrics import dice_coeff, iou_score, precision, recall

print("[1/6] Loading UNet++ model...")
unetpp_model = tf.keras.models.load_model(
    os.path.join("unet++", "models", "best_model.keras"),
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "dice_coeff"   : dice_coeff,
        "iou_score"    : iou_score,
        "precision"    : precision,
        "recall"       : recall,
    },
    compile=False
)

# DeepLabV3+
sys.path.insert(0, "deeplabv3")
# Re-import losses/metrics from deeplabv3 (they share the same names)
import importlib
import losses as dl_losses
importlib.reload(dl_losses)
import metrics as dl_metrics
importlib.reload(dl_metrics)

print("[2/6] Loading DeepLabV3+ model...")
deeplabv3_model = tf.keras.models.load_model(
    os.path.join("deeplabv3", "models", "best_model.keras"),
    custom_objects={
        "bce_dice_loss": dl_losses.bce_dice_loss,
        "iou_score"    : dl_metrics.iou_score,
    },
    compile=False
)

# ── Process images ──────────────────────────────────────────────────────────
print("[3/6] Processing images and computing skeleton properties...")

gt_data     = {"ccs": [], "branches": [], "endpoints": []}
unetpp_data = {"ccs": [], "branches": [], "endpoints": []}
deeplabv3_data = {"ccs": [], "branches": [], "endpoints": []}
valid_ids = []

for img_id in IMAGE_IDS:
    rgb_path  = os.path.join(RGB_DIR, f"{img_id}_RGB.jpg")
    mask_path = os.path.join(MASK_DIR, f"gauss_{img_id}_RGB.jpg")

    if not os.path.exists(rgb_path):
        print(f"  [WARN] RGB not found: {rgb_path}")
        continue
    if not os.path.exists(mask_path):
        print(f"  [WARN] Mask not found: {mask_path}")
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

    # Skeletonize all three
    gt_ccs, gt_br, gt_ep         = skeleton_properties(gt_binary)
    up_ccs, up_br, up_ep         = skeleton_properties(unetpp_mask)
    dl_ccs, dl_br, dl_ep         = skeleton_properties(deeplabv3_mask)

    valid_ids.append(img_id)
    gt_data["ccs"].append(gt_ccs);       gt_data["branches"].append(gt_br);       gt_data["endpoints"].append(gt_ep)
    unetpp_data["ccs"].append(up_ccs);   unetpp_data["branches"].append(up_br);   unetpp_data["endpoints"].append(up_ep)
    deeplabv3_data["ccs"].append(dl_ccs); deeplabv3_data["branches"].append(dl_br); deeplabv3_data["endpoints"].append(dl_ep)

    print(f"  Image {img_id}:  GT({gt_ccs}, {gt_br}, {gt_ep})  "
          f"UNet++({up_ccs}, {up_br}, {up_ep})  "
          f"DeepLabV3+({dl_ccs}, {dl_br}, {dl_ep})")

# ── Save detailed CSV ───────────────────────────────────────────────────────
print("[4/6] Saving detailed comparison CSV...")
csv_path = os.path.join(OUTPUT_DIR, "pearson_skeleton_data.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Image_ID",
        "GT_Connected", "GT_Branches", "GT_Endpoints",
        "UNetPP_Connected", "UNetPP_Branches", "UNetPP_Endpoints",
        "DeepLabV3_Connected", "DeepLabV3_Branches", "DeepLabV3_Endpoints"
    ])
    for i, img_id in enumerate(valid_ids):
        writer.writerow([
            img_id,
            gt_data["ccs"][i], gt_data["branches"][i], gt_data["endpoints"][i],
            unetpp_data["ccs"][i], unetpp_data["branches"][i], unetpp_data["endpoints"][i],
            deeplabv3_data["ccs"][i], deeplabv3_data["branches"][i], deeplabv3_data["endpoints"][i],
        ])
print(f"  Saved -> {csv_path}")

# ── Compute Pearson correlations ────────────────────────────────────────────
print("[5/6] Computing Pearson correlations...")
metrics = ["ccs", "branches", "endpoints"]
metric_labels = {
    "ccs":       "Connected Components",
    "branches":  "Branch Points",
    "endpoints": "Endpoints"
}

pearson_results = {}
for m in metrics:
    gt_arr = np.array(gt_data[m])

    up_arr = np.array(unetpp_data[m])
    r_up, p_up = pearsonr(gt_arr, up_arr)

    dl_arr = np.array(deeplabv3_data[m])
    r_dl, p_dl = pearsonr(gt_arr, dl_arr)

    pearson_results[m] = {
        "unetpp": (r_up, p_up),
        "deeplabv3": (r_dl, p_dl)
    }
    print(f"  {metric_labels[m]}:")
    print(f"    UNet++      : r = {r_up:.4f}, p = {p_up:.4f}")
    print(f"    DeepLabV3+  : r = {r_dl:.4f}, p = {p_dl:.4f}")

# ── Plot Pearson correlation scatter graphs ─────────────────────────────────
print("[6/6] Generating Pearson correlation plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Pearson Correlation: Ground Truth vs Predicted Skeleton Properties\n"
             "(Each point = one test image)",
             fontsize=15, fontweight="bold", y=0.98)

colors_up = "#2196F3"   # Blue for UNet++
colors_dl = "#FF5722"   # Red-orange for DeepLabV3+

for col, m in enumerate(metrics):
    gt_arr = np.array(gt_data[m])
    up_arr = np.array(unetpp_data[m])
    dl_arr = np.array(deeplabv3_data[m])
    r_up, p_up = pearson_results[m]["unetpp"]
    r_dl, p_dl = pearson_results[m]["deeplabv3"]

    # ---- Row 0: UNet++ ----
    ax = axes[0, col]
    ax.scatter(gt_arr, up_arr, c=colors_up, s=80, edgecolors='k', linewidths=0.5, zorder=3)
    # Fit line
    if np.std(gt_arr) > 0 and np.std(up_arr) > 0:
        z = np.polyfit(gt_arr, up_arr, 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(gt_arr.min(), gt_arr.max(), 100)
        ax.plot(x_range, p_line(x_range), '--', color=colors_up, alpha=0.7, linewidth=2)
    # Annotate image IDs
    for i, img_id in enumerate(valid_ids):
        ax.annotate(str(img_id), (gt_arr[i], up_arr[i]),
                     fontsize=7, ha='left', va='bottom', xytext=(4, 4),
                     textcoords='offset points')
    ax.set_xlabel(f"Ground Truth {metric_labels[m]}", fontsize=10)
    ax.set_ylabel(f"UNet++ Predicted {metric_labels[m]}", fontsize=10)
    ax.set_title(f"UNet++ - {metric_labels[m]}\nr = {r_up:.4f}, p = {p_up:.4f}",
                 fontsize=11, fontweight="bold", color=colors_up)
    ax.grid(True, alpha=0.3)

    # ---- Row 1: DeepLabV3+ ----
    ax = axes[1, col]
    ax.scatter(gt_arr, dl_arr, c=colors_dl, s=80, edgecolors='k', linewidths=0.5, zorder=3)
    if np.std(gt_arr) > 0 and np.std(dl_arr) > 0:
        z = np.polyfit(gt_arr, dl_arr, 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(gt_arr.min(), gt_arr.max(), 100)
        ax.plot(x_range, p_line(x_range), '--', color=colors_dl, alpha=0.7, linewidth=2)
    for i, img_id in enumerate(valid_ids):
        ax.annotate(str(img_id), (gt_arr[i], dl_arr[i]),
                     fontsize=7, ha='left', va='bottom', xytext=(4, 4),
                     textcoords='offset points')
    ax.set_xlabel(f"Ground Truth {metric_labels[m]}", fontsize=10)
    ax.set_ylabel(f"DeepLabV3+ Predicted {metric_labels[m]}", fontsize=10)
    ax.set_title(f"DeepLabV3+ - {metric_labels[m]}\nr = {r_dl:.4f}, p = {p_dl:.4f}",
                 fontsize=11, fontweight="bold", color=colors_dl)
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plot_path = os.path.join(OUTPUT_DIR, "pearson_correlation_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved -> {plot_path}")

# ── Summary table ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PEARSON CORRELATION SUMMARY")
print("=" * 65)
print(f"  {'Metric':<25} {'UNet++ (r)':<15} {'DeepLabV3+ (r)':<15}")
print("-" * 65)
for m in metrics:
    r_up = pearson_results[m]["unetpp"][0]
    r_dl = pearson_results[m]["deeplabv3"][0]
    print(f"  {metric_labels[m]:<25} {r_up:<15.4f} {r_dl:<15.4f}")
print("=" * 65)
print("\n[DONE] Pearson correlation analysis complete!")
