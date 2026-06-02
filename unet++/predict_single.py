"""
predict_single.py  –  Run prediction on ONE image
===================================================
Usage:
    python predict_single.py

Shows: RGB Input | Ground Truth Mask | Predicted Mask | Overlay
Saves result to: outputs/single_prediction.png
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR    = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)

from losses  import bce_dice_loss
from metrics import dice_coeff, iou_score, precision, recall

# ─────────────────────────────────────────────
# CONFIG  ← change IMAGE_IDS to any numbers 501–637
# ─────────────────────────────────────────────
IMAGE_IDS  = [501, 502, 503, 504, 505, 506, 507, 508, 509, 510]  # 10 test images
IMG_SIZE   = (256, 256)
THRESHOLD  = 0.5

MODEL_PATH = "models/best_model.keras"
RGB_DIR    = r"d:\labdatanew@aniket\137_rgb_mask\RGB"
MASK_DIR   = r"d:\labdatanew@aniket\137_rgb_mask\MASK"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. Load model
# ─────────────────────────────────────────────
print(f"[1/4] Loading model from {MODEL_PATH} …")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "dice_coeff"   : dice_coeff,
        "iou_score"    : iou_score,
        "precision"    : precision,
        "recall"       : recall,
    },
    compile=False
)
print(f"      [OK] Model loaded.")

# Loop over 10 images
for IMAGE_ID in IMAGE_IDS:
    print(f"\n--- Processing Image {IMAGE_ID} ---")
    
    # ─────────────────────────────────────────────
    # 2. Load one image + mask
    # ─────────────────────────────────────────────
    print(f"[2/4] Loading image {IMAGE_ID}_RGB.jpg …")
    
    rgb_path  = os.path.join(RGB_DIR,  f"{IMAGE_ID}_RGB.jpg")
    mask_path = os.path.join(MASK_DIR, f"{IMAGE_ID}_RGB.png")   # masks are PNG
    
    if not os.path.exists(rgb_path):
        print(f"      [WARN] Image not found: {rgb_path}, skipping...")
        continue
    
    # Load & preprocess image
    img_bgr  = cv2.imread(rgb_path)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_norm    = img_resized.astype(np.float32) / 255.0          # (256,256,3)
    
    # Load ground truth mask (if available)
    gt_mask = None
    if os.path.exists(mask_path):
        mask_raw    = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_resized = cv2.resize(mask_raw, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        gt_mask      = (mask_resized > 127).astype(np.float32)    # (256,256)
        print(f"      [OK] Ground truth mask loaded.")
    else:
        print(f"      [WARN] No mask found at {mask_path} - will show prediction only.")
    
    # ─────────────────────────────────────────────
    # 3. Predict
    # ─────────────────────────────────────────────
    print(f"[3/4] Running inference …")
    inp        = np.expand_dims(img_norm, axis=0)            # (1, 256, 256, 3)
    pred_prob  = model.predict(inp, verbose=0)[0, :, :, 0]  # (256, 256)  raw prob
    pred_mask  = (pred_prob > THRESHOLD).astype(np.float32) # (256, 256)  binary
    
    # Metrics (only if GT mask available)
    dice = 0.0
    iou = 0.0
    if gt_mask is not None:
        inter = np.sum(gt_mask * pred_mask)
        dice  = (2 * inter + 1e-6) / (np.sum(gt_mask) + np.sum(pred_mask) + 1e-6)
        union = np.sum(gt_mask) + np.sum(pred_mask) - inter
        iou   = (inter + 1e-6) / (union + 1e-6)
        print(f"      Dice : {dice:.4f}")
        print(f"      IoU  : {iou:.4f}")
    
    # ─────────────────────────────────────────────
    # 4. Visualise & Save
    # ─────────────────────────────────────────────
    print(f"[4/4] Saving visualisation …")
    
    # Build overlay: draw prediction contour on RGB
    overlay_u8  = img_resized.copy()
    contours, _ = cv2.findContours(
        pred_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay_u8, contours, -1, (255, 0, 0), 2)   # red contour
    
    # Create figure
    num_cols = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 6))
    fig.suptitle(
        f"UNet++ Prediction  —  Image {IMAGE_ID}" +
        (f"  |  Dice={dice:.4f}  IoU={iou:.4f}" if gt_mask is not None else ""),
        fontsize=13, fontweight="bold"
    )
    
    col = 0
    axes[col].imshow(img_resized);             axes[col].set_title("RGB Input",       fontsize=11)
    col += 1
    
    if gt_mask is not None:
        axes[col].imshow(gt_mask, cmap="gray"); axes[col].set_title("Ground Truth",   fontsize=11)
        col += 1
    
    axes[col].imshow(pred_prob, cmap="hot", vmin=0, vmax=1)
    axes[col].set_title("Prediction (heatmap)", fontsize=11)
    col += 1
    
    axes[col].imshow(overlay_u8)
    axes[col].set_title("Overlay (red = predicted boundary)", fontsize=11)
    
    for ax in axes:
        ax.axis("off")
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"single_prediction_{IMAGE_ID}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"      [OK] Saved -> {out_path}")
    plt.close()

print("\n[DONE] Finished predicting all 10 images.")
