"""
predict.py – Inference & Visualization for UNet++
===================================================
Usage:
    python predict.py

Loads the best saved model, runs inference on the test set,
and saves:
  • outputs/predictions.png    – grid of 6 samples (RGB / GT / Pred / Overlay)
  • outputs/pred_masks/        – individual binary masks as PNG
"""

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects

# ── Import from src/ ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dataset import load_test_data
from losses  import bce_dice_loss
from metrics import dice_coeff, iou_score, precision, recall

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MODEL_PATH      = "models/best_model.keras"
OUTPUT_DIR      = "outputs"
THRESHOLD       = 0.5
NUM_DISPLAY     = 6        # samples to visualise in grid
MIN_OBJ_SIZE    = 50       # remove_small_objects threshold (pixels)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. Load Model
# ─────────────────────────────────────────────
print("[1/5] Loading model …")
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
print(f"      Model loaded from : {MODEL_PATH}")
print(f"      Input  shape      : {model.input_shape}")
print(f"      Output shape      : {model.output_shape}")

# ─────────────────────────────────────────────
# 2. Load Test Data
# ─────────────────────────────────────────────
print("\n[2/5] Loading test split (137_rgb_mask) …")
X_test, Y_test = load_test_data()
print(f"      Test samples: {len(X_test)}")

# ─────────────────────────────────────────────
# 3. Predict
# ─────────────────────────────────────────────
print("\n[3/5] Running inference …")
pred_probs = model.predict(X_test, batch_size=4, verbose=1)   # (N, H, W, 1)
pred_masks = (pred_probs > THRESHOLD).astype(np.uint8)        # binary

# ─────────────────────────────────────────────
# 4. Post-processing – remove_small_objects
# ─────────────────────────────────────────────
print(f"\n[4/5] Post-processing (min_size={MIN_OBJ_SIZE}) …")
pred_masks_clean = np.zeros_like(pred_masks)

for i, m in enumerate(pred_masks):
    m_bool               = m.squeeze().astype(bool)
    m_clean              = remove_small_objects(m_bool, min_size=MIN_OBJ_SIZE)
    pred_masks_clean[i]  = np.expand_dims(m_clean.astype(np.uint8), axis=-1)

# ─────────────────────────────────────────────
# Per-sample Dice & IoU  (numpy, hard)
# ─────────────────────────────────────────────
def batch_dice(y_true, y_pred, smooth=1e-6):
    scores = []
    for gt, pr in zip(y_true, y_pred):
        gt    = gt.flatten().astype(np.float32)
        pr    = pr.flatten().astype(np.float32)
        inter = np.sum(gt * pr)
        score = (2.0 * inter + smooth) / (np.sum(gt) + np.sum(pr) + smooth)
        scores.append(score)
    return np.array(scores)


def batch_iou(y_true, y_pred, smooth=1e-6):
    scores = []
    for gt, pr in zip(y_true, y_pred):
        gt    = gt.flatten().astype(np.float32)
        pr    = pr.flatten().astype(np.float32)
        inter = np.sum(gt * pr)
        union = np.sum(gt) + np.sum(pr) - inter
        scores.append((inter + smooth) / (union + smooth))
    return np.array(scores)


dice_scores = batch_dice(Y_test, pred_masks_clean)
iou_scores  = batch_iou (Y_test, pred_masks_clean)

print(f"\n── Aggregate Metrics (post-processed) ───")
print(f"  Mean Dice  : {dice_scores.mean():.4f} ± {dice_scores.std():.4f}")
print(f"  Mean IoU   : {iou_scores.mean():.4f}  ± {iou_scores.std():.4f}")

# ─────────────────────────────────────────────
# 5. Visualisation Grid
# ─────────────────────────────────────────────
print(f"\n[5/5] Saving visualisation for {NUM_DISPLAY} samples …")

n   = min(NUM_DISPLAY, len(X_test))
fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
fig.suptitle("UNet++  –  Vessel Segmentation Results", fontsize=14, fontweight="bold")

col_titles = ["RGB Input", "Ground Truth", "Prediction", "Overlay"]

for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=12, fontweight="bold")

for i in range(n):
    img  = X_test[i]                        # (H, W, 3)
    gt   = Y_test[i].squeeze()              # (H, W)
    pred = pred_masks_clean[i].squeeze()    # (H, W) – post-processed binary
    prob = pred_probs[i].squeeze()          # (H, W) – raw probability

    # Overlay: prediction contour on RGB image
    overlay      = img.copy()
    contours, _  = cv2.findContours(
        pred.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    overlay_u8   = (overlay * 255).astype(np.uint8)
    cv2.drawContours(overlay_u8, contours, -1, (255, 0, 0), 1)
    overlay      = overlay_u8.astype(np.float32) / 255.0

    axes[i, 0].imshow(img)
    axes[i, 0].set_ylabel(f"#{i}  Dice={dice_scores[i]:.3f}", fontsize=9)

    axes[i, 1].imshow(gt,      cmap="gray")
    axes[i, 2].imshow(prob,    cmap="hot", vmin=0, vmax=1)
    axes[i, 3].imshow(overlay)

    for ax in axes[i]:
        ax.axis("off")

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "predictions.png")
plt.savefig(out_path, dpi=150)
print(f"      Saved → {out_path}")
plt.close()

# ─────────────────────────────────────────────
# 6. Save individual masks as PNG
# ─────────────────────────────────────────────
masks_dir = os.path.join(OUTPUT_DIR, "pred_masks")
os.makedirs(masks_dir, exist_ok=True)

for i, m in enumerate(pred_masks_clean):
    cv2.imwrite(
        os.path.join(masks_dir, f"pred_{i:04d}.png"),
        m.squeeze() * 255
    )

print(f"      Binary masks saved → {masks_dir}/")
print("\n✅ Prediction complete.")
