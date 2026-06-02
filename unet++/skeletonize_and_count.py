import os
import sys
import numpy as np
import cv2
import csv
import tensorflow as tf
from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.ndimage import convolve

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR    = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)

from losses  import bce_dice_loss
from metrics import dice_coeff, iou_score, precision, recall

IMAGE_IDS  = [501, 502, 503, 504, 505, 506, 507, 508, 509, 510]
IMG_SIZE   = (256, 256)
THRESHOLD  = 0.5

MODEL_PATH = "models/best_model.keras"
RGB_DIR    = r"d:\labdatanew_Seemant\137_rgb_mask\RGB"
OUTPUT_CSV = "outputs/vessel_counts.csv"

print(f"Loading model from {MODEL_PATH} ...")
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

results = []

for img_id in IMAGE_IDS:
    rgb_path = os.path.join(RGB_DIR, f"{img_id}_RGB.jpg")
    if not os.path.exists(rgb_path):
        print(f"[WARN] Image not found: {rgb_path}")
        continue
        
    img_bgr  = cv2.imread(rgb_path)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_norm    = img_resized.astype(np.float32) / 255.0

    inp = np.expand_dims(img_norm, axis=0)
    pred_prob = model.predict(inp, verbose=0)[0, :, :, 0]
    pred_mask = (pred_prob > THRESHOLD).astype(np.uint8)

    # Apply skeletonization
    skeleton = skeletonize(pred_mask)
    
    # 1. Number of connected structures
    labeled_skeleton, num_ccs = label(skeleton, return_num=True)
    
    # 2. Find branches and endpoints using neighbor count
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    neighbors = convolve(skeleton.astype(int), kernel, mode='constant')
    
    branch_points = (skeleton > 0) & (neighbors > 2)
    num_branches = np.sum(branch_points)
    
    end_points = (skeleton > 0) & (neighbors == 1)
    num_endpoints = np.sum(end_points)
    
    print(f"Image {img_id}: {num_ccs} connected networks, {num_branches} branch points, {num_endpoints} endpoints")
    results.append({
        'Image ID': img_id, 
        'Connected Networks Count': num_ccs,
        'Branch Points Count': num_branches,
        'End Points Count': num_endpoints
    })

# Save to CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    fieldnames = ['Image ID', 'Connected Networks Count', 'Branch Points Count', 'End Points Count']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\n[DONE] Results saved to {OUTPUT_CSV}")
