import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from losses import bce_dice_loss
from metrics import iou_score

os.makedirs("outputs", exist_ok=True)

# 1. Load the best model
model_path = "models/best_model.keras"
if not os.path.exists(model_path):
    model_path = "models/best_model.h5"

print(f"Loading model from {model_path}...")
model = tf.keras.models.load_model(
    model_path,
    custom_objects={"bce_dice_loss": bce_dice_loss, "iou_score": iou_score},
    compile=False
)

# 2. Setup image directory
IMG_DIR = r"d:\labdatanew_Seemant\137_rgb_mask\RGB"
IMG_SIZE = 256

for i in range(501, 511):
    img_name = f"{i}_RGB.jpg"
    img_path = os.path.join(IMG_DIR, img_name)
    
    if not os.path.exists(img_path):
        print(f"Image {img_path} not found.")
        continue
        
    print(f"Processing {img_name}...")
    
    # Read and preprocess image
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print(f"Warning: Could not read {img_path}")
        continue
        
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Resize for model prediction
    img_resized = cv2.resize(orig_img_rgb, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    
    # Predict
    input_tensor = np.expand_dims(img_norm, axis=0)
    pred = model.predict(input_tensor, verbose=0)
    
    # Binarize prediction
    pred_mask = (pred[0] > 0.5).astype(np.float32)
    
    # Use 256x256 resolution for all panels (same as UNet++)
    pred_mask_resized = pred_mask.squeeze()  # already 256x256
    
    # Create overlay at 256x256
    overlay = img_resized.copy()
    
    # Create a red overlay layer
    color_mask = np.zeros_like(img_resized)
    color_mask[:, :, 0] = 255  # Red channel
    
    # Alpha blend where mask is 1
    alpha = 0.4
    mask_indices = pred_mask_resized > 0.5
    
    # Blend
    overlay[mask_indices] = cv2.addWeighted(
        img_resized[mask_indices], 1 - alpha, 
        color_mask[mask_indices], alpha, 0
    )
    
    # Setup MASK_DIR (assuming it is parallel to IMG_DIR)
    MASK_DIR = IMG_DIR.replace("RGB", "MASK")
    mask_path = os.path.join(MASK_DIR, f"gauss_{img_name}")   # masks are gauss_NNN_RGB.jpg
    
    gt_mask_resized = None
    if os.path.exists(mask_path):
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None:
            gt_mask_resized = cv2.resize(gt_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    
    # Compute Dice/IoU if GT available
    dice = 0.0
    iou = 0.0
    if gt_mask_resized is not None:
        gt_bin = (gt_mask_resized > 127).astype(np.float32)
        pred_bin = (pred_mask_resized > 0.5).astype(np.float32)
        inter = np.sum(gt_bin * pred_bin)
        dice = (2 * inter + 1e-6) / (np.sum(gt_bin) + np.sum(pred_bin) + 1e-6)
        union = np.sum(gt_bin) + np.sum(pred_bin) - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        print(f"  Dice: {dice:.4f}  IoU: {iou:.4f}")
    
    # Plot in 2x2 grid
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(
        f"DeepLabV3+ Prediction  —  Image {i}" +
        (f"\nDice={dice:.4f}  IoU={iou:.4f}" if gt_mask_resized is not None else ""),
        fontsize=13, fontweight="bold"
    )
    
    plt.subplot(2, 2, 1)
    plt.imshow(img_resized)
    plt.title(f"Original: {img_name} (256x256)")
    plt.axis("off")
    
    plt.subplot(2, 2, 2)
    if gt_mask_resized is not None:
        plt.imshow(gt_mask_resized, cmap='gray')
        plt.title("Ground Truth Mask")
    else:
        plt.text(0.5, 0.5, "No GT Available", ha='center', va='center')
        plt.title("Ground Truth Mask")
    plt.axis("off")
    
    plt.subplot(2, 2, 3)
    plt.imshow(pred_mask_resized, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")
    
    plt.subplot(2, 2, 4)
    plt.imshow(overlay)
    plt.title("Prediction Overlay")
    plt.axis("off")
    
    plt.tight_layout()
    
    out_file = f"outputs/overlay_{img_name}.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved {out_file}")

print("Done generating overlays!")
