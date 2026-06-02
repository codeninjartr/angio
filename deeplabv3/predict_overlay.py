import os
import cv2
import numpy as np
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
IMG_DIR = "d:/seemant-labdata/137_rgb_mask/RGB"
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
    
    # Resize the mask back to the original image size for a high-quality overlay
    orig_h, orig_w = orig_img.shape[:2]
    pred_mask_resized = cv2.resize(pred_mask, (orig_w, orig_h))
    
    # Create overlay
    overlay = orig_img_rgb.copy()
    
    # Create a red overlay layer
    color_mask = np.zeros_like(orig_img_rgb)
    color_mask[:, :, 0] = 255  # Red channel
    
    # Alpha blend where mask is 1
    alpha = 0.4  # Adjust transparency of the overlay
    mask_indices = pred_mask_resized > 0.5
    
    # Blend the original image with the red color mask
    overlay[mask_indices] = cv2.addWeighted(
        orig_img_rgb[mask_indices], 1 - alpha, 
        color_mask[mask_indices], alpha, 0
    )
    
    # Plot side-by-side: Original vs Overlay vs Mask
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(orig_img_rgb)
    plt.title(f"Original: {img_name}")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask_resized, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Prediction Overlay")
    plt.axis("off")
    
    plt.tight_layout()
    
    out_file = f"outputs/overlay_{img_name}.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved {out_file}")

print("Done generating overlays!")
