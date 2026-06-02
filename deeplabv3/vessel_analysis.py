import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.ndimage import convolve
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

IMG_DIR = "d:/seemant-labdata/137_rgb_mask/RGB"
IMG_SIZE = 256

results = []

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
        continue
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Resize for model prediction
    img_resized = cv2.resize(orig_img_rgb, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    
    # Predict
    input_tensor = np.expand_dims(img_norm, axis=0)
    pred = model.predict(input_tensor, verbose=0)
    
    # Binarize prediction to get boolean mask for skeletonize
    pred_mask = (pred[0, :, :, 0] > 0.5)
    
    # Skeletonize the mask
    skeleton = skeletonize(pred_mask)
    
    # Count vessels using connected components on the skeleton
    labeled_skeleton, num_features = label(skeleton, return_num=True, connectivity=2)
    
    # Calculate end points and branch points using a 3x3 convolution
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    filtered = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    
    # End points have exactly 1 neighbor: center (10) + 1 neighbor (1) = 11
    end_points = np.sum(filtered == 11)
    
    # Branch points have 3 or more neighbors: center (10) + >=3 neighbors (3) >= 13
    branch_points = np.sum(filtered >= 13)
    
    # Store result
    results.append({
        "Image_ID": f"{i}_RGB",
        "Network_Count": num_features,
        "Branch_Points": branch_points,
        "End_Points": end_points
    })
    
    # Resize skeleton mask back to original image size for visualization
    orig_h, orig_w = orig_img.shape[:2]
    skeleton_uint8 = skeleton.astype(np.uint8)
    skeleton_resized = cv2.resize(skeleton_uint8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Create an overlay (Green skeleton)
    overlay = orig_img_rgb.copy()
    overlay[skeleton_resized == 1] = [0, 255, 0] # Green pixels for skeleton
    
    # Plot side-by-side
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(orig_img_rgb)
    plt.title(f"Original: {img_name}")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(skeleton, cmap='gray')
    plt.title(f"Skeletonized (Count: {num_features})")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Skeleton Overlay")
    plt.axis("off")
    
    plt.tight_layout()
    out_file = f"outputs/skeleton_{img_name}.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved {out_file} with {num_features} vessels detected.")

# Save results to CSV
df = pd.DataFrame(results)
csv_path = "outputs/vessel_counts.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved vessel counts to {csv_path}")
print(df)
