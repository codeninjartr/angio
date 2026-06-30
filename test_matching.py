import os
import cv2
import numpy as np

# We'll test with the first 5 images in 500_rgb_mask/RGB
dataset_dir = r"d:\labdatanew_Seemant\500_rgb_mask\RGB"
raw_base_dir = r"D:\gs\angiogenesis data"

def load_and_downscale(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Resize to a very small size for fast, robust comparison
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

def find_raw_images(base_dir):
    raw_files = []
    for root, dirs, files in os.walk(base_dir):
        # Focus on cropped folders since the dataset is cropped
        if "cropped" not in root.lower():
            continue
        for f in files:
            if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.bmp')) and "result" not in f.lower():
                raw_files.append(os.path.join(root, f))
    return raw_files

def test_match():
    test_ids = ["1_RGB.jpg", "5_RGB.jpg", "9_RGB.jpg", "10_RGB.jpg", "20_RGB.jpg"]
    print("Finding raw images on disk...")
    raw_paths = find_raw_images(raw_base_dir)
    print(f"Found {len(raw_paths)} raw cropped images.")
    
    # Load and downscale raw images
    print("Caching raw downscaled images...")
    raw_cache = []
    for p in raw_paths:
        ds = load_and_downscale(p)
        if ds is not None:
            raw_cache.append((p, ds))
            
    print(f"Successfully cached {len(raw_cache)} raw images.")
    
    for filename in test_ids:
        path = os.path.join(dataset_dir, filename)
        ds_target = load_and_downscale(path)
        if ds_target is None:
            print(f"Could not load dataset image: {filename}")
            continue
            
        best_mse = float('inf')
        best_path = None
        
        for r_path, r_ds in raw_cache:
            # Calculate Mean Squared Error
            mse = np.mean((ds_target - r_ds) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_path = r_path
                
        print(f"\nDataset image: {filename}")
        print(f"  Best Match: {best_path}")
        print(f"  MSE Score: {best_mse:.2f}")
        
        # Parse information from the path
        if best_mse < 100.0:  # Threshold for similarity
            rel = os.path.relpath(best_path, raw_base_dir)
            parts = rel.split(os.sep)
            print(f"  Decoded: Series={parts[0]}, Hours={parts[2]}, Concentration={parts[3]}")
        else:
            print("  Warning: No confident match found (MSE too high)")

if __name__ == "__main__":
    test_match()
