import os
import cv2
import numpy as np

IMG_SIZE = 256

# Absolute directory paths matching the workspace structure
TRAIN_IMAGE_DIR = "d:/seemant-labdata/500_rgb_mask/RGB"
TRAIN_MASK_DIR = "d:/seemant-labdata/500_rgb_mask/MASK"

VAL_IMAGE_DIR = "d:/seemant-labdata/137_rgb_mask/RGB"
VAL_MASK_DIR = "d:/seemant-labdata/137_rgb_mask/MASK"

def load_data(image_dir, mask_dir):
    """
    Loads all images from image_dir and their corresponding masks from mask_dir.
    Images are expected to be RGB.
    Mask filenames correspond to image names with 'gauss_' prepended.
    """
    images = []
    masks = []

    image_names = sorted(os.listdir(image_dir))

    for name in image_names:
        # Load RGB image
        img_path = os.path.join(image_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0

        # Load corresponding binary mask
        # Note: mask files are prepended with 'gauss_' (e.g., gauss_1_RGB.jpg matches 1_RGB.jpg)
        mask_path = os.path.join(mask_dir, f"gauss_{name}")
        if not os.path.exists(mask_path):
            # Fallback in case there is no gauss_ prefix
            mask_path = os.path.join(mask_dir, name)
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask {mask_path}")
            continue
            
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = mask.astype(np.float32) / 255.0
        
        # Binarize mask (threshold at 0.5)
        mask = (mask > 0.5).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

def get_train_val_datasets():
    """
    Returns X_train, Y_train, X_val, Y_val by reading:
    - 500_rgb_mask for training
    - 137_rgb_mask for validation
    """
    print("Loading training dataset...")
    X_train, Y_train = load_data(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR)
    
    print("Loading validation dataset...")
    X_val, Y_val = load_data(VAL_IMAGE_DIR, VAL_MASK_DIR)
    
    print(f"Dataset summary:")
    print(f"  Train: Images shape {X_train.shape}, Masks shape {Y_train.shape}")
    print(f"  Val:   Images shape {X_val.shape}, Masks shape {Y_val.shape}")
    
    return X_train, Y_train, X_val, Y_val
