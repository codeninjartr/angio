"""
dataset.py – Data loading & augmentation for UNet++ vessel segmentation
=========================================================================
Training data : d:/labdatanew@aniket/500_rgb_mask/
    RGB/   →  NNN_RGB.jpg
    MASK/  →  gauss_NNN_RGB.jpg

Test data     : d:/labdatanew@aniket/137_rgb_mask/
    RGB/   →  NNN_RGB.jpg
    MASK/  →  gauss_NNN_RGB.jpg

The mask filename is derived by prepending "gauss_" to the image filename.
"""

import os
import cv2
import numpy as np
import albumentations as A

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
IMG_SIZE = 256       # 256×256 for most GPUs; set to 512 for RTX/A100

# Absolute paths – fixed train / test directories
TRAIN_RGB_DIR  = r"d:\labdatanew_Seemant\500_rgb_mask\RGB"
TRAIN_MASK_DIR = r"d:\labdatanew_Seemant\500_rgb_mask\MASK"

TEST_RGB_DIR   = r"d:\labdatanew_Seemant\137_rgb_mask\RGB"
TEST_MASK_DIR  = r"d:\labdatanew_Seemant\137_rgb_mask\MASK"


# ─────────────────────────────────────────────
# Albumentations Augmentation Pipeline
# ─────────────────────────────────────────────
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    A.GridDistortion(p=0.2),
])


def apply_augmentation(image, mask):
    """
    Apply Albumentations transforms to an image-mask pair.

    Args:
        image : np.ndarray  (H, W, 3)  float32 in [0, 1]
        mask  : np.ndarray  (H, W, 1)  float32 in {0, 1}
    Returns:
        augmented image, augmented mask (same types and shapes)
    """
    # Albumentations expects uint8 images and 2D masks
    img_uint8  = (image * 255).astype(np.uint8)
    mask_2d    = (mask.squeeze() * 255).astype(np.uint8)

    augmented  = train_transform(image=img_uint8, mask=mask_2d)

    img_out    = augmented["image"].astype(np.float32) / 255.0
    mask_out   = augmented["mask"].astype(np.float32) / 255.0
    mask_out   = (mask_out > 0.5).astype(np.float32)
    mask_out   = np.expand_dims(mask_out, axis=-1)

    return img_out, mask_out


# ─────────────────────────────────────────────
# Mask filename resolver
# ─────────────────────────────────────────────
def _mask_name_for(rgb_filename: str) -> str:
    """
    Given an RGB filename (e.g. '100_RGB.jpg'),
    return the corresponding mask filename ('gauss_100_RGB.jpg').
    """
    return "gauss_" + rgb_filename


# ─────────────────────────────────────────────
# Core loader
# ─────────────────────────────────────────────
def _load_split(rgb_dir: str, mask_dir: str,
                split_name: str = "",
                augment: bool = False):
    """
    Load all image-mask pairs from a directory pair.

    Args:
        rgb_dir    : folder containing RGB images
        mask_dir   : folder containing grayscale binary masks
        split_name : label for progress messages ('train' / 'test')
        augment    : if True, apply Albumentations augmentation

    Returns:
        images : np.ndarray  (N, IMG_SIZE, IMG_SIZE, 3),  float32 in [0, 1]
        masks  : np.ndarray  (N, IMG_SIZE, IMG_SIZE, 1),  float32 in {0, 1}
    """
    images  = []
    masks   = []
    skipped = 0

    rgb_files = sorted(os.listdir(rgb_dir))

    for rgb_filename in rgb_files:

        # ── RGB image ──────────────────────────────────────
        img_path = os.path.join(rgb_dir, rgb_filename)
        img      = cv2.imread(img_path)

        if img is None:
            print(f"[WARNING] Cannot read image: {img_path} – skipping.")
            skipped += 1
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0

        # ── Mask ───────────────────────────────────────────
        mask_filename = _mask_name_for(rgb_filename)
        mask_path     = os.path.join(mask_dir, mask_filename)

        # Fallback: same name as RGB
        if not os.path.exists(mask_path):
            fallback = os.path.join(mask_dir, rgb_filename)
            if os.path.exists(fallback):
                mask_path = fallback
            else:
                print(f"[WARNING] No mask for: {rgb_filename}  "
                      f"(tried '{mask_filename}') – skipping.")
                skipped += 1
                continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"[WARNING] Cannot read mask: {mask_path} – skipping.")
            skipped += 1
            continue

        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)   # (H, W, 1)

        # ── Optional Augmentation ──────────────────────────
        if augment:
            img, mask = apply_augmentation(img, mask)

        images.append(img)
        masks.append(mask)

    tag = f"[{split_name.upper()}]" if split_name else "[INFO]"
    print(f"{tag} Loaded {len(images)} pairs  |  skipped {skipped}  "
          f"from {rgb_dir}")

    return (
        np.array(images, dtype=np.float32),
        np.array(masks,  dtype=np.float32)
    )


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def load_train_data(augment: bool = False):
    """
    Load all 500 training image-mask pairs.

    Args:
        augment : apply Albumentations augmentation on-load
    Returns:
        X_train : (N, 256, 256, 3)  float32
        Y_train : (N, 256, 256, 1)  float32
    """
    return _load_split(TRAIN_RGB_DIR, TRAIN_MASK_DIR,
                       split_name="train", augment=augment)


def load_test_data():
    """
    Load all 137 test image-mask pairs (no augmentation).

    Returns:
        X_test : (137, 256, 256, 3)  float32
        Y_test : (137, 256, 256, 1)  float32
    """
    return _load_split(TEST_RGB_DIR, TEST_MASK_DIR,
                       split_name="test", augment=False)


def get_split():
    """
    Convenience wrapper – returns (X_train, X_test, Y_train, Y_test).
    Training data comes from 500_rgb_mask; test data from 137_rgb_mask.
    """
    X_train, Y_train = load_train_data()
    X_test,  Y_test  = load_test_data()

    print(f"\n[INFO] Train: {X_train.shape}  |  Test: {X_test.shape}")

    return X_train, X_test, Y_train, Y_test
