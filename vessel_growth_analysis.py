"""
vessel_growth_analysis.py - Angiogenesis Growth Analysis for 1ug Concentration
=============================================================================
1. Matches the 637 dataset images (500 train, 137 test) to original raw directories in D:/gs/angiogenesis data.
2. Decodes their concentration (e.g. 1ug, 10ug) and time points (e.g. 0h, 2h, 4h, 8h, 24h, 32h).
3. Segments the 1ug images using the best UNet++ model.
4. Skeletonizes the predicted masks to extract connected networks, branches, and endpoints.
5. Calculates the average metrics at each time point.
6. Generates line plots and tabular images showing growth trends.
"""

import os
import sys
import re
import csv
from collections import defaultdict
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label
from scipy.ndimage import convolve

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR = r"d:\labdatanew_Seemant\unet++\src"
sys.path.insert(0, SRC_DIR)

from losses import bce_dice_loss
from metrics import dice_coeff, iou_score, precision, recall

MODEL_PATH = r"d:\labdatanew_Seemant\unet++\models\best_model.keras"
TRAIN_DIR = r"d:\labdatanew_Seemant\500_rgb_mask\RGB"
TEST_DIR = r"d:\labdatanew_Seemant\137_rgb_mask\RGB"
RAW_BASE_DIR = r"D:\gs\angiogenesis data"
OUTPUT_DIR = r"d:\labdatanew_Seemant\outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helper: load and downscale ───────────────────────────────────────────────
def load_and_downscale(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

# ── Find raw images in D:/gs/angiogenesis data ────────────────────────────────
def find_raw_images(base_dir):
    raw_files = []
    for root, dirs, files in os.walk(base_dir):
        # We focus on the "cropped" directories as the dataset is cropped
        if "cropped" not in root.lower():
            continue
        for f in files:
            if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.bmp')) and "result" not in f.lower():
                raw_files.append(os.path.join(root, f))
    return raw_files

# ── Main Growth Analysis ──────────────────────────────────────────────────────
def run_analysis():
    mapping_csv = os.path.join(OUTPUT_DIR, "dataset_metadata_mapping.csv")
    matched_records = []
    
    if os.path.exists(mapping_csv):
        print(f"[1/3] Found cached metadata mapping at {mapping_csv}. Loading cached records...")
        with open(mapping_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = row["split"]
                filename = row["filename"]
                if split == "train":
                    path = os.path.join(TRAIN_DIR, filename)
                else:
                    path = os.path.join(TEST_DIR, filename)
                
                matched_records.append({
                    "filename": filename,
                    "split": split,
                    "path": path,
                    "original_path": row["original_path"],
                    "concentration": row["concentration"],
                    "hour": int(row["hour"]),
                    "mse": float(row["mse"])
                })
        print(f"      Loaded {len(matched_records)} records from cache.")
    else:
        print("[1/6] Scanning raw images on D: drive...")
        raw_paths = find_raw_images(RAW_BASE_DIR)
        print(f"      Found {len(raw_paths)} raw cropped images.")
        
        print("[2/6] Downscaling and caching raw images for matching...")
        raw_cache = []
        for p in raw_paths:
            ds = load_and_downscale(p)
            if ds is not None:
                raw_cache.append((p, ds))
        print(f"      Cached {len(raw_cache)} raw images.")

        # List all dataset images
        dataset_images = []
        for f in sorted(os.listdir(TRAIN_DIR)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                dataset_images.append(("train", os.path.join(TRAIN_DIR, f), f))
        for f in sorted(os.listdir(TEST_DIR)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                dataset_images.append(("test", os.path.join(TEST_DIR, f), f))
                
        print(f"      Found {len(dataset_images)} total dataset images (500 train, 137 test).")

        print("[3/6] Matching dataset images to original metadata...")
        
        for split, path, filename in dataset_images:
            ds_target = load_and_downscale(path)
            if ds_target is None:
                continue
                
            best_mse = float('inf')
            best_path = None
            
            for r_path, r_ds in raw_cache:
                mse = np.mean((ds_target - r_ds) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_path = r_path
                    
            if best_mse < 200.0:  # Threshold for valid matches
                # Decode metadata from original path
                rel = os.path.relpath(best_path, RAW_BASE_DIR)
                parts = rel.split(os.sep)
                
                # Find index of "cropped" in path parts
                try:
                    cropped_idx = parts.index("cropped")
                    concentration = parts[cropped_idx + 1].replace(" ", "").lower()
                except ValueError:
                    # Fallback if "cropped" is not in path (unlikely)
                    concentration = "unknown"
                    
                # Decode time point from filename
                r_filename = os.path.basename(best_path).lower()
                hr_match = re.search(r'(\d+)\s*h', r_filename)
                if hr_match:
                    hour = int(hr_match.group(1))
                else:
                    # Fallback to look at numerical subdirectories
                    hour_found = False
                    for p in parts:
                        if p.isdigit():
                            hour = int(p)
                            hour_found = True
                            break
                    if not hour_found:
                        hour = 0
                
                matched_records.append({
                    "filename": filename,
                    "split": split,
                    "path": path,
                    "original_path": best_path,
                    "concentration": concentration,
                    "hour": hour,
                    "mse": best_mse
                })
            
        print(f"      Successfully matched {len(matched_records)}/{len(dataset_images)} images.")
        
        # Save complete mapping to CSV
        with open(mapping_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "split", "concentration", "hour", "mse", "original_path"])
            writer.writeheader()
            for r in matched_records:
                writer.writerow({
                    "filename": r["filename"],
                    "split": r["split"],
                    "concentration": r["concentration"],
                    "hour": r["hour"],
                    "mse": r["mse"],
                    "original_path": r["original_path"]
                })
        print(f"      Saved complete mapping to {mapping_csv}")
    
    # Filter for "1ug" concentration
    target_records = [r for r in matched_records if r["concentration"] == "1ug"]
    print(f"      Found {len(target_records)} images matching '1ug' concentration.")
    
    if not target_records:
        print("[ERROR] No images found for concentration '1ug'.")
        return
        
    print("[4/6] Loading UNet++ ResNet50 segmentation model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_coeff": dice_coeff,
            "iou_score": iou_score,
            "precision": precision,
            "recall": recall
        },
        compile=False
    )
    print("      Model loaded successfully.")
    
    print("[5/6] Segmenting and skeletonizing 1ug images...")
    results = []
    
    for i, r in enumerate(target_records):
        if (i+1) % 10 == 0 or (i+1) == len(target_records):
            print(f"      Processing image {i+1}/{len(target_records)}...")
            
        img_bgr = cv2.imread(r["path"])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        img_norm = img_resized.astype(np.float32) / 255.0
        
        # Segment
        inp = np.expand_dims(img_norm, axis=0)
        pred_prob = model.predict(inp, verbose=0)[0, :, :, 0]
        pred_mask = (pred_prob > 0.5).astype(np.uint8)
        
        # Post-process: remove small noise components
        m_bool = pred_mask.astype(bool)
        m_clean = remove_small_objects(m_bool, min_size=50)
        pred_mask_clean = m_clean.astype(np.uint8)
        
        # Skeletonize
        skeleton = skeletonize(pred_mask_clean)
        
        # Connected components (number of networks)
        _, num_ccs = label(skeleton, return_num=True)
        
        # Neighbor count convolution for branch and endpoints
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
        neighbors = convolve(skeleton.astype(int), kernel, mode='constant')
        
        branch_points = (skeleton > 0) & (neighbors > 2)
        num_branches = np.sum(branch_points)
        
        end_points = (skeleton > 0) & (neighbors == 1)
        num_endpoints = np.sum(end_points)
        
        results.append({
            "filename": r["filename"],
            "hour": r["hour"],
            "network_count": num_ccs,
            "branch_points": num_branches,
            "end_points": num_endpoints
        })
        
    # Group results by hour and calculate mean & std
    hourly_data = defaultdict(list)
    for res in results:
        hourly_data[res["hour"]].append(res)
        
    summary_data = []
    print("\n=== Growth Analysis Averages for 1ug ===")
    for hr in sorted(hourly_data.keys()):
        hr_res = hourly_data[hr]
        nets = [x["network_count"] for x in hr_res]
        brs = [x["branch_points"] for x in hr_res]
        eps = [x["end_points"] for x in hr_res]
        
        summary_data.append({
            "hour": hr,
            "count": len(hr_res),
            "net_mean": np.mean(nets), "net_std": np.std(nets),
            "br_mean": np.mean(brs), "br_std": np.std(brs),
            "ep_mean": np.mean(eps), "ep_std": np.std(eps)
        })
        
        print(f"Hour {hr}h (n={len(hr_res)}):")
        print(f"  Vessel Networks : {np.mean(nets):.2f} +/- {np.std(nets):.2f}")
        print(f"  Branch Points   : {np.mean(brs):.2f} +/- {np.std(brs):.2f}")
        print(f"  Capillary Ends  : {np.mean(eps):.2f} +/- {np.std(eps):.2f}")
        
    # Save growth data to CSV
    growth_csv = os.path.join(OUTPUT_DIR, "vessel_growth_data.csv")
    with open(growth_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hour", "sample_size", "networks_mean", "networks_std", "branches_mean", "branches_std", "endpoints_mean", "endpoints_std"])
        writer.writeheader()
        for s in summary_data:
            writer.writerow({
                "hour": f"{s['hour']}h",
                "sample_size": s["count"],
                "networks_mean": f"{s['net_mean']:.2f}",
                "networks_std": f"{s['net_std']:.2f}",
                "branches_mean": f"{s['br_mean']:.2f}",
                "branches_std": f"{s['br_std']:.2f}",
                "endpoints_mean": f"{s['ep_mean']:.2f}",
                "endpoints_std": f"{s['ep_std']:.2f}"
            })
    print(f"\n      Saved growth summary to {growth_csv}")
    
    print("[6/6] Generating growth plots and publication tables...")
    plot_growth_curves(summary_data)
    plot_growth_table(summary_data)
    
    print("\n[OK] Vessel growth analysis complete!")

# ── Helper: plot growth curves ───────────────────────────────────────────────
def plot_growth_curves(summary_data):
    hours = [s["hour"] for s in summary_data]
    
    net_means = [s["net_mean"] for s in summary_data]
    net_stds = [s["net_std"] for s in summary_data]
    
    br_means = [s["br_mean"] for s in summary_data]
    br_stds = [s["br_std"] for s in summary_data]
    
    ep_means = [s["ep_mean"] for s in summary_data]
    ep_stds = [s["ep_std"] for s in summary_data]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Morphological Vessel Growth Progression over Time (1ug Concentration)", fontsize=14, fontweight="bold")
    
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    
    # 1. Connected Networks
    axes[0].errorbar(hours, net_means, yerr=net_stds, fmt='-o', color=colors[0], ecolor='lightblue', elinewidth=2, capsize=4, linewidth=2)
    axes[0].set_title("Vessel Networks Count", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Time (Hours)")
    axes[0].set_ylabel("Average Networks Count")
    axes[0].grid(True, alpha=0.3)
    
    # 2. Branch Points
    axes[1].errorbar(hours, br_means, yerr=br_stds, fmt='-s', color=colors[1], ecolor='lightgreen', elinewidth=2, capsize=4, linewidth=2)
    axes[1].set_title("Bifurcations (Branch Points)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Time (Hours)")
    axes[1].set_ylabel("Average Branch Points")
    axes[1].grid(True, alpha=0.3)
    
    # 3. End Points
    axes[2].errorbar(hours, ep_means, yerr=ep_stds, fmt='-d', color=colors[2], ecolor='pink', elinewidth=2, capsize=4, linewidth=2)
    axes[2].set_title("Capillary Terminations (End Points)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Time (Hours)")
    axes[2].set_ylabel("Average End Points")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_plot = os.path.join(OUTPUT_DIR, "vessel_growth_plots.png")
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"      Saved growth curves plot to {out_plot}")

# ── Helper: plot publication-quality growth table ────────────────────────────
def plot_growth_table(summary_data):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.axis("tight")
    
    columns = ["Time Point", "Sample Size (n)", "Vessel Networks (Mean ± SD)", "Branch Points (Mean ± SD)", "End Points (Mean ± SD)"]
    
    cell_text = []
    for s in summary_data:
        cell_text.append([
            f"{s['hour']} hours",
            str(s["count"]),
            f"{s['net_mean']:.2f} ± {s['net_std']:.2f}",
            f"{s['br_mean']:.2f} ± {s['br_std']:.2f}",
            f"{s['ep_mean']:.2f} ± {s['ep_std']:.2f}"
        ])
        
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2c3e50")
        else:
            if row % 2 == 0:
                cell.set_facecolor("#f8f9fa")
            else:
                cell.set_facecolor("#ffffff")
                
    plt.title("Growth Analysis Summary (1ug Concentration) over Hours", y=0.9, fontsize=12, fontweight="bold")
    plt.tight_layout()
    
    out_table = os.path.join(OUTPUT_DIR, "vessel_growth_table.png")
    plt.savefig(out_table, dpi=150)
    plt.close()
    print(f"      Saved growth table image to {out_table}")

if __name__ == "__main__":
    run_analysis()
