"""
vessel_growth_analysis.py - Angiogenesis Growth Analysis for All Concentrations
=============================================================================
1. Loads or matches the 637 dataset images to original metadata (timepoints, concentrations).
2. Runs segmentation & skeletonization to extract:
   - Vessel Networks (connected components)
   - Branch Points (bifurcations)
   - End Points (capillary terminations)
3. Caches individual image metrics in outputs/image_skeleton_features.csv to make subsequent runs instantaneous.
4. Generates comparative growth progression plots comparing Control, 0.1ug, 1ug, and 10ug.
5. Generates a styled summary table image and CSV file.
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
        print(f"[1/5] Found cached metadata mapping at {mapping_csv}. Loading cached records...")
        with open(mapping_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = row["split"]
                filename = row["filename"]
                if split == "train":
                    path = os.path.join(TRAIN_DIR, filename)
                else:
                    path = os.path.join(TEST_DIR, filename)
                
                # Normalize concentration
                conc = row["concentration"].replace(" ", "").lower()
                if conc == "croppedcontrol":
                    conc = "control"
                
                matched_records.append({
                    "filename": filename,
                    "split": split,
                    "path": path,
                    "original_path": row["original_path"],
                    "concentration": conc,
                    "hour": int(row["hour"]),
                    "mse": float(row["mse"])
                })
        print(f"      Loaded {len(matched_records)} records from cache.")
    else:
        print("[1/5] Scanning raw images on D: drive...")
        raw_paths = find_raw_images(RAW_BASE_DIR)
        print(f"      Found {len(raw_paths)} raw cropped images.")
        
        print("      Downscaling and caching raw images for matching...")
        raw_cache = []
        for p in raw_paths:
            ds = load_and_downscale(p)
            if ds is not None:
                raw_cache.append((p, ds))

        dataset_images = []
        for f in sorted(os.listdir(TRAIN_DIR)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                dataset_images.append(("train", os.path.join(TRAIN_DIR, f), f))
        for f in sorted(os.listdir(TEST_DIR)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                dataset_images.append(("test", os.path.join(TEST_DIR, f), f))
                
        print("[2/5] Matching dataset images to original metadata...")
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
                    
            if best_mse < 200.0:
                rel = os.path.relpath(best_path, RAW_BASE_DIR)
                parts = rel.split(os.sep)
                
                try:
                    cropped_idx = parts.index("cropped")
                    concentration = parts[cropped_idx + 1].replace(" ", "").lower()
                except ValueError:
                    concentration = "unknown"
                
                if concentration == "croppedcontrol":
                    concentration = "control"
                
                r_filename = os.path.basename(best_path).lower()
                hr_match = re.search(r'(\d+)\s*h', r_filename)
                if hr_match:
                    hour = int(hr_match.group(1))
                else:
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
        
        # Save complete mapping
        with open(mapping_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "split", "concentration", "hour", "mse", "original_path"])
            writer.writeheader()
            for r in matched_records:
                writer.writerow(r)
        print(f"      Saved complete mapping to {mapping_csv}")

    # Check for feature cache
    features_csv = os.path.join(OUTPUT_DIR, "image_skeleton_features.csv")
    feature_cache = {}
    if os.path.exists(features_csv):
        print(f"[2/5] Found cached skeleton features at {features_csv}. Loading...")
        with open(features_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feature_cache[row["filename"]] = {
                    "network_count": int(row["network_count"]),
                    "branch_points": int(row["branch_points"]),
                    "end_points": int(row["end_points"])
                }
        print(f"      Loaded cached features for {len(feature_cache)} images.")

    # Identify images to run segmentation on
    to_segment = [r for r in matched_records if r["filename"] not in feature_cache]
    
    if to_segment:
        print(f"[3/5] Loading UNet++ model to segment {len(to_segment)} new images...")
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
        print("      Model loaded successfully. Segmenting and skeletonizing in batches...")
        
        # Batch processing to speed up inference on CPU
        batch_size = 32
        for start_idx in range(0, len(to_segment), batch_size):
            end_idx = min(start_idx + batch_size, len(to_segment))
            batch_records = to_segment[start_idx:end_idx]
            
            # Load images for the batch
            batch_images = []
            for r in batch_records:
                img_bgr = cv2.imread(r["path"])
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
                img_norm = img_resized.astype(np.float32) / 255.0
                batch_images.append(img_norm)
                
            batch_inp = np.array(batch_images)
            batch_preds = model.predict(batch_inp, batch_size=len(batch_images), verbose=0)[:, :, :, 0]
            
            for idx, r in enumerate(batch_records):
                pred_prob = batch_preds[idx]
                pred_mask = (pred_prob > 0.5).astype(np.uint8)
                
                # Post-process
                m_bool = pred_mask.astype(bool)
                m_clean = remove_small_objects(m_bool, min_size=50)
                pred_mask_clean = m_clean.astype(np.uint8)
                
                # Skeletonize
                skeleton = skeletonize(pred_mask_clean)
                _, num_ccs = label(skeleton, return_num=True)
                
                # Neighbor convolution
                kernel = np.array([[1, 1, 1],
                                   [1, 0, 1],
                                   [1, 1, 1]])
                neighbors = convolve(skeleton.astype(int), kernel, mode='constant')
                
                branch_points = (skeleton > 0) & (neighbors > 2)
                num_branches = np.sum(branch_points)
                
                end_points = (skeleton > 0) & (neighbors == 1)
                num_endpoints = np.sum(end_points)
                
                feature_cache[r["filename"]] = {
                    "network_count": num_ccs,
                    "branch_points": num_branches,
                    "end_points": num_endpoints
                }
            print(f"      Processed images {end_idx}/{len(to_segment)}...")
            
        # Write updated feature cache back
        with open(features_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "network_count", "branch_points", "end_points"])
            writer.writeheader()
            for filename, feat in sorted(feature_cache.items()):
                writer.writerow({
                    "filename": filename,
                    "network_count": feat["network_count"],
                    "branch_points": feat["branch_points"],
                    "end_points": feat["end_points"]
                })
        print(f"      Saved updated skeleton features to {features_csv}")
    else:
        print("[3/5] No new segmentations needed. All images retrieved from cache.")

    # Combine metadata and features
    all_results = []
    for r in matched_records:
        feat = feature_cache.get(r["filename"])
        if feat:
            all_results.append({
                "filename": r["filename"],
                "concentration": r["concentration"],
                "hour": r["hour"],
                "network_count": feat["network_count"],
                "branch_points": feat["branch_points"],
                "end_points": feat["end_points"]
            })

    # Group by concentration and hour
    grouped = defaultdict(lambda: defaultdict(list))
    for res in all_results:
        grouped[res["concentration"]][res["hour"]].append(res)

    # Calculate summary stats for each concentration and hour
    summary_stats = []
    target_concs = ["control", "0.1ug", "1ug", "10ug"]
    
    print("\n=== Growth Analysis Averages by Concentration ===")
    for conc in target_concs:
        print(f"\nConcentration: {conc.upper()}")
        for hr in sorted(grouped[conc].keys()):
            samples = grouped[conc][hr]
            nets = [x["network_count"] for x in samples]
            brs = [x["branch_points"] for x in samples]
            eps = [x["end_points"] for x in samples]
            
            summary_stats.append({
                "concentration": conc,
                "hour": hr,
                "count": len(samples),
                "net_mean": np.mean(nets), "net_std": np.std(nets),
                "br_mean": np.mean(brs), "br_std": np.std(brs),
                "ep_mean": np.mean(eps), "ep_std": np.std(eps)
            })
            print(f"  Hour {hr}h (n={len(samples)}):")
            print(f"    Vessel Networks : {np.mean(nets):.2f} +/- {np.std(nets):.2f}")
            print(f"    Branch Points   : {np.mean(brs):.2f} +/- {np.std(brs):.2f}")
            print(f"    Capillary Ends  : {np.mean(eps):.2f} +/- {np.std(eps):.2f}")

    # Save to vessel_growth_data.csv
    growth_csv = os.path.join(OUTPUT_DIR, "vessel_growth_data.csv")
    with open(growth_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "concentration", "hour", "sample_size", 
            "networks_mean", "networks_std", 
            "branches_mean", "branches_std", 
            "endpoints_mean", "endpoints_std"
        ])
        writer.writeheader()
        for s in summary_stats:
            writer.writerow({
                "concentration": s["concentration"],
                "hour": f"{s['hour']}h",
                "sample_size": s["count"],
                "networks_mean": f"{s['net_mean']:.2f}",
                "networks_std": f"{s['net_std']:.2f}",
                "branches_mean": f"{s['br_mean']:.2f}",
                "branches_std": f"{s['br_std']:.2f}",
                "endpoints_mean": f"{s['ep_mean']:.2f}",
                "endpoints_std": f"{s['ep_std']:.2f}"
            })
    print(f"\n[4/5] Saved growth summary to {growth_csv}")

    print("[5/5] Generating growth plots and publication tables...")
    plot_growth_curves(summary_stats)
    plot_growth_table(summary_stats)
    
    print("\n[OK] Vessel growth analysis complete!")

# ── Helper: plot growth curves ───────────────────────────────────────────────
def plot_growth_curves(summary_data):
    concs_data = defaultdict(list)
    for s in summary_data:
        concs_data[s["concentration"]].append(s)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Retinal Vessel Growth Dynamics Across Concentrations & Hours", fontsize=15, fontweight="bold")

    colors = {
        "control": "#2c3e50",
        "0.1ug": "#2980b9",
        "1ug": "#27ae60",
        "10ug": "#e74c3c"
    }
    markers = {
        "control": "o",
        "0.1ug": "s",
        "1ug": "d",
        "10ug": "^"
    }

    titles = [
        "Vessel Networks Count (Lower is more consolidated)",
        "Bifurcations / Branch Points (Higher is more branched)",
        "Capillary Terminations / End Points"
    ]
    ylabels = ["Average Networks Count", "Average Branch Points", "Average End Points"]
    keys = ["net", "br", "ep"]

    for i, key in enumerate(keys):
        ax = axes[i]
        ax.set_title(titles[i], fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (Hours)", fontsize=10)
        ax.set_ylabel(ylabels[i], fontsize=10)
        ax.grid(True, alpha=0.3)

        for conc in ["control", "0.1ug", "1ug", "10ug"]:
            data = sorted(concs_data[conc], key=lambda x: x["hour"])
            if not data:
                continue
            data = [d for d in data if d["hour"] != 32]
            
            hours = [d["hour"] for d in data]
            means = [d[f"{key}_mean"] for d in data]
            stds = [d[f"{key}_std"] for d in data]

            ax.errorbar(
                hours, means, yerr=stds, 
                fmt='-' + markers[conc], color=colors[conc], 
                ecolor=colors[conc], elinewidth=1.5, capsize=3, 
                linewidth=2, label=f"{conc.upper()} (n={sum(d['count'] for d in data)})"
            )
        
        ax.legend(loc="best")

    plt.tight_layout()
    out_plot = os.path.join(OUTPUT_DIR, "vessel_growth_plots.png")
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"      Saved comparative growth curves to {out_plot}")

# ── Helper: plot publication-quality growth table ────────────────────────────
def plot_growth_table(summary_data):
    sorted_data = sorted(summary_data, key=lambda x: (x["concentration"], x["hour"]))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    ax.axis("tight")
    
    columns = ["Concentration", "Time Point", "Sample Size (n)", "Vessel Networks (Mean ± SD)", "Branch Points (Mean ± SD)", "End Points (Mean ± SD)"]
    
    cell_text = []
    row_colors = []
    
    colors_map = {
        "control": "#ecf0f1",
        "0.1ug": "#dff0d8",
        "1ug": "#d9edf7",
        "10ug": "#f2dede"
    }

    for s in sorted_data:
        cell_text.append([
            s["concentration"].upper(),
            f"{s['hour']} hours",
            str(s["count"]),
            f"{s['net_mean']:.2f} ± {s['net_std']:.2f}",
            f"{s['br_mean']:.2f} ± {s['br_std']:.2f}",
            f"{s['ep_mean']:.2f} ± {s['ep_std']:.2f}"
        ])
        row_colors.append(colors_map[s["concentration"]])
        
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2c3e50")
        else:
            cell.set_facecolor(row_colors[row - 1])
                
    plt.title("Multi-Concentration Growth Analysis Summary (Retinal Vessel Kinetics)", y=0.98, fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    out_table = os.path.join(OUTPUT_DIR, "vessel_growth_table.png")
    plt.savefig(out_table, dpi=150)
    plt.close()
    print(f"      Saved styled growth table image to {out_table}")

if __name__ == "__main__":
    run_analysis()
