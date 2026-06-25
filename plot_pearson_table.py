"""
plot_pearson_table.py  -  Generate visual tables from Pearson correlation data
===============================================================================
Reads pearson_skeleton_data.csv and creates two publication-quality table images:
  1. Skeleton properties comparison table (GT vs UNet++ vs DeepLabV3+)
  2. Pearson correlation summary table
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

OUTPUT_DIR = r"d:\labdatanew_Seemant\outputs"
CSV_PATH   = os.path.join(OUTPUT_DIR, "pearson_skeleton_data.csv")

# ── Load CSV data ───────────────────────────────────────────────────────────
print("[1/3] Loading data from CSV...")
data = np.genfromtxt(CSV_PATH, delimiter=',', skip_header=1, dtype=None, encoding='utf-8')

image_ids = [str(row[0]) for row in data]
gt_cc   = [int(row[1]) for row in data]
gt_br   = [int(row[2]) for row in data]
gt_ep   = [int(row[3]) for row in data]
up_cc   = [int(row[4]) for row in data]
up_br   = [int(row[5]) for row in data]
up_ep   = [int(row[6]) for row in data]
dl_cc   = [int(row[7]) for row in data]
dl_br   = [int(row[8]) for row in data]
dl_ep   = [int(row[9]) for row in data]

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1: Skeleton Properties Comparison (Detailed Data Table)
# ══════════════════════════════════════════════════════════════════════════════
print("[2/3] Generating detailed comparison table...")

col_labels = [
    "Image\nID",
    "GT\nConn.", "GT\nBranch", "GT\nEndpt",
    "UNet++\nConn.", "UNet++\nBranch", "UNet++\nEndpt",
    "DLV3+\nConn.", "DLV3+\nBranch", "DLV3+\nEndpt"
]

# Only show first 10 rows to keep it readable
display_len = min(10, len(image_ids))
table_data = []
for i in range(display_len):
    table_data.append([
        image_ids[i],
        gt_cc[i], gt_br[i], gt_ep[i],
        up_cc[i], up_br[i], up_ep[i],
        dl_cc[i], dl_br[i], dl_ep[i]
    ])

fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('off')
fig.suptitle(
    "Skeleton Topological Properties: Ground Truth vs Predicted Masks\n"
    "(Connected Components, Branch Points, Endpoints - Showing First 10 of 137 Test Samples)",
    fontsize=14, fontweight='bold', y=0.97
)

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(color='white', fontweight='bold', fontsize=8)

# Color-code columns by group
gt_color   = '#e8f5e9'  # light green
up_color   = '#e3f2fd'  # light blue
dl_color   = '#fff3e0'  # light orange
id_color   = '#f5f5f5'  # light grey

for i in range(1, display_len + 1):
    table[i, 0].set_facecolor(id_color)
    for j in [1, 2, 3]:
        table[i, j].set_facecolor(gt_color)
    for j in [4, 5, 6]:
        table[i, j].set_facecolor(up_color)
    for j in [7, 8, 9]:
        table[i, j].set_facecolor(dl_color)

plt.tight_layout(rect=[0, 0, 1, 0.90])
table1_path = os.path.join(OUTPUT_DIR, "pearson_data_table.png")
plt.savefig(table1_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved -> {table1_path}")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2: Pearson Correlation Summary Table
# ══════════════════════════════════════════════════════════════════════════════
print("[3/3] Generating Pearson correlation summary table...")

metrics = {
    "Connected Components": (gt_cc, up_cc, dl_cc),
    "Branch Points":        (gt_br, up_br, dl_br),
    "Endpoints":            (gt_ep, up_ep, dl_ep),
}

summary_cols = [
    "Skeleton\nProperty",
    "UNet++\nr value",
    "UNet++\np value",
    "UNet++\nSignificance",
    "DeepLabV3+\nr value",
    "DeepLabV3+\np value",
    "DeepLabV3+\nSignificance"
]

summary_data = []
for name, (gt, up, dl) in metrics.items():
    r_up, p_up = pearsonr(gt, up)
    r_dl, p_dl = pearsonr(gt, dl)

    sig_up = "Yes (p<0.05)" if p_up < 0.05 else "No"
    sig_dl = "Yes (p<0.05)" if p_dl < 0.05 else "No"

    summary_data.append([
        name,
        f"{r_up:.4f}",
        f"{p_up:.4f}",
        sig_up,
        f"{r_dl:.4f}",
        f"{p_dl:.4f}",
        sig_dl
    ])

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
fig.suptitle(
    "Pearson Correlation Coefficient Summary\n"
    "Ground Truth vs Predicted Skeleton Properties (n=137 images)",
    fontsize=14, fontweight='bold', y=0.95
)

table2 = ax.table(
    cellText=summary_data,
    colLabels=summary_cols,
    cellLoc='center',
    loc='center'
)
table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1.0, 2.0)

# Style header
for j in range(len(summary_cols)):
    cell = table2[0, j]
    cell.set_facecolor('#1a237e')
    cell.set_text_props(color='white', fontweight='bold', fontsize=9)

# Color-code rows
for i in range(1, len(summary_data) + 1):
    table2[i, 0].set_facecolor('#f5f5f5')
    table2[i, 0].set_text_props(fontweight='bold')

    # UNet++ columns
    for j in [1, 2, 3]:
        table2[i, j].set_facecolor('#e3f2fd')
    # DeepLabV3+ columns
    for j in [4, 5, 6]:
        table2[i, j].set_facecolor('#fff3e0')

    # Highlight statistically significant r values in green
    for j, col_idx in [(1, 2), (4, 5)]:   # (r_col, p_col)
        p_val = float(summary_data[i-1][col_idx])
        if p_val < 0.05:
            table2[i, j].set_facecolor('#c8e6c9')
            table2[i, j].set_text_props(fontweight='bold', color='#1b5e20')
            table2[i, j+2].set_facecolor('#c8e6c9')
            table2[i, j+2].set_text_props(fontweight='bold', color='#1b5e20')

plt.tight_layout(rect=[0, 0, 1, 0.85])
table2_path = os.path.join(OUTPUT_DIR, "pearson_summary_table.png")
plt.savefig(table2_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved -> {table2_path}")

print("\n[DONE] All tables generated!")
