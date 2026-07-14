"""
spearman_correlation.py  -  Spearman Correlation Heatmaps
=========================================================
Reads the skeleton property data from pearson_skeleton_data.csv and computes
Spearman rank correlation between GT and Predicted skeleton properties
for both UNet++ and DeepLabV3+. Generates publication-quality heatmaps.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH   = r"d:\labdatanew_Seemant\outputs\pearson_skeleton_data.csv"
OUTPUT_DIR = r"d:\labdatanew_Seemant\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────────────
print("[1/4] Loading skeleton data from CSV...")
df = pd.read_csv(CSV_PATH)
print(f"  Loaded {len(df)} images")

# Column groups
gt_cols     = ["GT_Connected", "GT_Branches", "GT_Endpoints"]
unetpp_cols = ["UNetPP_Connected", "UNetPP_Branches", "UNetPP_Endpoints"]
dlv3_cols   = ["DeepLabV3_Connected", "DeepLabV3_Branches", "DeepLabV3_Endpoints"]

metric_labels = ["Connected\nComponents", "Branch\nPoints", "Endpoints"]

# ── Compute Spearman correlations ───────────────────────────────────────────
print("[2/4] Computing Spearman rank correlations...")

def compute_spearman_matrix(gt_columns, pred_columns, dataframe):
    """Compute Spearman correlation matrix between GT and Predicted columns."""
    n = len(gt_columns)
    corr_matrix = np.zeros((n, n))
    pval_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r, p = spearmanr(dataframe[gt_columns[i]], dataframe[pred_columns[j]])
            corr_matrix[i, j] = r
            pval_matrix[i, j] = p
    return corr_matrix, pval_matrix

# UNet++
unetpp_corr, unetpp_pval = compute_spearman_matrix(gt_cols, unetpp_cols, df)
# DeepLabV3+
dlv3_corr, dlv3_pval = compute_spearman_matrix(gt_cols, dlv3_cols, df)

# Print results
print("\n  UNet++ Spearman Correlation Matrix (GT vs Predicted):")
for i, gl in enumerate(metric_labels):
    for j, pl in enumerate(metric_labels):
        print(f"    GT {gl.replace(chr(10),' ')} vs Pred {pl.replace(chr(10),' ')}: "
              f"rho = {unetpp_corr[i,j]:.4f}, p = {unetpp_pval[i,j]:.4f}")

print("\n  DeepLabV3+ Spearman Correlation Matrix (GT vs Predicted):")
for i, gl in enumerate(metric_labels):
    for j, pl in enumerate(metric_labels):
        print(f"    GT {gl.replace(chr(10),' ')} vs Pred {pl.replace(chr(10),' ')}: "
              f"rho = {dlv3_corr[i,j]:.4f}, p = {dlv3_pval[i,j]:.4f}")

# ── Font sizes ──────────────────────────────────────────────────────────────
TITLE_SIZE  = 36
LABEL_SIZE  = 28
TICK_SIZE   = 22
LEGEND_SIZE = 22
ANNOT_SIZE  = 20
CBAR_LABEL  = 22
BAR_VAL_SIZE = 16

# ── Plot 1a: Heatmap — UNet++ ──────────────────────────────────────────────
print("\n[3/4] Generating Spearman correlation heatmaps...")

annot_up = np.empty_like(unetpp_corr, dtype=object)
for i in range(unetpp_corr.shape[0]):
    for j in range(unetpp_corr.shape[1]):
        r_val = unetpp_corr[i, j]
        p_val = unetpp_pval[i, j]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        annot_up[i, j] = f"{r_val:.3f}\n({sig})"

fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(unetpp_corr, ax=ax, annot=annot_up, fmt="",
            xticklabels=metric_labels, yticklabels=metric_labels,
            cmap="RdYlBu_r", center=0, vmin=-1, vmax=1,
            linewidths=2, linecolor="white",
            square=True, annot_kws={"size": ANNOT_SIZE},
            cbar_kws={"label": "Spearman ρ", "shrink": 0.8})
ax.set_title("UNet++\nSpearman Rank Correlation", fontsize=TITLE_SIZE, fontweight="bold", color="#2196F3", pad=20)
ax.set_xlabel("Predicted Properties", fontsize=LABEL_SIZE, fontweight="bold")
ax.set_ylabel("Ground Truth Properties", fontsize=LABEL_SIZE, fontweight="bold")
ax.tick_params(axis='both', labelsize=TICK_SIZE)
# Style colorbar label
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=TICK_SIZE)
cbar.set_label("Spearman ρ", fontsize=CBAR_LABEL)
fig.text(0.5, -0.02,
         "Significance: *** p < 0.001  |  ** p < 0.01  |  * p < 0.05  |  ns = not significant",
         ha='center', fontsize=16, style='italic', color='gray')
plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, "spearman_heatmap_unetpp.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved -> {path1}")

# ── Plot 1b: Heatmap — DeepLabV3+ ─────────────────────────────────────────
annot_dl = np.empty_like(dlv3_corr, dtype=object)
for i in range(dlv3_corr.shape[0]):
    for j in range(dlv3_corr.shape[1]):
        r_val = dlv3_corr[i, j]
        p_val = dlv3_pval[i, j]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        annot_dl[i, j] = f"{r_val:.3f}\n({sig})"

fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(dlv3_corr, ax=ax, annot=annot_dl, fmt="",
            xticklabels=metric_labels, yticklabels=metric_labels,
            cmap="RdYlBu_r", center=0, vmin=-1, vmax=1,
            linewidths=2, linecolor="white",
            square=True, annot_kws={"size": ANNOT_SIZE},
            cbar_kws={"label": "Spearman ρ", "shrink": 0.8})
ax.set_title("DeepLabV3+\nSpearman Rank Correlation", fontsize=TITLE_SIZE, fontweight="bold", color="#FF5722", pad=20)
ax.set_xlabel("Predicted Properties", fontsize=LABEL_SIZE, fontweight="bold")
ax.set_ylabel("Ground Truth Properties", fontsize=LABEL_SIZE, fontweight="bold")
ax.tick_params(axis='both', labelsize=TICK_SIZE)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=TICK_SIZE)
cbar.set_label("Spearman ρ", fontsize=CBAR_LABEL)
fig.text(0.5, -0.02,
         "Significance: *** p < 0.001  |  ** p < 0.01  |  * p < 0.05  |  ns = not significant",
         ha='center', fontsize=16, style='italic', color='gray')
plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, "spearman_heatmap_deeplabv3.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved -> {path2}")

# ── Plot 2: Comparison bar charts (separate) ───────────────────────────────
print("[4/4] Generating Pearson vs Spearman comparison charts...")

from scipy.stats import pearsonr

pearson_unetpp_diag = []
pearson_dlv3_diag = []
spearman_unetpp_diag = []
spearman_dlv3_diag = []

for i in range(3):
    r_p, _ = pearsonr(df[gt_cols[i]], df[unetpp_cols[i]])
    pearson_unetpp_diag.append(r_p)
    spearman_unetpp_diag.append(unetpp_corr[i, i])

    r_p, _ = pearsonr(df[gt_cols[i]], df[dlv3_cols[i]])
    pearson_dlv3_diag.append(r_p)
    spearman_dlv3_diag.append(dlv3_corr[i, i])

x = np.arange(3)
width = 0.35
short_labels = ["Connected\nComponents", "Branch\nPoints", "Endpoints"]

# ── Plot 2a: Bar chart — UNet++ ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
bars1 = ax.bar(x - width/2, pearson_unetpp_diag, width, label='Pearson r',
               color='#42A5F5', edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2, spearman_unetpp_diag, width, label='Spearman ρ',
               color='#1565C0', edgecolor='black', linewidth=0.8)
ax.set_ylabel("Correlation Coefficient", fontsize=LABEL_SIZE)
ax.set_title("UNet++\nPearson vs Spearman Comparison", fontsize=TITLE_SIZE, fontweight="bold", color="#2196F3")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=TICK_SIZE)
ax.tick_params(axis='y', labelsize=TICK_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
ax.set_ylim([-1, 1])
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.3f}', ha='center', va='bottom', fontsize=BAR_VAL_SIZE, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.3f}', ha='center', va='bottom', fontsize=BAR_VAL_SIZE, fontweight='bold')
plt.tight_layout()
path3 = os.path.join(OUTPUT_DIR, "pearson_vs_spearman_unetpp.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved -> {path3}")

# ── Plot 2b: Bar chart — DeepLabV3+ ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
bars1 = ax.bar(x - width/2, pearson_dlv3_diag, width, label='Pearson r',
               color='#FF7043', edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2, spearman_dlv3_diag, width, label='Spearman ρ',
               color='#D84315', edgecolor='black', linewidth=0.8)
ax.set_ylabel("Correlation Coefficient", fontsize=LABEL_SIZE)
ax.set_title("DeepLabV3+\nPearson vs Spearman Comparison", fontsize=TITLE_SIZE, fontweight="bold", color="#FF5722")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=TICK_SIZE)
ax.tick_params(axis='y', labelsize=TICK_SIZE)
ax.legend(fontsize=LEGEND_SIZE)
ax.set_ylim([-1, 1])
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.3f}', ha='center', va='bottom', fontsize=BAR_VAL_SIZE, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height:.3f}', ha='center', va='bottom', fontsize=BAR_VAL_SIZE, fontweight='bold')
plt.tight_layout()
path4 = os.path.join(OUTPUT_DIR, "pearson_vs_spearman_deeplabv3.png")
plt.savefig(path4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved -> {path4}")

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SPEARMAN RANK CORRELATION SUMMARY (Diagonal: same metric GT vs Pred)")
print("=" * 70)
print(f"  {'Metric':<25} {'UNet++ (rho)':<15} {'DeepLabV3+ (rho)':<15}")
print("-" * 70)
short_names = ["Connected Components", "Branch Points", "Endpoints"]
for i in range(3):
    print(f"  {short_names[i]:<25} {unetpp_corr[i,i]:<15.4f} {dlv3_corr[i,i]:<15.4f}")
print("=" * 70)
print(f"\n  Pearson vs Spearman Difference (UNet++):")
for i in range(3):
    diff = abs(pearson_unetpp_diag[i] - spearman_unetpp_diag[i])
    print(f"    {short_names[i]:<25} Delta = {diff:.4f}")
print(f"\n  Pearson vs Spearman Difference (DeepLabV3+):")
for i in range(3):
    diff = abs(pearson_dlv3_diag[i] - spearman_dlv3_diag[i])
    print(f"    {short_names[i]:<25} Delta = {diff:.4f}")
print("=" * 70)
print("\n[DONE] Spearman correlation analysis complete!")
print(f"  Heatmaps    -> {path1}, {path2}")
print(f"  Comparison  -> {path3}, {path4}")

