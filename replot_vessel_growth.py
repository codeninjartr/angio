"""
Quick script to regenerate vessel_growth_plots.png from the existing CSV data,
without needing TensorFlow or the full analysis pipeline.
"""
import csv
import os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = r"d:\labdatanew_Seemant\outputs"
csv_path = os.path.join(OUTPUT_DIR, "vessel_growth_data.csv")

# Read CSV
summary_data = []
with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        summary_data.append({
            "concentration": row["concentration"],
            "hour": int(row["hour"].replace("h", "")),
            "count": int(row["n_eggs"]),
            "net_mean": float(row["networks_mean"]),
            "net_std": float(row["networks_std"]),
            "br_mean": float(row["branches_mean"]),
            "br_std": float(row["branches_std"]),
            "ep_mean": float(row["endpoints_mean"]),
            "ep_std": float(row["endpoints_std"]),
        })

concs_data = defaultdict(list)
for s in summary_data:
    concs_data[s["concentration"]].append(s)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("CAM Assay: Tumor Angiogenesis Assessment Across Drug Concentrations & Time", fontsize=15, fontweight="bold")

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
    "Vessel Network Fragmentation\n(↑ = Anti-angiogenic / Tumor-starving)",
    "Vascular Branching Complexity\n(↑ = Pro-angiogenic / Tumor-feeding)",
    "Capillary Dead-Ends\n(↑ = Immature vasculature / Poor perfusion)"
]
ylabels = ["Avg. Disconnected Networks", "Avg. Branch Points", "Avg. End Points"]
keys = ["net", "br", "ep"]

for i, key in enumerate(keys):
    ax = axes[i]
    ax.set_title(titles[i], fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (Hours)", fontsize=10)
    ax.set_ylabel(ylabels[i], fontsize=10)
    ax.grid(True, alpha=0.3)

    for conc in ["0.1ug", "1ug", "10ug", "control"]:
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
            linewidth=2, label=f"{conc.upper()}"
        )

    ax.legend(loc="best")

plt.tight_layout()
out_plot = os.path.join(OUTPUT_DIR, "vessel_growth_plots.png")
plt.savefig(out_plot, dpi=150)
plt.close()
print(f"Saved updated plot to {out_plot}")
