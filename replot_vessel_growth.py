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

TITLE_SIZE  = 36
LABEL_SIZE  = 28
TICK_SIZE   = 22
LEGEND_SIZE = 22
LINE_WIDTH  = 3.5

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

plots = [
    {
        "filename": "vessel_growth_networks.png",
        "title": "Vessel Network Fragmentation\n(↑ = Anti-angiogenic / Tumor-starving)",
        "ylabel": "Avg. Disconnected Networks",
        "key": "net",
    },
    {
        "filename": "vessel_growth_branches.png",
        "title": "Vascular Branching Complexity\n(↑ = Pro-angiogenic / Tumor-feeding)",
        "ylabel": "Avg. Branch Points",
        "key": "br",
    },
    {
        "filename": "vessel_growth_endpoints.png",
        "title": "Capillary Dead-Ends\n(↑ = Immature vasculature / Poor perfusion)",
        "ylabel": "Avg. End Points",
        "key": "ep",
    },
]

for p in plots:
    fig, ax = plt.subplots(figsize=(12, 9))
    key = p["key"]

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
            ecolor=colors[conc], elinewidth=2, capsize=5,
            linewidth=LINE_WIDTH, markersize=10, label=f"{conc.upper()}"
        )

    ax.set_title(p["title"], fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Time (Hours)", fontsize=LABEL_SIZE)
    ax.set_ylabel(p["ylabel"], fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_SIZE)

    plt.tight_layout()
    out_plot = os.path.join(OUTPUT_DIR, p["filename"])
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"Saved: {out_plot}")

