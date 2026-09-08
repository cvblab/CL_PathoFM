"""Horizontal heatmap of the data-scarcity effect on CrowdGleason.

Numbers are computed directly from results_rerun_scarcity_crowdgleason/, the
same source and paired-(encoder,seed) method as revision/make_results.py.

Color: defaults to red/blue diverging. Pass `redgreen` on the command line for the
classic RdYlGn variant instead.

    python plot_scarcity_heatmap_crowdgleason.py [redblue|redgreen]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SCHEME = sys.argv[1] if len(sys.argv) > 1 else "redblue"
if SCHEME not in ("redblue", "redgreen"):
    raise SystemExit(f"unknown scheme {SCHEME!r}; use redblue or redgreen")

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

ROOT = os.environ.get(
    "SCARCITY_RESULTS_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                  "results_rerun_scarcity_crowdgleason")),
)
ENCS = ["CONCHv1.5", "UNI2", "VIRCHOW2"]
SEEDS = [42, 53, 78, 102, 294]
DIR_FMT = "experiment4_{enc}_subsets_ordered"
FRACS = [10, 25, 50, 100]
CRITERIA = [
    ("cl_prior", "Prior"),
    ("cl_adaptive_raw", "Adaptive"),
    ("cl_combined_staged", "Combined Staged"),
    ("cl_combined_optimized", "Combined Cosine"),
]


def f1(enc, exp, pct, i):
    p = os.path.join(ROOT, DIR_FMT.format(enc=enc), f"tmp_{exp}_frac{pct}_run{i}_test.json")
    return json.load(open(p))["f1"]


grid = np.zeros((len(CRITERIA), len(FRACS)))
for row, (exp, _) in enumerate(CRITERIA):
    for col, pct in enumerate(FRACS):
        base = np.array([f1(enc, "non_cl", pct, i) for enc in ENCS for i in range(len(SEEDS))])
        cl = np.array([f1(enc, exp, pct, i) for enc in ENCS for i in range(len(SEEDS))])
        grid[row, col] = (cl.mean() - base.mean()) / base.mean() * 100

if SCHEME == "redblue":
    cmap = LinearSegmentedColormap.from_list(
        "red_gray_blue", ["#d03b3b", "#f0efec", "#2a78d6"], N=256
    )
else:
    cmap = plt.get_cmap("RdYlGn")

bound = np.ceil(np.abs(grid).max())  # symmetric around 0 so gray stays "no effect"
fig, ax = plt.subplots(figsize=(11, 4.6))
im = ax.imshow(grid, cmap=cmap, vmin=-bound, vmax=bound, aspect="auto")

ax.set_xticks(range(len(FRACS)))
ax.set_xticklabels([f"{f}%" for f in FRACS])
ax.set_yticks(range(len(CRITERIA)))
ax.set_yticklabels([label for _, label in CRITERIA])
ax.set_xlabel("Training fraction")
ax.set_title("Effect under data scarcity (CrowdGleason)", fontweight="bold", pad=14)

ax.set_xticks(np.arange(-0.5, len(FRACS), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(CRITERIA), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

for row in range(len(CRITERIA)):
    for col in range(len(FRACS)):
        v = grid[row, col]
        lum = abs(v) / bound
        text_color = "white" if lum > 0.55 else "#2a2a28"
        ax.text(col, row, f"{v:+.2f}", ha="center", va="center",
                fontsize=12, color=text_color)

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label("Relative F1 Gain (%)", fontsize=11)

plt.tight_layout()
suffix = "" if SCHEME == "redblue" else "_redgreen"
out = os.path.join(ROOT, f"scarcity_heatmap_crowdgleason{suffix}")
plt.savefig(out + ".pdf", dpi=300, bbox_inches="tight")
plt.savefig(out + ".png", dpi=300, bbox_inches="tight")
print(f"Saved {out}.pdf and {out}.png")
