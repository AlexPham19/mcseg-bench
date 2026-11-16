"""Draw the documented benchmark workflow using no medical data.

Optional dependency: matplotlib (see requirements-plots.txt).
Run from the package root: python scripts/draw_pipeline.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

root = Path(__file__).resolve().parents[1]
fig, ax = plt.subplots(figsize=(15, 10.8))
fig.patch.set_facecolor("#f7f9fc")
ax.set_facecolor("#f7f9fc")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(.03, .965, "Microcalcification benchmark: shared execution pipeline", fontsize=21, weight="bold", color="#14263d")
ax.text(.03, .926, "Fit on TRAIN. Select and calibrate on DEV. Apply the frozen decisions to TEST.", fontsize=12, color="#4d627b")

rows = [
    ("1  PREPARE", "Organizer", "#e8eef8", ["Real mammograms\nand binary references", "Audit labels\nand patient/group linkage", "Preserve registered\ntrain / dev / test split", "Freeze manifest\npaths + SHA256 hashes"]),
    ("2  TRAIN", "Model runner", "#e6f3ec", ["TRAIN image/mask\npatches", "Fit reference U-Net\nBCE + soft Dice", "DEV IoU at 0.5\nselect checkpoint", "Frozen checkpoint\nmodel.pt"]),
    ("3  CALIBRATE", "Development only", "#fff1d9", ["DEV images", "Frozen model\nnative probability maps", "DEV GT + fixed grid\nchoose operating points", "Threshold lock\nthreshold_lock.json"]),
    ("4  PREDICT", "No test-label tuning", "#e8f1fb", ["TEST images", "Same frozen model\nnative probability maps", "Apply locked\noperating point", "Native-coordinate\nbinary prediction"]),
    ("5  EVALUATE", "Scoring only", "#eee9f7", ["Prediction + TEST GT\nreferences enter here", "Shared pixel and\ncomponent metrics", "Group confidence intervals\nand paired comparisons", "Metrics + report\nactual test FP burden"]),
]

start_x, width, gap, height = .215, .174, .023, .097
ys = [.79, .63, .47, .31, .15]
for (title, subtitle, color, cells), y in zip(rows, ys):
    ax.text(.03, y+.049, title, fontsize=12, weight="bold", color="#1b3350", va="center")
    ax.text(.03, y+.019, subtitle, fontsize=9.5, color="#5a6c82", va="center")
    for i, text in enumerate(cells):
        x = start_x + i*(width+gap)
        box = FancyBboxPatch((x,y),width,height,boxstyle="round,pad=0.009,rounding_size=0.012",facecolor=color,edgecolor="#bdcbdc",linewidth=1)
        ax.add_patch(box)
        ax.text(x+width/2,y+height/2,text,ha="center",va="center",fontsize=10.1,color="#17304a",linespacing=1.6)
        if i<3:
            ax.add_patch(FancyArrowPatch((x+width+.004,y+height/2),(x+width+gap-.003,y+height/2),arrowstyle="-|>",mutation_scale=12,linewidth=1.1,color="#6b7e96"))

ax.text(.03,.072,"Two registered objectives: contour IoU on positive images; component recall under development-selected FP budgets.",fontsize=11,color="#243d5a")
ax.text(.03,.037,"Implemented starter: reference U-Net + common calibration/evaluation. External-model adapters and verifier/refiner extensions remain separate work.",fontsize=9.3,color="#5b6d82")
out=root/"docs"
out.mkdir(exist_ok=True)
fig.savefig(out/"pipeline_overview.png",dpi=160,bbox_inches="tight",facecolor=fig.get_facecolor())
fig.savefig(out/"pipeline_overview.svg",bbox_inches="tight",facecolor=fig.get_facecolor())
plt.close(fig)
print("Wrote docs/pipeline_overview.png and docs/pipeline_overview.svg")
