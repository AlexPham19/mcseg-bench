"""Regenerate an aggregate tradeoff figure; no medical images or private IDs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
rows=json.loads((root/"reports/OBSERVED_AGGREGATES.json").read_text(encoding="utf-8"))["models"]
labels=["DINO raw","DINO + GRID","Historical best Dice","AND raw","AND + GRID","AND + MEAN","AND + MEAN + YOLO","TRIM raw","TRIM + GRID","TRIM + CG","HR-AND raw","HR-AND + CG"]
offsets=[(8,-16),(8,0),(-5,-22),(8,-18),(-105,25),(-55,-35),(9,-2),(8,2),(6,4),(-75,5),(8,3),(8,-20)]
fig,ax=plt.subplots(figsize=(13.6,7.3))
sc=ax.scatter([r["fp_per_image"] for r in rows],[100*r["component_recall"] for r in rows],
              c=[100*r["positive_macro_iou"] for r in rows],cmap="viridis",vmin=10,vmax=46,s=105,edgecolor="white",linewidth=1.2,zorder=3)
for r,name,offset in zip(rows,labels,offsets):
    ax.annotate(name,(r["fp_per_image"],100*r["component_recall"]),xytext=offset,textcoords="offset points",fontsize=9,
                arrowprops=dict(arrowstyle="-",color="#777777",lw=.65))
ax.set_xscale("log");ax.set_xlim(3.4,220);ax.set_ylim(57,95)
ax.set_xlabel("False-positive regions / image, all44 images (log scale)")
ax.set_ylabel("Component recall (%) — one-to-one, any pixel overlap")
ax.set_title("CSAW44 development: contour IoU and component recall select different points",pad=18)
ax.grid(alpha=.18,which="both");fig.colorbar(sc,ax=ax,label="Mean foreground IoU on24 positive images (%)")
fig.text(.08,.015,"Exploratory saved operating points; not held-out test or a complete Pareto frontier. FP regions are not clinical reader marks.",fontsize=9,color="#555555")
fig.tight_layout(rect=(0,.04,1,1))
out=root/"runs/regenerated_tradeoff.png";out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,dpi=160)
print(out)
