import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np
BLUE="#2a78d6"; ORANGE="#eb6834"; GREY="#666666"; GRID="#e5e5e5"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8})
c=pd.read_csv("../results/T9_convergence.csv"); s=pd.read_csv("../results/T9_seeds.csv")
fig,axs=plt.subplots(1,2,figsize=(7.2,2.8),dpi=300)
ax=axs[0]; ax.fill_between(c.paths,(c.estimate-1.96*c.se)*100,(c.estimate+1.96*c.se)*100,color=BLUE,alpha=0.18,lw=0)
ax.plot(c.paths,c.estimate*100,color=BLUE,lw=1.6); ax.set_xscale("log"); ax.set_xlabel("Number of simulated paths"); ax.set_ylabel("20-season ruin probability (%)")
ax.set_title("(a) Running estimate with 95% band",fontsize=8)
ax=axs[1]; ax.plot(s.seed,s.ruin*100,"o",color=BLUE,ms=4); ax.axhline(s.ruin.mean()*100,color=GREY,ls="--",lw=0.8)
ax.set_xlabel("Random seed (20,000 paths each)"); ax.set_ylabel("20-season ruin probability (%)"); ax.set_ylim(0.8,1.3); ax.set_title("(b) Twenty independent seeds",fontsize=8)
for a in axs:
    a.grid(color=GRID,lw=0.6); a.set_axisbelow(True)
    for sp in ["top","right"]: a.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig("FigureS9_convergence.png",bbox_inches="tight",facecolor="white"); print("ok")
