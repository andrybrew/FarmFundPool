import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np, matplotlib.ticker
BLUE="#2a78d6"; ORANGE="#eb6834"; INK="#222222"; GREY="#666666"; GRID="#e5e5e5"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.edgecolor":GREY,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK})
R="../results/"
# Figure 3: drought frequency
t=pd.read_csv(R+"T1_drought_frequency.csv")
names={"LampungTengah":"Central Lampung","Sidrap":"Sidenreng Rappang"}
order=["Bandung","Indramayu","Karawang","Subang","Cirebon","Sragen","Ngawi","Lamongan","Sidrap","LampungTengah","Pooled"]
fig,axs=plt.subplots(1,3,figsize=(7.2,3.3),dpi=300,sharey=True)
for ax,(f,lab) in zip(axs,[(0.85,"Mild: < 85% of normal"),(0.80,"Moderate: < 80% of normal"),(0.75,"Severe: < 75% of normal")]):
    s=t[t.threshold==f].set_index("district").loc[order]
    y=np.arange(len(order))[::-1]
    col=[ORANGE if d=="Pooled" else BLUE for d in order]
    for yi,(d,r),c in zip(y,s.iterrows(),col):
        ax.plot([r.lo*100,r.hi*100],[yi,yi],color=c,lw=1.6,solid_capstyle="round")
        ax.plot(r.p*100,yi,"o",ms=4.5,color=c,mec="white",mew=0.8)
    if f==0.85: ax.axvline(100/7,color=GREY,ls=":",lw=0.9); ax.text(100/7+0.6,10.45,"1/7",fontsize=6.5,color=GREY)
    ax.axvline(3,color=GREY,ls="--",lw=0.8)
    ax.set_title(lab,fontsize=8); ax.set_xlim(0,30); ax.grid(axis="x",color=GRID,lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("Seasons triggered (%)")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
axs[0].set_yticks(np.arange(len(order))[::-1]); axs[0].set_yticklabels([names.get(d,d) for d in order])
plt.tight_layout(); plt.savefig("Figure3_drought_frequency.png",bbox_inches="tight",facecolor="white"); plt.close()
# Figure 4: frontier
t=pd.read_csv(R+"T4_pricing_frontier.csv")
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),dpi=300)
for pay,c,mk in [("full",BLUE,"o"),("graded",ORANGE,"s")]:
    s=t[t.payout==pay]
    axs[0].plot(s.return_period,s.gross_rate*100,marker=mk,color=c,lw=1.6,ms=4,label=f"{pay} payout, gross")
    axs[0].plot(s.return_period,s.fair_rate*100,ls=":",color=c,lw=1.2,label=f"{pay} payout, expected loss")
    axs[1].plot(s.return_period,s.si_for_180k/1e6,marker=mk,color=c,lw=1.6,ms=4,label=f"{pay} payout")
axs[0].axhline(3,color=GREY,ls="--",lw=0.8); axs[0].text(31,3.6,"AUTP 3%",fontsize=6.5,color=GREY)
axs[1].axhline(6,color=GREY,ls="--",lw=0.8); axs[1].text(5,6.25,"AUTP sum insured Rp 6 million",fontsize=6.5,color=GREY)
axs[0].set_ylabel("Premium (% of sum insured)"); axs[1].set_ylabel("Sum insured bought by Rp 180,000 (Rp million)")
for ax in axs:
    ax.set_xscale("log"); ax.set_xticks([5,7,10,15,20,30,50]); ax.set_xticklabels(["5","7","10","15","20","30","50"]); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Trigger return period (1 in n seasons)"); ax.grid(color=GRID,lw=0.6); ax.set_axisbelow(True)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
axs[0].legend(fontsize=6.3,frameon=False); axs[1].legend(fontsize=6.5,frameon=False,loc="center left")
axs[0].set_title("(a) Premium rate",fontsize=8); axs[1].set_title("(b) Cover affordable at the AUTP premium",fontsize=8)
plt.tight_layout(); plt.savefig("Figure4_pricing_frontier.png",bbox_inches="tight",facecolor="white"); plt.close()
# Figure 5: tornado
t=pd.read_csv(R+"T7_tornado.csv")
lab={"p":"Drought frequency (3.3% to 14.3%)","r_res":"Yield on reserves (0% to 6%)","d0":"Normal-season default (1% to 6%)","d1":"Drought-season default (10% to 40%)","lgd":"Loss given default (30% to 70%)","r":"Loan rate (6% to 24%)","loan":"Loan size (Rp 3m to 10m)","expense":"Operating expenses (10% to 30%)","reins_mult":"Reinsurance price (1.5x to 3x)","r_inv":"Investor return (4% to 12%)","rho_c":"Within-cooperative correlation (0 to 0.3)","beta":"Lendable share β (0.5 to 1)"}
t["span"]=(t.ruin_high-t.ruin_low).abs(); t=t.sort_values("span")
base=t.base.iloc[0]*100
fig,ax=plt.subplots(figsize=(7.2,3.6),dpi=300)
for i,(_,r) in enumerate(t.iterrows()):
    lo,hi=r.ruin_low*100,r.ruin_high*100
    ax.barh(i,lo-base,left=base,color=BLUE,height=0.62); ax.barh(i,hi-base,left=base,color=ORANGE,height=0.62)
ax.set_yticks(range(len(t))); ax.set_yticklabels([lab[p] for p in t.parameter])
ax.axvline(base,color=INK,lw=0.8); ax.set_xscale("symlog",linthresh=1); ax.set_xlim(0,100)
ax.set_xticks([0,0.5,1,2,5,10,20,50,100]); ax.set_xticklabels(["0","0.5","1","2","5","10","20","50","100"])
ax.set_xlabel("20-season ruin probability (%), premium held at the baseline price")
ax.grid(axis="x",color=GRID,lw=0.6); ax.set_axisbelow(True)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE,label="low end of range"),Patch(color=ORANGE,label="high end of range")],fontsize=6.8,frameon=False,loc="lower right")
plt.tight_layout(); plt.savefig("Figure5_sensitivity.png",bbox_inches="tight",facecolor="white"); plt.close()
print("ok")
