import numpy as np, pandas as pd, json
from ffp_model import *
from experiments_cfg import P_LEVELS, CORR, BASE
rows=[]
for pl,p in P_LEVELS.items():
    for cl,ckw in CORR.items():
        P=replace(BASE,p=p,**ckw)
        pay,crd,rec=one_season_losses(P)
        D=derived(P)
        tot=pay+crd
        q_pay=np.quantile(pay,0.995); q_tot=np.quantile(tot,0.995)
        rows.append(dict(p_level=pl,corr=cl,var_pay=q_pay/C,var_int=(q_tot+D['expenses'])/C,
            es_pay=pay[pay>=q_pay].mean()/C, es_int=(tot[tot>=q_tot].mean()+D['expenses'])/C,
            prob_combined_exceeds_var_pay=float((tot>q_pay).mean()),
            mean_credit_loss_drought_season=float(crd[pay>0].mean()/C), mean_credit_loss_normal=float(crd[pay==0].mean()/C),
            zero_trailing_mcr_share=(1-p)**3))
pd.DataFrame(rows).to_csv('../results/T5_mcr.csv',index=False); print(pd.DataFrame(rows).round(3).to_string())
