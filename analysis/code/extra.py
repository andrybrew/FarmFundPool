import pandas as pd, numpy as np, json
from experiments_cfg import *
rows=[]
for pl,p in P_LEVELS.items():
    for cl,ckw in CORR.items():
        for att in [None,0.10]:
            P=replace(BASE,p=p,attach_q=att,**ckw)
            r=simulate(replace(P,k0=1.0,k0_mcr=True,coc=True))
            rows.append(dict(p_level=pl,corr=cl,attach_q=att,k0=r['k0_abs']/C,mcr_int=r['mcr_int']/C,ruin20=r['ruin'],ci_lo=r['ci_lo'],ci_hi=r['ci_hi'],premium=r['premium']/C))
t=pd.DataFrame(rows); t.to_csv('../results/T5b_capital_at_mcr.csv',index=False); print(t.round(3).to_string())
b=pd.read_csv('../results/T10_breakeven.csv') if __import__('os').path.exists('../results/T10_breakeven.csv') else None
