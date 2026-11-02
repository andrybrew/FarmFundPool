import numpy as np, pandas as pd
rows={}
for l in open('../data/chirps_monthly_raw.txt').read().strip().split('\n'):
    k,v=l.split(':'); a=np.array(list(map(int,v.split(','))),float)
    idx=pd.period_range('1981-01','2025-06',freq='M'); rows[k]=pd.Series(a,idx)
M=pd.DataFrame(rows); M.to_csv('../data/chirps_monthly.csv')
JFM=M[M.index.month<=3].groupby(M[M.index.month<=3].index.year).sum()
AMJ=M[(M.index.month>=4)&(M.index.month<=6)].groupby(M[(M.index.month>=4)&(M.index.month<=6)].index.year).sum()
JFM.to_csv('../data/chirps_JFM.csv'); AMJ.to_csv('../data/chirps_AMJ.csv')
pd.set_option('display.width',200)
print(JFM.loc[2017:2023].round(0))
print(JFM.describe().round(0))
