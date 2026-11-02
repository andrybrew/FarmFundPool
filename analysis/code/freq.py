import numpy as np, pandas as pd
from scipy.stats import beta, norm, kendalltau, rankdata
J=pd.read_csv('../data/chirps_JFM.csv',index_col=0)
J=J.loc[1981:2025]
norm9120=J.loc[1991:2020].median()
def cp(k,n,a=0.05):
    lo=beta.ppf(a/2,k,n-k+1) if k>0 else 0.0; hi=beta.ppf(1-a/2,k+1,n-k) if k<n else 1.0; return lo,hi
print("Exact CI for 1/7:",[round(x,3) for x in cp(1,7)])
R=J/norm9120
out=[]
for f in [0.6,0.7,0.75,0.8]:
    for k in J.columns:
        x=(R[k]<f).sum(); n=len(R)
        out.append((f,k,x,n,x/n,*cp(x,n)))
    allx=(R<f).values.sum(); N=R.size
    # pooled with district-cluster bootstrap over years
    rng=np.random.default_rng(1); bs=[]
    for _ in range(4000):
        yrs=rng.integers(0,len(R),len(R)); bs.append((R.values[yrs]<f).mean())
    out.append((f,'POOLED',allx,N,allx/N,np.quantile(bs,.025),np.quantile(bs,.975)))
df=pd.DataFrame(out,columns=['f','district','k','n','p','lo','hi'])
pd.set_option('display.width',200); pd.set_option('display.max_rows',100)
print(df.round(3).to_string())
# correlations of normal scores
Z=J.apply(lambda s: norm.ppf(rankdata(s)/(len(s)+1)))
C=Z.corr(); print(C.round(2))
iu=np.triu_indices(len(C),1); print("mean pairwise corr all",C.values[iu].mean().round(2))
wj=['Bandung','Indramayu','Karawang','Subang','Cirebon']; Cw=C.loc[wj,wj].values; iw=np.triu_indices(5,1); print("West Java mean",Cw[iw].mean().round(2))
# one-factor rho: average corr (latent)
for k in J.columns:
    t,p=kendalltau(J.index,J[k]); print(k,'trend tau',round(t,2),'p',round(p,3))
print(norm9120.round(0))
