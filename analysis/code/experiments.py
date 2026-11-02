"""Experiments for the JRF revision. Writes CSV tables to ../results."""
import json, os, time
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata, beta as beta_dist, qmc, spearmanr
from ffp_model import *

OUT = "../results"; os.makedirs(OUT, exist_ok=True)
PATHS = 20_000
J = pd.read_csv("../data/chirps_JFM.csv", index_col=0).loc[1981:2025]
NORMAL = J.loc[1991:2020].median()
R = J / NORMAL
Z = J.apply(lambda s: norm.ppf(rankdata(s) / (len(s) + 1)))
CN = tuple(np.round(Z.corr().values, 4).ravel())          # empirical 10-district latent correlation
P_LEVELS = {"severe (25% deficit)": 15 / 450, "moderate (20% deficit)": 26 / 450, "mild (15% deficit, original trigger)": 1 / 7}
CORR = {"national (10 districts, empirical)": dict(corr=CN, zones=10),
        "single province (rho = 0.8)": dict(rho=0.8, zones=10)}
BASE = Params(expense=0.15, r_inv=0.08, paths=PATHS)


def cp(k, n, a=0.05):
    lo = beta_dist.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


# ---------- T1. Drought frequency ----------
rows = []
for f in [0.75, 0.80, 0.85]:
    for d in J.columns:
        k = int((R[d] < f).sum()); n = len(R); lo, hi = cp(k, n)
        rows.append(dict(threshold=f, district=d, k=k, n=n, p=k / n, lo=lo, hi=hi))
    rng = np.random.default_rng(1); bs = [(R.values[rng.integers(0, len(R), len(R))] < f).mean() for _ in range(4000)]
    k = int((R < f).values.sum()); n = R.size
    rows.append(dict(threshold=f, district="Pooled", k=k, n=n, p=k / n, lo=np.quantile(bs, .025), hi=np.quantile(bs, .975)))
T1 = pd.DataFrame(rows); T1.to_csv(f"{OUT}/T1_drought_frequency.csv", index=False)
CM = Z.corr(); iu = np.triu_indices(10, 1)
wj = ["Bandung", "Indramayu", "Karawang", "Subang", "Cirebon"]
corr_summary = dict(all_mean=float(CM.values[iu].mean()), all_min=float(CM.values[iu].min()), all_max=float(CM.values[iu].max()),
                    wj_mean=float(CM.loc[wj, wj].values[np.triu_indices(5, 1)].mean()),
                    bandung_station_1in7=float(cp(1, 7)[0]), bandung_station_1in7_hi=float(cp(1, 7)[1]),
                    normals=NORMAL.round(0).to_dict())
CM.round(2).to_csv(f"{OUT}/T1b_correlation.csv")
json.dump(corr_summary, open(f"{OUT}/T1c_corr_summary.json", "w"), indent=1)
print("T1 done", time.ctime())

# ---------- T2. Status quo and pricing ----------
rows = []
for pl, p in P_LEVELS.items():
    for cl, ckw in CORR.items():
        A = simulate(replace(BASE, p=p, premium=0.03, expense=0.0, r_inv=0.0, mcr="payout", k0=1.0, **ckw))
        B = simulate(replace(BASE, p=p, k0=1.0, **ckw))  # fair + expenses, no CoC, investors paid from surplus
        rows.append(dict(p_level=pl, p=p, corr=cl, config="A: AUTP premium 3%", **{k: A[k] for k in
                    ["premium", "ruin", "ci_lo", "ci_hi", "epd", "deficit_given_ruin", "inv_irr_median", "inv_loss_prob", "mcr_pay", "mcr_int", "lend_suspended"]}))
        rows.append(dict(p_level=pl, p=p, corr=cl, config="B: fair premium + expenses", **{k: B[k] for k in
                    ["premium", "ruin", "ci_lo", "ci_hi", "epd", "deficit_given_ruin", "inv_irr_median", "inv_loss_prob", "mcr_pay", "mcr_int", "lend_suspended"]}))
T2 = pd.DataFrame(rows); T2.to_csv(f"{OUT}/T2_status_quo.csv", index=False)
print("T2 done", time.ctime())

# ---------- T3. Cost of a configuration meeting a solvency target ----------
rows = []
for pl, p in P_LEVELS.items():
    for cl, ckw in CORR.items():
        for rl, rkw in {"none": dict(attach_q=None), "stop-loss above 1-in-10 season": dict(attach_q=0.10),
                        "stop-loss above 1-in-5 season": dict(attach_q=0.20)}.items():
            P = replace(BASE, p=p, **ckw, **rkw)
            k = required_capital(P, target=0.01)
            r = simulate(replace(P, k0=k, coc=True))
            prem = r["premium"]
            rows.append(dict(p_level=pl, p=p, corr=cl, reinsurance=rl, k_req=k, premium=prem, rate=prem / C,
                             fair=r["fair"], expenses=r["expenses"], reins_net=r["reins_net"], coc=r["coc"],
                             farmer_20=0.2 * prem, state_80=0.8 * prem, ruin=r["ruin"], ci_hi=r["ci_hi"], epd=r["epd"],
                             mcr_pay=r["mcr_pay"], mcr_int=r["mcr_int"], attach=r["attach_abs"],
                             inv_irr_median=r["inv_irr_median"], inv_loss_prob=r["inv_loss_prob"], lend_suspended=r["lend_suspended"]))
            print(pl, cl, rl, round(k, 3), round(prem / C, 4), time.ctime())
            if rl != "stop-loss above 1-in-5 season":
                P2 = replace(P, r_res=0.06)
                k2 = required_capital(P2, target=0.01)
                r2 = simulate(replace(P2, k0=k2, coc=True))
                rows.append(dict(p_level=pl, p=p, corr=cl, reinsurance=rl + "; reserves earn 6%", k_req=k2, premium=r2["premium"], rate=r2["premium"] / C,
                                 fair=r2["fair"], expenses=r2["expenses"], reins_net=r2["reins_net"], coc=r2["coc"],
                                 farmer_20=0.2 * r2["premium"], state_80=0.8 * r2["premium"], ruin=r2["ruin"], ci_hi=r2["ci_hi"], epd=r2["epd"],
                                 mcr_pay=r2["mcr_pay"], mcr_int=r2["mcr_int"], attach=r2["attach_abs"],
                                 inv_irr_median=r2["inv_irr_median"], inv_loss_prob=r2["inv_loss_prob"], lend_suspended=r2["lend_suspended"]))
T3 = pd.DataFrame(rows); T3.to_csv(f"{OUT}/T3_required_capital_cost.csv", index=False)

# ---------- T4. Pricing frontier (trigger frequency x payout structure) ----------
rows = []
loading = (T3.query("p_level=='moderate (20% deficit)' and corr.str.startswith('national') and reinsurance=='stop-loss above 1-in-10 season'")["rate"].iloc[0]
           / T3.query("p_level=='moderate (20% deficit)' and corr.str.startswith('national') and reinsurance=='stop-loss above 1-in-10 season'")["p"].iloc[0])
for rp in [5, 7, 10, 15, 20, 30, 50]:
    p = 1 / rp
    for graded in [False, True]:
        Q = replace(BASE, p=p, graded=graded, corr=CN, zones=10, attach_q=0.10)
        k = required_capital(Q, target=0.01)
        r = simulate(replace(Q, k0=k, coc=True))
        rate = r["premium"] / C
        rows.append(dict(return_period=rp, p=p, payout="graded" if graded else "full", fair_rate=r["fair"] / C, gross_rate=rate,
                         k_req=k, si_for_180k=180_000 / rate, farmer_share=0.2 * r["premium"], state_share=0.8 * r["premium"]))
T4 = pd.DataFrame(rows); T4.to_csv(f"{OUT}/T4_pricing_frontier.csv", index=False)
print("T4 done", time.ctime())

# ---------- T5. Integrated vs payout-only MCR, dependence stress ----------
rows = []
for pl, p in P_LEVELS.items():
    for cl, ckw in CORR.items():
        P = replace(BASE, p=p, attach_q=0.10, **ckw)
        pay, cl_, rec = one_season_losses(P)
        D = derived(P)
        rows.append(dict(p_level=pl, corr=cl, mcr_pay=D["mcr_pay"] / C, mcr_int=D["mcr_int"] / C,
                         prob_combined_exceeds_mcr_pay=float(((pay - rec + cl_) > D["mcr_pay"]).mean()),
                         es_pay=D["es_pay"] / C, es_int=D["es_int"] / C,
                         zero_trailing_mcr_share=(1 - p) ** 3))
T5 = pd.DataFrame(rows); T5.to_csv(f"{OUT}/T5_mcr.csv", index=False)

rows = []
Pc = replace(BASE, p=P_LEVELS["moderate (20% deficit)"], attach_q=0.10, corr=CN, zones=10)
kc = required_capital(Pc, target=0.01)
base_prem = simulate(replace(Pc, k0=kc, coc=True))["premium"] / C
for lab, kw in {"Gaussian, empirical correlation": dict(),
                "Student-t (nu = 4), empirical correlation": dict(copula="t", nu=4),
                "Student-t (nu = 2), empirical correlation": dict(copula="t", nu=2),
                "Gaussian, one factor rho = 0.8": dict(corr=None, rho=0.8),
                "Student-t (nu = 4), rho = 0.8": dict(corr=None, rho=0.8, copula="t", nu=4),
                "Within-cooperative default correlation 0.2": dict(rho_c=0.2),
                "Drought-year default 40%": dict(d1=0.40),
                "Drought frequency doubled": dict(p=2 * P_LEVELS["moderate (20% deficit)"])}.items():
    r = simulate(replace(Pc, k0=kc, premium=base_prem, **kw))
    rows.append(dict(scenario=lab, ruin=r["ruin"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], epd=r["epd"], deficit_given_ruin=r["deficit_given_ruin"],
                     inv_loss_prob=r["inv_loss_prob"]))
T6 = pd.DataFrame(rows); T6.to_csv(f"{OUT}/T6_stress.csv", index=False)
json.dump(dict(k_central=kc, premium_central=base_prem), open(f"{OUT}/central.json", "w"))
print("T5-6 done", time.ctime())

# ---------- T7. One-at-a-time sensitivity (tornado) at the central configuration ----------
ranges = {"p": (0.033, 0.143), "r_res": (0.0, 0.06), "d0": (0.01, 0.06), "d1": (0.10, 0.40), "lgd": (0.30, 0.70), "r": (0.06, 0.24),
          "loan": (3e6, 10e6), "expense": (0.10, 0.30), "reins_mult": (1.5, 3.0), "r_inv": (0.04, 0.12),
          "rho_c": (0.0, 0.3), "beta": (0.5, 1.0)}
base = simulate(replace(Pc, k0=kc, premium=base_prem))["ruin"]
rows = []
for k, (lo, hi) in ranges.items():
    rl = simulate(replace(Pc, k0=kc, premium=base_prem, **{k: lo}))["ruin"]
    rh = simulate(replace(Pc, k0=kc, premium=base_prem, **{k: hi}))["ruin"]
    rows.append(dict(parameter=k, low=lo, high=hi, ruin_low=rl, ruin_high=rh, base=base))
T7 = pd.DataFrame(rows); T7.to_csv(f"{OUT}/T7_tornado.csv", index=False)
print("T7 done", time.ctime())

# ---------- T8. Global sensitivity: Latin hypercube + standardized rank regression ----------
names = list(ranges); lows = np.array([ranges[k][0] for k in names]); highs = np.array([ranges[k][1] for k in names])
X = qmc.scale(qmc.LatinHypercube(d=len(names), seed=11).random(1000), lows, highs)
y = []
for i, x in enumerate(X):
    kw = dict(zip(names, x)); kw = {k: float(v) for k, v in kw.items()}
    y.append(simulate(replace(Pc, k0=kc, premium=base_prem, paths=4000, seed=1000 + i, **kw))["ruin"])
y = np.array(y)
Xr = np.column_stack([rankdata(c) for c in X.T]); yr = rankdata(y)
Xs = (Xr - Xr.mean(0)) / Xr.std(0); ys = (yr - yr.mean()) / yr.std()
coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(ys)), Xs]), ys, rcond=None)
pred = np.column_stack([np.ones(len(ys)), Xs]) @ coef
r2 = 1 - ((ys - pred) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
T8 = pd.DataFrame(dict(parameter=names, srrc=coef[1:])).sort_values("srrc", key=abs, ascending=False)
T8["r2"] = r2; T8.to_csv(f"{OUT}/T8_global_srrc.csv", index=False)
pd.DataFrame(np.column_stack([X, y]), columns=names + ["ruin"]).to_csv(f"{OUT}/T8_lhs_samples.csv", index=False)
print("T8 done", time.ctime())

# ---------- T9. Convergence ----------
r = simulate(replace(Pc, k0=kc, premium=base_prem, paths=100_000), return_paths=True)
ru = r["ruined_paths"].astype(float)
ns = np.unique(np.logspace(3, 5, 25).astype(int))
conv = pd.DataFrame(dict(paths=ns, estimate=[ru[:n].mean() for n in ns], se=[np.sqrt(ru[:n].mean() * (1 - ru[:n].mean()) / n) for n in ns]))
conv.to_csv(f"{OUT}/T9_convergence.csv", index=False)
seeds = [simulate(replace(Pc, k0=kc, premium=base_prem, seed=s))["ruin"] for s in range(1, 21)]
pd.DataFrame(dict(seed=range(1, 21), ruin=seeds)).to_csv(f"{OUT}/T9_seeds.csv", index=False)
print("T9 done", time.ctime())

# ---------- T10. Lending break-even (closed form) ----------
rows = []
for pl, p in P_LEVELS.items():
    gap = max(0.0, p * C - 0.03 * C)
    for r_ in [0.06, 0.12, 0.18, 0.24]:
        for d1 in [0.10, 0.25, 0.40]:
            ed = (1 - p) * 0.03 + p * d1
            m = r_ / 2 * (1 - ed) - ed * 0.5
            rows.append(dict(p_level=pl, p=p, gap=gap, r=r_, d1=d1, margin=m, breakeven_loan=(gap / m if m > 0 else np.nan) if gap > 0 else 0.0))
pd.DataFrame(rows).to_csv(f"{OUT}/T10_breakeven.csv", index=False)
print("ALL DONE", time.ctime())
