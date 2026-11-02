"""Experiments for the JRF revision. Writes CSV tables to ../results."""
import json, os, time
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata, beta as beta_dist, qmc, spearmanr
from ffp_model import *

OUT = "../results"
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


