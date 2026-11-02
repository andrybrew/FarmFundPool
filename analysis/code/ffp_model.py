"""FarmFund Pool integrated risk model (JRF major revision).

All monetary values are per insured farmer (one hectare), in Rp, for one six-month season.
A season t proceeds as follows:
  1. Premiums are collected and operating expenses and the reinsurance premium are paid.
  2. Capital above the Minimum Capital Requirement (MCR) can be lent to cooperatives (eq. 6).
  3. Drought indicators are drawn for each index zone from a Gaussian or Student-t copula.
  4. Payouts are made from capital; the stop-loss reinsurer recovers the excess above the attachment.
  5. Loans mature: interest is earned on performing loans; defaulted principal is lost at LGD.
     Default rates depend on zone drought (conditional PD) and on a cooperative-level factor.
  6. If capital is at least the MCR, investors receive their required return (eq. 5 condition).
Ruin is recorded when capital becomes negative (payouts cannot be met in full).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, replace, asdict
from scipy.stats import norm, t as student_t

C = 6_000_000  # sum insured per hectare (AUTP)


@dataclass
class Params:
    p: float = 1 / 7            # zone drought (trigger) probability per season
    zones: int = 10
    rho: float = 0.5            # latent drought correlation between zones
    copula: str = "gauss"       # "gauss" or "t"
    nu: float = 4.0             # t-copula degrees of freedom
    graded: bool = False        # graded payout between trigger and exit
    exit_ratio: float = 0.25    # exit quantile = exit_ratio * p (full payout)
    premium: float | None = None  # gross premium rate on C; None = fair*(1+load)+expense+reins
    load: float = 0.0           # risk loading on the fair premium
    expense: float = 0.15       # operating expenses as share of the fair premium (admin, oracle, technology)
    loan: float = 6_000_000     # loan per insured farmer
    r: float = 0.18             # annual loan rate, six-month tenor
    d0: float = 0.03            # default rate, normal season
    d1: float = 0.25            # default rate, drought season (conditional PD)
    lgd: float = 0.5
    rho_c: float = 0.0          # within-cooperative default correlation (Vasicek)
    beta: float = 0.8           # share of capital above MCR that may be lent (eq. 6)
    k0: float = 1.0             # initial investor capital as share of C (or multiple of MCR if k0_mcr)
    k0_mcr: bool = False        # if True, initial capital = k0 * MCR
    coc: bool = False           # if True, premium includes investors' required return on initial capital
    r_res: float = 0.0          # annual yield earned on pool capital (0 = idle on-chain reserve)
    r_inv: float = 0.0          # investor required annual return (paid each season if capital >= MCR)
    attach: float | None = None  # stop-loss attachment as share of C (aggregate payout); None = no reinsurance
    attach_q: float | None = None  # alternatively: attachment at the (1 - attach_q) quantile of seasonal payout
    reins_mult: float = 2.0     # reinsurance price as multiple of expected recovery
    mcr: str = "integrated"     # "payout" (Q99.5 of payouts) or "integrated"
    corr: tuple | None = None   # full zone correlation matrix (flattened tuple); overrides zones/rho
    seasons: int = 20
    paths: int = 20_000
    seed: int = 20261026


_CHOL = {}


def _latent(rng, n, z, rho, copula, nu, corr=None):
    if corr is not None:
        k = int(round(len(corr) ** 0.5))
        if corr not in _CHOL:
            _CHOL[corr] = np.linalg.cholesky(np.array(corr).reshape(k, k))
        y = rng.standard_normal((n, k)) @ _CHOL[corr].T
    else:
        f = rng.standard_normal((n, 1))
        e = rng.standard_normal((n, z))
        y = np.sqrt(rho) * f + np.sqrt(1 - rho) * e
    if copula == "t":
        w = rng.chisquare(nu, size=(n, 1)) / nu
        y = y / np.sqrt(w)
        u = student_t.cdf(y, nu)
    else:
        u = norm.cdf(y)
    return u  # uniforms with the chosen dependence


def payout_fraction(u, P: Params):
    """Payout as share of C given zone uniform u (low u = dry)."""
    if not P.graded:
        return (u < P.p).astype(float)
    lo = P.exit_ratio * P.p
    return np.clip((P.p - u) / (P.p - lo), 0.0, 1.0)


def expected_payout_rate(P: Params):
    if not P.graded:
        return P.p
    lo = P.exit_ratio * P.p
    # E[clip((p-u)/(p-lo))] for u~U(0,1) = lo + (p-lo)/2
    return lo + (P.p - lo) / 2


def default_fraction(rng, n, drought_zone_flag, P: Params):
    pd = np.where(drought_zone_flag, P.d1, P.d0)
    if P.rho_c <= 0:
        return pd
    m = rng.standard_normal(pd.shape)
    return norm.cdf((norm.ppf(pd) - np.sqrt(P.rho_c) * m) / np.sqrt(1 - P.rho_c))


def attach_abs(P: Params, pay=None):
    if P.attach_q is not None:
        if pay is None:
            pay = one_season_losses(replace(P, attach=None, attach_q=None))[0]
        return float(np.quantile(pay, 1 - P.attach_q))
    return None if P.attach is None else P.attach * C


def one_season_losses(P: Params, n=200_000, seed=7):
    """Draw one-season payout share S, credit-loss share and recovery for MCR and reinsurance pricing."""
    rng = np.random.default_rng(seed)
    u = _latent(rng, n, P.zones, P.rho, P.copula, P.nu, P.corr)
    pay = payout_fraction(u, P).mean(1) * C
    df = default_fraction(rng, n, u < P.p, P).mean(1)
    cl = P.loan * df * P.lgd
    lost_int = P.loan * (P.r / 2) * df
    a = attach_abs(P, pay)
    rec = np.maximum(pay - a, 0.0) if a is not None else np.zeros(n)
    return pay, cl + lost_int, rec


def derived(P: Params):
    """Premium, expenses, reinsurance cost and MCR implied by P (per farmer per season)."""
    fair = expected_payout_rate(P) * C
    pay, cl, rec = one_season_losses(P)
    a = attach_abs(P, pay)
    reins_cost = P.reins_mult * rec.mean() if a is not None else 0.0   # reinsurance premium paid by the pool
    reins_net = reins_cost - rec.mean()                                 # net cost of reinsurance (loading)
    expenses = P.expense * fair
    if P.premium is None:
        prem = fair * (1 + P.load) + expenses + reins_net
    else:
        prem = P.premium * C
    net_pay = pay - rec
    mcr_pay = np.quantile(net_pay, 0.995)
    mcr_int = np.quantile(net_pay + cl, 0.995) + expenses + reins_cost
    mcr = mcr_int if P.mcr == "integrated" else mcr_pay
    k0_abs = P.k0 * mcr if P.k0_mcr else P.k0 * C
    coc = max(0.0, P.r_inv - P.r_res) / 2 * k0_abs if P.coc else 0.0   # frictional cost of capital
    if P.premium is None:
        prem = prem + coc
    return dict(fair=fair, premium=prem, coc=coc, k0_abs=k0_abs, expenses=expenses, reins_cost=reins_cost, reins_net=reins_net,
                attach_abs=a if a is not None else np.nan, exp_recovery=rec.mean(),
                mcr_pay=mcr_pay, mcr_int=mcr_int,
                es_pay=net_pay[net_pay >= mcr_pay].mean(),
                es_int=(net_pay + cl)[(net_pay + cl) >= np.quantile(net_pay + cl, 0.995)].mean())


def simulate(P: Params, return_paths=False):
    D = derived(P)
    mcr = D["mcr_int"] if P.mcr == "integrated" else D["mcr_pay"]
    rng = np.random.default_rng(P.seed)
    n = P.paths
    K = np.full(n, D["k0_abs"])
    ruined = np.zeros(n, bool)
    breach = np.zeros(n, bool)
    first_ruin_deficit = np.zeros(n)
    unpaid = np.zeros(n)
    inv_paid = np.zeros(n)
    min_k = K.copy()
    susp = 0.0
    seasonal_net = []
    for _ in range(P.seasons):
        alive = ~ruined
        K = K + D["premium"] - D["expenses"] - D["reins_cost"] + np.maximum(K, 0.0) * P.r_res / 2
        lend = np.minimum(P.loan, np.maximum(0.0, P.beta * (K - mcr)))
        susp += (lend <= 0).mean() / P.seasons
        u = _latent(rng, n, P.zones, P.rho, P.copula, P.nu, P.corr)
        pay = payout_fraction(u, P).mean(1) * C
        a = D["attach_abs"]
        rec = np.maximum(pay - a, 0.0) if not np.isnan(a) else 0.0
        df = default_fraction(rng, n, u < P.p, P).mean(1)
        credit = lend * (P.r / 2) * (1 - df) - lend * df * P.lgd
        K_before_inv = K - pay + rec + credit
        seasonal_net.append(K_before_inv - K)
        newly = alive & (K_before_inv < 0)
        first_ruin_deficit[newly] = -K_before_inv[newly]
        unpaid += np.where(alive, np.maximum(-K_before_inv, 0.0), 0.0)
        ruined |= K_before_inv < 0
        breach |= K_before_inv < mcr
        dist = np.where((K_before_inv >= mcr) & ~ruined,
                        np.minimum(P.r_inv / 2 * D["k0_abs"], K_before_inv - mcr), 0.0)
        inv_paid += dist
        K = np.where(ruined, np.minimum(K_before_inv, 0.0), K_before_inv - dist)
        min_k = np.minimum(min_k, K)
    pr = ruined.mean()
    se = np.sqrt(pr * (1 - pr) / n)
    z = 1.959964
    centre = (pr + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(pr * (1 - pr) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    terminal = np.where(ruined, 0.0, K)  # investor value at horizon (ruined = wiped out)
    inv_value = terminal + inv_paid
    out = dict(ruin=pr, se=se, ci_lo=max(0.0, centre - half), ci_hi=centre + half,
               breach=breach.mean(), lend_suspended=susp,
               deficit_given_ruin=(first_ruin_deficit[ruined].mean() / C) if ruined.any() else 0.0,
               epd=unpaid.mean() / C,  # expected unpaid claims over horizon, share of C
               inv_loss_prob=(inv_value < D["k0_abs"]).mean() if D["k0_abs"] > 0 else np.nan,
               inv_irr_median=_annual_return(inv_value, D["k0_abs"], P) if D["k0_abs"] > 0 else np.nan,
               **{k: v for k, v in D.items()}, mcr_used=mcr)
    if return_paths:
        out["ruined_paths"] = ruined
    return out


def _annual_return(value, k0_abs, P: Params):
    years = P.seasons / 2
    # simple (non-compounded) average annual return of the median path: (terminal + distributions - K0) / K0 / years
    return (np.median(value) / k0_abs - 1) / years


def with_(P: Params, **kw):
    return replace(P, **kw)


def required_capital(P: Params, target=0.01, lo=0.0, hi=3.0, tol=0.01):
    """Smallest initial capital (share of C) with 20-season ruin probability <= target.
    Premium includes the investors' required return on that capital (coc=True)."""
    P = replace(P, k0_mcr=False, coc=True)
    if simulate(replace(P, k0=hi))["ruin"] > target:
        return np.nan
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if simulate(replace(P, k0=mid))["ruin"] <= target:
            hi = mid
        else:
            lo = mid
    return hi
