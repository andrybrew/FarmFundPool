# FarmFund Pool

Smart contracts, data, and risk-analysis code for the FarmFund Pool, a blockchain-based parametric crop insurance design that pools insurance reserves with cooperative seasonal lending for Indonesian smallholders.

This repository supports:

- Lambey, F.R. and Alamsyah, A. (2025), "Parametric crop insurance with FarmFund Pool: a blockchain-based approach for climate resilience in Indonesia agriculture", *2025 International Conference on Data Science and Its Applications (ICoDSA)*, IEEE, pp. 932-937, doi: 10.1109/ICoDSA67155.2025.11157645.
- Alamsyah, A. and Lambey, F.R., "When can cooperative liquidity pools sustain parametric crop insurance? A blockchain-based design and risk analysis for Indonesian smallholders" (manuscript under review).

Imported from the original repository by F.R. Lambey (`rivaldilambey22/ParametricCropInsurance`); the full commit history is preserved.

## Contents

| Folder | What it contains |
|---|---|
| `contracts/` | Solidity contracts of the prototype deployed on the Polygon Amoy testnet (Registration, Insurance Policy, FarmFund Pool, Weather Trigger, Payout, mock stablecoin). Addresses are in `deployed-addresses.md`. |
| `analysis/code/` | Python code for the rainfall calibration and the Monte Carlo risk model (`ffp_model.py`, `experiments.py`), figures, and diagnostics. |
| `analysis/data/` | CHIRPS v2.0 monthly rainfall for ten rice districts (1981 to 2025) and recorded testnet transactions of the deployed contracts. |
| `analysis/results/` | All result tables (CSV/JSON) reported in the article and its supplementary file. |
| `analysis/figures/` | Figures of the article (Fig1 to Fig3) and supplementary file (FigS9 to FigS11). |

## Prototype versus model

The deployed prototype implements a simplified version of the pool: a fixed reserve of 30 percent of total coverage, administrator-approved loans against cooperative collateral, a time-based staking reward, and a trigger evaluated as index value ≥ threshold (correct for an index expressed as a rainfall deficit). The reserve, yield, and lending rules of the article (equations 1 to 5, including the integrated capital floor) are evaluated by simulation in `analysis/` and are the specification for the next contract version.

## Reproducing the risk analysis

Python 3 with numpy, pandas, scipy, and matplotlib. From `analysis/code/`:

```
python rain.py          # seasonal totals from the monthly CHIRPS series
python freq.py          # trigger frequencies, correlations, trends
python experiments.py   # tables T1 to T10 in ../results (seed 20261026; 20,000 paths x 20 seasons)
python t5fix.py         # capital requirement diagnostics (T5)
python extra.py         # ruin with capital held at the MCR (T5b)
python figs.py; python fig2.py; python figS.py   # figures
```

Rainfall source: CHIRPS v2.0 (Funk et al., 2015, *Scientific Data* 2, 150066), area averages retrieved through SERVIR ClimateSERV. District box coordinates are listed in `analysis/README.md`.

## Contact

Andry Alamsyah, School of Economics and Business, Telkom University, Bandung, Indonesia.
