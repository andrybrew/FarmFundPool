# FarmFund Pool risk analysis (JRF major revision, R1)

Reproduces every number, table, and figure in Sections 3.1 and 5.2 to 5.6 of the revised manuscript and Section S6 of Supplementary File S1.

## Data
- `data/chirps_monthly_raw.txt`: CHIRPS v2.0 monthly precipitation (mm), Jan 1981 to Jun 2025, area averages for ten districts, retrieved from SERVIR ClimateSERV (datatype 0, daily average aggregated to months). Box centers (lat, lon, half-width in degrees): Bandung (-6.92, 107.61, 0.05), Indramayu (-6.40, 108.20, 0.1), Karawang (-6.25, 107.40, 0.1), Subang (-6.40, 107.75, 0.1), Cirebon (-6.75, 108.50, 0.1), Sragen (-7.43, 111.02, 0.1), Ngawi (-7.40, 111.45, 0.1), Lamongan (-7.12, 112.41, 0.1), Sidenreng Rappang (-3.90, 119.80, 0.1), Central Lampung (-4.95, 105.25, 0.1).
- `data/chirps_JFM.csv`: January to March totals (built by `code/rain.py`).
- `data/onchain_gas.csv`: recorded transactions of the deployed contracts on Polygon Amoy (Polygonscan, retrieved 26 Oct 2026).

## Code (Python 3, numpy, pandas, scipy, matplotlib)
1. `rain.py`, `freq.py`: seasonal totals, trigger frequencies, correlations, trends.
2. `ffp_model.py`: the model (equations 1, 2, and 4 of the manuscript), simulation, required-capital solver.
3. `experiments.py` (constants in `experiments_cfg.py`): Tables T1 to T10 in `results/`. Seed 20261026, 20,000 paths, 20 seasons.
4. `t5fix.py`, `extra.py`: MCR diagnostics (T5) and ruin with capital at the MCR (T5b).
5. `fig2.py`, `figs.py`, `figS.py`: figures.

Run from `code/` with `results/` and `data/` as sibling folders (paths are relative: `../data`, `../results`).

Note: one cell of T3 (mild trigger, single province, reserves earning 6%) needs the bisection bound raised to 6.0; its value in the CSV was computed with `required_capital(..., hi=6.0)`.
