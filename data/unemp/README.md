# Metro unemployment panel + CES payroll covariate

## Model-ready file
- `unemployment_metro_model_panel_bls_only_name_matched.csv` — one row per
  metro-month (2000-01..2026-05, 390 MSAs):
  - `unemployment_rate` — BLS LAUS, NSA monthly metro unemployment rate (outcome).
  - `payroll_growth_12m` — BLS CES (State & Area) 12-month payroll-employment growth
    (covariate), matched to each LAUS metro by exact name (`match_method=exact_full`).

## Sources
- BLS LAUS, `la.data.60.Metro` (https://download.bls.gov/pub/time.series/la/).
- BLS CES State & Area employment (https://www.bls.gov/sae/).
The LAUS->CES name-matching build script should be committed alongside this file.

## Transformations (applied in `dlrhcs/unemp.py`)
pivot to metro x month; dedup; linear interpolation of the sparse BLS suppressions
(<= 6 months/metro); LEVEL-PRESERVING NSA deseasonalization (subtract per-metro
month-of-year means, add back the grand mean); winsorize+standardize the payroll
covariate; the covariate enters predetermined (one-month lag).

## Used by
`scripts/unemp_abc.py` (specs A/B/C).
