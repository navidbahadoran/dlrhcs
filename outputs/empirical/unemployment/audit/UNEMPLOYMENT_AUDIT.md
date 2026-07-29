# Unemployment Empirical Application Audit

## Bottom Line

The current repository documents much of the unemployment empirical code path, but it does not fully reproduce the manuscript unemployment application from immutable local artifacts. The derived monthly LAUS/CES unemployment file exists, but the raw BLS builder and raw BLS inputs are absent. The monthly population/GDP covariate file expected by `scripts/unemp_abc.py` is absent. The legacy unemployment output JSON files referenced by the manuscript are absent. The current loader also deseasonalizes NSA unemployment in code and interpolates sparse unemployment/payroll gaps.

Under a headline rule requiring monthly, frequency-matched, non-interpolated variables, the current population/GDP unemployment covariate specification should be treated as sensitivity-only, not headline.

## Local Data Facts

- Derived panel: `data/unemp/unemployment_metro_model_panel_bls_only_name_matched.csv`.
- Rows: 124,191.
- Metro/CBSA codes: 390.
- Months: 317, from 2000-01 to 2026-05.
- Duplicate `(cbsa_code, date)` keys: 948, with at least 948 extra duplicate rows.
- Missing values: unemployment rate 419; payroll employment 0; payroll growth 1-month 393; payroll growth 12-month 4,716; payroll change 12-month 4,716.
- Match method: all rows are `exact_full`.

## Code Path Map

1. `scripts/unemp_abc.py` loads the derived unemployment panel using `dlrhcs.unemp.load_unemp_panel` and runs specs A-D through `dlrhcs.empirical.run_ar2`.
2. `dlrhcs.unemp.load_unemp_panel` pivots the panel, silently resolves duplicate CBSA-month rows by last write, keeps units with total missing count no larger than `MAX_GAP=6`, interpolates/backfills/forward-fills kept gaps, deseasonalizes NSA unemployment with metro-specific month-of-year means, and winsorizes/standardizes payroll growth.
3. `dlrhcs.empirical.build_ar2` constructs the effective sample as `Ymat[2:]`, lag 1 as `Ymat[1:-1]`, lag 2 as `Ymat[0:-2]`, and covariates as one-month-lagged predetermined variables.
4. `scripts/spatial_kernel_se.py` can recompute geographic spatial-kernel SEs, but it expects the missing monthly population/GDP covariate file and writes an output JSON that is currently absent.

## Transformations Affecting Interpretation

- Interpolation: `_interp` fills leading, internal, and trailing missing values for kept units. The keep rule counts total missing observations, not maximum consecutive missing gap length.
- Deseasonalization: the local README and code identify the unemployment rate as NSA. Seasonal adjustment is performed in code by subtracting metro-specific month-of-year means and adding back the grand mean.
- Standardization: payroll covariates are winsorized at the 1st/99th percentiles and standardized over finite cells even when `require_cov=False`.
- Dynamic sample: even nominal rank vector `(1,0,1)` is estimated through the AR(2) wrapper and loses two initial months.

## Manuscript Consistency Risks

- The manuscript reports headline unemployment `N=315`, `T=240`, 2005-2024. The current output JSON files supporting that number are absent. The local loader alone keeps 388 units for 2005-2024 before missing covariate matching.
- Spec C depends on `data/zillow/metro_monthly_covariates_2000_present.csv`, which is absent. The available builder converts annual population/GDP to monthly values.
- The repository contains a CES payroll covariate in the unemployment panel, while manuscript text says payroll employment is avoided. This can be reconciled only by separating archived data contents from headline controls.
- Spatial-kernel unemployment SE scripts exist, but the corresponding output file is absent.

## Collinearity Audit Summary

For the feasible local monthly pair CES payroll growth and BPS permit growth, complete-case correlations are small. In 2005-2024, the two-way-demeaned correlation is -0.0021 and the VIF is 1.0012. In 2010-2024, the two-way-demeaned correlation is -0.0033 and the VIF is 1.0011. QCEW wage data are not available locally. The legacy population/GDP monthly file is absent.

## Recommended Headline Data Rule

Use source-monthly, source-seasonally-adjusted metro unemployment whenever coverage matches the intended sample. If the official SA metro LAUS file cannot be matched, use NSA LAUS only with a transparent, audited deseasonalization and an explicit list of altered/interpolated cells. Do not include annual population/GDP controls in headline unemployment specifications.

## Deliverables

- `UNEMPLOYMENT_FILE_INVENTORY.csv`
- `UNEMPLOYMENT_SOURCE_INVENTORY.csv`
- `UNEMPLOYMENT_CLAIM_EVIDENCE.csv`
- `UNEMPLOYMENT_COLLINEARITY_AUDIT.csv`
- `UNEMPLOYMENT_IMPLEMENTATION_PLAN.md`
- `audit_manifest.json`
