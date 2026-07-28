# Existing Housing Code Audit

Generated: 2026-07-28T16:31:17+00:00

## scripts/zillow_abc.py
- **current purpose:** Runs legacy housing A/B/C/D empirical specifications.
- **data inputs:** data/zillow/zillow_metro_top.csv; data/zillow/zillow_metro_bottom.csv; metro_monthly_covariates_2000_present.csv; cbsa_county_crosswalk_2023.csv
- **data outputs:** outputs/empirical/zillow_*.json
- **Zillow series used:** top-tier and bottom-tier ZHVI stacked as separate units
- **covariates used:** permits, population, GDP in spec C
- **whether interpolation occurs:** not in this script; upstream covariate file documents population interpolation and GDP monthly carry/lagging
- **whether seasonal adjustment occurs:** no local adjustment here; relies on Zillow tier files
- **whether winsorization occurs:** not in this script; covariate loader winsorizes
- **whether standardization occurs:** yes via load_zillow and covariate loader
- **whether the code can be retained:** yes, preserve for old top/bottom reproducibility
- **required revision:** do not change for this audit; future all-homes spec should call a separate loader

## dlrhcs/empirical.py
- **current purpose:** Legacy empirical loaders, AR(2) construction, targets, and run_ar2.
- **data inputs:** Zillow tier CSVs; metro unemployment CSVs
- **data outputs:** in-memory model panels and result dictionaries
- **Zillow series used:** top/bottom tiers in load_zillow
- **covariates used:** optional predetermined covariates from caller
- **whether interpolation occurs:** no direct interpolation
- **whether seasonal adjustment occurs:** no
- **whether winsorization occurs:** no direct winsorization
- **whether standardization occurs:** yes, load_zillow standardizes transformed series
- **whether the code can be retained:** yes
- **required revision:** add future all-homes model loader only after data audit approval

## dlrhcs/covariates.py
- **current purpose:** Legacy covariate matching and transformation for Zillow/unemployment specs.
- **data inputs:** metro_monthly_covariates_2000_present.csv; cbsa_county_crosswalk_2023.csv
- **data outputs:** covariate matrices aligned to panel units
- **Zillow series used:** region names from legacy Zillow tier loader
- **covariates used:** permits_units_growth_12m, population_growth_12m, real_gdp_growth_1y
- **whether interpolation occurs:** not directly; docstring states upstream annual-to-monthly interpolation for GDP/population
- **whether seasonal adjustment occurs:** no
- **whether winsorization occurs:** yes, 1st/99th percentile clipping
- **whether standardization occurs:** yes, z-scoring
- **whether the code can be retained:** yes for legacy outputs only
- **required revision:** future headline all-homes specification should not use this three-covariate loader

## data/zillow/zillow-covariate.py
- **current purpose:** Legacy upstream covariate builder.
- **data inputs:** Census BPS, BEA county GDP/population, Census CBSA delineation
- **data outputs:** metro_monthly_covariates_2000_present.csv and intermediates
- **Zillow series used:** none directly
- **covariates used:** permits, population, GDP
- **whether interpolation occurs:** yes for annual population; GDP is repeated/lagged into months
- **whether seasonal adjustment occurs:** no local X-13 adjustment
- **whether winsorization occurs:** no
- **whether standardization occurs:** no
- **whether the code can be retained:** yes as legacy provenance, but not for new headline data
- **required revision:** replace with no-interpolation monthly-only pipeline

## scripts/build_metro_panel.py
- **current purpose:** Builds legacy annual unemployment panel, not housing.
- **data inputs:** BLS LAUS raw files under data/metro
- **data outputs:** data/metro/metro_unemployment.csv
- **Zillow series used:** none
- **covariates used:** none
- **whether interpolation occurs:** no
- **whether seasonal adjustment occurs:** uses BLS annual average, no monthly seasonal adjustment
- **whether winsorization occurs:** no
- **whether standardization occurs:** downstream only
- **whether the code can be retained:** yes; unrelated to new housing acquisition
- **required revision:** none for this task

## tests/test_core.py
- **current purpose:** Core unit tests for simulation/estimation/reporting helpers.
- **data inputs:** synthetic fixtures only
- **data outputs:** none
- **Zillow series used:** none before this task
- **covariates used:** none
- **whether interpolation occurs:** no
- **whether seasonal adjustment occurs:** no
- **whether winsorization occurs:** no
- **whether standardization occurs:** no
- **whether the code can be retained:** yes
- **required revision:** add fixture tests for housing audit helpers
