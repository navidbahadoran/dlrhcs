# Zillow house-price data + covariates

## Model-ready files
- `zillow_metro_top.csv`, `zillow_metro_bottom.csv` — Zillow Home Value Index (ZHVI),
  top- and bottom-tier metro series (monthly, 2000-). Source: Zillow Research,
  https://www.zillow.com/research/data/ . Outcome = monthly log price growth.
- `metro_monthly_covariates_2000_present.csv` — metro (CBSA) monthly covariates:
  building permits, population, and real GDP, with 12-month Delta-log growth rates.
- `cbsa_county_crosswalk_2023.csv` — Census 2023 county->CBSA delineation.

## How the covariates are built (`zillow-covariate.py`)
1. Download monthly metro building permits (Census Building Permits Survey).
2. Download county GDP and population (BEA), aggregate counties -> CBSA via the
   2023 crosswalk.
3. Convert annual GDP/population to monthly (linear interpolation).
4. Compute 12-month Delta-log growth for each covariate.
Run: `pip install pandas requests openpyxl`; `export BEA_API_KEY=...`; `python zillow-covariate.py`.

## Used by
`scripts/zillow_abc.py` (specs A/B/C).  Metro->CBSA matching and covariate
winsorization/standardization are in `dlrhcs/covariates.py`.
