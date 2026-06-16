# Downloading Zillow ZHVI metro panel, price tiers
- Page: https://www.zillow.com/research/data/  (section "Home Values")
- Choose Data Type = "ZHVI ... by tier", Geography = "Metro & U.S.", and
  download BOTH the bottom-tier and top-tier cuts. The files are named like:
    bottom:  Metro_zhvi_uc_sfrcondo_tier_0.0_0.33_sm_sa_month.csv
    top:     Metro_zhvi_uc_sfrcondo_tier_0.67_1.0_sm_sa_month.csv
- Save as:  data/zillow_metro_bottom.csv  and  data/zillow_metro_top.csv
- The loader stacks each metro's top- and bottom-tier series as two units in one
  panel (outcome = monthly log price growth), with a tier indicator, so the
  top-vs-bottom price-tier contrast uses the paper's own contrast estimator.
  License: Zillow, citation required.
