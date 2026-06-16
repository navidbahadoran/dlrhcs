# Metro-area unemployment panel (BLS LAUS)

**Application 2** of the empirical section: a balanced panel of **383 U.S.
metropolitan statistical areas**, annual-average unemployment rate, **1990–2025
(T = 36)**.  This is the well-conditioned `N >> T` regime (the earlier state
panel had `T/N ≈ 12`, which over-fit the interactive low-rank block).

## Source
U.S. Bureau of Labor Statistics, Local Area Unemployment Statistics (LAUS),
flat files at <https://download.bls.gov/pub/time.series/la/> (public domain).
Download into this folder:

| file | role |
|------|------|
| `la.data.60.Metro.txt` | all metro-area observations (raw, ~41 MB) |
| `la.series`            | `series_id` → area title map |
| `la.area`, `la.measure`| code dictionaries (reference) |

> BLS blocks default download agents; fetch via a browser (as the raw vintage
> here was obtained), or `curl` with a descriptive `User-Agent` header.

## Cleaning (see also the paper's data-cleaning appendix)
Run:

```bash
python data/metro/build_metro_panel.py
```

which writes **`metro_unemployment.csv`** (`YEAR` + one column per metro) by:

1. keeping metropolitan areas (`MT`), not-seasonally-adjusted unemployment-rate
   series (`LAUMT…3`, measure code `03`);
2. using BLS's own **annual average** (period `M13`) — the within-year mean of
   the twelve monthly rates, so the series carry **no seasonal cycle** and need
   no seasonal adjustment from us;
3. keeping the balanced set of metros observed in **every** complete year
   1990–2025.

Per-series stationarization (ADF: difference once only if a unit root, else keep
the level) and standardization are applied downstream in
`dlrhcs.empirical.load_metro`, identically to the Zillow application.

`metro_unemployment.csv` is committed so the panel reproduces even without
re-downloading the 41 MB raw file; the raw `sha256` is printed by the build
script for vintage verification (