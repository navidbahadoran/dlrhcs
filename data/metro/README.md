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
script for vintage verification (full file ≈ 41,262,124 bytes, sha256[:16] =
`4d6b3e0d2c9031bc`).

> Note: a **partial** `la.data.60.Metro.txt` (~10 MB) may be present from an
> interrupted copy — delete it and re-download the full file before running the
> build script. The committed `metro_unemployment.csv` is already correct.

## Status
Built and validated but **not** wired into the default `run_all.py` empirical
stage — the paper currently ships the Zillow application only. To enable later,
add an Application-2 block in `run_all.py::stage_empirical` calling
`load_metro("data/metro/metro_unemployment.csv")` + `metro_groups`, mirroring the
Zillow block. Note the metro series are factor-dominated (one common factor ≈
76% of variance) and economically stationary-but-persistent in levels; treat the
rate as stationary (keep levels) rather than ADF-differencing, and expect a
near-unit-root persistence estimate.
