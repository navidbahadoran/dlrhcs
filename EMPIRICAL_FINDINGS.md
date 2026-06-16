# Empirical application — status & findings

The empirical pipeline runs a homogeneous panel (one variable across comparable
units) through a single procedure: download raw → clean (balance/align) →
**minimal per-series stationarization** (ADF: difference once only if a unit
root, else keep the level; log-return for positive multiplicative series) →
standardize. No database-supplied or arbitrary transforms. Full cleaning details
are in the paper's data appendix.

## Zillow metro house values — the paper's application

House-price levels are a unit root, so under the rule above nearly every series
becomes a monthly log-return. Headline numbers (stable across tuning settings):

- lag-1 mean **+1.22**, lag-2 mean **−0.40** (momentum then reversion)
- companion spectral radius **0.63** (< 1, stationary); a+b ≈ **0.82**;
  long-run multiplier ≈ **5.6**
- the top-vs-bottom price-tier contrast is estimated within one panel via the
  paper's contrast functional, reported with both White and within-period s.e.

This is the large-N, large-T regime where the theory predicts near-benchmark
precision and the estimator delivers it. Clean and supportive.

## Metro unemployment — built and parked (not in the default run)

A second application (BLS LAUS metro-area unemployment, 383 metros, 1990–2025) is
fully implemented (`dlrhcs.empirical.load_metro`, `data/metro/`) but is **not**
wired into `run_all.py`. Diagnostics showed it is strongly common-factor-driven
(one factor ≈ 76% of variance), and economically stationary-but-persistent in
levels (idiosyncratic AR(1) ≈ 0.55 after removing common factors) while ADF
fails to reject a unit root at T = 36. In levels it gives a near-unit-root
persistence estimate that needs care; differenced it is over-differenced into
near-white-noise. It is kept available for later use with that caveat documented
in `data/metro/README.md`.

## Why earlier candidates were dropped

- **FRED-QD** is a heterogeneous bag of different macro aggregates: it needs
  *mixed* per-series transforms (so a pooled mean blends different quantities),
  the idiosyncratic AR(1) after removing common factors is ≈ 0.08 (almost no
  signal), and the common block is weak. Wrong *shape* for a model built to
  estimate heterogeneous idiosyncratic dynamics.
- **State unemployment** (51 units, monthly since 1976) has T ≫ N (T/N ≈ 12),
  which over-fits the interactive low-rank block; the metro panel was the attempt
  to fix the shape, and is parked as above.

The shipped empirical section is Zillow only.
