# Housing All-Homes Production Report

## 1. Data and Sample
| field | value |
| --- | --- |
| Sample start | 2010-01-01 |
| Sample end | 2026-05-01 |
| Number of MSAs | 169 |
| Number of level months | 197 |
| Usable dynamic observations | 33124 |
| Number of folds | 10 |
| Fixed ranks | [1, 1, 1, 2] |
| Outcome transformation | log(zhvi_all_homes_sa) |
| Control transformations | lag_asinh_permits, lag_log_employment |
| Lag order | one monthly lag |
| BLS preliminary observations | 0 (validated by production run) |
| Seed | 2024 |
| Riesz maximum iterations | 2000 |
| Riesz tolerance | 1e-05 |
| Riesz ridge | 1e-06 |
| Cached-scale setting | False |
| Production runtime | 177.215 |

## 2. Estimation Specification

The report summarizes the completed full production run. The target definitions are verified against `housing_targets()` and `build_blocks()`: the reported targets are full-sample mean coefficients on lagged log ZHVI, lagged asinh building permits, and lagged log payroll employment. All controls enter with one monthly lag in the prepared panel specification.

## 3. Main Estimates
| estimand | estimate | white_se | white_ci_lower | white_ci_upper | xs_se | xs_ci_lower | xs_ci_upper | plugin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mean coefficient on lagged log ZHVI | 0.93818 | 0.00113906 | 0.935947 | 0.940412 | 0.00518689 | 0.928013 | 0.948346 | 0.889101 |
| Mean coefficient on lagged asinh building permits | 0.0584903 | 0.00107511 | 0.0563831 | 0.0605974 | 0.0048989 | 0.0488886 | 0.0680919 | 0.104958 |
| Mean coefficient on lagged log payroll employment | -5.02535e-06 | 3.01896e-05 | -6.41959e-05 | 5.41452e-05 | 3.01896e-05 | -6.41959e-05 | 5.41452e-05 | 9.20078e-08 |

## 4. Numerical Diagnostics
| diagnostic | value |
| --- | --- |
| Number of targets | 3 |
| Number of folds | 10 |
| Total Riesz solves | 30 |
| Convergence fraction | 1 |
| Mean iterations | 948 |
| Median iterations | 948.5 |
| Maximum iterations | 1021 |
| Number reaching maximum iterations without convergence | 0 |
| Mean relative residual | 8.27307e-06 |
| Median relative residual | 8.49689e-06 |
| Maximum relative residual | 9.9065e-06 |
| Number containing nonfinite values | 0 |
| production_valid | True |
| validation failures | none |

## 5. Reproducibility Information

- Report schema version: `housing_all_homes_report_v1`
- Production run signature hash: `cbf61cd011875adb1fa83f1e9a5781c68f1cbec0b4eb09e64c9a1f440b411f2b`
- Git commit: `4418612730c5b66403eded376584fa6f3706507f`
- Git dirty: `True`

## 6. Manuscript-Ready Factual Summary

The completed housing all-homes production run reports dynamic panel estimates of mean coefficients. The lagged log-ZHVI coefficient is 0.93818, indicating strong persistence in the monthly log home-value process. The permits coefficient is positive, with estimate 0.0584903. The payroll-employment coefficient is close to zero at the reported scale, with estimate -5.02535e-06. Statistical precision differs between the White standard errors and the cross-sectional standard errors reported in the production output. These statements describe estimated mean coefficients and do not assign a causal interpretation beyond the maintained identification assumptions of the existing estimator.
