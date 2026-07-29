# Unemployment Implementation Plan

## Principles

- Headline variables should be monthly, frequency-matched to the target, and source seasonally adjusted whenever feasible.
- No annual or quarterly covariate should be interpolated into monthly headline controls.
- Every empirical run should be reproducible from archived inputs, source hashes, metadata, and deterministic run settings.

## Recommended Patch Sequence

1. Add or recover the builder for the derived LAUS/CES file, raw BLS source filenames/URLs, series IDs, seasonal codes, release date/vintage, and exact match crosswalk.
2. Replace the headline unemployment outcome with BLS LAUS smoothed seasonally adjusted metro unemployment if coverage matches the sample.
3. If NSA LAUS is retained, add a deseasonalization audit table that records the adjustment method and all altered/interpolated cells.
4. Remove population/GDP from headline unemployment configs. Keep them sensitivity-only if rebuilt and documented, because they originate from annual data converted to monthly values.
5. Add an audited gap policy: record duplicate rows, dropped units, interpolated cells, leading/trailing fills, and final sample dimensions before estimation.
6. Make output writes deterministic and atomic for `scripts/unemp_abc.py` and `scripts/spatial_kernel_se.py`; add run metadata with code commit, source hashes, tuning, ranks, and source-seasonal status.
7. Add a monthly non-interpolated sensitivity grid: no covariate headline; CES payroll-growth sensitivity only if seasonal status and series IDs are recorded; optional BPS permit-growth sensitivity using the processed monthly CBSA permit file; reject QCEW unless a local reproducible source is added.
8. Add fixture tests for duplicate handling, gap interpolation, deseasonalization, covariate alignment, missing covariate failure, and output metadata.
9. Regenerate unemployment outputs only after the audit checks pass; then update manuscript numbers from immutable output objects.

## Minimum Pre-Production Checks

- `python -m compileall -q .`
- `python -m pytest -q` after installing pytest, or the project's direct test runner if pytest remains absent.
- Check that protected raw/derived inputs are not modified by preflight or reporting.
