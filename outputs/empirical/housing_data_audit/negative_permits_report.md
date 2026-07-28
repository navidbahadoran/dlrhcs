# Negative Seasonally Adjusted Permit Values

This audit relabels the diagnostic as `negative_seasonally_adjusted_permit_value`. These are not invalid raw permit observations: additive X-13 seasonal adjustment can produce negative adjusted values.

## Summary
- total count: 605
- affected CBSAs: 79
- first affected month: 2004-02-01
- last affected month: 2026-06-01
- minimum: -90.125404575521
- p1: -30.12473909679566
- p5: -10.474011989671379
- median among negative values: -0.646113915307396
- maximum negative value: -0.000416289010444802
- X-13 warning or failed segment count: 0

## X-13 Specification
Local X-13 run with transform { function = none } and x11 { mode = add save = (d11) }.

Observation-level diagnostics, including corresponding NSA permit values and affected candidate panels, are written to `negative_permits_diagnostics.csv`.

No deletion, truncation, winsorization, replacement, or transformation of these values was performed during this audit.
