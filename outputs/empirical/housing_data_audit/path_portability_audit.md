# Housing Path Portability Audit

Resolved repository root for this audit: `.`

No literal dependency on either laptop-specific repository root was found in production housing code after this portability pass.

| file | line/function | current behavior | depends on cwd | machine-specific absolute path | enters resume signature | informational metadata only | required correction |
|---|---|---|---|---|---|---|---|
| scripts/audit_zillow_data.py | main | resolves --data-root and --bls-local-dir | no | no | no | no | uses dlrhcs.paths; relative paths are repo-relative |
| scripts/report_housing_data.py | main | resolves --data-root and --output-root | no | no | no | no | uses dlrhcs.paths; relative paths are repo-relative |
| scripts/housing_all_homes.py | main/preflight/run_estimation | resolves panel, output, config, and candidate paths | no | no | yes | yes | resume signatures use checksums and repo-relative identities, not absolute roots |
| scripts/zillow_abc.py | module constants | legacy top/bottom runner derives ROOT from script location | no when launched as script | no | not applicable | some output metadata | preserved for reproducibility; no machine-specific literal path found |
| scripts/build_metro_panel.py | module constants | legacy metro script derives ROOT from script location | no when launched as script | no | not applicable | no | preserved for reproducibility; no machine-specific literal path found |
| dlrhcs/housing_data.py | run_housing_audit/X-13 helpers | accepts data_root from caller; subprocess X-13 receives explicit cwd | caller-controlled | no | no | source manifests only | caller now supplies resolved repo-relative paths from scripts |
| dlrhcs/empirical.py | load_zillow/run_ar2 | legacy loaders consume explicit paths supplied by scripts | caller-controlled | no | no | data fingerprint only | preserved |
| dlrhcs/covariates.py | load_zillow_covariates/load_cbsa_covariates | legacy covariate loaders consume explicit paths | caller-controlled | no | no | no | preserved |
| tests/test_core.py | test bootstrap | adds repository root to sys.path from test file location | no | no | no | no | test-only bootstrap |
| data/zillow/processed/candidate_panels* | metadata.json | candidate metadata keyed by candidate_id and checksums | no | no | no | yes | candidate path is recoverable repo-relative; no root name required |
| data/zillow/processed/estimation_panels* | metadata.json | input metadata records repo-relative and resolved paths plus checksums | no | no | no | yes | substantive identity uses checksum/schema/dimensions/date range/specification |

Path rules: user-supplied relative paths are interpreted relative to the resolved repository root; explicit absolute paths are used as supplied; resume-signature identity excludes absolute repository roots.
