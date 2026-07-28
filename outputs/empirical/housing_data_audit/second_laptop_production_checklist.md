# Second-Laptop Housing Production Checklist

Run these commands from Git Bash on the second laptop. They assume the repository is at `/d/Programming/dlrhcs`.

```bash
cd /d/Programming/dlrhcs
git pull
git status --short data/zillow
test -f data/zillow/processed/candidate_panels_final_only/start_2010/housing_panel_levels.csv
test -f data/zillow/processed/estimation_panels/housing_baseline_2010_final/housing_estimation_panel.csv
python scripts/housing_all_homes.py --preflight --repo-root . --panel-id start_2010
python scripts/housing_all_homes.py --smoke --repo-root . --config configs/full.json --seed 2024 --n-jobs 1
# Eventual production, after preflight and smoke pass:
python scripts/housing_all_homes.py --repo-root . --config configs/full.json --seed 2024 --n-jobs 1
```

Required data are the validated files under `data/zillow/processed`, including `candidate_panels_final_only/start_2010` and `estimation_panels/housing_baseline_2010_final`. If these files are absent from Git on the second laptop, copy them from this machine or regenerate them from already-downloaded local data only after confirming the audit inputs are present. Do not rerun production estimation until `production_preflight.json` reports `ready_for_production: true`.
