# Housing Paper Output Report

1. Zillow ZHVI and Census permit NSA begin in 2000; BLS official SA employment begins in 1990; matched current-CBSA permit SA begins in 2004.
2. Permit SA begins in 2004 for accepted current metropolitan CBSAs because 2000--2003 permit rows use legacy PMSA/MSA codes and X-13 segments for those codes are below the 84-month minimum.
3. Balanced panel beginning in 2000 feasible: 0 MSAs.
4. Largest 2000-start N/T: 0 by 318.
5. Largest 2004-start panel: 98 MSAs.
6. Largest 2005-start panel: 108 MSAs.
7. Largest 2010-start panel: 169 MSAs.
8. Non-dominated candidates: 54 rows in housing_pareto_frontier.csv.
9. Candidate maximizing N: pareto_54_201904_87m_196n.
10. Candidate maximizing T: longest_feasible_common_window.
11. Candidate maximizing NT: pareto_33_200903_208m_163n.
12. Candidate maximizing NT/(N+T): pareto_33_200903_208m_163n.
13. Preliminary BLS observations by candidate are recorded in candidate metadata/source tables and data-quality flags.
14. Latest month with all matched official-SA BLS employment observations final: 2026-05-01.
15. Final-only non-dominated candidates: 54 rows in housing_pareto_frontier_final_only.csv.
16. Negative seasonally adjusted permit values: 605 observations across 79 CBSAs.
17. X-13 problematic segments are not in the current all-three matched current-CBSA candidate panels unless flagged in the X-13 warning columns.
18. Candidate completeness checks require exactly N*T rows and no missing primary variables.
19. Candidate-ranking highlights are diagnostic only: largest N with T >= 180 -> all_vintage:at_least_180m; maximum T -> all_vintage:longest_feasible_common_window; longest panel with N >= 100 -> all_vintage:pareto_03_200408_263m_102n; maximum NT; maximum NT/(N+T) -> all_vintage:pareto_33_200903_208m_163n; closest N/T ratio to one -> all_vintage:pareto_44_201201_174m_175n; maximum N -> all_vintage:pareto_54_201904_87m_196n.
20. Tables and figures under outputs/empirical/housing_data_audit are ready for manuscript review but are not copied into manuscript tables.
21. No interpolation, imputation, winsorization, standardization, forecasting, or backcasting was performed.
