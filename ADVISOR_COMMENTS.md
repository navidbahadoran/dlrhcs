# Advisor comments — master checklist (simulation + empirical)

Merged and deduplicated from `Simulation.txt` and `New Text Document.txt`.
Priority: **P1** critical/changes results · **P2** important clarity/reporting · **P3** polish.
Type: **Text** (no rerun) · **Code+Run** (I build, you run) · **Decision** (judgment call) · **Diagnose**.

---

## 1. DGP: two labeled data-generating processes  *(P1)*
- [ ] **DGP 1** — independent heteroskedastic: `u_it ~ N(0, σ_i²)`, `σ_i² ~ U(0.5,1.5)`. *(Code+Run)*
- [ ] **DGP 2** — weakly cross-sectionally dependent **and** heteroskedastic (spatial). *(Code+Run)*
- [ ] Label them **DGP 1, DGP 2** throughout. *(Text)*
- [ ] Errors must be heteroskedastic; spatial dependence must be a considered case. *(Code+Run)*

## 2. DGP: full written description in the MAIN paper  *(P1)*
- [ ] Explain the whole DGP in words so a reader can replicate it. *(Text)*
- [ ] Detail how `A_0`, `B_0`, `H_0` parameters are generated (not just "exact rank-one … order √(TN)"). *(Text)*
- [ ] Define **ρ_y**; explain the `max|a_ti| ≤ 0.92 ρ_y` rescaling. *(Text)*
- [ ] Outline every component of `x_it = c_x f_{x,t} λ_{x,i} + σ_x e_it` (how `c_x`, `f_{x,t}`, `λ_{x,i}`, `σ_x`, `e_it` are generated). *(Text)*
- [ ] Define "residual identifying variation." *(Text)*
- [ ] Delete **Table 2** (`tab:sim_design`); move the settings into prose. *(Text)*

## 3. Simulation runs  *(P1)*
- [ ] **R ≥ 1000 for every cell** (currently 500 at 400). *(Code+Run)*
- [ ] **Full T×N grid** (all combinations, not just the diagonal). *(Code+Run)*
- [ ] Run both DGPs. *(Code+Run)*

## 4. Simulation reporting  *(P1–P2)*
- [ ] Report **true values** of every target in the MC table. *(P1, Text)*
- [ ] Report **empirical size** in addition to coverage. *(P1, Code+Run)*
- [ ] Report **mean + standard deviation** of the coefficients. *(P2, Code+Run)*
- [ ] **Rank-correct-probability table** in the main paper, by sample size. *(P1, Text — data exists)*
- [ ] Report the **full** convergence study (Tables 9 & 10), not a compact subset. *(P2, Text)*
- [ ] Define "**pc diagnostic coverage**" (period-cluster). *(P2, Text)*
- [ ] Terminology: **autoregression coefficient `a_it`** and **slope coefficient `b_it`** (not "lag coefficient"). *(P2, Text)*
- [ ] Remove the **"+" sign** on all positive numbers (sim + empirical). *(P3, Text)*

## 5. Simulation: cluster vs spatial s.e.  *(P1, Decision)*
- Conflict in the comments: one asks to **remove** the cluster s.e. from the sim; another asks to **report both** spatial-kernel and cluster (since both dependence forms are considered).
- [ ] **Decide**: with a metric-bearing DGP, report spatial-kernel as primary; report cluster only if a cluster DGP is included — and justify it. *(Decision → Text/Code)*

## 6. Simulation anomalies to diagnose  *(P1, Diagnose)*
- [ ] Table 15: coverage ≈ 0.70 even at the **correct** rank — check. *(Diagnose)*
- [ ] Table 16: coverage **worsens** N=200→300 — likely R=300 Monte-Carlo noise (fixed by R≥1000). *(Diagnose)*

## 7. Fixed J vs J_TN → ∞  *(P1, Decision/theory)*
- [ ] "Major concern": check whether the theory can support a **fixed** `J_TN`. *(Decision)*
- [ ] Consider finer `J` (e.g., 5, 10) in the sensitivity. *(Code+Run)*

## 8. Optimization / numerical stability  *(P2)*
- [ ] "Removes the occasional bad stationary point" — report **how often** this occurs. *(Diagnose+Text)*
- [ ] Explain optimization convergence and the local-extremum risk (sim). *(Text)*
- [ ] Report **max companion radius** (instability) and numerical-instability diagnostics — **both** application and simulation. *(Code+Run+Text)*

## 9. Do not cross-reference internal results in the sim section  *(P2, Text)*
- [ ] Remove all lemma / proposition / assumption references from the **simulation** section. *(Text)*

## 10. Empirical: data construction  *(P1, Decision)*
- [ ] **Seasonal adjustment** for **both** housing and unemployment. *(Decision/Code)*
- [ ] **No interpolation** of any series. *(Decision/Code)*
- [ ] Do **not** standardize `x_it` in the empirical application. *(Decision/Code)*
- [ ] **Reconsider GDP** as a housing covariate. *(Decision)*
- [ ] Detailed **data-download instructions** for housing, unemployment, and all covariates. *(Text)*
- [ ] Explain the **window size `q`** choice for `i` and `T` (sim + both applications). *(Text)*
- [ ] Is `q_i` the same for all `i`? If so, why? (advisor: doesn't make sense to be identical). *(Decision/Text)*
- [ ] Housing **tiers**: how defined, and why use both? *(Text)*
- [ ] Why only **390 of 610** metro-tier units match coordinates? *(Text/Code — improve match)*
- [ ] Report **data outliers**. *(Code+Text)*
- [ ] How is the **spatial standard error** computed? (make explicit). *(Text)*

## 11. Empirical: "problematic" results to address  *(P1, Decision)*
- [ ] COVID-excluded persistence **0.988** (near unit root) — flagged "problematic." *(Decision)*
- [ ] Housing covariates moving cumulative persistence **0.831 → 0.801** with insignificant coefficients — flagged "problematic." *(Decision)*

## 12. Empirical: writing  *(P2–P3, Text)*
- [ ] Do **not** say empirical settings were "taken similar to the Monte Carlo." *(Text)*
- [ ] **Summarize** the over-long Table 17 note. *(Text)*
- [ ] Remove "+" signs on positive empirical numbers. *(P3, Text)*

---

### Suggested phase order
1. **Phase A (Text, now):** §2 wording, §4 terminology/definitions/±signs, §9 remove refs, §12 writing, tier/window/spatial-se explanations, delete Table 2.
2. **Phase B (Code+Run):** §1 two DGPs, §3 grid/R≥1000, §4 size+coverage/mean+SD/rank-prob, §6 diagnose, §8 instability. → you run → I wire tables + rewrite the sim section.
3. **Phase C (Decisions):** §5 s.e. choice, §7 fixed-J, §10 data (seasonal/interpolation/standardize/GDP), §11 the two "problematic" results.
