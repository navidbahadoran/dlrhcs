# Implementation Spec — `dlrhcs_sim` Replication Package

**Paper:** Cross-Fitted Debiased Inference for Dynamic Panels with Low-Rank Heterogeneous Coefficients.
**Purpose of this document:** a complete, self-contained blueprint of the estimator, the cross-fitting scheme, the debiasing/studentization, the rank selection, the Monte Carlo DGP, and the two empirical applications — precise enough that an agent (Claude Code / Cowork) can build a runnable, reproducible package without re-deriving anything from the manuscript. Every block cross-references the paper's equation labels so each piece can be checked against the source.

---

## 0. Read this first — research-integrity ground rule

The numbers currently printed in the paper's tables/figures (`tab:sim_convergence`, `tab:sim_precision`, `tab:sim_purge`, `fig:purge_sensitivity`, `tab:emp_fredqd`, `tab:emp_zillow`) are **author-supplied targets, not code output**. This package GENERATES the numbers; it does not reverse-engineer them.

The workflow is therefore:
1. Implement the estimator and DGP correctly.
2. Run it.
3. **Replace** the paper's reported numbers with whatever the code actually produces.

Never tune the code to hit a pre-written number. The journal's Data Editor reruns this package on a clean machine; any mismatch between deposited code output and printed tables is the failure mode to avoid. If the real output differs from the paper's claims (e.g. coverage off nominal, empirical persistence different), that is a finding to report honestly, not a bug to suppress.

---

## 1. Notation and core objects

| Symbol | Meaning |
|---|---|
| `T`, `N` | time, cross-section sizes (raw) |
| `P`, `K` | lag order, number of exogenous regressors |
| `M = P + K` | number of coefficient blocks (one per lag, one per regressor) |
| `Tp = T - P` | effective time length (`\Tplus`) |
| `I = {P+1..T} × {1..N}` | effective sample, `Tp·N` cells `a=(t,i)` |
| `Γ^(m)` | `Tp×N` coefficient surface for block `m` (lag-loading `A^(ℓ)` or slope `B^(k)`) |
| `H` | `Tp×N` interactive nuisance surface |
| `Θ = (Γ^(1..M), H)` | full parameter (a tuple of surfaces) |
| `Z^(m)` | `Tp×N` "design" matrix for block `m`: the **lagged outcome** `y_{i,t-ℓ}` for a lag block, or the **regressor** `x_{it,k}` for a slope block |
| `r_m, r_H` | ranks of the blocks |
| `J` | number of cross-fitting folds |
| `q` | forward exclusion window length |
| `ℓ_TN = sqrt(log(Tp·N))` | localization factor (appears only in tuning constants) |

**Design map** `A` (`eq:operator_model`, `subsec:operator_form`). For a parameter `Θ`,
```
A(Θ)_{ti} = Σ_{m=1..M} Z^(m)_{ti} · Γ^(m)_{ti} + H_{ti}        (Hadamard / cell-wise product)
```
This is the fitted outcome. `A` is **linear in Θ** given the `Z^(m)`. Its adjoint `A*` maps a residual matrix `R` (`Tp×N`) to the tuple of surfaces `(Z^(m) ⊙ R)_m` and `R` for the `H` block. Implement `A` and `A*` as the two primitives everything else calls.

> **Key structural point** (`subsec:model_objects`, warm-start discussion): the object that is low-rank in *raw outcome* space is **not** any single surface — the surfaces enter multiplied by observed `Z^(m)`. Low-rank structure lives in each `Γ^(m)` and `H` individually, recovered only after a linear pass that removes the Hadamard weighting. Do not SVD the outcome matrix `Y`.

**Companion form** (`eq:companion_matrix`). For a lag vector `a=(a_1..a_P)`, the companion matrix `C(a)` is the standard `P×P` companion (first row `a`, sub-diagonal identity). With `P=1`, `C(a)=a`.

---

## 2. Suggested module layout

```
dlrhcs/
  design.py        # A, A_adjoint, build Z^(m) from data (incl. lagged outcomes)
  factorridge.py   # alternating factor-ridge ALS + warm start  (§4, §5)
  folds.py         # scattered folds + forward exclusion window  (§6)
  ranks.py         # cross-fitted rank criterion + data-driven roadmap  (§7)
  targets.py       # target directions D_nu; tangent space; Riesz weights  (§8)
  onestep.py       # one-step debiased estimator + variance estimators  (§9,§10)
  pipeline.py      # full feasible procedure end-to-end  (§11)
  dgp.py           # Monte Carlo data-generating process  (§12)
  mc.py            # Monte Carlo harness + validation checkpoint  (§12)
  empirical.py     # FRED-QD and Zillow applications  (§13)
configs/           # YAML/JSON: seeds, (Tp,N) grid, R, tuning constants
run_all.{sh,py}    # one-command reproduction → writes all table/figure files
```

---

## 3. The model and the Monte Carlo DGP (`subsec:sim_dgp`, `eq:sim_dgp`, `tab:sim_design`)

**Model** (`eq:model`):
```
y_{it} = Σ_{ℓ=1..P} a_{0,ti,ℓ} y_{i,t-ℓ} + Σ_{k=1..K} x_{it,k} β_{0,ti,k} + h_{0,ti} + u_{it}
```

**Baseline DGP** (`P=K=1`, the smallest setting with the dynamic generated-regressor problem):
```
y_{it} = a_{0,ti} y_{i,t-1} + x_{it} β_{0,ti} + h_{0,ti} + u_{it},   u_{it} ~ iid N(0, σ_u²)
```

Construction (match `tab:sim_design` exactly):
- **Burn-in:** simulate `50` extra pre-sample periods, discard them, so the period-`P` row is a genuine observed initial condition.
- **Surfaces** `A_0, B_0, H_0`: exact **rank-1** matrices `U Σ V'` with
  - smooth time factors (e.g. low-frequency deterministic curves on a grid over `t`),
  - random **incoherent** unit loadings (e.g. iid bounded, then column-normalized; check `max_i ‖e_i'V‖² ≤ C/N`),
  - singular value of order `sqrt(Tp·N)` (set `σ_1 = c·sqrt(Tp·N)`).
- **Lag-loading** `A_0`: built **positive and bounded**, then rescaled so `max_{t,i}|a_{0,ti}| ≤ 0.92·ρ_y` (enforces stability `eq:dynamic_stability_ass`).
- **Slope/nuisance** `B_0, H_0`: rank-1, entrywise rms `0.50`, with `V_B ⊥ V_H` (orthogonal loading spaces → slope and nuisance cleanly identified; lag-loading is the hard block).
- **Regressor** (`eq:sim_x_dgp`): `x_{it} = c_x · f_{x,t} · λ_{x,i} + σ_x · e_{it}`, `e_{it} ~ iid N(0,1)`, standardized over the effective sample. `c_x = 0.3`. (Common factor part + residual identifying variation; residual keeps tangent images non-collinear, `ass:image_no_collinearity`.)
- **Baseline constants:** `ρ_y = 0.85`, `σ_u = 0.30`. High-SNR regime `σ_1 ≍ sqrt(Tp·N) ≫ σ_u`.

**Panel grid / replications** (`tab:sim_design`): `R = 1000` at `(Tp,N) ∈ {79,119,159}²`; pilots `R=120` at `59`, `R=45` at `109`.

**Variants** to support (`tab:sim_design`, the dependence-robust study): heteroskedastic innovations (deterministic `ς²_{it}` depending on `G_0`), and a within-period cross-sectionally dependent variant (independent across time slices but correlated within a period — for the `xs` variance study). Both must keep the conditional-mean-zero / martingale structure (`ass:dynamic`, `ass:dependent`).

> The DGP is conditional-on-`G_0` (the exogenous frame = surfaces, regressor paths, burn-in initials, fold assignment). Given `G_0`, innovations are independent mean-zero with deterministic conditional variances. This is what makes the predictable-weight CLT apply (`subsec:benchmark_clt`).

---

## 4. Core estimator: alternating factor-ridge (`subsec:factor_ridge`)

**Objective** (`eq:factor_ridge_objective`), on a training set `S` (will be a purged fold `I^pur_{-j}`), at candidate ranks `r`:
```
Q(F_{1..M}, Λ_{1..M}, F_H, Λ_H; r)
 = ½ Σ_{(t,i)∈S} [ y_{it} − Σ_m Z^(m)_{ti} · fv_{t,m}'λ_{i,m} − fv_{t,H}'λ_{i,H} ]²
   + ½ Σ_m ( ϱ_F‖F_m‖² + ϱ_Λ‖Λ_m‖² ) + ½ ϱ_H^F‖F_H‖² + ½ ϱ_H^Λ‖Λ_H‖²
```
where `Γ^(m) = F_m Λ_m'` (`F_m: Tp×r_m`, `Λ_m: N×r_m`), `fv_{t,m}'` = row `t` of `F_m`, `λ_{i,m}'` = row `i` of `Λ_m`. The factor ridge **is** the factorized nuclear-norm penalty (`‖M‖_* = min_{M=LR'} ½(‖L‖²+‖R‖²)`).

**Alternating ridge updates** (closed-form, batched):

*Row update* (`eq:row_update`), holding loadings fixed, for each `t`:
```
stack  z_t  = (fv_{t,1}', …, fv_{t,M}', fv_{t,H}')'           # length Σr_m + r_H
design d_it = (Z^(1)_{ti}·λ_{i,1}', …, Z^(M)_{ti}·λ_{i,M}', λ_{i,H}')'
ẑ_t = ( Σ_{i:(t,i)∈S} d_it d_it' + R_F )^{-1} ( Σ_{i:(t,i)∈S} d_it y_{it} )
R_F = diag(ϱ_F I_{r_1}, …, ϱ_F I_{r_M}, ϱ_H^F I_{r_H})
```
Then write `ẑ_t` back into row `t` of each `F_m`, `F_H`.

*Column update* (`eq:col_update`), holding factors fixed, for each `i`:
```
stack  w_i  = (λ_{i,1}', …, λ_{i,M}', λ_{i,H}')'
design c_it = (Z^(1)_{ti}·fv_{t,1}', …, Z^(M)_{ti}·fv_{t,M}', fv_{t,H}')'
ŵ_i = ( Σ_{t:(t,i)∈S} c_it c_it' + R_Λ )^{-1} ( Σ_{t:(t,i)∈S} c_it y_{it} )
R_Λ = diag(ϱ_Λ I_{r_1}, …, ϱ_Λ I_{r_M}, ϱ_H^Λ I_{r_H})
```

**Loop:** warm start (§5) → alternate row/column sweeps until the objective stabilizes. Per `tab:sim_design`: ridge `ϱ = 0.02` (all four), `4` random restarts, keep the lowest-objective fit. Expect convergence in `O(log(Tp·N))` sweeps and a **monotonically non-increasing** objective across sweeps (assert this in tests; `subsec:sim_estimation`).

Implementation notes:
- Each block's row/column dimension is tiny (`r_m` fixed, ~1–3), so the solves are small `(Σr_m+r_H)×(Σr_m+r_H)` systems — vectorize over `t` (resp. `i`).
- Use a Cholesky/`solve`, not an explicit inverse.
- Restarts: perturb the warm start (or random init) for restarts `2..4`; retain min objective. This removes bad stationary points the weakly-identified lag block can produce in small panels.

---

## 5. Truncated-SVD warm start (warm-start paragraph in `sec:estimation`)

Two steps. **Do not** SVD `Y`.

1. **Linear surface recovery** (minimum-norm ridge solve in surface space):
```
Θ^lin_{-j} = argmin_M  ‖ Π^pur_{-j}{ Y − A(M) } ‖_F²  + τ_TN ‖M‖²
```
This is an unconstrained (per-cell, all-ranks) ridge regression recovering each surface `Γ^(m),lin`, `H^lin` from the Hadamard design. `τ_TN` small, only to stabilize the linear inverse. **Not** a nuclear-norm program. In the `P=K=1` Hadamard model this is a per-cell / low-dimensional linear system — set it up as a ridge least squares for the stacked surface coordinates.

2. **Per-block truncated SVD** at the selected ranks:
```
U^(0)_m Σ^(0)_m V^(0)_m' = SVD_{r_m}( Γ^(m),lin_{-j} ),   m = 1..M, H
Θ^(0) = the resulting rank-r factors
```
This `Θ^(0)` seeds the ALS loop in §4. (It is what enters the basin analysis `prop:als_optimization`(i); the entrywise rate is `lem:oracle` + the secant eigenvalue.)

---

## 6. Folds + forward exclusion window (`subsec:folds`) — the signature piece

**Scattered folds** (`eq:purged_training_appx` context). Assign every cell `(t,i)∈I` to a fold `σ(t,i)∈{1..J}`, **scattered over the time-unit grid** (deterministic checkerboard interleaving, or fixed-seed random draw), balanced `|σ^{-1}(j)|/(Tp·N) → 1/J`. **Not contiguous time blocks** (a time block leaves its own dates with no training support → singular information map). `J ∈ {6,8,10}` in practice.

**Forward exclusion window** (`eq:purged_training_appx`): the training set for fold `j` removes the fold itself **and every same-unit cell within `q` periods *after* a fold-`j` cell**:
```
I^pur_{-j} = { (t,i) ∈ I :  σ(t,i) ≠ j   AND   σ(s,i) ≠ j  for all  max(P+1, t−q) ≤ s < t }
```
In words: drop a cell from training if it is held out, or if the same unit was held out at any of the preceding `q` dates. This deletes exactly the future same-unit descendants through which a held-out `u_{it}` propagates. **This is the heart of the method** — get the indexing exactly right and write a unit test against a small hand-checked grid.

Bookkeeping:
- `p_{j} = |I_j| / (Tp·N)` (realized fold share; `Σ_j p_j = 1`).
- `n^pur_{-j} = |I^pur_{-j}|`, `α_j = Tp·N / n^pur_{-j}` (rescales training sums to full-sample scale).
- Retained share `≈ (1−1/J)^{q+1}`; keep it above a floor (~0.35), else raise `J`.

> **Gram normalization convention** (stated in `subsec:folds`): every Gram is a *per-cell average* of normalized design products `X_a(Δ)=sqrt(Tp·N)·[A Δ]_a`. The full-sample average equals the raw Frobenius inner product `⟨AΔ,AΞ⟩`; the factors `α_j` and `p_j^{-1}` convert purged-training and held-out raw Grams to that same per-cell scale. Keep all Grams on this scale to avoid an off-by-`Tp·N` error.

---

## 7. Rank selection + data-driven roadmap (`sec:rank_selection`, `app:roadmap`)

**Cross-fitted prediction criterion.** For candidate `r` in a finite box `R`:
- out-of-fold loss (`eq:rank_cv_loss`):
```
L̂(r) = (1/(Tp·N)) Σ_{j=1..J} ‖ Π_j{ Y − A(Θ̂^0_{-j}(r)) } ‖_F²
```
  where `Θ̂^0_{-j}(r)` is the §4 estimator trained on `I^pur_{-j}` at rank `r`.
- effective dimension penalty (`eq:rank_dimension`):
```
d(r) = Σ_{m=1..M} r_m(Tp+N−r_m) + r_H(Tp+N−r_H)
```
- selector (`eq:rank_selector`):
```
r̂ = argmin_{r∈R} { L̂(r) + κ_TN · d(r)/(Tp·N) },
```
  ties → smaller `d(r)`, then lexicographic.

**Data-driven roadmap** (`app:roadmap`, Steps 0–4) — implement exactly:

- **Step 0 — persistence.** One full-sample fit at a generous working rank `r^wk` (each component = max entertained). Form estimated companion matrices `Ĉ_{ti}` (`eq:companion_matrix`) and
```
ρ̂_* = min{ 0.99,  max_{1≤h≤H_TN} max_{t,i} ‖ Ĉ_{ti}Ĉ_{t-1,i}…Ĉ_{t-h+1,i} ‖_op^{1/h} },
H_TN = ⌈log(Tp·N)⌉,  inner max over (t,i) with t−h+1 ≥ 1.
```
  (Cap 0.99 guards near-unit-root. Use the companion-product modulus, **not** `Σ_ℓ|â_{ti,ℓ}|`.) Also return residual scale `σ̂²`.
- **Step 1 — window.** `q = ⌈ log(Tp·N) / |log ρ̂_*| ⌉` (gives `ρ̂_*^q ≤ (Tp·N)^{-1}`). Cap `q` at a moderate value for very persistent panels.
- **Step 2 — folds.** Choose `J` so `(1−1/J)^{q+1} ≥ τ_tr`, `τ_tr ∈ [0.35,0.6]` → typically `J∈{6,8,10}`.
- **Step 3 — candidate box.** Per block, screen singular values of the working fit: `r̄_m` = smallest `r` with `σ̂_{m,r+1} ≤ τ_sv·σ̂_{m,1}`, `τ_sv ∈ [0.1,0.2]`. Box `R = Π_m {0..r̄_m+1} × {0..r̄_H+1}` (the `+1` brackets the truth).
- **Step 4 — penalty.** `κ_TN = c_κ · σ̂² · ℓ_TN² · log log(Tp·N)`, default `c_κ=1`. Report sensitivity over `c_κ ∈ {0.5,1,2}`.

---

## 8. Targets, tangent space, and debiasing (Riesz) weights (`subsec:folds`, `eq:feasible_fold_gram`)

Every target is a linear (or smooth) functional of `Θ_0`. Linear targets are written `φ_ν(Θ) = ⟨D_ν, Θ⟩` for a **direction** `D_ν` (a tuple of surfaces, mostly zero).

**The eight scalar targets** in the MC (entry + mean for each of `A`,`B`, plus contrasts/IRF):
- **Entry**: `φ = e_t' Γ^(m) e_i` → `D_ν` has a single 1 in block `m` at cell `(t,i)`.
- **Group/full mean** (`eq:onestep`, dynamic and static mean lines): `θ^A_{t,G,ℓ} = e_t' A^(ℓ) π_G`, with `π_G` a weight vector over units (`π_G = (1/|G|)·1_G`; full mean = `(1/N)·1`). `D_ν` = block `ℓ`, row `t`, columns weighted by `π_G`.
- **Between-group contrast**: `ν_Δ = ν_1 − ν_2` → `D_{ν_Δ} = D_{ν_1} − D_{ν_2}`.
- **Impulse response / long-run multiplier** (`cor:irf_body`, `eq:irf_clt`): smooth functions of the lag loadings at an evaluation point. With `P=1`: horizon-`h` response `ψ_h(a) = a^h` (general `ψ_h(a)=e_1'C(a)^h e_1`), long-run multiplier `m(a) = 1/(1−Σ_ℓ a_ℓ)`. Handled by **delta method** on top of the joint CLT for the lag loadings (see §10).

**Local tangent space** `T_0` (needed for the Riesz solve). At the low-rank point, block `m` with SVD `Γ^(m)=U_m Σ_m V_m'`, the tangent space of the rank-`r_m` manifold is
```
T_m = { U_m B' + A V_m'  :  A ∈ R^{Tp×r_m}, B ∈ R^{N×r_m} },
P_{T_m}(X) = U_m U_m' X + X V_m V_m' − U_m U_m' X V_m V_m'.
```
`T_0 = T_1 × … × T_M × T_H`; `P_{T_0}` applies block-wise. **Implement and unit-test `P_{T_0}` carefully — it is where bugs hide.** (Feasible version uses estimated `Û_m,V̂_m` from `Θ̂^0_{-j}`; the projector is `P_{T̂_{-j}}`.)

**Feasible debiasing weights** (`eq:feasible_fold_gram`):
```
Ĝ_{ν,-j}  = α_j · P_{T̂_{-j}} A* Π^pur_{-j} A P_{T̂_{-j}}      # local information map on tangent space
q̂_{ν,-j}  = Ĝ_{ν,-j}^{+} P_{T̂_{-j}} D_ν                      # Riesz solve (truncated inverse = numerical safeguard only)
Ψ̂_{ν,-j}  = A( q̂_{ν,-j} )                                    # observation-space weights (Tp×N matrix)
```
- This is exactly solving `(X'X)q = (target direction)` restricted to the tangent space. Parametrize `T̂_{-j}` by an explicit orthonormal basis, assemble `Ĝ` as a dense matrix in that basis (its dimension is `O(Σ r_m·(Tp+N))` — manageable for the MC sizes; for the large Zillow panel, exploit the block/Kronecker structure or solve iteratively).
- `Ĝ^+`: use a genuine solve; truncate only tiny eigenvalues as numerical regularization (under the restricted-eigenvalue condition the map is nonsingular w.p.→1 and truncation is inactive).
- The weight is **predictable**: `Ψ̂_{ν,-j}` evaluated on a held-out cell uses only that cell's predetermined design and a direction computed from `I^pur_{-j}`. This predictability is the whole point — do not let any held-out fold information enter `q̂_{ν,-j}`.

---

## 9. One-step debiased estimator (`eq:onestep`)

```
R̂^0_{-j}     = Y − A( Θ̂^0_{-j}(r̂) )                    # out-of-fold residual panel
φ̂^0_{ν,-j}   = ⟨ D_ν , Θ̂^0_{-j}(r̂) ⟩                   # plug-in target value

φ̌_ν = Σ_{j=1..J} p_j [ φ̂^0_{ν,-j} + p_j^{-1} ⟨ Π_j Ψ̂_{ν,-j} , Π_j R̂^0_{-j} ⟩ ]
```
`⟨Π_j·,Π_j·⟩` = sum over fold-`j` cells only. `p_j^{-1}` restores full-sample scale. (For group-mean targets, the plug-in term is `e_t'Â^(ℓ)_{-j}π_G` etc. — same formula with the mean direction.)

---

## 10. Variance estimators / studentizers (`eq:variance_estimator`, `eq:xs_estimator_main`)

Cellwise cross-fitted weights and residuals:
```
Ψ̂^cf_{ν,a} = [Ψ̂_{ν,-j(a)}]_a,      û^cf_a = [R̂^0_{-j(a)}]_a       # j(a) = fold containing cell a
```

**Baseline (heteroskedasticity-robust) — White/sandwich form** (`eq:variance_estimator`):
```
ŝ²_ν   = Σ_{a∈I} (Ψ̂^cf_{ν,a})² (û^cf_a)²
ŝ²_{ν,+} = max{ ŝ²_ν , (Tp·N)^{-2} }                              # floor; asymptotically inactive
```
Interval: `φ̌_ν ± z_{1-α/2} · ŝ_{ν,+}`. t-stat `(φ̌_ν − φ_ν(Θ_0))/ŝ_{ν,+} → N(0,1)` (`thm:feasible`).

**Cross-sectional (within-period) dependence-robust — spatial kernel** (`eq:xs_estimator_main`):
```
ŝ²_{ν,xs} = Σ_{a,b∈I} K_xs( d(a,b)/b_TN ) · Ψ̂^cf_{ν,a} Ψ̂^cf_{ν,b} û^cf_a û^cf_b
```
`K_xs` compactly supported (e.g. Bartlett/Parzen), `d` a product metric (e.g. `|t−s|+|i−j|`), `b_TN` a bandwidth. Report both s.e.'s for every empirical target (paper labels them "xs s.e."/"xs CI").

**IRF/LRM delta method** (`cor:irf_body`): get the joint covariance `Σ̂` of the lag-loading entries entering the function (from the cellwise weights/residuals, same machinery), then
```
ŝ²_h = ∇ψ_h(Φ̂)' Σ̂ ∇ψ_h(Φ̂),     ŝ_m = m(Φ̂)² · (1_P' Σ̂ 1_P)^{1/2}
```
with `ψ_h(a)=e_1'C(a)^h e_1`, `m(a)=1/(1−Σ_ℓ a_ℓ)`.

---

## 11. Full feasible pipeline (`pipeline.py`)

```
def estimate(Y, Z_list, P, K, targets, tuning):
    # 1. roadmap Step 0: working fit → ρ̂*, σ̂²   (§7)
    # 2. q (Step1), J (Step2), candidate box R (Step3), κ (Step4)
    # 3. build scattered folds σ and {I^pur_{-j}}  (§6)
    # 4. rank selection: for each r∈R, each fold j: ALS fit on I^pur_{-j}; CV loss; pick r̂  (§7,§4,§5)
    # 5. refit Θ̂^0_{-j}(r̂) on each purged fold; residuals R̂^0_{-j}  (§4)
    # 6. for each target ν: tangent proj, feasible info map, Riesz weights Ψ̂_{ν,-j}  (§8)
    # 7. one-step φ̌_ν  (§9)
    # 8. variances ŝ²_ν (and ŝ²_{ν,xs}); intervals; IRF/LRM via delta method  (§10)
    return estimates, ses, intervals, r̂, q, J, diagnostics
```
Diagnostics to log per run: objective monotonicity flag, #sweeps to converge, retained training share, smallest eigenvalue of `Ĝ` (truncation should be inactive), selected ranks.

---

## 12. Monte Carlo harness + the validation checkpoint (`mc.py`)

For each `(Tp,N)` in the grid, for `R` replications with recorded seeds:
1. `dgp.simulate(Tp, N, params, seed)` → `Y`, true surfaces, true targets.
2. `pipeline.estimate(...)` → `φ̌_ν`, `ŝ_ν`, intervals.
3. Accumulate per target: bias `mean(φ̌_ν − φ_ν,true)`, RMSE, mean `ŝ_ν` vs MC sd of `φ̌_ν`, empirical coverage of nominal 95% interval.

Also run:
- **Infeasible oracle** (`subsec:sim_oracle`): identical, but pass the **true** tangent spaces `T_0` (true `U,V`) into the Riesz solve instead of estimated ones — isolates the influence-function/CLT logic from first-stage error.
- **Exclusion-window sensitivity** (`tab:sim_purge`, `fig:purge_sensitivity`): sweep `q` from 0 upward at fixed `(Tp,N)`; show coverage/bias as a function of `q` — the own-error-leakage mechanism (`prop:dynamic_leakage`). Expect an interior optimum.

> ### The checkpoint that decides whether you have a paper
> Run the **oracle** first on one moderate size (e.g. `(79,79)`). The theory predicts (`subsec:sim_oracle`): essentially unbiased for all targets, **coverage in ≈[0.93,0.96]**, and mean `ŝ_ν` matching MC sd to ~2 digits. If the oracle hits this, the influence-function + martingale-CLT core is implemented correctly and the method is sound — then move to the feasible study. If it does **not**, stop and debug (almost always the tangent projection `P_{T_0}`, the Riesz solve, the fold/window indexing, or a Gram-normalization scale) before running anything large. This single experiment is worth more than the entire rest of the build.

---

## 13. Empirical applications (`sec:empirical`)

Both use the **AR(2)** heterogeneous low-rank form (`P=2,K=0`): `ỹ_{it} = a_{0,ti} ỹ_{i,t-1} + b_{0,ti} ỹ_{i,t-2} + h_{0,ti} + u_{it}` (second lag is the predetermined generated regressor; `H_0` interactive). Same cross-fitted debiased one-step.

- **FRED-QD** (`subsec:emp_macro`, `mccracken2021`): St. Louis Fed quarterly macro database. Apply the database's own stationarity transform codes (FRED-MD convention, `mccracken2016`). Retain the balanced subset `N=176` series over `T=267` quarters; standardize each series. `H_0` = common macro factors; lag loadings = idiosyncratic persistence after common comovement removed. Report all eight-style targets with both s.e.'s → `tab:emp_fredqd`.
- **Zillow** (`subsec:emp_*` / `tab:emp_zillow`): large regional housing panel (house-price momentum); both dimensions large → near-benchmark precision.

> The paper currently states selected ranks `(1,1,2)` and specific numbers (e.g. avg idiosyncratic lag-1 persistence `−0.01`, xs CI `[−0.48,0.45]`). **These must be reproduced by the code on the actual downloaded data; if the real output differs, update the paper.** Pin the exact data vintage/download date and the transform codes in the config — public databases revise, and "which vintage" is a common replication failure.

---

## 14. Reproducibility scaffolding (Data-Editor grade)

- **Seeds:** one master seed per config; derive per-replication seeds deterministically (e.g. `SeedSequence`). Record them in the config files (the paper claims "from the seeds recorded in its configuration files").
- **Determinism:** fix BLAS threading or at least verify cross-thread reproducibility; SVD/eig sign conventions can flip — canonicalize signs after every SVD (e.g. fix the sign of the largest-magnitude entry of each singular vector) so `Û`, `V̂`, and downstream weights are reproducible.
- **Environment:** pin exact versions (`requirements.txt` / `renv.lock` / `Project.toml`) and record OS/BLAS. "It ran in my session" is the #1 verification failure.
- **One command:** `run_all` regenerates every table/figure file from scratch into an `output/` dir. Tables should be written by the code, not transcribed.
- **Data:** include a scripted download (with checksum) for FRED-QD/Zillow, or document the exact retrieval; never commit large raw data without licence check.
- **README:** hardware, expected runtime per table, how to reproduce each number.

---

## 15. Test checklist (write these before scaling up)

1. `A` / `A*` adjoint identity: `⟨A(Θ), R⟩ = ⟨Θ, A*(R)⟩` to machine precision.
2. Forward-exclusion index set on a tiny hand-checked grid (verify the same-unit, `q`-back rule exactly).
3. ALS objective monotonic non-increasing across sweeps; converges in `O(log)` sweeps.
4. Tangent projector `P_{T_0}` idempotent (`P²=P`), self-adjoint, and range = the manifold tangent space.
5. Riesz identity: for the **infeasible** weights on a known `Θ_0`, `⟨Ψ_ν, A(Δ)⟩ = ⟨D_ν, Δ⟩` for admissible tangent `Δ` (the Riesz-representer property).
6. Recover a noiseless surface (`σ_u=0`) exactly at the true rank.
7. Oracle MC coverage ≈ nominal (the §12 checkpoint).
8. Gram scale: `p_j^{-1}⟨Π_j AΔ, Π_j AΔ⟩` and `α_j⟨A^pur Δ, A^pur Δ⟩` and `⟨AΔ,AΔ⟩` are all the same order (per-cell-average convention).

---

### Cross-reference index (paper labels → this spec)

`eq:model` §1,§3 · `eq:operator_model`/`subsec:operator_form` §1 · `eq:companion_matrix` §1 · `eq:factor_ridge_objective` §4 · `eq:row_update`/`eq:col_update` §4 · warm start (`sec:estimation`) §5 · `subsec:folds`/`eq:purged_training_appx` §6 · `sec:rank_selection` (`eq:rank_cv_loss`,`eq:rank_dimension`,`eq:rank_selector`) §7 · `app:roadmap` Steps 0–4 §7 · `eq:feasible_fold_gram` §8 · `eq:onestep` §9 · `eq:variance_estimator`/`eq:xs_estimator_main` §10 · `cor:irf_body`/`eq:irf_clt` §8,§10 · `subsec:sim_dgp`/`eq:sim_dgp`/`tab:sim_design` §3,§12 · `subsec:sim_oracle` §12 · `sec:empirical` §13.
