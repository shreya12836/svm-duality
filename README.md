# Duality Geometry of Linear and Kernel Support Vector Machines

ED401 Advanced Analysis, Project 1. Data: UCI Banknote Authentication.

> **Central question.** How do continuous duality, separating functionals,
> Lagrange duality and RKHS representation explain the geometry and sparsity
> of support vector machines?

This repository solves the soft-margin SVM five ways (exact primal, exact dual,
SMO, stochastic sub-gradient and an ℓ₁-norm LP) on real banknote data. It
checks every optimality condition numerically and produces every figure and
table used in the report.

## Quick start

```bash
python -m pip install -r requirements.txt   # Python >= 3.11
python run_experiment.py                    # ~1 min; writes results/
python -m pytest                            # 96 tests (~50 s)
```

Command-line options:

```bash
python run_experiment.py --seed 7 --out results_seed7   # different split / Pegasos seed
python run_experiment.py --config configs/default.yaml  # explicit config
python run_experiment.py --no-memory                    # skip the peak-memory pass
python -m pytest -m "not integration"                   # unit tests only
python -m pytest --cov=svmdual --cov-report=term-missing
```

Every hyperparameter, the seed, the grids and the dataset checksum are set in
[`configs/default.yaml`](configs/default.yaml).

## Notation

Training data (xᵢ, yᵢ), i = 1…n, with xᵢ ∈ ℝ⁴ and yᵢ ∈ {−1, +1}. A kernel *k* has
RKHS H_k.

| Symbol | Meaning |
|---|---|
| k(x, x′) = ⟨x, x′⟩ | linear kernel (parametric; H_k ≅ ℝ⁴) |
| k(x, x′) = exp(−γ‖x − x′‖²) | RBF kernel (nonparametric; H_k infinite-dimensional) |
| C | cost of violating the margin |
| γ | inverse squared kernel width |
| K, Q | Gram matrix K_ij = k(xᵢ, xⱼ); Q_ij = yᵢ yⱼ K_ij |
| αᵢ | dual variable (Lagrange multiplier) of margin constraint i |
| f(x) = Σᵢ αᵢ yᵢ k(xᵢ, x) + b | decision function (representer theorem) |

### Primal (P)

```text
min_{f ∈ H_k, b, ξ}  ½‖f‖²_H + C Σᵢ ξᵢ    s.t.  yᵢ (f(xᵢ) + b) ≥ 1 − ξᵢ,  ξᵢ ≥ 0
```

### Dual (D)

```text
max_α  Σᵢ αᵢ − ½ αᵀQα    s.t.  0 ≤ αᵢ ≤ C,  Σᵢ αᵢ yᵢ = 0
```

### KKT conditions

Residuals are reported in `results/tables/kkt_residuals.csv`.

| Condition | Statement | Column |
|---|---|---|
| dual feasibility | 0 ≤ αᵢ ≤ C | `dual_box_violation` |
| | Σ αᵢ yᵢ = 0 | `equality_residual` |
| stationarity | w = Σ αᵢ yᵢ φ(xᵢ),  αᵢ + μᵢ = C | `stationarity_w`, `stationarity_xi` |
| complementary slackness | αᵢ (yᵢ f(xᵢ) − 1 + ξᵢ) = 0 | `comp_slack_margin` |
| | (C − αᵢ) ξᵢ = 0 | `comp_slack_slack` |
| SMO optimality | m(α) − M(α) ≤ 0 | `max_violating_pair` |
| strong duality | P − D = 0 | `duality_gap` |

**Sparsity.** By complementary slackness, αᵢ > 0 only for points on or inside
the margin, so f depends only on the support vectors. Free SVs (0 < αᵢ < C) lie
exactly on f = ±1. Bounded SVs (αᵢ = C) violate the margin.

**Geometry (Bennett & Bredensteiner 2000).** Write s = Σ_{yᵢ=+1} αᵢ = Σ_{yᵢ=−1} αᵢ.
Then w = s(c − d), where c and d are the closest points of the two *reduced*
convex hulls: convex combinations whose weights are at most D = C/s. Replacing
‖·‖₂ by ‖·‖₁ (the 1-norm SVM) measures the margin 2/‖w‖₁ in the ℓ∞ norm instead.

## Solvers

| Solver | File | Type | Role in the comparison |
|---|---|---|---|
| `cvx_primal` | `solvers/cvx.py` | exact, interior point (Clarabel) | **primal**; returns α, μ as multipliers |
| `cvx_dual` | `solvers/cvx.py` | exact, interior point | **dual**; b from KKT on free SVs |
| `smo` | `solvers/smo.py` | iterative, own implementation (WSS2 of Fan et al. 2005) | records the **primal–dual gap** at every iteration |
| `pegasos` | `solvers/pegasos.py` | stochastic sub-gradient, kernelised | **approximate** vs exact |
| `l1_svm` | `solvers/l1svm.py` | LP (HiGHS) | **non-Euclidean** (ℓ₁ / ℓ∞) vs Euclidean |
| `libsvm` | `sensitivity.py` | scikit-learn `SVC` | independent reference; fits the C × γ grid |

For the RBF kernel, the CVXPY primal is solved through the exact empirical
feature map K = ΦΦᵀ (an eigendecomposition). By the representer theorem this is
equivalent to the RKHS problem restricted to the span of the training points.

Notes on the comparison:

- **Pegasos** has no offset b (as in the original paper), and λ = 1/(nC). Its
  suboptimality P(f_T, 0) − P* is therefore always ≥ 0. It is run with 5
  seeds, and the mean ± sd is reported.
- **Runtime** is the median of 3 runs after a warm-up solve. SMO and Pegasos
  are pure-Python loops and libsvm is compiled C++. Runtimes therefore compare
  implementations, not algorithms.
- **Memory** (`py_heap_peak_mb`) is measured with `tracemalloc` in a separate
  run. It covers Python and NumPy allocations only, not memory inside the
  native solvers (Clarabel, HiGHS, libsvm).

## Results (seed 42; regenerate with `python run_experiment.py`)

**Model selection** uses validation balanced accuracy, with ties broken by
validation ROC-AUC, then *fewest support vectors*, then smaller C and γ. The
tie-break matters: 40 of the 77 RBF grid cells reach 1.000 on the 270
validation notes. "Fewest SVs" is principled because the leave-one-out error is
bounded by #SV/n. It selects **linear C = 100** and **RBF C = 1000, γ = 0.01**.

| Solver | Kernel | Runtime (s) | Primal P | P − P* | #SV / n | Test bal. acc. [95 % CI] |
|---|---|---|---|---|---|---|
| cvx_primal | linear | 0.029 | 944.545 | 0 | 12 / 808 | 0.993 [0.983, 1] |
| cvx_dual | linear | 0.029 | 944.545 | ≈ 0 | 12 / 808 | 0.993 [0.983, 1] |
| smo | linear | 1.79 | 944.546 | 1.1e-3 | 12 / 808 | 0.993 [0.983, 1] |
| libsvm | linear | 0.005 | 944.547 | 2.3e-3 | 12 / 808 | 0.993 [0.983, 1] |
| pegasos | linear | 0.031 | 10084 | 9140 | 108 / 808 | 0.976 [0.959, 0.993] |
| l1_svm | linear | 0.015 | 848.3 (ℓ₁ objective) | – | 12 / 808 | 0.993 [0.983, 1] |
| cvx_primal | RBF | 0.35 | 2339.7 | 0 | 11 / 808 | 1.000 |
| cvx_dual | RBF | 0.44 | 2339.7 | 2.9e-3 | 9 / 808 | 1.000 |
| smo | RBF | 0.42 | 2339.7 | 7.5e-3 | 9 / 808 | 1.000 |
| libsvm | RBF | 0.003 | 2339.8 | 9.6e-2 | 9 / 808 | 1.000 |
| pegasos | RBF | 0.040 | 48050 | 45710 | 191 / 808 | 0.993 [0.983, 1] |

The full tables, including macro-F1, ROC-AUC, PR-AUC and every KKT residual,
are in [`results/summary.md`](results/summary.md). What the numbers show:

1. **Strong duality holds numerically.** For every exact solver (CVXPY, SMO,
   libsvm) the relative duality gap (P − D)/P is ≤ 5e-5, and ≤ 1.3e-6 for the
   CVXPY solves. Complementary-slackness residuals scale with C. For CVXPY they
   are ≤ 3e-6·C. Interior-point multipliers are never exactly 0 or C, so SV
   counts use the threshold αᵢ > 1e-6·C. This explains why cvx_primal reports
   11 SVs where the other solvers report 9.
2. **Sparsity.** Both selected models depend on about 1 % of the training
   notes: 12 SVs (linear) and 9 SVs (RBF). In the linear sweep, #SV/n falls
   from 0.45 at C = 0.01 to 0.015 at C ≥ 100 while the margin 2/‖w‖ shrinks.
   This is the reduced-hull trade-off, with weights capped at D = C/s.
3. **ℓ₁ vs ℓ₂ geometry.** At the same C, the 1-norm SVM keeps the same 12
   support vectors and test accuracy. All 4 weights stay nonzero,
   w = (−7.33, −10.68, −9.08, −0.87), close in direction to the ℓ₂ solution
   (−7.22, −9.73, −8.45, −0.59), so every wavelet feature carries signal. Its
   margin 2/‖w‖₁ = 0.072 is measured in ℓ∞ and is not comparable with the
   Euclidean 0.135. The 1-norm SVM is not tuned separately.
4. **SMO gap curve.** Both gap traces decrease linearly on a log scale. The RBF
   problem converges in about 1,900 iterations and the linear one in about
   9,200, with visible oscillation.
5. **Exact vs approximate.** Pegasos guarantees suboptimality
   Õ(1/(λT)) = Õ(C/epochs). At the selected large C this bound is vacuous:
   λT = 0.5 (linear) and 0.05 (RBF). Over 5 seeds the relative suboptimality
   is 11.0 ± 4.6 (linear) and 22.6 ± 9.4 (RBF). Pegasos also gives up sparsity
   (13–24 % SVs), because every sampled margin violator keeps a weight.
6. **Uncertainty.** With 270 test notes, the 0.993 vs 0.976 difference is
   within the bootstrap CIs. For RBF there are 0 test errors, so the bootstrap
   interval collapses to [1, 1]. The "rule of three" gives a more honest 95 %
   upper bound on the error rate: 3/270 ≈ 1.1 %. Use `--seed` to repeat over
   different splits.

### Limitations

- **Near-duplicates.** Only exact duplicate rows are removed. Several
  images of the same banknote, if present, could fall into different splits.
- **Selection noise.** Selection rests on a single validation split. Accuracy
  ties at the 1-error level are common (see the heatmaps).
- **Restricted RBF primal.** The kernel primal is solved in the empirical
  feature space of the training points, which the representer theorem
  justifies. Eigenvalues below 1e-10·λ_max are dropped.

## Required outputs → files

| Required output | File |
|---|---|
| Separating-hyperplane figure | `results/figures/separating_hyperplane.png` |
| Support-vector and margin visualisation | `separating_hyperplane.png` (linear), `rbf_margin_support_vectors.png` (RBF) |
| Primal–dual gap curve | `results/figures/primal_dual_gap_curve.png` |
| KKT residual report | `results/tables/kkt_residuals.csv`, `results/summary.md` |
| Support-vector sparsity ratio | `sparsity_ratio` in `solver_comparison.csv` and `sensitivity_grid.csv` |
| Sensitivity analysis for C and γ | `sensitivity_linear_C.png`, `sensitivity_rbf_C_gamma.png`, `sensitivity_grid.csv` |
| Exact vs approximate optimisation | `results/figures/pegasos_vs_exact.png` |
| Experiment log (hyperparameters, runtime, convergence, seed, hardware) | `results/experiment_log.json` |

## Repository layout

```text
svm-duality/
├── configs/default.yaml      # every hyperparameter, seed, grids, data checksum
├── data/
│   ├── README.md             # source, licence, checksum, preprocessing
│   └── raw/banknote_authentication.csv
├── src/svmdual/
│   ├── data.py               # verify, load, deduplicate, stratified split, scaling
│   ├── kernels.py            # linear / RBF kernels, empirical feature map
│   ├── model.py              # SVMSolution, objectives, optimal / KKT offset b
│   ├── solvers/              # cvx.py, smo.py, pegasos.py, l1svm.py
│   ├── kkt.py                # KKT residuals, sparsity, margin, reduced-hull points
│   ├── metrics.py            # balanced accuracy, macro-F1, ROC-AUC, PR-AUC
│   ├── sensitivity.py        # libsvm C × γ sweep and validation-based selection
│   ├── experiment.py         # solver suite, evaluation, figure stages
│   ├── plots.py              # all figures
│   ├── report.py             # tables, summary.md, experiment log
│   └── config.py             # typed YAML config
├── tests/                    # pytest suite (98 % line coverage)
├── results/                  # generated: figures/, tables/, summary.md, experiment_log.json
├── run_experiment.py         # CLI entry point
├── requirements.txt
└── pyproject.toml
```

## Reproducibility

- All randomness is seeded from the config: the split, Pegasos sampling, and
  libsvm (deterministic).
- The SHA-256 of the dataset is checked before every run.
- The test split is not used until the final evaluation.
- `results/experiment_log.json` records the resolved config, package versions,
  hardware, per-solver runtime and convergence status.

## References

1. Bennett, K.P., Bredensteiner, E.J. (2000). Duality and Geometry in SVM Classifiers. *ICML 2000*, 57–64.
2. Lohweg, V. et al. (2013). Banknote authentication with mobile devices. *Proc. SPIE 8665*. doi:10.1117/12.2001444
3. Cortes, C., Vapnik, V. (1995). Support-Vector Networks. *Machine Learning* 20, 273–297.
4. Lohweg, V. (2012). Banknote Authentication [Dataset]. UCI ML Repository. doi:10.24432/C55P57
5. Bennett, K.P., Mangasarian, O.L. (1992). Robust linear programming discrimination of two linearly inseparable sets. *Optim. Methods Softw.* 1, 23–34.
6. Fan, R.-E., Chen, P.-H., Lin, C.-J. (2005). Working set selection using second order information for training SVM. *JMLR* 6, 1889–1918.
7. Shalev-Shwartz, S., Singer, Y., Srebro, N., Cotter, A. (2011). Pegasos: primal estimated sub-gradient solver for SVM. *Math. Program.* 127, 3–30.
8. Schölkopf, B., Herbrich, R., Smola, A.J. (2001). A generalized representer theorem. *COLT 2001*, LNCS 2111, 416–426.
