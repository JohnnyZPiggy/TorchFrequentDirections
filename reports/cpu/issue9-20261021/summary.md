# Issue #9: CPU QR + Small SVD independent evidence

**Engineer disposition: retain direct SVD. CPU numerical checks PASS; universal performance screen FAIL. GPT scientific acceptance/review is pending.**

Base: `a48bc4fe39c572d51c9ca9868235fc89ee2a7dc5`. Exact clean tested source: `f5894820dcfdf5bd977384ea24033c10a7793aa2`. Evidence publication adds no Python changes. Production package, finite checks, clearing, defaults, `.sae`, `AGENTS.md`, and `sae.yaml` are unchanged.

[R1–R3 rationale and reproduction](../../../experiments/qr_svd/README.md) · [raw experiment](experiment.json) · [all timing medians/p95](benchmark.csv) · [clean execution](execution.json) · [unit log and oracle observations](unit.json) · [snapshot integrity](snapshot.json) · [byte manifest](manifest.json)

## Mathematical and state checks

- 38 unit tests PASS, including injected QR/SVD failure, nonfinite output, finite-input rejection before mutation, overflow-safe scaling, exact fallback/empty paths, equal SVD counts, oracle corruption detection and paired-order controls.
- 216 fixture/dtype/ell/chunk-pattern streams, 840 prefix observations and 48 factor checks PASS. 11744 individual k-bounds checked across both implementations, before benchmark gates.
- float32/float64; seeded random, exact-rank construction, 1e-16 spectral decay, ties at sigma_ell and epsilon-spaced ties, zero/empty/short input, noncontiguous rows, ell=1, ell>d, one feature, m>d direct fallback and 1025-row drift streams. Batch, rowwise and irregular updates include empty chunks. Snapshots are repeatable, nonmutating and alias-independent; later updates continue on the same objects.
- Each implementation separately passes explicit float64 Gram PSD/spectral-tail and right-subspace projection checks for every integer k<ell, k<=d. Cross-implementation Gram Frobenius gap uses the same `32 eps(dtype) max(n,d,2ell,1) ||A||_F^2` tolerance. Largest gap/tolerance over stream/factor checks: 0.0136109.
- The oracle uses no FD shrink and never compares singular-vector signs or raw sketch rows. No stability failure was observed under this energy-scaled absolute contract; this does not establish relative accuracy of near-null components or arbitrary long-stream stability.
- 131 pinned SAE snapshot hashes PASS; eight preexisting template relocation links remain recorded by the checker. No framework files were changed.

## CPU performance

AMD Ryzen 7 4800H under WSL2, Python 3.12.3, torch 2.11.0+cpu; intra-op and inter-op threads both 1. Full BLAS/LAPACK build and environment are in the raw report. Same process/tensors/seed; 2 warmups per variant, 20 pairs per timing block, ten AB and ten BA in seeded randomized order. Each observation is one complete call; no subtraction of instrumented components. Raw ordered samples, sample standard deviations, median, nearest-rank p95 and paired bootstrap intervals are retained.

Predeclared screen: every primary wide full-stream configuration must have median paired speedup >1.05 and descriptive 95% lower bound >1.0, after all math gates pass. Result: **FAIL**. The 128-feature cases prevent an across-matrix claim; small QR cases regress and m>d controls mostly measure wrapper overhead/noise. Larger shapes can motivate a separately scoped dispatch/integration hypothesis, which needs GPT review and fresh evidence; this PR proposes no dispatch threshold.

Full streaming medians are reported as paired speedup [descriptive 95% interval]. Counts ell-1 / ell+1 exercise both snapshot paths after multiple full-buffer shrinks (four in primary wide cases). Fresh construction, validation, chunk loop, clearing and final snapshot are included.

| dtype | d | ell | final count | paired speedup [interval] | direct / QR median ms |
|---|---:|---:|---:|---|---|
| float32 | 128 | 8 | 7 | 0.986 [0.950, 1.020] | 1.214 / 1.201 |
| float32 | 128 | 8 | 9 | 0.959 [0.914, 1.013] | 1.418 / 1.534 |
| float32 | 512 | 16 | 15 | 1.463 [1.398, 1.575] | 3.563 / 2.339 |
| float32 | 512 | 16 | 17 | 1.390 [1.292, 1.442] | 4.034 / 2.821 |
| float32 | 1024 | 32 | 31 | 1.554 [1.483, 1.645] | 14.340 / 9.413 |
| float32 | 1024 | 32 | 33 | 1.558 [1.492, 1.692] | 17.743 / 10.994 |
| float32 | 2048 | 64 | 63 | 1.986 [1.914, 2.032] | 75.648 / 38.387 |
| float32 | 2048 | 64 | 65 | 1.911 [1.842, 1.937] | 83.444 / 43.240 |
| float32 | 8 | 2 | 1 | 0.789 [0.765, 0.845] | 1.182 / 1.412 |
| float32 | 8 | 2 | 3 | 0.768 [0.743, 0.891] | 1.409 / 1.751 |
| float32 | 4 | 8 | 7 | 0.861 [0.803, 1.082] | 0.743 / 0.836 |
| float32 | 4 | 8 | 9 | 0.950 [0.890, 1.041] | 0.914 / 0.960 |
| float64 | 128 | 8 | 7 | 0.919 [0.823, 1.005] | 2.341 / 2.524 |
| float64 | 128 | 8 | 9 | 0.907 [0.842, 1.062] | 2.411 / 2.833 |
| float64 | 512 | 16 | 15 | 1.355 [1.275, 1.482] | 4.364 / 3.179 |
| float64 | 512 | 16 | 17 | 1.385 [1.317, 1.450] | 5.046 / 3.548 |
| float64 | 1024 | 32 | 31 | 1.639 [1.570, 1.672] | 17.964 / 11.223 |
| float64 | 1024 | 32 | 33 | 1.642 [1.597, 1.666] | 20.500 / 12.672 |
| float64 | 2048 | 64 | 63 | 1.815 [1.764, 1.914] | 112.265 / 60.823 |
| float64 | 2048 | 64 | 65 | 1.931 [1.887, 1.994] | 121.286 / 63.850 |
| float64 | 8 | 2 | 1 | 0.836 [0.795, 0.886] | 0.841 / 0.991 |
| float64 | 8 | 2 | 3 | 0.896 [0.835, 0.927] | 0.976 / 1.124 |
| float64 | 4 | 8 | 7 | 0.968 [0.920, 1.002] | 0.573 / 0.595 |
| float64 | 4 | 8 | 9 | 0.962 [0.904, 1.002] | 0.632 / 0.668 |

Worst energy-relative cross-variant Gram Frobenius gaps for the ill-spectrum and longer-stream cases follow. These are diagnostics, not relative tiny-eigenvalue guarantees.

| dtype | case | max Gram gap / input energy |
|---|---|---:|
| float32 | decay_1e16 | 1.18659e-06 |
| float32 | tie_at_ell | 5.48716e-07 |
| float32 | near_tie | 4.48527e-07 |
| float32 | rank_two | 6.46135e-07 |
| float32 | long_drift | 3.21847e-07 |
| float64 | decay_1e16 | 1.70641e-15 |
| float64 | tie_at_ell | 1.31522e-15 |
| float64 | near_tie | 7.10287e-16 |
| float64 | rank_two | 2.28252e-15 |
| float64 | long_drift | 4.99834e-16 |

Factor-only tests cover 16×128, 32×512, 64×1024 and 128×2048, plus 4×8 and 16×4 controls, both dtypes. No-shrink controls use ell=m to retain the full factor and have their own independent exact-covariance gate. Instrumented QR, small SVD, Vh recovery and shrink/output samples are nonadditive diagnostics; their medians must not be summed or substituted for end-to-end times.

Intervals are descriptive, single-session, unadjusted for multiple comparisons. No CPU affinity/frequency isolation, alternate thread counts or cross-host replication was performed; WSL/shared-host variability limits crossover inference.

## Execution, repairs and scope

An initial clean pilot at `988ef4ead2ce8458002603e7fdf430d84cb63cf3` passed 36 unit tests and the numerical gate. Inspection found the no-shrink timing control retained only ell<m rows and inherited a shrink=true gate. Before publication, the control was changed to ell=m with a dedicated gate, two oracle/benchmark tests were added, and the entire final clean run was repeated. No runtime math failure, backend repair or performance-based tuning occurred. Only final-source measurements are published as acceptance evidence.

Final experiment elapsed 53.018 s, below 900-second internal and 1000-second subprocess bounds. All subprocess exit codes zero. Temporary clean Linux checkout removed after artifacts were copied. No GPU, Colab or TPU allocation/execution, billing, Actions, dependencies, framework changes or self-merge. The existing CPU validator records CUDA availability as false using CPU-only torch; the experiment itself calls no accelerator API.

Architect review requested: numerical tolerance/ill-spectrum coverage, source and artifact identity, full-stream fairness, negative controls and the failed universal screen. Accept/reject the evidence and choose any next scoped study. No production backend integration is requested.
