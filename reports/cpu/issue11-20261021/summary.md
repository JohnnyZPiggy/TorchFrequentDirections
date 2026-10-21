# Issue #11: independent CPU shape/thread evidence

**RETAIN DIRECT DEFAULT / INCONCLUSIVE for integration. No automatic threshold is proposed.**

Base `17c0238e6971bb5b2559121e4f9039039f27dfdc`; clean tested source `f49068088b2b970da6ba0f585d132d6440d8f7da`. Publication adds evidence/documentation only. Production package, original prototype/driver, public API, default backend and pinned SAE Framework are unchanged. GPT Architect owns acceptance/merge.

[Research plan/reproduction](../../../experiments/qr_svd/SHAPE_THREADS.md) · [all timing blocks](benchmark.csv) · [audit/classifications](verification.json) · [execution log](execution.json) · [shape/spectral raw](shape.json) · [1 thread](threads1.json) · [2 threads](threads2.json) · [4 threads](threads4.json) · [8 threads](threads8.json) · [manifest](manifest.json)

## Numerical and control evidence

42 unit tests PASS; 760 explicit float64 independent oracle evaluations / 15748 per-k checks PASS across measured configurations and supplementary prefix/chunk checks. Every measured configuration gates direct and QR variants separately on Gram PSD, spectral-tail and right-subspace projection bounds for all eligible integers k<ell, k<=d, then on cross-variant Gram Frobenius gap. No FD shrink in the oracle. Tolerance: `32 eps(input dtype) max(n,d,2ell,1) ||A||_F²`. Tiny-eigenvalue relative accuracy: NOT_ESTABLISHED.

Spectral fixtures at ell=16,d=64/512 use random orthogonal embeddings of global rank-two, 1e-16 decay, tied/epsilon-spaced sigma_ell and zero spectra in both dtypes; inputs are noncontiguous. The local shrink matrices do not necessarily inherit the global tied spectrum. Prefix snapshots, subsequent updates, rowwise versus irregular partitions, empty chunks, ell=1 and ell>d are checked separately. This is finite test coverage, not an arbitrary-stream proof.

Instrumentation verifies four full-buffer shrinks and final occupancies ell-1 / ell+1; the latter has one finalization SVD, the former none. Actual calls/shapes for constant-n streams are retained. Short and empty controls execute zero QR/SVD; m>d fallback executes direct SVD and zero QR. No-shrink short finalization copies occupied rows using the production path. Repeated snapshots are nonmutating, independent and no-grad. 131 framework hashes PASS.

## Method and environment

Ryzen 7 4800H, WSL2, Python 3.12.3, torch 2.11.0+cpu. Fresh processes for intra-op 1/2/4/8; inter-op=1. OMP/MKL/OpenBLAS thread environment equals intra-op; dynamic threading disabled. CPU affinity/cgroup quota/build/OS details are recorded in every process result. No affinity, turbo or frequency isolation; shared Windows/WSL host load is uncontrolled.

520 timing blocks, two warmups per implementation and 20 balanced randomized pairs per block; two repeat batches with reversed block order. Same inputs/dtypes/seeds within pairs and across thread processes. Both factor-only and whole update+sketch scopes are explicit. Factor-only excludes state construction; stream timings include construction, validation, chunk loop, empty updates, clearing and snapshot. Stage diagnostics are intrusive/NONADDITIVE. Raw ordered pairs, medians, nearest-rank p95, sample SD and 2000-draw paired bootstrap intervals are retained. Intervals are descriptive, exploratory, unadjusted for multiple comparisons.

Predeclared screen: both batches median paired speedup>1.05 and descriptive lower>1. Placebo flags also detect symmetric slowdown (median<1/1.05, upper<1). Controls and repeat batches are required before a local candidate claim; even a local screen does not establish a universal threshold.

## Fixed-ell and fixed-d scans (one thread)

Entries are batch-1 / batch-2 median paired speedups. Width scan fixes ell=16; height scan fixes d=1024. Four-shrink streams vary n with ell; n=257 controls hold row count fixed. Aspect rho=d/(2ell). All n and actual shrink/finalization counts are in raw evidence.

| dtype | ell | d | rho | four short | four long | n=257 | all 3 repeat screens |
|---|---:|---:|---:|---|---|---|---|
| float32 | 16 | 32 | 1 | 0.858/0.840 | 0.838/0.846 | 0.915/0.829 | False |
| float32 | 16 | 64 | 2 | 0.901/0.900 | 0.887/0.916 | 0.904/0.867 | False |
| float32 | 16 | 128 | 4 | 1.016/0.968 | 0.998/1.006 | 0.951/0.904 | False |
| float32 | 16 | 256 | 8 | 1.110/1.184 | 1.196/1.155 | 1.220/1.134 | False |
| float32 | 16 | 512 | 16 | 1.355/1.356 | 1.394/1.371 | 1.336/1.331 | True |
| float32 | 16 | 1024 | 32 | 1.572/1.689 | 1.592/1.707 | 1.750/1.709 | True |
| float32 | 8 | 1024 | 64 | 1.508/1.492 | 1.600/1.533 | 1.644/1.567 | True |
| float32 | 32 | 1024 | 16 | 1.484/1.613 | 1.573/1.631 | 1.638/1.640 | True |
| float32 | 64 | 1024 | 8 | 1.609/1.622 | 1.635/1.616 | 1.604/1.589 | True |
| float64 | 16 | 32 | 1 | 0.831/0.844 | 0.857/0.852 | 0.872/0.876 | False |
| float64 | 16 | 64 | 2 | 0.927/0.919 | 0.916/0.930 | 0.909/0.933 | False |
| float64 | 16 | 128 | 4 | 1.032/1.048 | 1.016/1.005 | 1.023/1.035 | False |
| float64 | 16 | 256 | 8 | 1.159/1.147 | 1.178/1.212 | 1.185/1.175 | True |
| float64 | 16 | 512 | 16 | 1.215/1.307 | 1.225/1.337 | 1.337/1.328 | True |
| float64 | 16 | 1024 | 32 | 1.554/1.607 | 1.565/1.558 | 1.507/1.551 | True |
| float64 | 8 | 1024 | 64 | 1.406/1.438 | 1.522/1.408 | 1.442/1.543 | True |
| float64 | 32 | 1024 | 16 | 1.552/1.595 | 1.561/1.613 | 1.577/1.591 | True |
| float64 | 64 | 1024 | 8 | 1.556/1.522 | 1.526/1.526 | 1.506/1.518 | True |

## Targeted threads and placebos

Stage-1 batch-one scores select three distinct shapes by minimum, closest-to-one and maximum median over both dtypes/four-shrink occupancies. This selection is exploratory/data-dependent. Selected (ell,d): losing [16, 32], near [16, 128], winning [64, 1024]. All selection scores remain in shape.json.

| threads | dtype | ell | d | four short | four long | n=257 | screened / 3 | A/A flags / 2 |
|---:|---|---:|---:|---|---|---|---:|---:|
| 1 | float32 | 16 | 32 | 0.871/0.863 | 0.881/0.892 | 0.858/0.881 | 0 | 0 |
| 1 | float64 | 16 | 32 | 0.847/0.866 | 0.850/0.848 | 0.845/0.881 | 0 | 0 |
| 1 | float32 | 16 | 128 | 0.990/0.995 | 0.996/1.017 | 0.995/0.996 | 0 | 0 |
| 1 | float64 | 16 | 128 | 1.013/1.003 | 1.007/0.990 | 1.026/1.051 | 0 | 0 |
| 1 | float32 | 64 | 1024 | 1.600/1.586 | 1.622/1.607 | 1.600/1.574 | 3 | 0 |
| 1 | float64 | 64 | 1024 | 1.578/1.569 | 1.556/1.558 | 1.528/1.535 | 3 | 0 |
| 2 | float32 | 16 | 32 | 0.825/0.830 | 0.822/0.852 | 0.843/0.856 | 0 | 0 |
| 2 | float64 | 16 | 32 | 0.880/0.851 | 0.850/0.847 | 0.842/0.835 | 0 | 0 |
| 2 | float32 | 16 | 128 | 0.918/0.852 | 0.929/0.834 | 0.917/0.947 | 0 | 0 |
| 2 | float64 | 16 | 128 | 0.863/0.863 | 0.881/0.869 | 0.903/0.920 | 0 | 0 |
| 2 | float32 | 64 | 1024 | 1.599/1.571 | 1.655/1.698 | 1.665/1.670 | 3 | 0 |
| 2 | float64 | 64 | 1024 | 1.506/1.571 | 1.569/1.528 | 1.487/1.511 | 3 | 0 |
| 4 | float32 | 16 | 32 | 0.876/0.871 | 0.836/0.834 | 0.836/0.812 | 0 | 0 |
| 4 | float64 | 16 | 32 | 0.832/0.855 | 0.865/0.840 | 0.850/0.853 | 0 | 0 |
| 4 | float32 | 16 | 128 | 1.001/1.003 | 1.019/1.013 | 1.011/1.021 | 0 | 0 |
| 4 | float64 | 16 | 128 | 1.042/1.016 | 1.065/1.050 | 1.075/1.023 | 0 | 1 |
| 4 | float32 | 64 | 1024 | 1.698/1.659 | 1.716/1.678 | 1.724/1.705 | 3 | 0 |
| 4 | float64 | 64 | 1024 | 1.595/1.630 | 1.654/1.646 | 1.548/1.527 | 3 | 0 |
| 8 | float32 | 16 | 32 | 0.861/0.849 | 0.845/0.849 | 0.840/0.808 | 0 | 0 |
| 8 | float64 | 16 | 32 | 0.848/0.863 | 0.854/0.837 | 0.820/0.838 | 0 | 0 |
| 8 | float32 | 16 | 128 | 1.007/0.987 | 1.044/1.011 | 1.032/1.026 | 0 | 0 |
| 8 | float64 | 16 | 128 | 1.027/1.002 | 1.039/1.064 | 1.038/1.070 | 0 | 0 |
| 8 | float32 | 64 | 1024 | 1.628/1.590 | 1.548/1.667 | 1.646/1.630 | 3 | 0 |
| 8 | float64 | 64 | 1024 | 1.655/1.612 | 1.672/1.689 | 1.599/1.577 | 3 | 0 |

A/A flags: **1**; selected-thread full-stream slowdown flags: **30**. All positive/negative controls and failed screens are retained. A/A deviations expose run noise/order/host drift; short/fallback wrapper overhead is also visible and cannot be interpreted as a decomposition saving. Gain-to-loss reversals from repeated thread-1 gain to higher-thread slowdown: **0**. Known losing shapes are counted separately from those reversals. Any such reversals or placebo flags prevent integration confidence. Exact flags are in verification.json.

## Spectral and negative-control results

Spectral entries retain both repeat-batch speedups and energy-relative Gram differences. Numerical pass does not imply a performance pass. All controls use their instrumented intended paths.

| dtype | ell | d | spectrum / edge | batch 1 / 2 speedup | Gram gap / input energy |
|---|---:|---:|---|---|---:|
| float32 | 16 | 64 | spectrum_rank_two | 0.915/0.965 | 3.2015e-07 |
| float32 | 16 | 64 | spectrum_decay_1e16 | 0.890/0.912 | 2.5261e-07 |
| float32 | 16 | 64 | spectrum_tie | 0.915/0.920 | 4.3548e-07 |
| float32 | 16 | 64 | spectrum_near_tie | 0.899/0.900 | 3.4877e-07 |
| float32 | 16 | 64 | spectrum_zero | 0.888/0.895 | 0 |
| float32 | 16 | 512 | spectrum_rank_two | 1.375/1.375 | 1.1046e-06 |
| float32 | 16 | 512 | spectrum_decay_1e16 | 1.324/1.405 | 7.5888e-07 |
| float32 | 16 | 512 | spectrum_tie | 1.381/1.400 | 6.6171e-07 |
| float32 | 16 | 512 | spectrum_near_tie | 1.347/1.343 | 4.7575e-07 |
| float32 | 16 | 512 | spectrum_zero | 0.872/0.912 | 0 |
| float64 | 16 | 64 | spectrum_rank_two | 0.951/0.923 | 2.0835e-15 |
| float64 | 16 | 64 | spectrum_decay_1e16 | 0.900/0.909 | 6.4683e-16 |
| float64 | 16 | 64 | spectrum_tie | 0.924/0.924 | 1.0921e-15 |
| float64 | 16 | 64 | spectrum_near_tie | 0.928/0.906 | 7.3289e-16 |
| float64 | 16 | 64 | spectrum_zero | 0.963/0.909 | 0 |
| float64 | 16 | 512 | spectrum_rank_two | 1.292/1.343 | 1.195e-15 |
| float64 | 16 | 512 | spectrum_decay_1e16 | 1.388/1.293 | 8.1837e-16 |
| float64 | 16 | 512 | spectrum_tie | 1.307/1.300 | 1.2313e-15 |
| float64 | 16 | 512 | spectrum_near_tie | 1.319/1.311 | 9.4377e-16 |
| float64 | 16 | 512 | spectrum_zero | 0.876/0.910 | 0 |
| float32 | 1 | 16 | edge | 0.884/0.870 | 0 |
| float32 | 8 | 3 | edge | 0.992/0.974 | 0 |
| float64 | 1 | 16 | edge | 0.857/0.887 | 0 |
| float64 | 8 | 3 | edge | 0.991/1.013 | 0 |

| phase | threads | dtype | tiny | fallback | no-factor short | empty |
|---|---:|---|---|---|---|---|
| shape | 1 | float32 | 0.894/0.875 | 0.983/0.987 | 0.972/0.960 | 0.928/0.940 |
| shape | 1 | float64 | 0.892/0.887 | 1.012/0.991 | 0.965/0.981 | 0.923/0.938 |
| threads | 1 | float32 | 0.913/0.900 | 1.003/0.950 | 0.963/0.977 | 0.932/0.934 |
| threads | 1 | float64 | 0.862/0.852 | 0.991/1.048 | 0.971/0.963 | 0.918/0.926 |
| threads | 2 | float32 | 0.909/0.880 | 0.973/0.994 | 0.971/0.971 | 0.937/0.935 |
| threads | 2 | float64 | 0.842/0.893 | 0.950/0.948 | 0.968/0.975 | 0.920/0.932 |
| threads | 4 | float32 | 0.860/0.864 | 0.980/0.963 | 0.965/0.962 | 0.941/0.930 |
| threads | 4 | float64 | 0.845/0.868 | 1.003/0.991 | 0.968/0.995 | 0.925/0.927 |
| threads | 8 | float32 | 0.889/0.873 | 0.970/0.980 | 0.953/0.964 | 0.935/0.938 |
| threads | 8 | float64 | 0.908/0.847 | 1.025/0.968 | 0.994/0.966 | 0.924/0.940 |

## Limits, execution and handoff

The measurements characterize this host and thread configuration only. Width/height trends must be read alongside constant-n, repeated-batch and placebo results, not collapsed to a single aspect-ratio threshold. Global tied/decaying spectra and absolute bounds do not certify tiny spectral components or stability over unbounded streams. GPU behavior is NOT_RUN.

Observed width trend at fixed ell=16: d=32/64 loses on four-shrink streams, d=128 is near parity, d>=256 gains in the sampled full streams. The fixed-d=1024 scan does not support treating ell changes as a universal linear gain. The selected near shape (16,128) loses at two threads while its one-thread repeated timings were near parity. The winning (64,1024) shape retains sampled gains across these thread counts, but that is not a validated dispatch boundary. One float64 A/A block at four threads reports paired speedup 0.9415, exposing a false apparent slowdown despite identical implementations; it is a descriptive exploratory flag without multiplicity correction. New-loss configurations relative to thread 1: 6; no repeated one-thread gain switched to a screened higher-thread loss in this run.

All seven subprocesses completed with exit code zero under finite per-process/total bounds. Temporary clean checkout removed after artifact copy. No mathematical/runtime failures or repair retries were observed in this run. Existing CPU validator queries availability=false using CPU-only torch; no accelerator allocation, tensors/kernels, Colab/TPU, Actions, billing, new dependency, framework/sibling edits or self-merge.

A report-only helper quoting error was corrected during publication preparation; measured source and raw numerical/timing results were unchanged.

PyTorch emitted the existing missing-NumPy initialization warning; all used operations are torch CPU operations and no dependency was installed. See retained stderr logs.

Classifications: `{"gpu": "NOT_RUN", "integration": "INCONCLUSIVE / RETAIN_DIRECT_DEFAULT", "math": "PASS", "production_backend": "UNCHANGED", "shape_crossover": "INCONCLUSIVE", "threads": "SUPPORTED (execution and scoped CPU math)"}`.

GPT Architect: independently review the mathematical gates/path counts, source/provenance, orthogonal and constant-n scans, multi-thread/repeat/placebo evidence and negative findings; decide evidence acceptance and a separately scoped next study. No backend integration is requested.
