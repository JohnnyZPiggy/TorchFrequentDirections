# Issue #11: shape and thread study plan

This CPU-only EXPLORE extension reuses the immutable Issue #9 prototype, explicit
float64 Gram/full-SVD/projection oracle and balanced paired timing helpers. It
does not change production or propose a backend dispatch threshold. Issue #11
is the GPT-scoped hypothesis; scientific acceptance remains with GPT Architect.

R1: Earlier measurements changed both ell and d. They cannot identify independent
width, height or threading effects. Fast FD also changes processed row count when
holding the number of full-buffer shrinks fixed.

R2: The QR/small-SVD identity and finite-precision risks remain as derived in
[the original study](README.md). Hold ell=16 while scanning d=32,64,128,256,512,1024;
hold d=1024 while scanning ell=8,16,32,64. Four-shrink streams terminate at ell-1
and ell+1 occupied rows. Separate n=257 controls expose the varying-data-volume
confound. The exact row-count formula uses retained=min(ell-1,d), including fallback.

R3: Every measured configuration must first pass separate direct/candidate FD
inequalities for all integer 0<=k<ell, k<=d and cross-variant Gram comparison.
Tolerance remains 32 eps(dtype) max(n,d,2ell,1) times input squared Frobenius energy.
Path instrumentation verifies full-shrink, snapshot-SVD, QR and total SVD counts.
Rank-two, decay to 1e-16, tied/epsilon-spaced threshold spectra, zeros, noncontiguous
inputs, empty updates, ell=1 and ell>d have separate gates and timings. Prefix,
rowwise and irregular partition checks cover snapshots and continued updates.
Tiny-eigenvalue relative accuracy is NOT_ESTABLISHED by this absolute contract.

Priority: complete Stage 1 and controls, then targeted thread processes, then
publish all completed/failed/NOT_RUN outcomes within the Issue's 3600-second batch.
The shape phase includes spectral controls and nonadditive stage diagnostics.
For Stage 2 choose three distinct shapes using Stage-1 batch-one median scores
(median over dtypes and both four-shrink occupancies): minimum, closest to one
among remaining shapes, and maximum among remaining shapes. This selection is
exploratory and data-dependent; the second batch is not used for selection.
Fresh processes test intra-op 1,2,4,8 with inter-op=1; OMP/MKL/OpenBLAS environment
limits match the process's thread setting. No CPU affinity or host turbo changes.
CPU affinity, cgroup budget, WSL/CPU/build details and thread settings are recorded.

Each stage completes its gates before timing. Every block has two batches with
two warmups per implementation and 20 balanced randomized AB/BA pairs per batch;
batch two reverses workload order. Inputs remain identical across variants and
repeated thread settings. Whole-stream timing includes fresh state, validation,
chunk iteration, empty updates, clearing and final snapshot. Factor-only timing
excludes state construction. Tiny, short/no-factor, empty, m>d fallback, and
direct/direct A/A placebos are explicit controls. Fast FD no-shrink finalization
uses the existing raw occupied-row copy and must perform zero decompositions.

Predeclared exploratory screen: both batches paired median speedup >1.05 and
descriptive bootstrap 95% lower bound >1.0. Placebo stability means neither
batch meets that same improvement screen nor the symmetric slowdown screen
(median<1/1.05, upper<1). A selected shape is only a local candidate if its
repeated full-stream and constant-n blocks pass and placebos are stable. Any
thread regression or control drift forces RETAIN DIRECT DEFAULT / INCONCLUSIVE
for integration. No multiple-testing correction; intervals are exploratory,
single-host evidence and cannot establish a universal crossover or GPU behavior.

```sh
python -m experiments.qr_svd.shape_threads shape /outside/shape.json
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE \
  python -m experiments.qr_svd.shape_threads threads /outside/thread4.json \
  --threads 4 --selection /outside/shape.json
```

Use a clean checkout, existing CPU PyTorch and an external timeout of 920 seconds
per process (the reused driver enforces 900 seconds internally). No dependency,
accelerator API or framework modification is needed. The external batch runner
also enforces the total remaining Issue deadline and records skipped stages.

Completed study: [Issue #11 evidence](../../reports/cpu/issue11-20261021/summary.md).
MATH PASS; 1/2/4/8-thread execution completed. Shape crossover/integration remains
INCONCLUSIVE under the repeated-screen/control criteria; retain direct default.
GPU NOT_RUN. GPT acceptance is pending.
