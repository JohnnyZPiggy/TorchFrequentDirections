# Issue #13: bounded T4 QR + Small SVD evidence

**Scoped GPU math VALIDATE_PASS; performance exploratory/INCONCLUSIVE for integration. Retain Direct SVD default.**

Base `0e27f957151e30824bffb46e220e02a5dcfa1f39`; exact clean CPU/T4 tested source `23b5aa13ef926ecd5df36908f2277fc6e1c77b00`. Publication adds evidence/docs only. Production package, finite checks, clearing/pickle behavior, CPU-only historic prototype, public API and pinned SAE snapshot remain unchanged.

[Research/reproduction](../../../experiments/qr_svd/CUDA_STUDY.md) · [raw downloaded results](fd-cuda-qr.json) · [correctness](correctness.json) · [all timings](benchmark.csv) · [download/source audit](verification.json) · [controller stop/readback](colab-control.json) · [independent readback](host-execution.json) · [manifest](manifest.json)

## Mathematical and CPU gate

A separate explicit experimental adapter subclasses the historic CPU QR algebra while calling the unchanged production initializer for CPU/CUDA state. The historic CPU constructor continues to reject CUDA. No implicit dispatch, production backend or finite-check edits. For C^T=QR and R^T=U diag(s) W^T, use Vh=W^T Q^T and the unchanged normalized ell-th threshold shrink. m>d/empty paths retain direct fallback.

44 clean CPU tests PASS, including independent FD bounds, exact parity with historic CPU QR, input atomicity/no-grad/alias behavior and injected QR/SVD/nonfinite failures. Standalone pinned adapter CPU gates and controller preflight PASS. 131 SAE file hashes PASS. Source hashes include production, adapter, historic algebra, oracle and timing helper.

Real Tesla T4, capability [7, 5], Python 3.13.15, torch 2.11.0+cu130, CUDA 13.0, driver 580.82.07. Both float32/float64 live QR/SVD reconstruction and orthogonality probes PASS on cuda:0; free/total memory, synchronization and environment are recorded. TF32 disabled, float32 matmul precision highest; CPU oracle threads=1/inter-op=1.

68 GPU fixture/factor records plus 38 benchmark math gates PASS. Both algorithms independently satisfy CPU float64 explicit Gram PSD/spectral-tail/right-projection inequalities for every integer 0<=k<ell, k<=d, then Gram Frobenius gap. Tolerance `32 eps(input dtype) max(n,d,2ell,1) ||A||_F²`, fixed before timing. Includes rank deficiency, severe decay, global ties/near ties, exact local tied shrink threshold, zero/empty/short input, ell=1/ell>d, genuinely noncontiguous target storage, multiple shrinks, prefix/continued updates, repeated independent snapshots and injected backend faults/invalid-input rejection. Tiny-singular-value relative accuracy remains NOT_ESTABLISHED.

## GPU paired performance

38 Direct/QR blocks measured; 4 direct/direct A/A blocks; 0 optional/budget blocks NOT_RUN. Two warmups per variant, 20 balanced randomized AB/BA pairs per block; synchronize before/after every timed call. Complete fresh-state update+sketch uses GPU-resident inputs, including finite checks, shrink, clearing and finalization. Generation and H2D/D2H are excluded; oracle copies outside timings. Factor-only timing excludes state construction. Intrusive QR/small-SVD/Vh/shrink diagnostics are NONADDITIVE. No pure-kernel/CUPTI or compile claims.

Raw trial order, all samples/paired ratios, medians/p95/sample SD and 2000-draw descriptive paired-bootstrap intervals are retained and independently recomputed. Exploratory screen: paired median direct/QR>1.05 and lower interval>1. No multiple-comparison correction or independent-session replication; this is no universal crossover or integration acceptance.

| dtype | m×d | ell | kind | n | paired speedup [descriptive 95%] | direct / QR median ms | screen |
|---|---|---:|---|---:|---|---|---|
| float32 | 32×64 | 16 | factor | 32 | 1.526 [1.507,1.543] | 2.351 / 1.549 | True |
| float32 | 32×64 | 16 | short | 83 | 1.351 [1.245,1.396] | 9.611 / 6.850 | True |
| float32 | 32×64 | 16 | long | 85 | 1.253 [1.242,1.316] | 8.213 / 6.448 | True |
| float32 | 32×128 | 16 | factor | 32 | 1.359 [1.329,1.378] | 1.387 / 1.021 | True |
| float32 | 32×128 | 16 | short | 83 | 1.238 [1.210,1.264] | 6.967 / 5.637 | True |
| float32 | 32×128 | 16 | long | 85 | 1.263 [1.229,1.295] | 8.123 / 6.530 | True |
| float32 | 32×512 | 16 | factor | 32 | 1.543 [1.497,1.562] | 1.399 / 0.904 | True |
| float32 | 32×512 | 16 | short | 83 | 1.360 [1.342,1.370] | 6.976 / 5.147 | True |
| float32 | 32×512 | 16 | long | 85 | 1.346 [1.322,1.382] | 7.937 / 5.843 | True |
| float32 | 64×1024 | 32 | factor | 64 | 1.019 [1.004,1.033] | 2.029 / 1.985 | False |
| float32 | 64×1024 | 32 | short | 163 | 0.975 [0.963,0.988] | 9.218 / 9.385 | False |
| float32 | 64×1024 | 32 | long | 165 | 0.996 [0.983,1.025] | 10.645 / 10.629 | False |
| float32 | 16×8 | 8 | factor | 16 | 1.004 [0.980,1.039] | 0.589 / 0.594 | False |
| float32 | 16×8 | 8 | short | 43 | 0.988 [0.966,1.006] | 3.729 / 3.769 | False |
| float32 | 16×8 | 8 | long | 45 | 0.984 [0.975,1.005] | 4.331 / 4.391 | False |
| float32 | 128×2048 | 64 | factor | 128 | 1.010 [1.005,1.018] | 4.383 / 4.339 | False |
| float32 | 128×2048 | 64 | short | 323 | 0.998 [0.994,1.002] | 17.741 / 17.840 | False |
| float32 | 128×2048 | 64 | long | 325 | 0.999 [0.985,1.004] | 19.737 / 19.711 | False |
| float32 | 32×128 | 16 | no_factor | 15 | 0.941 [0.895,1.007] | 0.435 / 0.441 | False |
| float64 | 32×64 | 16 | factor | 32 | 1.195 [1.189,1.209] | 6.904 / 5.788 | True |
| float64 | 32×64 | 16 | short | 83 | 1.191 [1.188,1.198] | 29.291 / 24.564 | True |
| float64 | 32×64 | 16 | long | 85 | 1.212 [1.204,1.217] | 32.031 / 26.471 | True |
| float64 | 32×128 | 16 | factor | 32 | 1.229 [1.226,1.231] | 7.186 / 5.852 | True |
| float64 | 32×128 | 16 | short | 83 | 1.203 [1.200,1.210] | 29.745 / 24.774 | True |
| float64 | 32×128 | 16 | long | 85 | 1.200 [1.197,1.203] | 32.436 / 27.024 | True |
| float64 | 32×512 | 16 | factor | 32 | 1.205 [1.201,1.209] | 6.923 / 5.740 | True |
| float64 | 32×512 | 16 | short | 83 | 1.209 [1.201,1.215] | 27.126 / 22.455 | True |
| float64 | 32×512 | 16 | long | 85 | 1.240 [1.236,1.255] | 30.817 / 24.824 | True |
| float64 | 64×1024 | 32 | factor | 64 | 0.974 [0.972,0.977] | 16.630 / 17.061 | False |
| float64 | 64×1024 | 32 | short | 163 | 1.016 [1.012,1.018] | 63.870 / 62.841 | False |
| float64 | 64×1024 | 32 | long | 165 | 1.000 [0.996,1.001] | 70.715 / 70.789 | False |
| float64 | 16×8 | 8 | factor | 16 | 0.997 [0.985,1.000] | 1.073 / 1.088 | False |
| float64 | 16×8 | 8 | short | 43 | 0.984 [0.973,0.992] | 5.828 / 5.911 | False |
| float64 | 16×8 | 8 | long | 45 | 0.996 [0.987,1.011] | 7.046 / 7.056 | False |
| float64 | 128×2048 | 64 | factor | 128 | 1.019 [1.019,1.020] | 47.590 / 46.642 | False |
| float64 | 128×2048 | 64 | short | 323 | 1.010 [1.009,1.011] | 172.083 / 170.457 | False |
| float64 | 128×2048 | 64 | long | 325 | 1.018 [1.016,1.019] | 190.288 / 186.948 | False |
| float64 | 32×128 | 16 | no_factor | 15 | 0.927 [0.892,0.986] | 0.426 / 0.443 | False |

A/A screen flags: 0 (including symmetric slowdown test); details in verification.json. Short/no-factor and m>d fallback controls remain in the tables. Do not interpret wrapper/noise differences in these controls as QR savings. Candidate performance claims require control-aware GPT review; production remains direct regardless of positive local screens.

Observed local result: m=32 cases show whole-stream gains (~1.19–1.36×); m=64/d=1024 and m=128/d=2048 streams are approximately at parity and fail the predeclared >1.05 screen. For float32 m=32,d=512,ell=16,n=85, paired speedup is 1.346 [1.322,1.382], with direct/QR medians 7.937/5.843 ms. Short/no-factor and fallback cases do not support QR savings; float64 short/no-factor shows overhead/variability. These shape-specific T4 findings do not follow the CPU crossover and do not justify a production threshold.

## Entitlement, source, download and cleanup

Fresh authenticated CLI 0.7.4 CCU-info readbacks verified zero assignments and positive existing compute-unit balance immediately before the single controller allocation. Only managed Colab assign/exec/download/stop/usage operations were invoked; no purchase/pay, top-up, billing changes or Cloud/Enterprise resources. Existing unit balance and after-run readback are preserved in host-execution.json. The official [Colab resource policy](https://research.google.com/colaboratory/faq.html#resource-limits) describes exhaustion reverting to free-tier limits/termination; no new paid entitlement was enabled.

Balance readbacks are rounded and may lag accounting; equal displayed balances are not evidence of zero compute-unit consumption. No purchase or incremental monetary commitment was made.

Pinned controller task/allocation/work/cleanup bounds 600/540/420/60 s; outer controller timeout 620 s leaves emergency stop/readback room within the 660 s host cap. One allocation attempt only; a local exclusive lock prevents retries. Kernel and live QR/SVD probes ran inside that same session before science/timing. No GitHub writes occurred during the session.

Downloaded artifact: 476872 bytes, SHA-256 `5b5546b5ab3ddb9914768ff6d428cc470aff31b99a5f5cf441d5dc459438626b`. Emitted remote size/hash, controller download hash and independent host hash agree. Controller driver and artifact-manifest hashes match exact Git blobs; all imported source hashes verified. Controller stop succeeded and authenticated zero assignments were confirmed; a separate fresh authenticated readback independently returned zero after stop. Lifecycle 93.148 s; host elapsed 96.877 s. No allocation retries, new dependencies, Framework/sibling edits, default changes or self-merge.

One report-helper quotation error was corrected during publication preparation; the tested CPU/T4 source and raw downloaded artifact were unchanged.

GPT Architect: review numerical scope/tolerances, exact tested/publication diff, timing fairness and negative controls, source/download identity and authenticated cleanup; decide evidence acceptance and next study. This engineer report does not approve its own PR.
