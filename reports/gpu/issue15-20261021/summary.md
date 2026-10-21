# Issue #15: pure shrink capture, real fusion and full-workload limits

**CPU math PASS; CUDA partial-shrink math/codegen PASS. FULL_STREAM_COMPILED_NOT_AVAILABLE. End-to-end compile screen FAIL / INCONCLUSIVE; retain production Direct SVD.**

Base `907ffc725e0dd281bb6a7d60d435f6f37356240e`; exact clean CPU/T4 tested source `0fbf27bf23d349aafe2bd14e4ffbd0f934f395f6`. Publication adds docs/evidence only. Production package/default API, finite checks, clearing, public pickle behavior and immutable SAE snapshot are unchanged. GPT owns acceptance/merge.

[Rationale/reproduction](../../../experiments/compile_fd/README.md) · [raw original download](fd-compile.json) · [all timing blocks](benchmark.csv) · [correctness](correctness.json) · [graph/kernel diagnostics](compile-diagnostics.json) · [independent audit](verification.json) · [controller](colab-control.json) · [independent cleanup](host-execution.json) · [manifest](manifest.json)

## Scope and CPU diagnosis

47 clean CPU unit tests PASS; standalone AOT eager graph mathematics PASS across both dtypes and changed widths; 131 SAE hashes and controller preflight PASS. Pure tensor shrink retains normalized scaling/ell-th subtraction, clamp, sqrt, Vh scaling, fresh output and no-grad; SVD, finite Python bool checks, loops/copy/zero/snapshots stay outside the compiled subgraph. Eager test-wrapper outputs match production in CPU regressions, including ell=1, ell>d, m>d, m<ell, zero/rank/decay/ties, empty, overflow, alias and error paths.

CPU Dynamo explain: pure shrink **1 graph/0 breaks**, guarded factor **2 graphs/1 break**, whole stream **4 graphs/3 breaks**. Explain identifies Tensor.item/scalar reads in finite Python bool checks; the retained stderr points at shrink.py:32. Separate fullgraph factor/stream attempts fail with Unsupported / "Unexpected type in sourceless builder"; these are distinct observations, not proof SVD itself is unsupported. capture_scalar_outputs/suppress_errors were not changed. AOT eager emits 2 captured graphs; it is not Inductor fusion. Actual local CPU Inductor fails InvalidCxxCompiler: no g++/clang/C++ compiler. No dependency was installed.

CUDA compiles only the pure helper with counted real Inductor, fullgraph=True, dynamic=True, mode=default. Full CUDA stream fullgraph was not claimed/tested after CPU entrypoint failures. Whole update+sketch timings below invoke eager Direct SVD/state plus compiled shrink. CPU eager and CUDA eager use the same isolated Direct wrapper, not a QR algorithm; production equivalence is tested separately. No implicit eager fallback on compiled-helper failures.

## CUDA coverage and actual fusion

Real Tesla T4 capability [7, 5], Python 3.13.15, torch 2.11.0+cu130, CUDA 13.0, driver 580.82.07. Float32/64 QR/SVD probes and actual CUDA Inductor output probes PASS. CPU baseline: Intel(R) Xeon(R) CPU @ 2.00GHz, affinity [0, 1], intra/inter-op=1; full BLAS build/thread environment recorded. TF32 disabled, matmul precision highest, compile_threads=1, unique fresh Inductor/Triton cache directories.

4 backend graphs / 4 generated kernels; 0 new graphs during primary steady timing. Dynamic shapes/dtypes and fallback branch reuse are tracked per configuration. The graph excludes linalg_svd/finite checks/state loops; CPU FX capture containing an SVD operator is not SVD fusion.

| dtype | d | pure eager / partial kernel launches | full eager / partial launches |
|---|---:|---|---|
| float32 | 128 | 10 / 1 | 845 / 800 |
| float64 | 128 | 10 / 1 | 879 / 834 |
| float32 | 512 | 10 / 1 | 709 / 664 |
| float64 | 512 | 10 / 1 | 745 / 700 |

Actual Chrome-trace `kernel` category counts and generated Triton kernel names substantiate pointwise shrink fusion (10 launches to 1 in the profiled cases). Full workloads retain hundreds of external SVD kernels and host finite-check/dispatch waits; no whole-FD fusion claim. Profiler is intrusive/nonadditive. Aggregate CUDA runtime/host-op diagnostics are retained, raw compiled caches/private traces are excluded. Runtime event detail is capped at the first 120 selected runtime events per trace; kernel counts use the full trace.

## Cold versus steady results

Float32/64 first compile+execute probes: 4.775256 / 0.706678 seconds. Including subsequent first calls that created new graphs, measured codegen/first-call wall totals **6.332792 s**. Imports/source loading/benchmarking are excluded from that attribution; total remote process wall is 197.797 s. Cached guard-call timings are separately labelled, not counted again as compilation. No steady compilation was hidden in the primary timing.

30 configurations and 102 paired timing blocks complete, 0 NOT_RUN. Each block has two warmups/role and 20 balanced seeded randomized AB/BA pairs. Raw ordered sample arrays, paired ratios, median/p95/SD and 2000-draw descriptive intervals independently recomputed. GPU synchronization before/after every timed call; CPU/GPU inputs resident, allocation/state construction included for whole streams, H2D/D2H/generation excluded. CUDA-event intervals include dispatch gaps and are not pure kernel times. Twenty event/peak memory samples per CUDA variant/configuration and nearest-rank p95 memory are in raw evidence. Memory is PyTorch peak allocated bytes, not total NVML/cuSolver/reserved-device use; maximum observed per-case p95 allocated memory was 31.663 MiB for both CUDA variants.

Entries below are separately measured paired medians. CPU baselines vary between comparison blocks and GPU latency also changes with pairing context (see audit), so these ratios must NOT be multiplied or their spread attributed to compilation. Host/load/frequency/interleaving causes are NOT_DIAGNOSED. For ell=64,d=2048,float32,final=63, CUDA eager median is 41.146 ms when paired with CPU versus 19.466 ms when paired with CUDA partial; CPU/partial ratio 5.713 is not a compile gain. The same-device compile ratio in its paired block is 1.012. This material protocol dependence limits device/composite comparisons. Intervals are exploratory, unadjusted for multiple comparisons, one host/session and fixed workload-block order.

| dtype | ell,d | final | CPU/CUDA eager | CUDA eager/partial [95%] | CPU/CUDA partial |
|---|---|---:|---:|---|---:|
| float32 | 16,64 (stream) | 15 | 0.287 | 1.010 [0.980,1.042] | 0.436 |
| float32 | 16,64 (stream) | 17 | 0.457 | 1.006 [0.997,1.022] | 0.438 |
| float64 | 16,64 (stream) | 15 | 0.100 | 1.002 [0.998,1.004] | 0.104 |
| float64 | 16,64 (stream) | 17 | 0.106 | 1.006 [1.002,1.014] | 0.106 |
| float32 | 16,128 (stream) | 15 | 0.514 | 0.996 [0.977,1.016] | 0.513 |
| float32 | 16,128 (stream) | 17 | 0.496 | 1.016 [0.983,1.040] | 0.497 |
| float64 | 16,128 (stream) | 15 | 0.106 | 1.010 [1.006,1.012] | 0.112 |
| float64 | 16,128 (stream) | 17 | 0.107 | 1.008 [1.004,1.010] | 0.111 |
| float32 | 16,512 (stream) | 15 | 0.802 | 1.023 [1.006,1.053] | 0.818 |
| float32 | 16,512 (stream) | 17 | 0.766 | 1.034 [1.023,1.042] | 0.810 |
| float64 | 16,512 (stream) | 15 | 0.191 | 1.006 [0.996,1.010] | 0.188 |
| float64 | 16,512 (stream) | 17 | 0.189 | 1.007 [1.005,1.012] | 0.185 |
| float32 | 8,4 (stream) | 7 | 0.282 | 0.920 [0.887,0.949] | 0.252 |
| float32 | 8,4 (stream) | 9 | 0.276 | 0.875 [0.870,0.895] | 0.238 |
| float64 | 8,4 (stream) | 7 | 0.158 | 0.942 [0.920,0.963] | 0.168 |
| float64 | 8,4 (stream) | 9 | 0.179 | 0.944 [0.913,0.960] | 0.178 |
| float32 | 2,8 (stream) | 1 | 0.277 | 1.044 [1.035,1.063] | 0.292 |
| float32 | 2,8 (stream) | 3 | 0.274 | 1.046 [1.024,1.068] | 0.289 |
| float64 | 2,8 (stream) | 1 | 0.171 | 1.030 [1.014,1.046] | 0.198 |
| float64 | 2,8 (stream) | 3 | 0.185 | 1.035 [1.021,1.041] | 0.208 |
| float32 | 32,1024 (stream) | 31 | 1.312 | 1.019 [1.008,1.022] | 2.147 |
| float32 | 32,1024 (stream) | 33 | 1.931 | 1.017 [1.011,1.024] | 2.080 |
| float64 | 32,1024 (stream) | 31 | 0.286 | 1.003 [1.000,1.004] | 0.282 |
| float64 | 32,1024 (stream) | 33 | 0.278 | 1.003 [1.002,1.006] | 0.291 |
| float32 | 64,2048 (stream) | 63 | 2.828 | 1.012 [1.006,1.018] | 5.713 |
| float32 | 64,2048 (stream) | 65 | 2.735 | 1.013 [1.005,1.021] | 4.566 |
| float64 | 64,2048 (stream) | 63 | 0.852 | 1.002 [1.001,1.003] | 0.867 |
| float64 | 64,2048 (stream) | 65 | 0.825 | 1.000 [0.997,1.003] | 0.833 |
| float32 | 16,128 (no_factor) | 15 | 0.387 | 0.932 [0.908,1.011] | 0.398 |
| float64 | 16,128 (no_factor) | 15 | 0.393 | 1.023 [1.001,1.038] | 0.366 |

Full-workload >1.05/lower>1 screens passed: **0**; A/A flags: **1** (symmetric >5% slowdown also flagged). The float32 no-factor A/A block is noisy, while m>d controls regress. Pure shrink microbenchmarks improve locally but factor/full-FD results do not justify integration: shrink paired ratios 1.545–1.610 and isolated factor ratios 1.005–1.049 in the profiled subset. These component ratios are not full-workload ratios. Meaningful full-workload break-even is NOT_ESTABLISHED because no workload passes the predeclared performance/control screen. Do not multiply microbenchmark savings or profiler costs into end-to-end savings.

## Integrity, cleanup, failures and handoff

Downloaded 691185 bytes; SHA-256 `488f3c0de0f184d8cedc8da876b08ba9b87feca73bbf7b99a51cd07b93c8056a` matches emitted marker, controller download, host and Git evidence bytes. Driver/allowlist/import hashes verified against exact source. Fresh authenticated CLI 0.7.4 CCU-info returned zero sessions and positive existing units before one allocation. Controller stop and zero readback succeeded; a separate fresh authenticated readback again returned zero. Task/allocation/work/cleanup caps 600/540/420/60 s, outer controller 620 s plus stop/readback reserve within 660 s. Allocation lifecycle 229.454 s, host 233.634 s. Existing credits only; no purchase, billing-setting change, top-up, allocation retry, Actions, dependency, Framework/sibling edit, GPU-time GitHub write or self-merge.

Temporary clean checkouts and transport/cache directories were removed after verified artifact copy.

Preallocation repair: the first standalone source pin omitted the oracle’s transitive prototype module (pilot-source-import.json); fixed once, then all clean CPU gates repeated. A final CPU-host metadata addition also received complete clean CPU checks before T4. Expected fullgraph errors and unavailable CPU Inductor were preserved rather than hidden. No post-allocation source repair/reallocation.

Limitations: explicit compiled-CUDA correctness covers the measured random/control grid; CPU spectral/edge tests do not establish every GPU near-null relative component or arbitrary streams. Full CUDA stream capture is not available here; this is partial-helper evidence. Production Direct SVD stays unchanged. GPT Architect independently reviews math, coverage, actual fusion versus SVD limits, cold/steady scope, pairing-context drift/noisy controls and cleanup, then decides evidence acceptance/next scope.
