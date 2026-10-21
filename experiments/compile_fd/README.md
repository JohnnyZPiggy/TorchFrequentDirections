# Issue #15: pure tensor shrink compile experiment

This isolated EXPLORE study follows the scoped Issue #15 design. Production
Direct SVD, public API/default, finite checks, clearing/pickle behavior and SAE
Framework remain unchanged. GPT Architect decides scientific acceptance/merge.

R1: FD has external SVD, sequential state updates and mandatory finite tensor
reductions converted to Python bool. Capturing an FX graph is not proof of
Inductor code generation, kernel fusion or full-workload speedup. Earlier QR
results are a different algorithmic choice; this experiment uses Direct SVD only.

R2: Isolate normalized singular-value scaling, subtraction of sigma_ell squared,
clamp/sqrt, Vh row scaling and newly allocated ell×d output in `tensor_shrink`.
The exact-arithmetic formula is unchanged. Empty/m<ell/ell>d shapes retain zero
padding/exact fallback. Validation, SVD, dynamic loops, copy/zero and snapshots
remain eager in `TensorShrinkFD`; only its explicitly supplied shrink function
may compile. No finite check is removed to force fullgraph capture.

R3: CPU tests cover float32/64, ell=1/ell>d, m>d/m<ell, zero/empty/rank/decay/ties,
overflow-safe scaling, no-grad/alias safety and input/output/backend failures.
CPU AOT eager graph output and actual CUDA generated output use independent
CPU float64 Gram PSD/spectral-tail/right-projection bounds for all eligible k,
with the fixed 32 eps(dtype) max(n,d,2ell,1) times input-energy tolerance and
cross-variant Gram Frobenius gap. This absolute contract establishes neither
tiny-eigenvalue relative accuracy nor arbitrary-stream stability.

Local diagnosis preserves graph-break and fullgraph failure evidence. The local
CPU lacks a C++ compiler: AOT eager graph math passes, but CPU Inductor is
UNAVAILABLE (InvalidCxxCompiler). No compiler/dependency was installed. CPU FX
capture containing SVD is not evidence of fused SVD. CUDA probes separately
verify real T4/QR/SVD and Inductor output before timing. CUDA compiles only the
pure helper with fullgraph=True, dynamic=True, mode=default; complete FD is
**not** a fullgraph-compiled model. No suppress_errors or scalar-capture change.

Whole-workload cells use the same experimental Direct wrapper on CPU eager,
CUDA eager and CUDA partial-shrink compile. Production eager parity is tested
separately. Three ratios are independently paired: CPU/CUDA eager, CUDA eager/
partial and CPU/CUDA partial. Separate blocks can have different CPU timing
baselines and GPU latency changes with pairing context; never multiply the ratios
or attribute cross-block drift to compile.
GPU inputs are resident; allocation/state initialization is included in whole
calls, generation/H2D/D2H excluded. Synchronization brackets every timed call;
two warmups/role precede 20 balanced randomized AB/BA pairs. CUDA events include
dispatch gaps; memory samples/p95 and profiler diagnostics have separate scopes.

Compilation uses new temporary Inductor/Triton cache directories and one compiler
thread. Timed first subgraph calls separate codegen/first execution from warmed
latency; guard/cache-hit calls are labelled separately. Backend graph counts
track reuse/recompiles. Profiler exports are reduced to aggregate actual kernel
launch/name and host/runtime diagnostics, with no raw caches/trace uploads.
Profiler costs are intrusive/nonadditive. Kernel reduction or a microbenchmark
gain cannot be substituted for measured complete FD savings.

Predeclared exploratory performance screen: mathematical gates pass, stable A/A
placebo, complete-workload CUDA eager/partial median>1.05 and descriptive paired
bootstrap lower>1. No multiple-testing correction or independent-session
replication. Cold-cost break-even is meaningful only for a stable positive
whole-workload gain; this run establishes none.

Run CPU gates from a clean checkout with existing torch:

```sh
python scripts/validate_cpu.py
python scripts/check_snapshot.py
SAE_CPU_ONLY=1 SAE_LOCAL_SOURCE="$PWD" SAE_CANDIDATE_SHA="$(git rev-parse HEAD)" \
  SAE_ARTIFACT_DIR=/tmp/fd15-cpu python experiments/compile_fd/driver.py
```

The exact tested source is `0fbf27bf23d349aafe2bd14e4ffbd0f934f395f6`. The committed
standalone driver and artifact allowlist are transported by the immutable
controller. Fresh authenticated zero assignments/positive existing credit and
the no-purchase boundary preceded one T4 allocation. Limits were task/allocation/
work/cleanup 600/540/420/60 seconds; bounded compile calls and timing cutoff
protect cleanup. No allocation retry. Controller stop plus authenticated zero
readback and separate independent fresh authenticated zero readback succeeded.
No further allocation is required or authorized by this reproduction note.

[Raw results, graphs/kernels, timings, cold cost and cleanup](../../reports/gpu/issue15-20261021/summary.md)
show real shrink fusion but no accepted complete-workload compile gain. Production
Direct SVD stays unchanged; GPT review is pending.
