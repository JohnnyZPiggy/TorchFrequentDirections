# Issue #17 follow-up: isolated graph probes and full strict streaming benchmark

Refs #17; based on merged PR #18/main `a9a9dc24f69f02f6305b17d83e83fa357d7ad530`.
Tested exact source `d62ae254b0c9252cb68f13ffff8017ae79c32dc6`; later evidence/documentation commits do not change
tested executable code. Engineer EXPLORE results await GPT Architect acceptance.

## Completed validation

Clean WSL CPU: 52 tests, 216 production-oracle observations, 72 candidate fixture
gates, eight separate CPU stage processes; immutable SAE snapshot 131 files PASS.
Failure/timeout/malformed-child tests prove later children still start. Production
and ScratchFD are byte-unchanged; strict finite predicates, invalid-input atomicity,
snapshot ownership, physical zero clearing and pickle storage are preserved.
No default/API/framework/sibling/billing/Actions edits, no dependencies added.

Real T4 fresh benchmark: status **PASS**, CUDA RNG health
**PASS** after all graph children;
72 GPU fixture gates, 26/26 workload shapes measured,
110 paired blocks independently recomputed (20 balanced pairs each).
CPU+CUDA fixture/workload comparisons: 392 individual float64 oracles,
4224 every-k covariance/projection bounds passed. Tolerance remains
32 eps(input dtype) max(n,d,2ell,1) ||A||_F^2; tested cases are not an arbitrary-stream
proof or near-zero relative-accuracy guarantee. GPU injected backend-fault cases
are not separately executed; CPU safety/fault tests pass.

## Independent graph-stage outcomes

Each stage/dtype has a fresh bounded exec, no shared CUDA process state; first
operation and context-exit errors are separately retained. All stages omit host
strict finite/error decisions and are **PROBE_ONLY / NON_EQUIVALENT to full update**.
SVD uses preallocated out U/S/Vh. Two side-stream warmups, fixed 32x64 input,
ell=16, stable buffers; successful replay requires five independently changed
inputs matching eager output. These probes do not establish full-stream graphs.

| Stage | Dtype | Actual status | Last phase |
|---|---|---|---|
| pointwise | float32 | CAPTURE_REPLAY_PASS_PROBE_ONLY | complete |
| pointwise | float64 | CAPTURE_REPLAY_PASS_PROBE_ONLY | complete |
| svd | float32 | CAPTURE_FAILED_PROBE_ONLY | capture |
| svd | float64 | CAPTURE_FAILED_PROBE_ONLY | capture |
| svd_shrink | float32 | CAPTURE_FAILED_PROBE_ONLY | capture |
| svd_shrink | float64 | CAPTURE_FAILED_PROBE_ONLY | capture |
| buffer | float32 | CAPTURE_FAILED_PROBE_ONLY | capture |
| buffer | float64 | CAPTURE_FAILED_PROBE_ONLY | capture |

See [exact primary/teardown errors and cold/replay samples](graph-diagnostics.json).
Both pointwise probes completed five changed-input replays; all six SVD-containing
probes first failed during `cusolverDnSgesvdj` (float32) or `cusolverDnDgesvdj`
(float64), reporting CUSOLVER_STATUS_EXECUTION_FAILED. Context teardown then
reported cudaErrorStreamCaptureInvalidated. Warmups passed; localization is the
captured SVD call under the preallocated-out API, before shrink/reset operations.
Do not infer universal cuSOLVER incompatibility; conclusions apply to this exact
torch/CUDA/device/API/stream setup. Capture failure is nonblocking to other probes
and the fresh eager benchmark. No backend patch or capture/allocation retry.

## Complete update + sketch timing and controls

Fresh state, all updates and final sketch, original strict checks and physical
clearing are inside each synchronized host-wall call. Same resident inputs and
partitions for E0 original CUDA eager and E1 ScratchFD. Generation/H2D excluded;
CPU one-thread eager is a distinct device/context comparison. No multiplication
of cross-context ratios. Two warmups + twenty balanced randomized AB/BA pairs.
Shape order, context order and pair order are seeded/randomized and preserved.
A/A before/after brackets both contexts for each shape, with global A/A controls.
First-workload calls and cold CUDA setup are separate fields, not added to steady
samples. Capture has independent cold/replay timings in its own child processes.

**0 exploratory >1.05/lower-bound>1 stable screens;
11 shapes INCONCLUSIVE** due to controls. More than 5%
A/A median deviation, baseline-anchor drift or pairing-context CUDA drift rejects
a speed claim; final global A/A can invalidate all screens. Descriptive bootstrap
is not independent-session or multiple-comparison-adjusted evidence. No general
acceleration claim or production backend promotion follows from this batch.

Across all 26 shapes, CUDA eager/Scratch paired medians range 0.9530--1.0257,
below the 1.05 screen even before drift rejection. Bracketing eager median drift
ranges -9.39% to +6.62%; CPU-paired CUDA versus CUDA-paired CUDA drift ranges
-1.88% to +90.64%. These are contextual observations, not a causal diagnosis.
The global A/A medians are 0.99094 before and 1.00017 after. Thus a globally calm
placebo does not establish every within-shape/context baseline is stable.

[benchmark.csv](benchmark.csv) gives per-shape/context median, p95, sample SD,
paired intervals, drift and status. [Raw GPU JSON](fd-isolated.json) retains every
ordered sample, seed, cold first call, initial/final A/A and occupancy path.
[Kernel diagnostics](kernel-diagnostics.json) are separate intrusive profiles;
runtime wait/launch detail is capped at 120 by the existing helper. CPU op counts
include isfinite/item/SVD. Clock/temperature observations are not causal controls.
All four profiled shapes retain identical kernel counts for original and Scratch:
715/779/739/803. Each has 16 isfinite ops, 21 item ops and five SVD calls on both
paths. Local scratch reuse did not remove those launches or strict host decisions.
Profiler self-CPU times are dispatch observations, not device-only SVD time.
CPU model/affinity and allocation-level memory metrics were not captured in this
new worker; one-thread settings and exact runtime versions are retained. Do not
generalize CPU/GPU ratios to another CPU, transfer-inclusive workload or session.

| d | Dtype | Original kernels | Scratch kernels |
|---|---|---:|---:|
| 512 | torch.float32 | 715 | 715 |
| 64 | torch.float32 | 779 | 779 |
| 512 | torch.float64 | 739 | 739 |
| 64 | torch.float64 | 803 | 803 |

## Source, transport and cleanup

Run `fd17b-20261021-122737`, one NEW allocation only for this explicitly dispatched follow-up.
Fresh authenticated CLI 0.7.4 verified zero active assignments and existing balance
62.36 CU; managed existing-entitlement route only, no new monetary
commitment. Ending displayed balance 62.36 CU is not a precise
consumption claim (rounding/lag). Limits task/allocation/work/cleanup 600/540/420/60s,
host cap 660s. Allocation lifecycle 236.704s;
host elapsed 239.807s; scientific parent 213.371s.
Stop exit 0; controller authenticated zero plus separate independent authenticated
zero at 2026-10-21T12:31:30.945280+00:00. All GPU-time GitHub writes were avoided.

Original GPU download 765313 bytes SHA-256 `d0330028d92136ed479571db0ee2a05aa8d3d11114efbb7108cf812c4716ec0a`. Remote emitted marker,
controller download, host original and staged Git bytes are checked; source module,
driver and allowlist SHA-256 are retained in manifest/controller/audit. No credential,
session URL or raw profiler trace is published. Old failure evidence is unchanged.

Commands on the clean exact-SHA checkout:

```text
python scripts/validate_cpu.py
python scripts/check_snapshot.py
SAE_CPU_ONLY=1 SAE_CANDIDATE_SHA=d62ae254b0c9252cb68f13ffff8017ae79c32dc6 SAE_LOCAL_SOURCE=<clean-source> SAE_ARTIFACT_DIR=<out> python experiments/full_gpu/isolated_driver.py
python .sae/tools/compute/colab_control.py --repo-path <clean-source> --source-sha d62ae254b0c9252cb68f13ffff8017ae79c32dc6 --run-id fd17b-20261021-122737 --out <out> --driver experiments/full_gpu/isolated_driver.py --artifacts experiments/full_gpu/isolated_artifacts.json --cli /home/zhangru/.local/bin/colab --device T4 --task-seconds 600 --allocation-seconds 540 --cleanup-seconds 60 --kernel-timeout 30 --work-timeout 420
```

These record execution already performed, not permission to repeat allocation.

## Evidence and handoff

[Isolation and mathematical contract](../../../experiments/full_gpu/ISOLATED.md),
[raw CPU](cpu-full.json), [CPU regression](cpu_validation.json),
[snapshot](snapshot.json), [correctness](correctness.json), [audit](audit.json),
[controller](colab-control.json), [independent cleanup](host-execution.json),
[preflight](preflight.json), [local execution](local-execution.json),
[manifest](manifest.json). GPT Architect reviews actual math, control failures,
graph phase localization and scope before deciding acceptance or a next hypothesis.
Codex has not merged; Issue #17 is not prematurely closed.
