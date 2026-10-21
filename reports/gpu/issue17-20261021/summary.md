# Issue #17 bounded Fast FD GPU EXPLORE: failed graph probe, no performance result

**Engineer outcome: GPU_BATCH_FAIL_CLOSED; CPU_VALIDATED; GPU fixture correctness
PASS; full performance NOT_RUN.** Refs #17; do not close the issue as completed.
No production algorithm, default, finite-check semantics, clearing/pickle contract,
SAE snapshot, sibling project, billing setting or Actions changed. No merge.

Tested source `6a4900c25314af856ee4e37a59b875d05dd6d768`, dispatch main `396a998c37220cbe0ba2bc161e9464330a5efec4`.
Source was pushed before allocation; CPU and remote run used identical committed
driver/module blobs. Later commits contain documentation/evidence only.

## Actual evidence

- Clean WSL CPU: 50/50 tests, 216 production-oracle observations, and 72 candidate
  fixture gates passed. `check_snapshot.py`: 131 immutable SAE files verified,
  eight known template-link exceptions. Independent fault injection, invalid-input
  atomicity, all three partitions, no-grad, snapshots, exact state/pickle storage,
  severe decay/ties/extreme scale, ell=1 and ell>d passed on CPU.
- One real Tesla T4, capability 7.5, driver 580.82.07, Python 3.13.15,
  torch 2.11.0+cu130 / CUDA 13.0. Float32/64 QR/SVD probe passed. All 72 GPU
  fixture gates passed before graph capture, with exact same-device buffer parity,
  independent CPU float64 oracles, NaN/Inf rejection and nonmutating snapshots.
  Completed CPU+GPU candidate fixtures: 288 individual oracles / 1952
  per-k bounds. This is finite fixture evidence, not arbitrary-stream proof.
- First float32 fixed-shape `(32,64)`, ell=16 tensor-transition graph capture failed
  after 0.084929s: `CUDA error: operation failed due to a previous error
  during capture` / `cudaErrorStreamCaptureInvalidated`. No successful capture,
  no replay. The tensor diagnostic deliberately excluded strict host finite
  predicates and was always NON_EQUIVALENT to a production update.
- Next dtype setup (`torch.randn(..., device='cuda')`) raised
  `RuntimeError: Offset increment outside graph capture encountered unexpectedly.`
  The driver exited fail-closed before entering `study()`. Capturing graph errors
  in the same process did not protect subsequent CUDA work. We cannot locate the
  first failing SVD/shrink/reset operation or claim cuSOLVER is universally
  graph-incompatible from this error. No backend patch or retry was attempted.
- **0 A/A blocks, 0 AB/BA pairs, 0 shape timing cells, 0 kernel profiles** executed.
  CPU/GPU speedup, candidate gain, context drift, steady replay and stage timing
  are NOT_RUN, not zero-cost or negative speed measurements. No performance screen
  is accepted. The committed driver contains the randomized shape/context/pair
  and sandwich protocol, but that portion is unverified on T4.

## Meaning of the candidate

[Candidate and synchronization analysis](../../../experiments/full_gpu/README.md)
explains local scratch reuse: the same normalized shrink in the same operation
order, precomputing the threshold before overwriting the local quotient, and
writing products to fresh output storage. No persistent workspace, check removal,
buffer schedule change or stale zero tail. Strict input/factor/short-snapshot bool
reductions and SVD backend waits remain. H1/H3 performance is unanswered; H2 has
only the actual failed capture; H4 has no new measured kernel breakdown. Prior
Issue #15/PR #16 observations motivate this batch but are not new measurements.

## Lifecycle, provenance and integrity

Run `fd17-20261021-121300`; task/allocation/work/cleanup limits 600/540/420/60s;
host cap 660s. Allocation lifecycle 29.322s;
host execution 33.695s; scientific driver 15.245s.
Fresh authenticated CLI 0.7.4 read zero active assignments and existing 62.36 CU;
only the managed existing-entitlement assignment route was used. No new monetary
commitment. Displayed ending 62.36 CU can reflect rounding/lag, not proof of zero
consumption. One allocation, stop exit 0, controller authenticated zero and separate
fresh independent authenticated zero at 2026-10-21T12:13:34.577388+00:00.

Downloaded scientific file: 128747 bytes, SHA-256
`43a8613c198152a0b44e7f151bab53574baad618b6117c87d488d95c9c5399d4`. The immutable controller checks emitted marker length/hash before
scientific status; it reached `ARTIFACT_SOURCE_OR_STATUS_FAILED`, not integrity
failure. Therefore download integrity passed but the artifact correctly failed
acceptance. The controller does not retain its artifact_sha256 map for FAIL
science, and remote stdout marker is not separately saved. Host original/download
bytes and staged Git bytes are independently hashed; these limitations are
explicit in [audit.json](audit.json). Raw download is preserved byte-for-byte.

Exact reproduction commands on the clean pinned checkout (using existing torch):

```text
python scripts/validate_cpu.py
python scripts/check_snapshot.py
SAE_CPU_ONLY=1 SAE_CANDIDATE_SHA=6a4900c25314af856ee4e37a59b875d05dd6d768 SAE_LOCAL_SOURCE=<clean-source> SAE_ARTIFACT_DIR=<out> python experiments/full_gpu/driver.py
python .sae/tools/compute/colab_control.py --repo-path <clean-source> --source-sha 6a4900c25314af856ee4e37a59b875d05dd6d768 --run-id fd17-20261021-121300 --out <controller-out> --driver experiments/full_gpu/driver.py --artifacts experiments/full_gpu/artifacts.json --cli /home/zhangru/.local/bin/colab --device T4 --task-seconds 600 --allocation-seconds 540 --cleanup-seconds 60 --kernel-timeout 30 --work-timeout 420
```

These record the commands already used; do not reallocate within this batch.

## Architect handoff

Full Issue #17 experiment remains incomplete. Review the failure packet and
strict candidate correctness, retain production Direct SVD. A future scoped
repair should isolate graph attempts in a bounded child process (failed capture
may leave CUDA RNG/runtime state unsuitable for continuation), localize each graph
phase, and ensure strict baseline/control/performance work completes independently
of graph diagnostics. That repair and its CUDA execution are NOT implemented or
validated here. No further allocation is authorized by this report itself.

## Evidence index

[Raw GPU artifact](fd-full-gpu.json), [raw CPU fixture artifact](cpu-full.json),
[CPU validation](cpu_validation.json), [snapshot](snapshot.json),
[correctness](correctness.json), [benchmark NOT_RUN](benchmark.csv),
[graph diagnostics](graph-diagnostics.json), [controller](colab-control.json),
[independent host readback](host-execution.json), [preflight](preflight.json),
[local execution](local-execution.json), [audit](audit.json), [manifest](manifest.json).
