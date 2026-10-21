# Issue #7 — occupied-prefix proof; inconclusive clear-removal performance

**Engineering disposition: RETAIN BASELINE / DO_NOT_MERGE_RUNTIME_CANDIDATE.**
Logical update/sketch safety checks PASS; performance has small mixed signals,
not a robust default-optimization case. Implicit CPU pickle also preserves old
unused candidate rows, preventing a blanket serialization-equivalence claim.
GPT Architect scientific/evidence review remains pending.
[Issue #7](https://github.com/JohnnyZPiggy/TorchFrequentDirections/issues/7).

Baseline `f6ef317edd2e9ad1a42d8bc37a9a2e6229f88448`; actually tested no-clear candidate `1fab940d4f3987e0b03b7b5fad8540a9ccd51fb2`.
The candidate is exactly the baseline FD source minus one `self._buffer.zero_()`
line; the driver verifies the full-file hash of this transformation. All finite
checks, initial torch.zeros, shrink formulas, exceptions and public API are byte-identical.
After measurements, runtime was restored to the baseline in
`6c6cb724356e909913fd1fc4f6db379923a0ba8d`. The submitted runtime therefore does **not** remove
clearing. Prototype code remains reproducible at the immutable tested SHA.
Later publication commits add documentation/evidence only.

## Mathematical proof and all read paths

See the [Issue #7 R1–R3 audit](../../../docs/research/frequent-directions-tensorization.md).
Logical state is occupied count r and C=buffer[:r]; unused tail is not logical data.
_factor allocates independent output, so writing its retained prefix cannot alias
or destroy the source factor. Appended slots are completely overwritten before
count grows. Full-buffer reads occur only after all 2ell rows are occupied. Both
snapshot branches read only [:count] and produce independently allocated outputs;
short output padding is freshly zero-initialized. Reference FD is untouched.
Induction preserves all subsequent factor inputs (or Gram-equivalent states under
SVD basis ambiguity) and original FD covariance/projection bounds.

The audit includes ell=1/count=0, ell>d, empty input, zero/rank-deficient streams,
noncontiguous input, several fill cycles, chunk boundaries and observations followed
by continued updates. Constructor initialization stays intact. No production
defined serializer exists, but implicit Python pickle does serialize private
physical state. The CPU synthetic roundtrip preserved candidate old unused rows,
while baseline unused rows remained zero; both public snapshots stayed correct
and zero-padded. [Candidate pickle observation](pickle-candidate.json) and
[baseline observation](pickle-baseline.json) record booleans/hashes only; no raw
pickle/tensor/tail bytes were saved. This is an additional compatibility boundary,
not proof of a leak through sketch(). No serialization sanitizer/API is added.
Other diagnostic/error outputs do not include unused rows; private traces stay
local. Invalid input rejects before mutation;
backend/nonfinite-factor failures precede the clear/reset point and propagate in
both versions. Failed instances are discarded under the existing API contract.
Removing input/output finite checks or their synchronization was never attempted.

## Actual correctness evidence

Tested candidate clean CPU source: **30 tests PASS**, original oracle/seed sweep
and 216 baseline observations retained; 131-file pinned SAE snapshot/root links PASS.
Local A/B safety, fault checks and preflight PASS. One preliminary instrumentation
test failed because a pre-bound zero_ method bypassed a spy; the test now resolves
the method inside the patched lambda. FD code and finite checks were not changed
by this repair. [Sanitized failure](preliminary-failure.json) is retained; final
candidate evidence supersedes it. After restoring the baseline, a separate clean
CPU run at `39ebb07e27260552e4839692b143d76d0175e01e` passed **31 tests**, including the
new retained-runtime pickle regression; the runtime
FD source was verified byte-identical to the original baseline.

Actual **Tesla T4**, compute capability 7.5, cuda:0; Python 3.13.15,
torch 2.11.0+cu130 / CUDA 13.0. Same-host CPU: Xeon @2.00GHz, one PyTorch thread,
TF32 disabled. float32/float64 execute real SVDs in the correctness/workload checks;
no separate identity-SVD or NVIDIA driver-version probe is claimed.
**108 safety cases PASS** (54 CPU,54 CUDA), **24 workload cases PASS**, and
four device/dtype fault groups each check both actual baseline/candidate classes.
Every scheme has 1860 inspected factor inputs across both devices; 3480 stepwise
independent oracle calls plus 48 workload oracle calls cover both schemes.
Unused candidate rows are repeatedly poisoned with NaN; finite factor-input checks,
Gram equality, independent PSD/spectral-tail/right-projection bounds and public
snapshot alias isolation all pass. Poison/raw trace tensors never leave the checker.
SVD/finite-call counts are identical in every full workload; explicit post-shrink
clear calls go from 12/56/6/3 to zero in the affected shapes, zero→zero in controls.
Initial allocation's lower-level zero initialization is unchanged and not counted
as an explicit Python `zero_()` call.

[Correctness summary](correctness.json), [candidate CPU](cpu_validation.json),
[retained-baseline CPU](retained-baseline-cpu.json), [raw observations](fd-zero-clear.json).
Inherited dimension/energy-scaled tolerances are empirical, not a uniform
finite-precision theorem for arbitrary streams. See all k/tail/projection values
and safety maxima in the linked data.

## Full update+snapshot paired measurements — all cases

Per variant, two warmups and **20 balanced shuffled AB/BA pairs**; 10 trials start
with each variant. Same process/runtime/resident input/dtype; synchronization before
and after each CUDA timing. Allocation+update+snapshot and finite checks are included;
input generation/transfers/source install are excluded. CPU/CUDA case order is fixed.
Actual previous module is hash-pinned and loaded alongside the candidate; no prior
Issue #3/#5 timings are substituted.

Ratios below are **medians of paired baseline/candidate ratios** (>1 favors candidate),
with descriptive paired-bootstrap 95% intervals. They differ from ratios of independent
medians. Bootstrap uses 2000 resamples, is not adjusted for multiple comparisons,
and cannot establish cross-session/device generality.

| n/d/ell | dtype | Full shrinks | CPU paired ratio [interval] | T4 paired ratio [interval] |
|---|---|---:|---:|---:|
| 7/64/8 | float32 | 0 | 1.0121 [1.0043, 1.0218] | 0.9964 [0.9238, 1.1080] |
| 7/64/8 | float64 | 0 | 0.9928 [0.9848, 0.9993] | 0.9883 [0.8975, 1.0529] |
| 15/64/8 | float32 | 0 | 0.9870 [0.9154, 1.0788] | 0.9972 [0.9919, 1.0043] |
| 15/64/8 | float64 | 0 | 0.9874 [0.9219, 1.0054] | 1.0008 [0.9948, 1.0064] |
| 64/32/4 | float32 | 12 | 1.0197 [1.0046, 1.0407] | 1.0076 [0.9981, 1.0198] |
| 64/32/4 | float64 | 12 | 1.0206 [0.9786, 1.0564] | 1.0125 [1.0056, 1.0191] |
| 512/64/8 | float32 | 56 | 1.0061 [0.9966, 1.0180] | 1.0090 [1.0035, 1.0161] |
| 512/64/8 | float64 | 56 | 1.0113 [1.0005, 1.0204] | 1.0015 [0.9906, 1.0100] |
| 256/512/32 | float32 | 6 | 1.0029 [0.9988, 1.0151] | 1.0032 [0.9988, 1.0067] |
| 256/512/32 | float64 | 6 | 1.0047 [0.9996, 1.0093] | 1.0011 [0.9978, 1.0052] |
| 258/1024/64 | float32 | 3 | 1.0014 [0.9991, 1.0071] | 1.0057 [0.9963, 1.0093] |
| 258/1024/64 | float64 | 3 | 0.9993 [0.9851, 1.0075] | 1.0007 [0.9987, 1.0014] |

The predeclared >=1% paired-median / lower-bound>1 screen flags **3 affected
cases**, but also **1 no-clear control**. These are candidate signals,
not acceptance. Gains are small (~1–2%); many intervals include one, and controls
show comparable movement despite no clear removal. The CPU local readiness run
did not establish a full-workload gain either. This is **INCONCLUSIVE_SMALL_SIGNALS**,
not proof that clearing costs zero or that no workload can ever benefit. We retain
the baseline and recommend **not merging the runtime candidate** on this evidence
and the unresolved implicit-pickle physical-state boundary.
GPT Architect may independently accept the evidence or request a narrowly justified follow-up.

## Separate clear-only API cost

Forty balanced paired trials on hot resident buffers compare a bound `zero_()` API
with a Python no-op. Buffer allocation is outside timing. Values below are T4
synchronized host-wall **microseconds**, including launch/wait/API overhead:

| Physical buffer | dtype | zero_ median us | no-op median us |
|---|---|---:|---:|
| 8×32 | float32 | 16.56 | 7.33 |
| 8×32 | float64 | 16.41 | 7.37 |
| 16×64 | float32 | 16.49 | 7.41 |
| 16×64 | float64 | 16.68 | 7.39 |
| 64×512 | float32 | 16.29 | 7.31 |
| 64×512 | float64 | 16.42 | 7.33 |
| 128×1024 | float32 | 16.40 | 7.34 |
| 128×1024 | float64 | 16.28 | 7.32 |
| 512×4096 | float32 | 47.57 | 7.40 |
| 512×4096 | float64 | 88.18 | 7.61 |

These measurements demonstrate a real standalone clear cost, including a large
512×4096 memory-only case not used as an SVD/FD throughput claim. They cannot be
multiplied by shrink count or subtracted from FD latency: overlap, synchronization,
cache state and Python/device scheduling differ. [profile.csv](profile.csv)
contains intrusive/nonadditive clear and SVD host times/call counts; primary paired
timings are uninstrumented. Pure GPU kernel/CUPTI attribution, synchronization
removal, TPU, compile/fusion and new APIs/dependencies are NOT RUN/not changed.
[benchmark.csv](benchmark.csv) retains raw samples, median/p95/min/max/SD and paired
statistics; raw JSON also retains trial order and paired ratios. Nearest-rank p95
uses the 19th of 20 or 38th of 40 sorted samples. All cases, including negative
controls and weak/regressing values, are retained.

## Exact-source transport, cleanup and scope

Run `fd7-20261020-212203`: 2026-10-20T21:22:03.107674+00:00 to 2026-10-20T21:23:28.691605+00:00 UTC.
Driver elapsed 60.479 s; host elapsed 89.238 s.
One allocation, no retries; 600 s task / 540 s allocation / 420 s workload /
60 s cleanup reserve / 660 s independent host bound. Existing positive entitlement
and authenticated zero assignments checked before allocation; no new billing.
Mandatory kernel reachability probe preceded scientific work. Reused hash-pinned
installer/oracle/support, exact baseline and candidate SHA, no-deps/no-index pip target,
and committed project checker/manifest. No benchmark-time GitHub writes.

Downloaded JSON: 508967 bytes,
SHA-256 `4c98c9b80ab234c2e1e4a33157f95fd2a4d3f95f4d249007c05ef59ed0051110`; emitted byte/hash, downloaded bytes, source and status were
verified by the controller and rechecked locally. [Controller](colab-control.json)
stop succeeded; its authenticated usage readback and an independent fresh usage
readback both confirmed zero assignments, independently at
2026-10-20T21:23:28.624771+00:00. [Host evidence](host-execution.json).
This is timestamped state, not continuous monitoring. No private roots,
credentials/session URLs, actual tail bytes or raw trace tensors are published.
Pinned SAE rules/snapshot and public finite-check semantics remain unchanged.

## Resumption audit — 2026-10-21

This resumed Engineer cycle recovered the incomplete local branch and prior
2026-10-20 CPU/T4 artifacts. It did **not** allocate another GPU or relabel the
historical GPU measurements as a new run. Fresh clean CPU clones reproduced the
candidate/baseline pickle difference, and the retained default plus its new
serialization regression passed 31 tests. Exact source transformations,
download hashes, raw paired statistics and timestamped cleanup were reverified.
See [resumption verification](resumption-verification.json). Current remote account
state was not queried; authenticated zero belongs to the historical run's timestamp.

## Reproduce the measured prototype, not the retained runtime

The experiment driver intentionally pins the one-line no-clear prototype and
rejects a different FD file hash. To reproduce, use a disposable clean checkout
of `1fab940d4f3987e0b03b7b5fad8540a9ccd51fb2` (the final PR runtime keeps clearing):

```sh
git checkout --detach 1fab940d4f3987e0b03b7b5fad8540a9ccd51fb2
python scripts/validate_cpu.py
python scripts/check_snapshot.py
SAE_CPU_ONLY=1 SAE_LOCAL_SOURCE="$PWD" SAE_CANDIDATE_SHA=1fab940d4f3987e0b03b7b5fad8540a9ccd51fb2 SAE_ARTIFACT_DIR=/tmp/fd7-local python experiments/zero_clear/driver.py
```

Actual live invocation (local roots shown as variables):

```sh
python "$SOURCE/.sae/tools/compute/colab_control.py" --repo-path "$SOURCE" \
  --source-sha 1fab940d4f3987e0b03b7b5fad8540a9ccd51fb2 --run-id fd7-20261020-212203 --out "$NEW_OUTPUT" \
  --driver experiments/zero_clear/driver.py --artifacts experiments/zero_clear/artifacts.json \
  --cli "$EXISTING_COLAB_074" --device T4 --task-seconds 600 --allocation-seconds 540 \
  --cleanup-seconds 60 --kernel-timeout 30 --work-timeout 420
```

Use a fresh run ID/output directory for reproduction, with one session and
authenticated cleanup guards. Codex submits evidence only; no self-approval,
Issue closure, main write or self-merge. GPT Architect owns acceptance and merge.
