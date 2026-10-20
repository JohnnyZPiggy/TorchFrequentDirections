# Issue #3 — real T4 CPU/GPU validation

Status: **ENGINEER_MEASUREMENTS_PASS / ARCHITECT_REVIEW_PENDING**.
Issue: https://github.com/JohnnyZPiggy/TorchFrequentDirections/issues/3
Tested implementation/driver commit: `608975335817431d841e0ad313a21d4930304a46`. Later report commits do not change tested code.
Run ID: `fd3-20261020-203909`; UTC 2026-10-20T20:39:09.175900+00:00 to 2026-10-20T20:40:34.441534+00:00.

## Scope and recovered role

Codex Engineer continued the already claimed Issue and existing `feat/3-gpu-validate` branch.
PR #2's merged CPU baseline and mathematical acceptance were recovered before execution.
GPT Architect decides scientific acceptance and merging; Codex does not self-approve or merge.
No algorithm/public API changes, framework changes, new dependencies, Actions, TPU or compilation.

SAE source and project snapshot both pin `a60ed92d80a8866b413b928d609cc3afb82ec408`.
All 131 tracked framework files / 605207 Git bytes were read and inventoried;
Python/JSON/SVG parsed; nine Python sources and normative role/workflow contracts reviewed.
The per-file hashes, outlines and diagram labels are in [framework review](framework-review.json).
Historical Mermaid, Colab examples and some old templates retain superseded Scientist-approval
language; current constitution/project rules govern. Eight unresolved template links are existing
relocation placeholders. Immutable snapshots were preserved; no framework fix is bundled here.
Translations/historical data were cross-checked, not treated as proof of this project's execution.

## Implementation and local evidence

The existing project-owned adapter installs only allowlisted, hash-verified source files
from the exact public commit archive, using no-deps/no-index pip into a temporary target.
The controller separately verifies the committed driver and artifact-manifest blobs.
This continuation adds a wider 128/512/32 workload, nonzero scientific failure exit,
and local Git HEAD verification with a negative provenance test.

[CPU validation](cpu-validation.json): **21 tests PASS**, clean detached source,
216 baseline oracle observations; Python 3.12.3 / torch 2.11.0+cpu / one thread.
[Local execution](local-execution.json): 37 pinned framework tests PASS, 131 snapshot files
and root links PASS, CPU science driver PASS, no-allocation transport preflight PASS.
An earlier incorrectly transcribed local SHA and WSL-cleared /tmp results were discarded;
exact HEAD verification and persistent isolated paths fixed provenance before this final run.
No additional package or authentication setup was needed.

## Actual environment and independent correctness

Observed **Tesla T4**, compute capability 7.5, 15,637,086,208 device bytes;
tensor device cuda:0. Colab Python 3.13.15 / torch 2.11.0+cu130 / CUDA 13.0;
CPU Intel Xeon @ 2.00GHz, one PyTorch thread; TF32 disabled.
Both float32/float64 SVD identity reconstruction probes had zero observed error.
Separate NVIDIA driver version was not queried.

**28 fixture/method/dtype results plus 16 benchmark method/dtype results PASS**, each
with CPU and GPU independent checks plus same-algorithm covariance comparison.
Fixtures include partial buffers, ill-conditioning, near ties, zero/short input,
ell=1 and ell>d. Independent CPU float64 Gram PSD/eigenvalues, full-SVD tail bounds,
and explicit right-subspace projection check the GPU result.
See [correctness summary](correctness.json) and untouched [download](fd-science.json).
Tolerances inherit the CPU-reviewed dimension/energy-scaled slack; they are empirical,
not a rigorous finite-precision theorem or unlimited-stream guarantee.

## Same-algorithm CPU/GPU timings

All times below are **Fast FD median milliseconds**, same input/dtype/runtime,
constructor + full update + final snapshot. GPU resident excludes transfers;
end-to-end includes pageable H2D input and D2H sketch. Generation is excluded.
CPU/GPU >1 means the GPU was faster in this measurement.

| n/d/ell | dtype | CPU | GPU resident | GPU + transfers | CPU/GPU | GPU reference/Fast |
|---|---|---:|---:|---:|---:|---:|
| 64/32/4 | float32 | 2.544 | 9.731 | 9.910 | 0.261 | 3.94 |
| 64/32/4 | float64 | 2.754 | 14.901 | 14.981 | 0.185 | 4.20 |
| 128/64/8 | float32 | 4.511 | 14.680 | 15.594 | 0.307 | 7.37 |
| 128/64/8 | float64 | 4.430 | 44.125 | 44.855 | 0.100 | 5.60 |
| 256/128/16 | float32 | 16.353 | 21.218 | 22.002 | 0.771 | 15.05 |
| 256/128/16 | float64 | 9.348 | 102.418 | 102.371 | 0.091 | 6.93 |
| 128/512/32 | float32 | 10.030 | 7.318 | 7.365 | 1.371 | 21.68 |
| 128/512/32 | float64 | 8.514 | 49.893 | 49.717 | 0.171 | 12.30 |

Most tested small cases are slower on GPU. The wider float32 Fast case showed
1.371x resident and 1.362x
transfer-included CPU/GPU median ratios. Its raw CPU range was
9.693–10.261 ms and GPU range
7.222–7.507 ms.
This is a bounded pilot observation, not an established general crossover or universal speedup.
Every tested float64 Fast case was slower on this GPU than its one-thread CPU baseline.

One warmup and five synchronized wall-clock repeats per measurement; raw samples,
median, nearest-rank p95, min/max and sample SD are in [benchmark.csv](benchmark.csv).
With only five samples p95 equals the maximum, not a reliable tail estimate or confidence interval.
Fixed shape/method/device order and shared Colab CPU load can bias comparisons;
different timing blocks can make transfer-included medians appear lower than resident ones.
No subtraction/additive interpretation is justified. CPU all-core performance, sustained
throughput, larger matrices, pinned transfers and random/interleaved trial order are untested.

[profile.csv](profile.csv) separates synchronized host SVD, input validation, full update
and snapshot diagnostics. Monkeypatch/synchronization profiling is intrusive/nonadditive
and is never the speedup timing. Finite-check calibration is standalone and does not
attribute every output check. Pure CUDA kernel-only attribution and torch.compile are NOT RUN.

## Integrity, bounded execution and cleanup

Exactly **one** allocation, no retry; task 600 s, allocation lifecycle cap 540 s,
workload 300 s, cleanup reserve 60 s, independent host bound 660 s.
Host elapsed 88.216 s; driver elapsed 55.216 s.
Default kernel marker preceded scientific execution. Existing-account positive entitlement
and zero assignments were authenticated before allocation; no billing/account changes.
Benchmark driver has no GitHub writes. Publication is a separate engineer branch/PR action,
not a generic automatic Publisher implementation.

Untouched remote JSON: 332101 bytes, SHA-256
`96e85876d74f2456c25dff336a07539c5641b47dfa53ab6c02e806819b3ce1ff`. Emitted byte count/hash, downloaded bytes and exact source/status
were verified by the controller and rechecked locally. Safe controller event records are
in [colab-control.json](colab-control.json); independent host record in
[host-execution.json](host-execution.json).
Stop succeeded; controller authenticated usage readback and an independent fresh usage readback
both confirmed zero assignments. Independent confirmation: 2026-10-20T20:40:34.400214+00:00.
This is state at those timestamps, not continuous monitoring.

## Reproduce and hand off

In a clean checkout of the tested SHA with existing PyTorch:

```sh
python scripts/validate_cpu.py
python scripts/check_snapshot.py
python -m unittest discover -s .sae/tools/compute/tests -v
SAE_CPU_ONLY=1 SAE_LOCAL_SOURCE="$PWD" SAE_CANDIDATE_SHA="$(git rev-parse HEAD)" SAE_ARTIFACT_DIR=/tmp/fd-local python experiments/fd_gpu/driver.py
```

Actual live controller invocation (local paths expressed as variables):

```sh
python "$SOURCE/.sae/tools/compute/colab_control.py" --repo-path "$SOURCE" \
  --source-sha 608975335817431d841e0ad313a21d4930304a46 --run-id fd3-20261020-203909 \
  --out "$NEW_OUTPUT" --cli "$EXISTING_COLAB_074" --device T4 \
  --driver experiments/fd_gpu/driver.py --artifacts experiments/fd_gpu/artifacts.json \
  --task-seconds 600 --allocation-seconds 540 --cleanup-seconds 60 \
  --kernel-timeout 30 --work-timeout 300
```

Use a new output/run ID for a rerun, preserving one-session and authenticated-cleanup guards.
No data/paper/reference-code copies or credential/session URLs are published.
Framework tests are local mocks, not provider tests; the live scientific run is distinct.
GPT Architect should review tolerances, source identity, workload/variance limitations and
accept or return this evidence on Issue #3. No self-merge or scientific acceptance is claimed.
