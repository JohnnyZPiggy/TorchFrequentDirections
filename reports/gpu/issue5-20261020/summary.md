# Issue #5 — lossless short-buffer snapshot

Status: **ENGINEER_VALIDATION_PASS / ARCHITECT_REVIEW_PENDING**.
Issue: [#5](https://github.com/JohnnyZPiggy/TorchFrequentDirections/issues/5).
Candidate code/test/driver SHA: `687ea56596d1b47633b2abf5f94b3feffe04ec31`.
Actual baseline SHA: `73203cb3415bc0c8733e7f4e7c0d1d63a431a03f`; baseline FD SHA-256
`df1b7af6832574409856edaac2533c9c23e5fb9c8f0936826d383eff952cdff7`.
Both were loaded in the same process/runtime; no mixing of PR #4 timing values.
Run `fd5-20261020-205608`: UTC 2026-10-20T20:56:08.305444+00:00 to 2026-10-20T20:57:19.081266+00:00.

## Mathematical contract and code

The [R1–R3 study](../../../docs/research/frequent-directions-tensorization.md)
records the Architect's Issue #5 scoped RESEARCH_READY rationale and Codex proof-oriented checks.
For occupied C with m<=ell, B_new=[C;0], so B_new.T B_new=C.T C. The old reduced
SVD factor [Sigma Vh;0] has the same exact Gram. Padding adds no directional loss,
so FD covariance and top-k projection inequalities are unchanged, including empty,
rank-deficient, ell=1 and ell>d cases. Tied eigenspaces need not choose identical
bases. Finite-precision old-SVD equality is tested with tolerance, while copied
occupied rows and zero padding are exact.

The observable representation changes: short output rows need not be orthogonal
or in descending SVD order. The documented contract is Gram/subspace invariants,
not canonical row orientation; inspected project callers do not require it.
PCA still requires extracting the right singular subspace, whose downstream SVD
is outside this snapshot benchmark. Snapshots remain independent/nonmutating and
do not affect continued updates or future shrink schedules.

Core implementation is eight added lines in `FrequentDirections.sketch` including
its docstring. Input validation/invalid-input atomicity, finite output checks,
dtype/device, no-autograd and count>ell shrink are retained. New finite corruption
test confirms FloatingPointError detection. Secondary zero-clear, finite-check
removal/reordering, compile/fusion and alternative SVD are **not changed or claimed**.
No framework/public-signature/dependency change. The old Issue #3 profiler count expectation
was updated for the deliberately removed final SVD; its historical evidence is untouched.

## CPU and real GPU correctness

[Local execution](local-execution.json): clean detached source, **26 unit tests PASS**,
216 original baseline oracle observations, snapshot byte/link verification PASS,
local CPU A/B PASS (78 edge +16 workload cases), CLI 0.7.4 preflight PASS.
No implementation repair or failed numerical test occurred in this batch.
The publication verifier initially compared a Windows CRLF worktree using WSL
Git and falsely flagged snapshot text; checking actual staged Git blobs resolved
that harness mismatch. All published artifact bytes/hashes and tested code were
then independently verified; no scientific result or snapshot rule was weakened.
The unchanged pinned framework regressions are not rerun; their 37-test evidence
is in the prior Issue #3 report. The immutable 131-file snapshot was rechecked.

Actual Colab **Tesla T4**, compute capability 7.5, cuda:0;
Python 3.13.15 / PyTorch 2.11.0+cu130 / CUDA 13.0; CPU Xeon @2.00GHz,
one PyTorch CPU thread, TF32 disabled. Both float32/64 identity SVD probes PASS.
Separate NVIDIA driver-version query NOT RUN.
**156 edge cases +32 workload cases PASS**, each checking both pinned baseline
and candidate using independent float64 CPU Gram PSD/eigenvalues, full-SVD tails,
explicit right projection and old/new Gram comparison. Boundary tests cover
0,1,ell-1,ell,ell+1 requested rows; some occupancies are unreachable after automatic
full-buffer shrink (e.g. ell=1/count=2), so raw results record actual occupancy.
Repeated/aliased observations and continued updates after several fills also PASS.
See [correctness summary](correctness.json), [CPU regression](cpu_validation.json)
and [unaltered downloaded observations](fd-snapshot.json).
Inherited energy/dimension-scaled tolerances are empirical, not a new uniform
finite-precision theorem. No exact equality of old/new factor entries is asserted.

## Paired A/B observations

For each variant, two warmups and **20 randomly ordered paired trials**,
same resident input, dtype, algorithm and host environment. Each CUDA timing
explicitly synchronizes before/after; it measures host/Python/launch/wait latency,
not pure GPU kernels. Input generation, source install and H2D/D2H transfers are
excluded from this A/B; unlike Issue #3, this task isolates snapshot optimization.
CPU/GPU device order is fixed; baseline/candidate order is shuffled within each
device, and the full per-trial order/samples are retained. This single-session
pilot is not a universal speedup or a causal kernel breakdown.

Table: baseline/candidate median-time ratios (>1 means candidate faster).
Only active short-buffer workloads appear here; inactive controls are in the CSV/raw JSON.

| n/d/ell | dtype | CPU snapshot | T4 snapshot | CPU full stream | T4 full stream | Full SVD calls |
|---|---|---:|---:|---:|---:|---:|
| 64/32/4 | float32 | 2.46 | 3.55 | 1.012 | 1.045 | 13 → 12 |
| 64/32/4 | float64 | 2.73 | 6.08 | 1.063 | 1.032 | 13 → 12 |
| 124/64/8 | float32 | 3.30 | 3.84 | 1.046 | 1.017 | 14 → 13 |
| 124/64/8 | float64 | 3.18 | 5.60 | 1.018 | 1.013 | 14 → 13 |
| 253/128/16 | float32 | 4.45 | 4.47 | 0.999 | 1.029 | 15 → 14 |
| 253/128/16 | float64 | 4.18 | 7.42 | 1.012 | 1.011 | 15 → 14 |
| 98/512/32 | float32 | 10.42 | 6.47 | 1.211 | 1.171 | 3 → 2 |
| 98/512/32 | float64 | 7.14 | 27.27 | 1.139 | 1.130 | 3 → 2 |

Nonempty short snapshot SVD counts are **1→0**, empty already **0→0**,
and count>ell remains **1→1**. Finite-call counts are unchanged in every
edge/workload snapshot and full workload. Thus timing savings do not come from
skipping finite validation. All prior representative shapes 64/32/4,128/64/8,
256/128/16,128/512/32 are included, plus nearby n values exercising the other branch.

For 98/512/32 float32 on T4, snapshot median was
1.074552→0.166154 ms;
full stream was 6.075131→5.186932 ms
(1.171x). Same-host CPU full stream:
8.091350→6.682635 ms
(1.211x).
The GPU float64 snapshot-only ratio can be much larger because it removes one
expensive SVD, but its full-stream gain is far smaller; do not substitute one for the other.

Inactive full-stream ratios range 0.975–1.051;
their SVD counts/algorithmic path are unchanged, so treat those deviations as
measurement variation/branch overhead, not a claimed optimization benefit.
Some active full-stream improvements are similarly small or absent; one saved
snapshot SVD need not dominate a long update. Shared CPU load and sequential
trials limit inference; no confidence interval or universal crossover is claimed.

[benchmark.csv](benchmark.csv) retains all 20 samples, median, nearest-rank p95,
min/max, sample SD and paired-ratio medians. Raw paired ratios/order are in the JSON.
For 20 samples p95 is the 19th sorted value.
[profile.csv](profile.csv) records intrusive synchronized SVD wall-time/counts and
finite-call counts. Separate whole-API `bool(isfinite(...).all())` input/output
calibrations include host waiting; they are nonadditive, not isolated kernel times.
Do not subtract these from the primary paired timings. Pure kernel/CUPTI profiling,
torch.compile, TPU, all-core CPU and zero-clear experiments are NOT RUN.

## Provenance, lifecycle and publication

Project driver reuses the hash-pinned Issue #3 installer/oracle, fetches the
actual prior baseline FD file with exact SHA/hash, and installs allowlisted,
hash-verified candidate source via no-deps/no-index pip into a temporary target.
Controller independently verifies candidate driver/manifest Git blobs. No
benchmark-time GitHub writes, paper/dataset copies, credential/session URLs or
account/billing changes. Existing positive entitlement and authenticated zero
assignments verified before allocation. One allocation, no retries.

Task cap 600 s, allocation cap 540 s, workload cap 420 s, cleanup reserve 60 s,
independent host bound 660 s. Driver elapsed 43.246 s;
host elapsed 72.510 s. Default kernel reachability probe
preceded workload. Download: 896485 bytes,
SHA-256 `7ae49be64bcbd63aaa9c6ec9834c67bcf949f350f8f384762864696e086888a9`, checked against emitted bytes/hash, source/status and independently
rechecked locally. [Controller](colab-control.json) stop succeeded; controller and
[independent authenticated readback](host-execution.json) both confirmed zero
assignments, independently at 2026-10-20T20:57:19.006093+00:00.
This is timestamped cleanup evidence, not continuous monitoring.

## Reproduce and Architect handoff

At the tested candidate SHA, with an existing PyTorch environment:

```sh
python scripts/validate_cpu.py
python scripts/check_snapshot.py
SAE_CPU_ONLY=1 SAE_LOCAL_SOURCE="$PWD" SAE_CANDIDATE_SHA="$(git rev-parse HEAD)" SAE_ARTIFACT_DIR=/tmp/fd5-local python experiments/snapshot/driver.py
```

Actual live command, local roots represented as variables:

```sh
python "$SOURCE/.sae/tools/compute/colab_control.py" --repo-path "$SOURCE" \
  --source-sha 687ea56596d1b47633b2abf5f94b3feffe04ec31 --run-id fd5-20261020-205608 \
  --out "$NEW_OUTPUT" --cli "$EXISTING_COLAB_074" --device T4 \
  --driver experiments/snapshot/driver.py --artifacts experiments/snapshot/artifacts.json \
  --task-seconds 600 --allocation-seconds 540 --cleanup-seconds 60 \
  --kernel-timeout 30 --work-timeout 420
```

Use a fresh run ID/output directory for reproduction; preserve one-session and cleanup guards.
Report commits follow the tested code commit and do not change implementation/tests/driver.
GPT Architect reviews the mathematical representation contract, oracle tolerances,
raw paired timing/variance and final source diff, then decides acceptance/merge.
Codex claims completed engineering measurements only; no self-approval or self-merge.
