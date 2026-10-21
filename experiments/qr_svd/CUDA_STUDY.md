# Issue #13: explicit CUDA QR adapter

This experiment implements the scoped readiness decision in GitHub Issue #13.
GPT Architect owns scientific acceptance and merge. The production package and
historic CPU prototype are unchanged; there is no automatic backend selection.

R1: CPU shape/thread findings in Issues #9/#11 do not establish CUDA behavior.
Fast FD remains sequential between shrinks and performs finite input/output
checks, buffer clearing and independent snapshots. CPU and CUDA SVD may use
different solvers; no CPU crossover is transplanted to T4.

R2: For C in R^(m×d), m<=d, reduced QR gives C^T=QR, Q^TQ=I_m.
SVD(R^T)=U diag(s) W^T implies C=U diag(s) W^T Q^T and Vh=W^TQ^T.
Use the existing normalized scale-before-square ell-th threshold shrink, clamp
and fresh output; m>d and empty input take the existing direct factor path.
Exact arithmetic preserves Gram matrices; signs and degenerate bases need not
match. Finite precision, near-null spectra and accumulated shrink error require
the independent gates and remain bounded-test assumptions.

R3: `ExperimentalQRSmallSVD` explicitly opts into CPU/CUDA state through the
unchanged production initializer and inherits the pinned device-generic algebra.
The historic CPU-only constructor still rejects CUDA. This adapter does not
change update/sketch, finite checks, clearing or pickle behavior in production.
CPU standalone/fault/regression checks precede any allocation. Every measured
CUDA workload gates both variants using CPU float64 explicit Gram PSD, full-SVD
tail and right-subspace projection bounds for every integer 0<=k<ell, k<=d.
Tolerance fixed in the tested payload: 32 eps(input dtype) max(n,d,2ell,1)
times input Frobenius energy; the same bound applies to Gram Frobenius gap.
This absolute contract does not establish relative accuracy of tiny singular
components. Exact local tied thresholds are checked separately from global
stream spectra. Fixture input storage is genuinely noncontiguous on CUDA.

The standalone payload reads exact Git SHA source files, verifies embedded
SHA-256 hashes and imports only those modules, without installing dependencies.
The immutable controller independently verifies transported driver and artifact
manifest Git blobs. Its single allocation holds kernel/QR/SVD probes, math gates
and performance. One exclusive host lock prevents allocation retries.

GPU-resident Direct/QR factor and whole-stream timings exclude generation and
transfers. Whole streams create new states, include validation, shrink/clearing
and final snapshots; factor-only excludes state construction. Every timed call
synchronizes CUDA before and after. Two warmups/role and 20 balanced randomized
AB/BA pairs are fixed in the reused timing helper; raw sample order, paired
ratios, median/p95/SD and descriptive bootstrap intervals are retained.
The payload predeclares the >1.05 paired median / lower interval>1 exploratory
screen; short/fallback and direct/direct A/A controls qualify interpretation.
No multiplicity adjustment or independent-session replication. QR/small-SVD/
Vh/shrink stage diagnostics are intrusive and NONADDITIVE, not kernel-only times.
TF32 is disabled and float32 matmul precision is highest for both variants.

Reproduce CPU gates first from a clean tested checkout (output outside source):

```sh
python scripts/validate_cpu.py
python scripts/check_snapshot.py
SAE_CPU_ONLY=1 SAE_LOCAL_SOURCE="$PWD" SAE_CANDIDATE_SHA="$(git rev-parse HEAD)" \
  SAE_ARTIFACT_DIR=/tmp/fd13-cpu python experiments/qr_svd/cuda_driver.py
```

For an explicitly authorized future experiment, use only the pinned controller
after fresh authenticated zero-assignment/positive-existing-CU checks and the
no-purchase boundary. This example is not an instruction to allocate again:

```sh
python .sae/tools/compute/colab_control.py --repo-path "$PWD" \
  --source-sha 23b5aa13ef926ecd5df36908f2277fc6e1c77b00 \
  --driver experiments/qr_svd/cuda_driver.py \
  --artifacts experiments/qr_svd/cuda_artifacts.json \
  --run-id fd13-reproduction --out /tmp/fd13-reproduction --device T4 \
  --task-seconds 600 --allocation-seconds 540 --cleanup-seconds 60 \
  --kernel-timeout 30 --work-timeout 420
```

The experiment itself has a 340-second scientific bound and skips timing blocks
after 285 seconds to protect cleanup. The host bounds the controller to 620 s,
with emergency stop/fresh readback reserve inside 660 s. Stop and controller
authenticated zero-session evidence plus a separate fresh authenticated readback
are mandatory, including on failure. Never retry the allocation in this Issue.
The current run finished all cases and confirmed cleanup; no additional run is
needed for publication. [Evidence and limitations](../../reports/gpu/issue13-20261021/summary.md).
