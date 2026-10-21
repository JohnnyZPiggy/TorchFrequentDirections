# Issue #1 — CPU bootstrap engineering evidence

Issue #7 follow-up: occupied-prefix/factor-read and fault checks passed for the
one-line no-clear prototype, but the CPU/T4 performance evidence was inconclusive
and implicit pickle preserves its stale private tail. The submitted runtime is
byte-identical to the clearing baseline; all finite checks remain untouched.
The [study update](../research/frequent-directions-tensorization.md) distinguishes
logical equivalence from serialization, and the
[evidence report](../../reports/gpu/issue7-20261020/summary.md) recommends rejecting
the runtime candidate. A fresh 31-test clean CPU run includes a pickle regression;
the exact earlier candidate SHA and real T4 evidence are retained for review.

Issue #5 follow-up: `sketch()` now copies and zero-pads an occupied prefix of
at most ell rows, retaining finite output checks and the longer-buffer shrink.
The [R1–R3 update](../research/frequent-directions-tensorization.md) documents
exact Gram equivalence and the noncanonical row-representation change.
Candidate `687ea56596d1b47633b2abf5f94b3feffe04ec31` passed 26 clean-source
CPU tests, independent CPU/T4 oracle checks and shuffled paired A/B measurements;
see the [Issue #5 report](../../reports/gpu/issue5-20261020/summary.md).
Secondary zero-clear and finite-check removal are deferred. GPT review is pending.

Issue #3 follow-up: the core algorithm below is unchanged. A project-owned
exact-source driver and artifact manifest now exercise same-algorithm CPU/GPU
correctness, transfers, synchronized repeated timing and intrusive SVD/update
diagnostics. The real T4 run, clean-source 21-test CPU regression and session
cleanup are recorded in the [GPU experimental report](../../reports/gpu/issue3-20261020/summary.md).
Its tested source is `608975335817431d841e0ad313a21d4930304a46`;
GPT Architect scientific/engineering review is pending. The Issue #1 results
below remain historical CPU-only evidence.

Owner: Codex Engineer. Decision/merge owner: GPT Architect. Status: REVIEW;
actual run identities/results are in [execution](../../reports/issue1/execution.json)
and [CPU observations](../../reports/issue1/cpu-validation.json).
Scientific scope is the provisional decision in [Issue #1](https://github.com/JohnnyZPiggy/TorchFrequentDirections/issues/1),
reconstructed in [R1–R3](../research/frequent-directions-tensorization.md).

## Framework and role recovery

Live repo was public, owner JohnnyZPiggy, actual push/admin permission true;
main initially held only MIT LICENSE at 325aa82 (not wholly empty). Claimed Issue
in comment 6300081388; branch `feat/1-fd-cpu-bootstrap`. No main write or merge.
Fixed SAE source `a60ed92d80a8866b413b928d609cc3afb82ec408`: all **131** tracked
files / **605207** bytes read/inventoried and compared to exact Git blobs.
Executable code (9 Python files including tests, adapter and offline diagnostics)
reviewed; Python AST, JSON evidence and all SVG XML parsed. Normative role,
bootstrap, workflows, templates and bilingual documentation examined; figures
and historical evidence reviewed as data, not executed activity in this project.
The [per-file inventory](../../reports/issue1/framework-inventory.json) preserves
byte hashes and structural review index. `scripts/check_snapshot.py` rechecks the
complete target copy, syntax/data parsers and root links. Its hashes were generated
against the pinned framework's local Git objects, not merely the copied files.
Generated, Git-ignored `__pycache__` directories are excluded from the file-set
comparison; all 131 source/data bytes remain checked, no snapshot text is edited.

Normative rules: Codex implements/tests/computes and submits PR evidence; GPT
decides research/numerical acceptance and eligible merges. No routine Scientist
approval. Snapshot drift found: Mermaid .mmd, Colab examples and some old GitHub
math/engineering templates retain superseded Scientist science/compute approval
language; dated capability notes predate README's newer live evidence. They stay
immutable. English constitution/current workflows/governance govern, as stated in
root AGENTS. Eight raw unresolved links all belong to the relocatable
`examples/project/AGENTS.md` template's `.sae/...` links; they resolve when adapted
to this project's root. No other relative Markdown file-link failure observed
by the checker (fragment anchors/external URLs are outside that check).

## Local input rights and Build vs Reuse (2026-10-20)

All four requested local inputs exist. Hashes/sizes of paper.pdf, TeX archive,
code ZIP and every file in its extracted folder are in
[provenance](../../reports/issue1/input-provenance.json). No private local path,
paper, archive, dataset or third-party source is published.

- Paper: arXiv1501.01711v2, four authors, 2015. Algorithm/proof sections were
  inspected locally and bibliographic metadata checked at the primary arXiv page.
  No explicit redistribution license was found in the local TeX archive; no
  permission for republishing those files is inferred from access.
- Local `frequent-direction-master` identifies **Shohei Hido, copyright 2013**;
  README cites Liberty2013. It is not verified to be official code by the four
  2015 paper authors. LICENSE has the two BSD redistribution conditions and
  disclaimer: copied source must retain copyright, conditions and disclaimer;
  binaries must reproduce these in documentation/materials. Inspection is not
  redistribution. New implementation derives from paper formulas, no source text
  copied; third-party code is not vendored or installed/run as an oracle.
- Its semantics differ: ell-row buffer, threshold at **zero-based floor(ell/2)**,
  rather than requested ell-th singular value in a 2ell buffer; row-sum rounded
  zero detection can misclassify nonzero cancelling rows; ell>=n is rejected;
  reduced SVD can reduce row count when ell>d. Python2 test printing/time.clock
  sample also require porting. Treating this code as the scientific oracle would
  conceal capacity/indexing differences.
- **NATIVE_REUSE + BUILD small FD recurrence:** use installed torch.linalg.svd
  and tensor broadcasting, standard-library unittest/full-SVD oracle; do not add
  a numerical library for a small shrink. [Official PyTorch 2.11 SVD docs](https://docs.pytorch.org/docs/2.11/generated/torch.linalg.svd.html)
  confirm reduced shapes, descending s and Vh right-vector orientation; sign/basis
  ambiguity motivates invariant comparisons. Existing PyTorch 2.11.0+cpu reused.
  No dependency upgrade, new CUDA backend, CLI fork or older-project migration.

## Implementation and verification scope

`torchfrequentdirections/fd.py`: reference/fast functions plus streaming object;
explicit occupied count, nonmutating finalization, dtype/device/shape/finiteness
checks, normalized shrink and ell>d extension. Tests cover independent PSD and
two bounds, exact/low rank, random/decay/degeneracy, zero/empty/short, ell1,
ell=d/ell>d, noncontiguous input, signed rows, chunking, merge, snapshot/input
ownership, no retained autograd, invalid-input atomicity, overflow and SVD-count
reduction. A seeded sweep adds 12 shapes in both dtypes to the fixed fixtures.
No new public merge API or benchmark framework.

`benchmarks.py` measures CPU constructor+update+snapshot, excludes generation,
uses seed1729, one torch thread, two warmups and individual repeated wall times.
It does not publish, install, assert correctness or claim GPU acceleration;
correctness is the separate validation script. An instrumented test verifies
fewer SVD calls; timing and asymptotic work are distinct claims.

Reproduce with commands in [README](../../README.md). Exact committed-source
validation, clean wheel/install/import/benchmark smoke and bundled framework
regressions are captured in execution.json. Generated evidence is separate from
the tested source commit. No autograd, compile, accelerator, long-stream stress or
old-PyTorch compatibility validation is claimed. NumPy is absent in the reused
CPU environment; torch emits an initialization warning, but no numpy API is used.

GPU/TPU: **NOT RUN**, optional GPU deliberately left to a scoped follow-up; no
Colab commands/session creation/stop/readback occurred in this batch. Therefore
current remote server state is **NOT QUERIED**, not claimed authenticated zero.
No new paid resource, Actions, background service or research approval flow.
Batch repairs/failures/timings and handoff accounting are in execution.json;
no per-attempt approval handoff. GPT review and merge are pending.

## Observed final local batch

Tested source: `77754cc983e51d20b3dc31f225c81567f8ffe69b`, detached clean Linux
checkout. Python 3.12.3 / torch 2.11.0+cpu, AMD Ryzen 7 4800H; one torch thread.
**16 tests PASS**, including 216 recorded fixture/method/dtype/capacity results
and **956** k-specific bound comparisons, plus the 12-seed shape sweep inside
the suite. Numerical validation + observation capture took **0.806 s**.
Worst observed PSD/covariance/projection excesses were respectively 0.267 / 0.274 /
0.00155 times the documented tolerance (rounded upward); all below one.
These are empirical finite-precision checks, not exact PSD/proof claims.
Windows Python 3.12.14 framework regressions: **37 PASS**, 23.487 s suite time.
Final combined validation/build/install/bench process: **47.824 s**; this is not
the full task duration. UTC timestamps and per-operation wall times are primary.

Offline wheel build with installed uv 0.13.1/setuptools 78.1.0, fresh target install,
outside-checkout isolated import and installed benchmark all **PASS**. Dependencies
reuse the existing CPU torch environment; this is not a fresh dependency install.
The legacy TOML license table builds successfully but emits a setuptools
deprecation warning; a future packaging change should adopt SPDX metadata before
moving to a backend that removes that syntax. Root MIT text is included in wheel.

Actual commands (temporary paths represented by variables; no network/download):

```sh
# PYTHON is the existing Python 3.12.3 + torch environment; UV the observed uv executable.
git clone --quiet --no-hardlinks --no-checkout "$REPO" "$SOURCE"
git -C "$SOURCE" checkout --quiet --detach 77754cc983e51d20b3dc31f225c81567f8ffe69b
# cwd SOURCE, PYTHONDONTWRITEBYTECODE=1
"$PYTHON" scripts/validate_cpu.py
"$PYTHON" scripts/check_snapshot.py
"$UV" build --wheel --no-build-isolation --offline --no-python-downloads --python "$PYTHON" --out-dir "$WHEELS" "$SOURCE"
"$UV" pip install --offline --no-deps --no-python-downloads --python "$PYTHON" --target "$TARGET" "$WHEEL"
```

The import/CLI smoke used `python -I`, inserted only TARGET ahead of the existing
dependency environment, and asserted the imported package came from TARGET.
Six CPU benchmark cases used n/d/ell = 128/32/4, 256/64/8, 512/64/16 in both
dtypes, two warmups and five samples each. For example, float64 256/64/8 median
was **32.929 ms reference / 5.505 ms Fast**. This is this CPU implementation and
small workload only, not a general performance or GPU claim; all samples are in
execution.json. Benchmark execution PASS is separate from the numerical oracle.

Environment repairs were local: UTF-8 PDF output, report CRLF normalization,
missing pip/ensurepip and non-login WSL uv path, and cross-filesystem source
identity. Two clean-tree harness attempts failed before publication: even after
restoring LICENSE's Git bytes, WSL status reported a change with identical Git
blob hash and empty diff. The final validation uses an independent Linux checkout;
no algorithm repair or failed numerical assertion occurred. An automated policy
blocked recursive generated-cache deletion; ignored caches were retained and only
`__pycache__` excluded from the snapshot file-set check. No rule or source byte was
weakened to accept a changed snapshot. Source identity, package contents and all
131 pinned file hashes were checked in that independent checkout.
