# Frequent Directions: R1–R3 engineering reconstruction

Issue: [#1](https://github.com/JohnnyZPiggy/TorchFrequentDirections/issues/1).
Scientific decision: **core formulas provisionally RESEARCH_READY**, recorded by
GPT Architect/Mentor in that Issue. Codex records implementation reasoning and
numerical evidence here; this is not an independent GPT scientific acceptance.
Source: [Ghashami, Liberty, Phillips & Woodruff, arXiv:1501.01711v2](https://arxiv.org/abs/1501.01711v2),
Algorithms 2.1/3.1, Properties 1–3 and §3.1. Local PDF pages 5/7 visually checked,
relevant proof/merge sections and TeX source inspected. No paper text is republished.

## R1 — Objects, assumptions and dependency

Rows of finite real `A ∈ R^(n×d)` arrive in order; `ell ≥ 1`, `d ≥ 1`.
Return `B ∈ R^(ell×d)`. Each shrink depends on the previous sketch and following
input rows. SVD and broadcasting are tensor operations, but the stream recurrence
cannot be replaced by independent batched SVDs while preserving this algorithm.
Neither a matrix-free `Ax` interface nor gradient calculation is required here.

For temporary `C ∈ R^(m×d)`, let `r=min(m,d)`. Reduced SVD yields
`U:(m,r), s:(r,), Vh:(r,d)`, with descending nonnegative `s`.
Right singular vectors are ROWS of `Vh`, not columns of U.

## R2 — Tensor formula and alternatives

1. Reference: start with zero `ell×d`; replace its last zero row with each input
   row; SVD and shrink after every insertion (Algorithm 2.1).
2. Shrink: zero-pad singular values conceptually; `delta=s[ell-1]^2`, or zero if
   `r<ell`. Compute `w_i=sqrt(max(s_i^2-delta,0))`; canonical sketch rows are
   `w[:,None]*Vh`. Only `ell-1` directions survive a positive-threshold shrink.
   PyTorch uses `full_matrices=False`, avoiding unused full orthogonal bases.
3. Fast: fill a `2ell×d` buffer; on full, shrink at the SAME ell-th singular
   value, preserve at most `min(ell-1,d)` directions and append new rows. No
   data-dependent zero-row detection is used (Algorithm 3.1).
4. Snapshot extension: Algorithm 3.1 returns its temporary buffer, whereas this
   API promises `ell` rows. If the pending count exceeds ell, apply one further
   identical shrink before padding; otherwise represent all pending rows exactly
   by copying occupied rows and padding zeros. This extension obeys the same loss properties below. Snapshot
   evaluation is nonmutating; repeated observation does not change the stream.
5. `ell>d`: missing ell-th singular value is zero, so retain all d directions
   and pad zeros. This is an exact-covariance extension outside the useful
   compression regime; repeated SVD roundoff still accumulates.

For any shrink, directional covariance loss is nonnegative and at most delta;
energy loss is at least `ell*delta`. Summing deltas gives `Delta` and telescopes
these properties across insertions. At a final partial shrink with `m>ell`, the
first ell singular values each lose delta; remaining loss is between zero and
delta. Thus the same three properties apply. Loss-free short snapshots have
delta=0. For `ell>d` the zero-delta extension preserves covariance in exact
arithmetic. By the paper's argument, for integer `0≤k<ell` (take `k≤d`):

```
0 <= A.T@A - B.T@B <= Delta*I
Delta <= ||A-A_k||_F^2 / (ell-k)
||A - A@V_k@V_k.T||_F^2 <= ell/(ell-k) * ||A-A_k||_F^2
```

Here `V_k=Vh_B[:k].T`; k=0 means zero projection, k=d means full-space projection.
The projection statement is about A projected onto B's top-k right space, not
`||A-B||`. Different schedules need not have identical B or covariance matrices.
§3.1 merge: applying FD to stacked sketches preserves these properties against
the concatenated original streams; a regression checks that invariant.

| Path | SVD work for ell≤d | State | Tradeoff |
|---|---|---|---|
| Reference, native SVD | O(n*d*ell²) | O(d*ell) plus SVD workspace | Simple baseline, per-row barrier |
| Buffered Fast | O(n*d*ell + d*ell²) including finalization | O(d*ell) with factor-two buffer plus workspace | About one shrink per ell+1 new rows after first fill |
| ell>d exact fallback | native SVD O(m*d²) per call, m≤2ell | O(d*ell) plus workspace | No compression advantage; bounded correctness extension |

No custom Gu–Eisenstat update, randomized SVD, dense n×n intermediates or
third-party algorithm package is introduced. Fast reduces decomposition count;
actual latency still depends on shapes, backend and validation overhead.

## R3 — Numerical contract, evidence and open hypotheses

Supported inputs: float32/float64 dense finite tensors. Inference only: updates
detach and execute under no_grad. Repeated/zero singular values make vector
orientations nonunique; no cross-hardware elementwise reproducibility claim.
SVD failures propagate. GPU scalar finite checks would synchronize; CUDA behavior
and suitable exact-SVD backend are untested. No torch.compile or TPU promise.

Implementation normalizes singular values by the largest (with tiny lower bound)
before squaring to avoid overflow from finite values such as 1e20 in float32.
Clamp subtraction roundoff at zero; near-threshold directions may be lost.
Finite-input SVD can still overflow/fail for extreme scales; nonfinite factors
raise FloatingPointError. This is finite-precision validation, not a new proof.

The oracle uses independent float64 full-SVD tail energy, symmetric Gram
eigvalsh and explicit right-space projection, never the shrink formula. Assert
PSD lower bound and both inequalities with absolute slack
`32*eps(input_dtype)*max(n,d,2ell,1)*||A||_F²`. Slack scales with energy (zero
input has zero tolerance) and dimensions, allowing accumulated SVD roundoff.
It is an empirical small-matrix tolerance, not a uniform backward-error theorem
for arbitrary long streams. Exact rank and ell>d covariance get separate tighter
float64 checks. A deliberately truncated partial buffer is a failing negative
control. Random seeds, raw errors and tolerance are in the linked evidence.

See [engineering report](../engineering/frequent-directions-engineering.md) and
[CPU observations](../../reports/issue1/cpu-validation.json) for actual outcomes.
GPU transfer vs SVD cost, throughput over n/d/ell, memory pressure and fair
CPU/GPU crossover are **HYPOTHESES / NOT RUN**. This CPU-first Issue permits
optional GPU; none is needed to claim only the local milestone. GPT Architect
must independently review this reconstruction, final-partial-buffer extension,
tolerances and evidence before acceptance/merge or an accelerator follow-up.

## Issue #5 — lossless short-buffer snapshot (R1–R3)

Architect dispatch and scoped **RESEARCH_READY** rationale:
[Issue #5](https://github.com/JohnnyZPiggy/TorchFrequentDirections/issues/5).
Codex records the proof-oriented implementation below; final scientific acceptance
remains with GPT Mentor/Architect.

R1: after any number of buffered updates, let C be the occupied m-by-d prefix.
The public API promises an independent ell-by-d factor with FD Gram/projection
bounds, not canonical row orientation. No inspected project caller requires a canonical factor;
the tests compare Gram matrices, right subspaces and FD invariants. The old short
snapshot computes C=U Sigma Vh and returns [Sigma Vh;0].

R2: for 0<=m<=ell, choose B_new=[C;0]. Then, in exact arithmetic,
`B_new.T B_new = C.T C = V Sigma^2 V.T = B_old.T B_old`, because U has
orthonormal columns in the reduced SVD. This also covers m=0, rank deficiency,
ell=1 and ell>d. Consequently both factors have identical eigenvalues and top-k
right eigenspaces when the spectral gap is positive. Ties allow different choices
inside a shared eigenspace; the original projection inequality still holds for a
top-k right singular subspace. Padding adds no loss and no new Delta. It does not
change future shrink schedules: snapshots never mutate buffer/count.

The new path copies exact finite entries (no decomposition roundoff) but Gram
products still round. Equality to the numerically computed old SVD factor is only
up to roundoff, so tests use independent float64 Gram and projection oracles.
The representation change is observable: rows are no longer SVD-oriented or
orthogonal, and row slicing is not a top-k projection API. Callers needing right
singular vectors must decompose the returned sketch. Such a decomposition remains
necessary for PCA, so this optimization does not claim to eliminate downstream PCA SVD.

R3: keep the m>ell shrink fallback and all input validation, invalid-update
atomicity, finite output checks, dtype/device, no-autograd and alias isolation.
The m=0 path stays exact zeros without SVD. Tests cover boundary occupancy,
rank-deficient/zero cases, repeated observations and continued updates after
several fills. A/B uses the exact previous main source in the same runtime,
snapshot-only versus full update+snapshot, both active and inactive paths.
The hypothesis is one saved SVD for each nonempty short snapshot; empty already
used zero SVDs. Latency/synchronization benefit needs actual CPU/GPU measurements.
No change to finite checks or full-buffer zeroing is included. Secondary zero-clear,
compile/fusion and alternate SVD candidates are deferred to retain causal attribution.

## Issue #7 — occupied-prefix equivalence before removing full-buffer zeroing

Conditional research decision: GPT Mentor/Architect's
[Issue #7](https://github.com/JohnnyZPiggy/TorchFrequentDirections/issues/7).
This is a single-line candidate, not a performance conclusion or merge approval.

R1 / read-path audit: constructor buffers stay torch.zeros-initialized. Fast update
writes appended rows before increasing count; a full-buffer _factor reads only
after count reaches 2ell. Snapshot reads [:count], then either creates a fresh
zero-padded short factor or factors that prefix. Reference FD has separate state
and is not modified. _validate reads incoming rows, shape/dtype/device and finite
status, not unoccupied buffer data. _factor always allocates independent output
using new_zeros before reading SVD results; its output cannot alias the old buffer.
No production serializer exposes the private buffer. Existing test-only full-buffer
clones check nonmutation internally; numerical oracles/diagnostic reports receive
only snapshots and scalar counts/timings, never raw unused rows. Arbitrary mutation
of private/configuration fields is outside this maintained-state contract.

R2: define the logical state as (r,C), C=buffer[:r]. Two physical buffers are
equivalent if they share r and C (or their corresponding FD Gram under SVD basis
ambiguity), regardless of their unused tail. A full shrink computes an independent
factor F, retains q=min(ell-1,d), and writes F[:q]. Clearing the rest is unobservable.
On append, [r:r+t] is completely overwritten before r grows. Induction on appends
and full shrinks shows that no old tail enters any later factor/SVD input; the
same shrink sequence therefore obeys the original covariance/projection guarantees.
Snapshots create fresh output from the logical prefix, so they cannot reveal tail
bytes, including when ell=1 resets occupancy to zero or ell>d retains only d rows.
No retained-prefix cleaning is required because F storage is independent and the
entire retained prefix is overwritten. Newly allocated buffers remain initialized.

Exception audit: invalid inputs are rejected before mutation. SVD/backend or
nonfinite-factor failures occur before the clear/reset point in both versions;
they leave the same fully occupied valid input buffer and propagate the same
exception class. Such instances are discarded per the public contract. Snapshot
finite checks remain unchanged. No finite check is removed, moved or weakened.

R3: tests deliberately poison unused rows with NaN, inspect every factor input,
compare old/new Gram matrices and independent FD bounds after every update/snapshot,
and verify alias isolation, noncontiguous input, rank deficiency, zero rows,
ell=1/ell>d, full cycles and failure paths. NaN poison and raw trace tensors remain
test-local; they are never saved in reports. The actual previous source is the A/B
baseline. Separate resident zero_ versus no-op timings do not imply additive FD
savings. Full update+snapshot A/B includes no-shrink controls and wide/large-ell
cases, balanced shuffled pairs, SVD/finite/clear counts and real CPU/T4 evidence.
Candidate signals require an observed >=1% median paired gain with a descriptive
paired-bootstrap lower bound above one; all workloads/controls must be reported.
The interval is not cross-session or multiple-comparison-adjusted proof. A null
result is valid and should retain the baseline rather than force this optimization.
