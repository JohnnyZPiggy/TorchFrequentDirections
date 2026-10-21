# CPU, T4 and compilation: an evidence-based performance study

[Documentation hub](../README.md) · [Exact selected measurements](measurements.md) ·
[Original raw reports](../README.md#accepted-evidence-map)

Current synthesis, 2026-10-21, based on accepted evidence through PR #19.
**Direct SVD remains the production backend.** The lossless short-buffer snapshot
is shipped; QR-SVD, compiled shrink, ScratchFD and graph stages remain experiments.
The historical reports are immutable; their original pending-review language is
not a statement about present acceptance.

## How to compare measurements

The [generated measurement tables](measurements.md) identify reference/candidate,
ell, d, n, dtype, scope, medians, pairing, interval, CPU threads, transfer policy,
tested source SHA and original CSV/summary for each section. They are selected
observations, not a pooled ranking. Complete positive/negative/control rows remain
in each linked raw CSV/JSON.

- **Pure shrink** starts with SVD outputs; **factor-only** includes a decomposition;
  **full update+sketch** includes fresh construction, validation, sequential state,
  clearing and finalization. None is interchangeable with another.
- GPU host-wall timings synchronize before/after calls and include Python/dispatch
  waits. Resident inputs exclude generation and transfers. Only PR #4 also reports
  explicitly transfer-included measurements; stage timings are nonadditive.
- Median paired ratios differ from ratios of medians. Early PR #4 uses five
  fixed-order repeats, not randomized pairs. Later studies use two warmups and
  twenty pairs per block; descriptive bootstrap intervals do not supply independent
  sessions or multiple-comparison correction.
- Device, thread configuration, source, stream partition and CPU/GPU pairing context
  matter. **Never multiply ratios from independent timing blocks.** Positive local
  screens do not define a production dispatch threshold or universal crossover.

## Production Direct FD and the shipped snapshot optimization

The real-T4 baseline established scoped CPU/GPU mathematical agreement, not a
general GPU advantage. In PR #4 the one-thread Xeon CPU beat T4 on most small
cases; n=128,d=512,ell=32,float32 was a local exception (CPU/GPU median ~1.371
resident, ~1.362 transfer-included). Every sampled float64 Fast case lost on T4.
Five repeats and fixed block order limit tail estimates and causal conclusions.
[Original evidence](../../reports/gpu/issue3-20261020/summary.md).

PR #6 removes an SVD only when a snapshot has `m<=ell` pending rows. Copying C and
padding zeros preserves C.T C; snapshots remain independent and nonmutating.
For the n=98,d=512,ell=32,float32 T4 example, the historical **ratio of medians**
is ~6.47 for snapshot-only and ~1.171 for the complete stream; these are separate
measurements. Do not label 6.47 as end-to-end acceleration. Empty snapshots already
needed no SVD; long snapshots still shrink. Downstream PCA decomposition is not
included in these savings. [Original evidence](../../reports/gpu/issue5-20261020/summary.md).

The no-clear candidate passed logical update/sketch checks but exported different
unused physical tail state through implicit pickle. Small mixed CPU/T4 timing
signals did not justify adopting it. PR #8 retained the original clearing runtime;
accepted counterexample evidence is not a production no-clear optimization.
[Original evidence](../../reports/gpu/issue7-20261020/summary.md).

## QR + Small SVD: useful local results, no default switch

For a wide m-by-d factor input C, factor C.T=QR and then R.T=U diag(s) W.T;
the right factor for C is W.T Q.T. Apply the same normalized ell-th shrink.
This replaces a wide SVD with QR plus a smaller SVD, but adds QR, reconstruction
and dispatch overhead. Empty/short and m>d paths include documented direct fallback.

CPU studies used Ryzen 7 4800H/WSL2, Python 3.12.3, torch 2.11.0+cpu. PR #10
found small-shape regressions and wider local gains; the universal screen failed.
PR #12 independently scanned fixed ell/fixed d, constant-n controls, severe spectra
and 1/2/4/8 threads, two reverse-order batches. Gains and losses depend on shape
and threading; the near-parity shape can regress at two threads. One A/A flag
exposed false apparent slowdown. Neither an aspect-ratio threshold nor a backend
switch was accepted. [CPU #9](../../reports/cpu/issue9-20261021/summary.md) ·
[CPU #11](../../reports/cpu/issue11-20261021/summary.md).

![Comparable T4 Direct/QR full-stream results](../assets/gpu-benchmarks.svg)

*PR #14: every float32 long-final-buffer compression configuration in one T4
study, including larger parity/negative controls. Points: median paired Direct/QR;
whiskers: descriptive 95% paired bootstrap. Each row has its own n/ell/d.
[Selection, values and sources](../assets/README.md).*

Real T4 QR correctness passed in float32/64. For m=32 (ell=16), full-stream paired
ratios were locally ~1.19–1.36 across both dtypes and occupancies. The m=64,d=1024
and m=128,d=2048 streams failed the >1.05 screen, near parity. Four A/A blocks
had no flags, but there was no independent-session replication. These findings
do not inherit the CPU crossover and do not establish default integration.
[Original T4 evidence](../../reports/gpu/issue13-20261021/summary.md).

## Inductor fusion is measured; full FD acceleration is not

PR #16 tested `torch.compile` with real CUDA Inductor, fullgraph=True and dynamic
shapes **only for the pure pointwise shrink helper**. Four backend graphs/kernels
were generated; zero new graphs appeared during primary steady timing. Observed
pure-shrink paired ratios were ~1.545–1.610; factor-only ~1.005–1.049. All **30
full-workload >1.05/lower>1 screens failed**; one A/A flag and fallback regressions
further limit inference.

![Independent pure shrink and full-stream kernel-count scopes](../assets/kernel-fusion.svg)

*One pure shrink: 10 eager launches to one Triton launch. A separately profiled
full stream: 709 to 664 for float32, ell=16,d=512. These are kernel counts, not
10x time savings or whole-FD fusion. [Figure provenance](../assets/README.md).*

SVD, finite-check Python bool conversions, buffer/state loops and snapshots remain
eager. CPU Dynamo explain reported pure shrink 1 graph/0 breaks, guarded factor
2/1 and whole stream 4/3. Separate CPU fullgraph attempts failed with an Unsupported
builder error; that is not proof that SVD cannot appear in a graph. AOT-eager
capture is not Inductor fusion. Local CPU Inductor could not compile because a C++
compiler was absent. CUDA full-stream fullgraph was not subsequently established.

Measured codegen/new-graph first calls totaled ~6.333 seconds, separately from
steady timing; imports, source loading and total remote process wall are different
scopes. GPU event intervals also contain dispatch gaps. In one float32 ell=64,
d=2048,final=63 case, CUDA eager took 41.146 ms in the CPU-paired block versus
19.466 ms in the CUDA-paired block. The CPU/partial ratio 5.713 is **not** a compile
gain: the same-device eager/partial paired ratio was 1.012. Do not multiply those
independent ratios. The cause of context dependence was not diagnosed.
[Original compilation evidence](../../reports/gpu/issue15-20261021/summary.md).

## CUDA Graphs and strict ScratchFD controls

The first PR #18 graph attempt failed capture, followed by a CUDA RNG offset
error in the same process. Its 72 GPU fixture gates passed, but **zero benchmark,
A/A, replay or profiler blocks ran**. There is no zero-latency measurement to plot.
[Retained failure record](../../reports/gpu/issue17-20261021/summary.md).

PR #19 isolated eight stage/dtype probes and launched the strict benchmark in a
ninth fresh process. Pointwise float32/64 capture and five changed-input replays
passed. All six SVD-containing stages first failed at captured
`cusolverDnSgesvdj`/`cusolverDnDgesvdj` with CUSOLVER_STATUS_EXECUTION_FAILED;
teardown then reported capture invalidation. This localizes the tested
preallocated-out SVD call, not universal cuSOLVER/CUDA Graph incompatibility.
Graph tensor stages omit strict host finite/error decisions and are explicitly
**non-equivalent diagnostic workloads**, not a strict full-stream graph.

The separate benchmark passed RNG/device probes and all 26 shapes. It compared
original CUDA eager against semantically strict ScratchFD local scratch reuse,
including every update/check/clear and snapshot. Across 110 twenty-pair blocks,
shape order, context order and AB/BA pairs were randomized, with global A/A plus
per-shape A/A before/after anchors. Pairing-context and baseline drift >5% reject
a performance conclusion. **0/26 screens passed; 11/26 were control-inconclusive.**
Eager/Scratch paired medians ranged 0.9530–1.0257. CPU-paired CUDA versus CUDA-paired
CUDA drift reached +90.64%; global A/A near one did not prove every local context
stable. No general device or ScratchFD speedup follows. Four profiles retained
identical kernel counts (715/779/739/803), 16 finite ops, 21 item ops and five SVDs
on each path. CPU model/affinity were not captured in that worker; one-thread
settings and versions were. [Original follow-up](../../reports/gpu/issue17-followup-20261021/summary.md).

## Mathematical contract

For C=U diag(s) Vh, with r=min(m,d), conceptually zero-pad s beyond r.
Let delta=s[ell-1]^2 (one-based sigma_ell; delta=0 when ell>r) and
w_i=sqrt(max(s_i^2-delta,0)). Retain at most min(ell-1,d) directions after a
**full** buffer shrink. Each later shrink depends on this retained state: independent
batched SVDs would change the recurrence.

The code normalizes by max(s_1,tiny) before squaring to avoid finite-value square
overflow. A short snapshot `B=[C;0]` has B.T B=C.T C in exact arithmetic and adds
no loss; it can have rank ell. A long snapshot applies the same threshold, without
changing stream state. If ell>d, missing singular values are zero: exact covariance
in exact arithmetic, with repeated-SVD roundoff in practice.

Each shrink has nonnegative directional covariance loss at most delta and energy
loss at least ell*delta. Telescoping over insertions, with Delta the sum of deltas,
gives the documented FD bounds for integer 0<=k<ell, k<=d:

```text
0 <= A.T A - B.T B <= Delta I
Delta <= ||A-A_k||_F^2 / (ell-k)
||A-A V_k V_k.T||_F^2 <= ell/(ell-k) * ||A-A_k||_F^2
```

V_k is B's top-k right singular space, not a slice of raw output rows. Ties allow
multiple bases; no cross-device elementwise reproducibility is promised.
The [original publication](https://arxiv.org/abs/1501.01711v2) and
[retained reconstruction](../research/frequent-directions-tensorization.md) give
the exact-arithmetic reasoning and project extensions.

Numerical validation uses independent CPU float64 Gram eigensolves, full-SVD tails,
every eligible k projection and cross-variant Gram gaps. The fixed absolute slack
is `32*eps(input dtype)*max(n,d,2ell,1)*||A||_F^2`; zero input has zero slack.
This energy/dimension-scaled empirical gate is **not** tiny-spectrum relative
accuracy, a uniform backward-error result or an arbitrary-stream theorem.
NaN/Inf/shape/dtype rejection, no-grad, aliases, continued updates and injected
failures have separate coverage. Production input checks are atomic; a backend
failure can leave a partially updated instance, which must be discarded.

## Source and validation ledger

The SHA below is the **actually tested source**, not the later merge/publication
head. Links retain original manifests and full counts. “Tests” counts CPU unit
tests at that milestone, not GPU tests; fixture/oracle counts are separate.

| Accepted PR / scope | Tested source SHA | Recorded CPU suite | Original manifest / evidence |
|---|---|---:|---|
| #2 CPU baseline | `77754cc983e51d20b3dc31f225c81567f8ffe69b` | 16 | [Manifest](../../reports/issue1/manifest.json), [CPU](../../reports/issue1/cpu-validation.json) |
| #4 T4 baseline | `608975335817431d841e0ad313a21d4930304a46` | 21 | [Manifest](../../reports/gpu/issue3-20261020/manifest.json) |
| #6 short snapshot | `687ea56596d1b47633b2abf5f94b3feffe04ec31` | 26 | [Manifest](../../reports/gpu/issue5-20261020/manifest.json) |
| #8 no-clear candidate | `1fab940d4f3987e0b03b7b5fad8540a9ccd51fb2` | 30 candidate; 31 retained runtime | [Manifest](../../reports/gpu/issue7-20261020/manifest.json) |
| #10 CPU QR | `f5894820dcfdf5bd977384ea24033c10a7793aa2` | 38 | [Manifest](../../reports/cpu/issue9-20261021/manifest.json) |
| #12 CPU shape/threads | `f49068088b2b970da6ba0f585d132d6440d8f7da` | 42 | [Manifest](../../reports/cpu/issue11-20261021/manifest.json) |
| #14 T4 QR | `23b5aa13ef926ecd5df36908f2277fc6e1c77b00` | 44 | [Manifest](../../reports/gpu/issue13-20261021/manifest.json) |
| #16 partial compile | `0fbf27bf23d349aafe2bd14e4ffbd0f934f395f6` | 47 | [Manifest](../../reports/gpu/issue15-20261021/manifest.json) |
| #18 first graph batch | `6a4900c25314af856ee4e37a59b875d05dd6d768` | 50 | [Manifest](../../reports/gpu/issue17-20261021/manifest.json) |
| #19 isolated follow-up | `d62ae254b0c9252cb68f13ffff8017ae79c32dc6` | 52 | [Manifest](../../reports/gpu/issue17-followup-20261021/manifest.json) |

Merge SHAs can be inspected through the [accepted PR links](../README.md#accepted-evidence-map).
For example PR #19 merged as `f97070dddc1cb1788d4b1fab85c58da650a9f552`, whereas
`d62ae254...` is the measured executable source. The no-clear source was measured
then removed from production before publication. Original hashes/raw data remain
unchanged; this docs-only synthesis ran no new GPU or performance workload.
