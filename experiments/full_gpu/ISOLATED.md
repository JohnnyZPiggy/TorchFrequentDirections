# Issue #17 follow-up: disposable CUDA Graph probes

The previous shared-process capture attempt invalidated capture and then CUDA
random generation failed. `isolated_driver.py` creates no torch/CUDA context.
It hash-checks exact Git source, then invokes eight sequential fresh Python execs:
pointwise-only, SVD-only, SVD+normalized-shrink, full tensor-buffer reset/write,
each in float32 and float64. Each child is killed and waited on after ten seconds.
Neither failure, timeout, nonzero exit nor invalid JSON prevents the next child.
These are ordinary finite task subprocesses, not background services.

Each graph child has fixed `(32,64)` input, ell=16, preallocated U/S/Vh, factor,
pointwise and physical buffer storage; two side-stream warmups precede actual
CUDAGraph capture. Its stage and phase are saved before risky calls. An operation
exception is saved **inside** capture before context exit can replace it; teardown
errors are recorded separately. A failed child returns without any further CUDA
calls or RNG generation. Successful capture replays five separately seeded inputs,
checks against fresh eager results and verifies stable storage addresses. Capture
and replay timings are separate. SVD-only validates singular values, not vector
orientation; the combined stages validate factors/buffer with fixed epsilon bounds.
All graph stages are PROBE_ONLY and omit strict host finite/error/atomicity checks;
none is a full-stream graph or a production-equivalent optimization.

A ninth fresh exec runs **only strict eager math and performance**, never graph
capture. Its CUDA RNG and QR/SVD device probe explicitly test a healthy context
after all previous children exit. The original ScratchFD class is byte-unchanged;
its local-scratch math, finite predicates and storage contract remain as described
in [README.md](README.md). CPU tests validate preallocated stage algebra, exact
candidate/reference state, serialization, safety and independent float64 bounds.
Actual CUDA capture is not inferred from CPU operation success.

The benchmark reuses `driver.study`: randomized shape order and per-shape context
order; balanced twenty AB/BA pairs, two warmups; A/A before and after each shape,
plus global A/A controls. Both contexts include complete fresh construction,
update and sketch, synchronized host-wall timing, resident inputs and one CPU
thread. Generation/H2D are excluded. First-workload calls and CUDA initialization
are separate from warmed samples. No compile backend or production changes.
More than 5% placebo median, anchor drift or CUDA-baseline pairing-context drift
makes the screen INCONCLUSIVE. Raw paired data and descriptive seeded bootstrap
intervals are retained; no cross-context ratio multiplication or causal thermal
claim. Global final A/A can invalidate every screen.

Completed benchmark blocks are checkpointed outside timing, so a later failure
does not discard earlier raw evidence. The shape-start cutoff is 245 seconds in
the benchmark process, with a 300-second child bound. Graph children have a
90-second parent elapsed cutoff; immutable controller work bound is 420 seconds,
allocation 540, task 600 and cleanup reserve 60. Source download is also bounded.
The outer host controller has a 660-second cap, stop and independent authenticated
cleanup readback. This batch allows one new T4 allocation and no allocation retry.

Profiler calls are separate intrusive diagnostics, not additive elapsed stages.
Actual kernel counts and selected SVD/isfinite/item CPU ops are kept; the existing
helper caps individual runtime wait/launch entries at 120. GPU clocks, temperature,
power and utilization are observed without privileges, not locked or controlled.

See [the follow-up evidence](../../reports/gpu/issue17-followup-20261021/summary.md)
for actual capture outcomes, benchmark coverage, uncertainty and provenance.
