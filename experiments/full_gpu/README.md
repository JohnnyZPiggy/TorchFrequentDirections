# Issue #17: strict Fast FD scheduling and graph exploration

This is an isolated EXPLORE experiment, not a production backend. The production
Direct SVD implementation, framework snapshot and default API are unchanged.

## Mathematical and state boundary

For singular values s, let c=max(s[0], tiny), z=s/c, and t=z[ell-1]^2.
Both implementations compute c sqrt(max(z^2-t,0)) and multiply the same right
singular vectors. ScratchFD computes t **before** overwriting z, reuses only this
local quotient tensor, and writes the product directly to a fresh zero output.
No shared input, SVD output or object storage is overwritten by these operations.
There is no persistent workspace and no extra serialized state. This reduces
temporary allocations; it does not remove kernel launches by compilation.

The inherited update validates the complete input before mutation, preserves
every factor output finite check before buffer clearing/copying, and keeps the
original zeroed unused storage and snapshot ownership. The occupancy recurrence
and shrink frequency are identical. Backend SVD failure still requires discarding
the affected instance, as in production. Reference-mode calls are also inherited.
In-place scratch arithmetic is not a claim of universally bitwise reproducible
SVD across devices. CPU tests require exact parity on the tested same-device
paths; GPU gates additionally use independent float64 covariance/projection bounds.

The independent oracle uses A and returned B, not either implementation's shrink:
PSD(A' A-B' B), every eligible k spectral-tail/(ell-k) covariance bound, right
subspace residual bound and cross-variant Gram difference. Tolerance is fixed at
32 eps(input dtype) max(n,d,2ell,1) ||A||_F^2, including absolute treatment of zeros.
This finite fixture grid is not a proof for arbitrary streams or tiny relative
singular values. Tests cover partitions, occupancies, extreme scales, ties,
noncontiguous inputs, no-grad, snapshots, serialization, invalid-input atomicity,
injected backend/nonfinite outputs and corrupt short snapshots.

## Synchronization and CUDA Graphs

Production fd.py input validation (line 44), factor result (line 60), and short
snapshot result (line 105) each convert a CUDA reduction to a Python bool. These
are mandatory host decisions, normally represented by item/local-scalar reads
and synchronization. SVD also has backend status/dispatch behavior requiring
measurement. Buffer zero/copy, output allocation and all streaming Python loops
remain; removing validation or physical clearing would change the contract.

`graph_transition` intentionally omits host finite/atomicity predicates. It is
**NON_EQUIVALENT**, a diagnostic of one fixed-shape SVD + normalized shrink + full
buffer reset/write, not a full update/sketch graph or drop-in optimization. Capture
uses stable input/output addresses, warmed side stream, actual CUDAGraph, new
external input copies and a complete reset on each replay. A capture exception
is recorded verbatim with bounded length; no backend patches or allocation retry.
Successful replay must match a fresh eager buffer for five changed inputs. Strict
full-stream capture is unavailable because the mandatory host predicates remain.
E3 compiled shrink is intentionally omitted: #15 already measured that partial
fusion; this batch targets strict scratch scheduling and graph viability.

## Timing protocol

The committed driver bootstraps exact Git blobs with SHA-256 checks. CPU tests
and both validation scripts run on a separate clean exact-SHA Linux checkout.
An immutable controller handles the sole optional T4 allocation, finite 600/540/
420/60-second task/allocation/work/cleanup limits, artifact download and stop.
Independent authenticated account readback follows controller cleanup.

Initial CUDA A/A precedes candidates. A seeded shuffle determines shape/block
order, and a second shuffle chooses CPU-paired versus CUDA-only context order.
Within each shape, A/A before and after bracket both balanced AB/BA contexts;
each block has two warmups and twenty pairs. Fresh state construction, every
update, finite check, and sketch are timed with device synchronization before
and after each call. Resident inputs, random generation and H2D are excluded.
One CPU thread is used. First workload calls, first CUDA probe and graph setup
are distinct from steady-state samples. No cross-context ratio multiplication.

Every raw pair, ordering, median/p95/sample SD and seeded descriptive bootstrap
is retained. More than 5% A/A median deviation, bracketing baseline drift or
CPU-context CUDA baseline drift marks performance INCONCLUSIVE. Final material
A/A also invalidates all screens. Otherwise the exploratory screen requires
paired E0/E1 median >1.05 and lower bootstrap bound >1, plus math/safety gates.
Profiler kernel counts and selected host wait calls are intrusive separate calls,
not additive latency components. Runtime detail lists are capped by the reused
profiler helper at 120 entries; CPU op counts and total kernel counts remain.
Temperature/clocks/power/utilization are unprivileged nvidia-smi observations,
not causal attribution or controlled clock locking. No transfer-inclusive claim.

See [the evidence packet](../../reports/gpu/issue17-20261021/summary.md) for actual
outcomes, exact source identity, failures, raw samples and cleanup verification.
