# Documentation

**Current status as of 2026-10-21**, based on main
`f97070dddc1cb1788d4b1fab85c58da650a9f552` and accepted evidence through PR #19.
Production remains Direct SVD. Accepting a research report does not enable its
prototype or promise acceleration.

## Read by purpose

| Goal | Entry point |
|---|---|
| Install and use the public API | [Project README](../README.md) |
| Understand current guarantees | [Mathematical contract](performance/README.md#mathematical-contract) |
| Read the derivation, snapshot proof and no-clear counterexample | [Foundational research record](research/frequent-directions-tensorization.md) |
| Audit original implementation and input rights | [Engineering provenance](engineering/frequent-directions-engineering.md) |
| Compare CPU, T4, Inductor and graph findings | [Integrated study](performance/README.md) |
| Inspect medians, protocols and tested SHAs | [Generated measurements](performance/measurements.md) |
| Recreate three original figures | [Figure sources](assets/README.md) |
| Verify this docs-only change | [Verification](VERIFICATION.md) |

Foundational docs and all `reports/` directories remain contemporaneous records.
Older “GPU NOT RUN” or “review pending” statements describe that milestone, **not
current status**. Use this index for later disposition and each original report
for exact measured source and limitations. Test, fixture, oracle, per-k inequality
and timing-block counts describe different scopes and must not be conflated.

## Accepted evidence map

All linked PRs are merged. Tested and merge SHAs differ: see the
[source ledger](performance/README.md#source-and-validation-ledger).

| PR | Meaning of acceptance | Original evidence | Experiment entry point |
|---|---|---|---|
| [#2](https://github.com/JohnnyZPiggy/TorchFrequentDirections/pull/2) | CPU production baseline and math | [Issue #1](../reports/issue1/cpu-validation.json), [engineering](engineering/frequent-directions-engineering.md) | [CPU validator](../scripts/validate_cpu.py) |
| [#4](https://github.com/JohnnyZPiggy/TorchFrequentDirections/pull/4) | Scoped real-T4 correctness and CPU/GPU pilot | [Issue #3](../reports/gpu/issue3-20261020/summary.md) | [GPU driver](../experiments/fd_gpu/driver.py) |
| [#6](https://github.com/JohnnyZPiggy/TorchFrequentDirections/pull/6) | **Shipped** short-prefix snapshots | [Issue #5](../reports/gpu/issue5-20261020/summary.md) | [Snapshot driver](../experiments/snapshot/driver.py) |
| [#8](https://github.com/JohnnyZPiggy/TorchFrequentDirections/pull/8) | No-clear evidence accepted; runtime rejected | [Issue #7](../reports/gpu/issue7-20261020/summary.md) | [No-clear driver](../experiments/zero_clear/driver.py) |
| [#10](https://github.com/JohnnyZPiggy/TorchFrequentDirections/pull/10) | CPU QR math, mixed performance; keep Direct | [Issue #9](../reports/cpu/issue9-20261021/summary.md) | [CPU QR study](../experiments/qr_svd/README.md) |
| [#12](https://github.com/JohnnyZPiggy/TorchFrequentDirections/pull/12) | CPU shape/thread and negative controls | [Issue #11](../reports/cpu/issue11-20261021/summary.md) | [Shape/thread study](../experiments/qr_svd/SHAPE_THREADS.md) |
| [#14](https://github.com/JohnnyZPiggy/TorchFrequentDirections/pull/14) | Scoped T4 QR math/gains; no dispatch | [Issue #13](../reports/gpu/issue13-20261021/summary.md) | [CUDA QR study](../experiments/qr_svd/CUDA_STUDY.md) |
| [#16](https://github.com/JohnnyZPiggy/TorchFrequentDirections/pull/16) | Pointwise Inductor fusion; no full-FD speed screen | [Issue #15](../reports/gpu/issue15-20261021/summary.md) | [Compilation study](../experiments/compile_fd/README.md) |
| [#18](https://github.com/JohnnyZPiggy/TorchFrequentDirections/pull/18) | Bounded graph failure; **no timing blocks** | [First #17 batch](../reports/gpu/issue17-20261021/summary.md) | [Full-GPU study](../experiments/full_gpu/README.md) |
| [#19](https://github.com/JohnnyZPiggy/TorchFrequentDirections/pull/19) | Isolated SVD failure + strict timing controls; no ScratchFD gain | [#17 follow-up](../reports/gpu/issue17-followup-20261021/summary.md) | [Isolated study](../experiments/full_gpu/ISOLATED.md) |

Entry points are archived research code, not public API recommendations or
instructions to allocate hardware. GPU reproduction has a separate bounded-source,
entitlement and cleanup protocol; reading docs or running figure tools allocates
no GPU.

## Governance and provenance

[Project rules](../AGENTS.md) and [configuration](../sae.yaml) pin SAE-Framework
`a60ed92d80a8866b413b928d609cc3afb82ec408`. The
[inventory](../reports/issue1/framework-inventory.json) records 131 files and
historical template exceptions. Original reports, manifests, download hashes and
tested SHAs are preserved byte-for-byte.
