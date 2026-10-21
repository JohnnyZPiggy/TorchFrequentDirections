# TorchFrequentDirections

[English](README.md) | [简体中文](README.zh-CN.md)

Streaming matrix sketches in native PyTorch: reference and buffered Frequent
Directions for covariance approximation and right-subspace estimation.
Includes CPU and NVIDIA T4 experiments, PyTorch compilation studies, and reproducible performance reports.

![MIT license](https://img.shields.io/badge/license-MIT-0f766e)
![Research software](https://img.shields.io/badge/status-research_software-334155)

[Documentation](docs/README.md) · [Mathematics](docs/performance/README.md#mathematical-contract) ·
[Performance study](docs/performance/README.md) · [Measured data](docs/performance/measurements.md)

## Start with the production API

From this checkout, with Python >=3.10 and a compatible PyTorch >=2.0 already
installed, install the local package:

```sh
python -m pip install --no-deps --no-build-isolation .
```

The local build requires setuptools >=68. Alternatively, `pip install .` can
resolve build and runtime dependencies.

```python
import torch
from torchfrequentdirections import FrequentDirections, fast_fd, reference_fd

generator = torch.Generator().manual_seed(1729)
a = torch.randn(100, 20, generator=generator, dtype=torch.float64)
b = fast_fd(a, ell=8)
reference = reference_fd(a, ell=8)

stream = FrequentDirections(8, a.shape[1], dtype=a.dtype, device=a.device)
stream.update(a[:37]).update(a[37:])
snapshot = stream.sketch()
assert b.shape == reference.shape == snapshot.shape == (8, 20)
assert snapshot.dtype == a.dtype and snapshot.device == a.device
assert not snapshot.requires_grad
```

`reference_fd` shrinks after each row; `fast_fd` and the default stream use a
buffer with capacity `2*ell` and **Direct SVD**. These schedules need not return the
same factors. `sketch()` returns an independent copy without changing state;
updates may continue afterward. Configure `device` and `dtype` to match input:
there is no implicit conversion. CUDA requires a suitable existing torch install;
the example above is deliberately CPU-only.

## What is shipped, and what is research?

| Capability | Current status |
|---|---|
| Reference + buffered Fast FD; strict finite checks | Production, CPU and scoped real-T4 correctness evidence |
| Short-buffer snapshot without SVD | **Shipped**: lossless copy/padding for occupied count <=ell; PR #6 |
| QR + Small SVD | Experimental CPU/CUDA modules; shape/thread-dependent gains and losses; no dispatch |
| `torch.compile` / Inductor | Experimental **pointwise shrink** fusion measured on T4; no accepted full-FD acceleration |
| ScratchFD local tensor reuse | Experimental; 0/26 full-workload screens, 11 control-inconclusive |
| CUDA Graphs | Pointwise capture/replay passed; captured SVD failed in the tested setup; no strict full-stream graph |
| Removing physical buffer clearing | Rejected: mixed/null timing evidence and different implicit pickle tail state |

The [current evidence index](docs/README.md#accepted-evidence-map) records merged
studies through PR #19. The index also tracks the status of experimental modules and historical reports.

## Algorithm and guarantee

![Sequential buffered Fast FD and its two nonmutating snapshot paths](docs/assets/fd-algorithm.svg)

*A full buffer triggers one dependent shrink, then streaming continues on the
same state. A short snapshot copies; a longer one needs an extra shrink.
[Figure source and regeneration](docs/assets/README.md).*

For finite real `A` with `n` rows and `d` features, output `B` has shape `(ell,d)`.
In exact arithmetic, for every integer `0 <= k < ell`, `k <= d`,

```text
0 <= A.T A - B.T B <= ||A - A_k||_F^2 / (ell - k) * I
||A - A V_k V_k.T||_F^2 <= ell / (ell - k) * ||A - A_k||_F^2
```

Matrix inequalities use positive-semidefinite order; `A_k` is a best rank-k
approximation, and `V_k` contains B's top-k **right singular vectors**.
Short snapshots need not be SVD-oriented or rank `ell-1`: slicing raw sketch rows
does not select PCA components. Signs and tied-subspace bases can vary.

The implementation follows [Ghashami, Liberty, Phillips & Woodruff, Frequent
Directions, arXiv:1501.01711v2](https://arxiv.org/abs/1501.01711v2), with documented
fixed-size snapshot and `ell>d` extensions. See the [derivation and numerical
boundary](docs/performance/README.md#mathematical-contract) and retained
[foundational research record](docs/research/frequent-directions-tensorization.md).

## Potential Applications in LLMs and VLMs

Frequent Directions offers a compact streaming summary of high-dimensional
features. Its covariance and right-subspace approximation properties suggest
several directions for large-scale model analysis and optimization, building on
the [original FD work](https://arxiv.org/abs/1501.01711) and related
[Sketchy optimizer research (NeurIPS 2023)](https://papers.nips.cc/paper_files/paper/2023/hash/ef72fa6579401ffff9da246a5014f055-Abstract-Conference.html).

| Research direction | How FD can help |
|---|---|
| LLM activation analysis | Sketch hidden-state vectors online; approximate their uncentered second moment with `B.T @ B` and inspect dominant representation subspaces. |
| Optimizer statistics | Summarize gradient or factor statistics for research into low-rank preconditioning, motivated in part by Sketchy. |
| VLM representation analysis | Sketch dimension-compatible projected image and text embeddings to study dominant directions and representation shifts. |
| Continual learning | Retain compact feature or gradient subspace summaries across tasks for drift analysis and exploration of regularization methods. |

These are potential integrations rather than shipped LLM/VLM workflows.
Centered covariance requires mean tracking; optimizer integration requires a
separate training implementation. The sketch occupies `O(ell * d)` elements,
while end-to-end memory and runtime depend on the surrounding model and
training pipeline.

## Performance: measured results

![T4 full update plus sketch: paired Direct versus QR-SVD timings](docs/assets/gpu-benchmarks.svg)

*Real T4, float32, complete resident-input calls (PR #14). The figure reports
paired medians and descriptive bootstrap intervals.
[Measurements, source SHA and protocol](docs/performance/measurements.md).*

The default buffered implementation uses **Direct SVD**. Research experiments
also evaluate QR + Small SVD, pointwise `torch.compile` fusion, ScratchFD,
and CUDA Graphs:

- **Short snapshots:** the shipped fast path avoids an extra SVD when the
  occupied buffer has at most `ell` rows.
- **QR + Small SVD:** performance varies by matrix shape, device and threading;
  this remains an experimental backend.
- **Pointwise compilation:** measured T4 shrink kernels reduced launches from
  10 to 1. Full-workload compilation passed 0/30 performance screens.
- **ScratchFD:** full-workload screens passed 0/26, including 11
  control-inconclusive cases.
- **CUDA Graphs:** pointwise capture/replay worked; SVD capture failed in the
  tested setup.

The [integrated performance study](docs/performance/README.md) contains
experimental configurations, positive and negative findings, and comparisons
across CPU and T4.

## Input, numerical and validation boundaries

- Dense finite real **2D float32/float64** tensors, positive integer `ell` and `d`.
  Empty/short streams and noncontiguous dense input are supported. `ell>d` is an
  exact-covariance extension in exact arithmetic, without compression advantage.
- Inference only: updates detach input and keep no autograd graph. Half/bfloat16,
  complex/sparse inputs and training through the sketch are unsupported.
- Invalid input rejects before mutation. Discard the instance after SVD/backend
  failure. Finite input does not guarantee every extreme-scale SVD succeeds;
  output finite checks remain strict.
- Recorded CPU validation: Python 3.12.3 / torch 2.11.0+cpu. Real-T4 studies:
  Python 3.13.15 / torch 2.11.0+cu130, CUDA 13.0, float32/64. These are bounded
  configurations, not promises for every GPU, torch version or long stream.
- Independent float64 covariance/tail/projection oracles use dimension/energy
  tolerance, not a proof of tiny-singular-value relative accuracy or arbitrary
  finite-precision streams. TPU/other accelerators remain unverified. Full strict
  FD compilation/graph acceleration is unavailable in the tested paths, distinct
  from measured partial fusion.

For local checks (CPU-only torch suffices):

```sh
python -m unittest discover -s tests -v
python scripts/validate_cpu.py
python scripts/check_snapshot.py
python docs/tools/build_figures.py --check
python docs/tools/verify_docs.py --run-example
```

These commands run local tests and documentation checks.

## License and research workflow

Project code and original figures are [MIT licensed](LICENSE); the fixed
[SAE V3.3 snapshot](.sae/AGENTS.md) retains its own MIT license. No original paper,
private dataset or third-party implementation is redistributed. No GitHub Actions
or background compute is enabled. [Project rules](AGENTS.md) assign implementation
and evidence to Codex, and independent scientific/merge review to GPT Architect.

## Claude Engineer Smoke Test

This note verifies the local Claude Code Engineer workflow (branch, test, commit, push, draft PR). It is documentation only and has no effect on the library.
