# TorchFrequentDirections

Streaming matrix sketches in native PyTorch: reference and buffered Frequent
Directions for covariance approximation and right-subspace estimation.
CPU, real-T4 and compilation studies explore performance; GPU acceleration is
**not** a default performance guarantee.

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

This uses the checkout, not a claimed PyPI release. The build requires setuptools
>=68; `pip install .` can resolve build/runtime dependencies when needed. Packaging
version ranges are not validation claims for every release.

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
studies through PR #19. Acceptance of research evidence does **not** promote its
prototype to the API. Historical reports retain their original review-pending
language; the index gives their later disposition.

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

## Performance: read the scope before the ratio

![T4 full update plus sketch: paired Direct versus QR-SVD timings](docs/assets/gpu-benchmarks.svg)

*PR #14, one T4 study, float32, complete resident-input calls. Points and whiskers
are paired medians and descriptive bootstrap intervals, not universal speedups.
QR is a research prototype. [Medians, source SHA and protocol](docs/performance/measurements.md).*

The shipped short-snapshot optimization saves a final SVD only on its active path;
snapshot-only savings are not full-stream savings. QR gains depend on shape and
device. Compiling pointwise shrink reduced 10 launches to one in measured cases,
yet **0/30 full-workload compile screens passed**. ScratchFD also passed no
performance screen. CPU/GPU pairing-context drift limits device comparisons.
The [integrated study](docs/performance/README.md) retains positive, negative and
inconclusive results; it never multiplies ratios from different blocks.

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

These commands do not allocate cloud hardware. This documentation update runs
figure/link/example checks only, without new performance experiments.

## License and research workflow

Project code and original figures are [MIT licensed](LICENSE); the fixed
[SAE V3.3 snapshot](.sae/AGENTS.md) retains its own MIT license. No original paper,
private dataset or third-party implementation is redistributed. No GitHub Actions
or background compute is enabled. [Project rules](AGENTS.md) assign implementation
and evidence to Codex, and independent scientific/merge review to GPT Architect.
