# TorchFrequentDirections

Inference-only reference and buffered Frequent Directions, using native PyTorch
SVD. First CPU engineering batch for [Issue #1](https://github.com/JohnnyZPiggy/TorchFrequentDirections/issues/1);
scientific acceptance and merge remain with GPT Architect.

```sh
python -m pip install .
python -m unittest discover -s tests -v
python scripts/validate_cpu.py
python scripts/check_snapshot.py
python -m torchfrequentdirections.benchmarks --n 256 --d 64 --ell 8 --repeat 5
```

```python
import torch
from torchfrequentdirections import fast_fd, reference_fd, FrequentDirections

a = torch.randn(100, 20, dtype=torch.float64)
b = fast_fd(a, ell=8)             # (8, 20); preserves dtype/device
baseline = reference_fd(a, ell=8)
stream = FrequentDirections(8, 20, dtype=a.dtype, device=a.device)
stream.update(a[:37]).update(a[37:])
snapshot = stream.sketch()         # independent copy; does not finalize stream state
```

Inputs: dense, finite, real 2D float32/float64, positive integer `ell` and feature
dimension. Empty/short streams are valid; outputs always have `ell` rows. Missing
singular values for `ell > d` are zero, giving an exact-covariance fallback up to
roundoff. No input dtype/device conversion or autograd graph is retained. Invalid
input is checked before mutation; discard a streaming instance after an SVD/backend
failure. Singular-vector signs/bases and reference/Fast outputs can differ.

Fast FD uses a temporary `2*ell` buffer and an explicit occupied-row count; rows
whose entries sum to zero remain data. A partial final buffer larger than `ell`
receives an additional shrink, rather than unsafe row truncation. A short buffer
of at most `ell` rows is represented without shrink, so it may have rank `ell`.

[Mathematical specification](docs/research/frequent-directions-tensorization.md),
[engineering/provenance and evidence](docs/engineering/frequent-directions-engineering.md),
[SAE rules](AGENTS.md), [complete fixed snapshot](.sae/AGENTS.md).

Tested: CPU PyTorch 2.11.0+cpu, Python 3.12.3, float32/float64. Packaging permits
torch>=2.0 / Python>=3.10, but older releases are unverified. GPU/TPU, autograd,
torch.compile, half precision and accelerator performance are NOT RUN/unsupported
in this milestone. Benchmark timings are CPU observations, not GPU speedup claims.
No GitHub Actions or background computation. Root code is MIT; bundled SAE retains
its own MIT license. No original paper or third-party implementation is distributed.
