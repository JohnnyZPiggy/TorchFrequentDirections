# TorchFrequentDirections

Inference-only reference and buffered Frequent Directions, using native PyTorch
SVD. CPU and bounded T4 baselines were accepted in PRs #2/#4, and the lossless
short-snapshot optimization in PR #6. The buffer-clear study for
[Issue #7](https://github.com/JohnnyZPiggy/TorchFrequentDirections/issues/7)
is submitted for GPT Architect review.

The [buffer-clear study](reports/gpu/issue7-20261020/summary.md) retains the
existing clearing behavior: logical update/sketch equivalence passed, but overall
CPU/T4 gains were inconclusive and implicit pickle preserved unused candidate
rows. The one-line runtime candidate is not proposed for adoption.

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
Short Fast snapshots copy the occupied rows and pad zeros without SVD. Their row
orientation can differ from earlier releases; covariance and right-subspace
guarantees, independent ownership and continued-update semantics are preserved.
For PCA, extract the right singular subspace of the returned sketch; slicing its
raw rows does not select principal components.

[Mathematical specification](docs/research/frequent-directions-tensorization.md),
[engineering/provenance and evidence](docs/engineering/frequent-directions-engineering.md),
[SAE rules](AGENTS.md), [complete fixed snapshot](.sae/AGENTS.md).

Tested: CPU PyTorch 2.11.0+cpu / Python 3.12.3 and a real Tesla T4 with
PyTorch 2.11.0+cu130 / Python 3.13.15, float32/float64. See the
[exact-source numerical and CPU/GPU report](reports/gpu/issue3-20261020/summary.md)
for measured shapes, transfers, variability, cleanup and limitations. Most small
tested cases were slower on GPU; the wider float32 Fast case showed a modest
gain against a one-thread CPU baseline. This is not a general speedup guarantee.
The [snapshot A/B report](reports/gpu/issue5-20261020/summary.md) compares this
optimization with the exact previous source in the same CPU/T4 environment,
including short and unchanged long-buffer paths and 20 shuffled paired trials.
Packaging permits torch>=2.0 / Python>=3.10, but older releases are unverified.
TPU, torch.compile, half precision and other accelerators remain NOT RUN;
autograd is unsupported.
No GitHub Actions or background computation. Root code is MIT; bundled SAE retains
its own MIT license. No original paper or third-party implementation is distributed.
