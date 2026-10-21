"""Issue #9: bounded CPU-only correctness gate followed by paired benchmarks.

Run from a clean checkout: python -m experiments.qr_svd.driver OUTPUT.json
The caller also imposes a process timeout. No accelerator API is called.
"""
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

import torch

from experiments.qr_svd.prototype import QRSmallSVD
from torchfrequentdirections import FrequentDirections

SEED = 20261021
START = time.monotonic()
DEADLINE = 900


def budget():
    if time.monotonic() - START > DEADLINE:
        raise TimeoutError("900-second experiment bound")


def oracle(a, b, ell):
    """Explicit CPU float64 Gram eigensolve and full-input SVD; no FD shrink."""
    dtype = a.dtype
    a, b = a.double(), b.double()
    energy = float(a.square().sum())
    tolerance = 32 * torch.finfo(dtype).eps * max(*a.shape, 2 * ell, 1) * energy
    gap = a.T @ a - b.T @ b
    eig = torch.linalg.eigvalsh((gap + gap.T) / 2)
    singular = torch.linalg.svdvals(a)
    _, _, vh = torch.linalg.svd(b, full_matrices=False)
    checks = []
    for k in range(min(ell - 1, a.shape[1]) + 1):
        tail = float(singular[k:].square().sum())
        residual = float((a - (a @ vh[:k].T) @ vh[:k]).square().sum())
        bound = tail / (ell - k)
        checks.append(dict(k=k, tail_squared=tail, covariance_bound=bound,
                           projection_squared=residual, projection_bound=ell * bound,
                           passed=float(eig.min()) >= -tolerance and
                           float(eig.abs().max()) <= bound + tolerance and
                           residual <= ell * bound + tolerance))
    return dict(energy=energy, tolerance=tolerance, minimum_eigenvalue=float(eig.min()),
                spectral_error=float(eig.abs().max()), checks=checks,
                passed=all(c['passed'] for c in checks))


def compare(a, outputs, ell):
    reports = [oracle(a, b, ell) for b in outputs]
    x, y = [b.double() for b in outputs]
    # Frobenius norm bounds spectral norm; stricter than the individual FD oracle.
    gap = float(torch.linalg.vector_norm(x.T @ x - y.T @ y))
    tolerance = reports[0]['tolerance']
    return dict(oracles=reports, gram_gap_frobenius=gap, gram_gap_tolerance=tolerance,
                passed=all(r['passed'] for r in reports) and gap <= tolerance)


def fixtures(dtype):
    rng = torch.Generator().manual_seed(SEED)

    def rand(n, d):
        return torch.randn(n, d, dtype=dtype, generator=rng)

    q, _ = torch.linalg.qr(rand(97, 8))
    v, _ = torch.linalg.qr(rand(17, 8))
    spectra = {
        'decay_1e16': torch.logspace(0, -16, 8, dtype=dtype),
        'tie_at_ell': torch.tensor([2, 1, 1, 1, 1, .1, .01, 0], dtype=dtype),
        'near_tie': torch.tensor([2, 1.1, 1, 1 - 4 * torch.finfo(dtype).eps,
                                  1 - 8 * torch.finfo(dtype).eps, .1, 0, 0], dtype=dtype),
        'rank_two': torch.tensor([2, 1, 0, 0, 0, 0, 0, 0], dtype=dtype)}
    result = {name: (q * s) @ v.T for name, s in spectra.items()}
    result.update(random=rand(97, 17), zero=torch.zeros(97, 17, dtype=dtype),
                  noncontiguous=rand(97, 34)[:, ::2], empty=rand(0, 17),
                  short=rand(3, 17), long_drift=rand(1025, 17),
                  fallback=rand(97, 3), one_feature=rand(31, 1))
    return result


def validate():
    records = []
    for dtype in (torch.float32, torch.float64):
        for name, a in fixtures(dtype).items():
            for ell in (1, 4, a.shape[1] + 2):
                for pattern in ('batch', 'rowwise', 'irregular'):
                    budget()
                    streams = [cls(ell, a.shape[1], dtype=dtype)
                               for cls in (FrequentDirections, QRSmallSVD)]
                    n = len(a)
                    if pattern == 'batch':
                        stops = [n]
                    elif pattern == 'rowwise':
                        stops = list(range(1, n + 1)) or [0]
                    else:
                        stops = sorted({0, min(2, n), min(7, n), min(19, n), n})
                    start = 0
                    observations = []
                    # Check every irregular boundary; bounded checkpoints for rowwise.
                    checkpoints = {0, 1, 2, 7, 19, n // 2, n - 1, n}
                    for stop in stops:
                        for stream in streams:
                            stream.update(a[start:stop])
                            stream.update(a[stop:stop])
                        start = stop
                        if pattern != 'rowwise' or stop in checkpoints:
                            snapshots = []
                            for stream in streams:
                                state, count = stream._buffer.clone(), stream._count
                                b = stream.sketch()
                                second = stream.sketch()
                                if not torch.equal(b, second) or not torch.equal(state, stream._buffer):
                                    raise AssertionError('mutating/nonrepeatable snapshot')
                                if count != stream._count or b.requires_grad:
                                    raise AssertionError('state/autograd contract')
                                second.fill_(123)
                                if not torch.equal(b, stream.sketch()):
                                    raise AssertionError('snapshot alias')
                                snapshots.append(b)
                            observation = compare(a[:stop], snapshots, ell)
                            observation.update(stop=stop, counts=[s._count for s in streams])
                            observations.append(observation)
                    record = dict(dtype=str(dtype), case=name, ell=ell, pattern=pattern,
                                  observations=observations,
                                  passed=all(o['passed'] for o in observations))
                    records.append(record)
    # Isolate shrink and no-shrink factor paths, including degenerate threshold.
    factors = []
    for dtype in (torch.float32, torch.float64):
        for name, source in fixtures(dtype).items():
            a = source[:8]
            for shrink in (False, True):
                ell = 4 if shrink else max(8, len(a))
                outputs = [cls(ell, a.shape[1], dtype=dtype)._factor(a, shrink=shrink)
                           for cls in (FrequentDirections, QRSmallSVD)]
                result = compare(a, outputs, ell)
                result.update(case=name, dtype=str(dtype), shrink=shrink)
                factors.append(result)
    return dict(streams=records, factors=factors,
                passed=all(r['passed'] for r in records + factors))


def summary(samples):
    return dict(samples_ms=samples, median_ms=statistics.median(samples),
                p95_ms=sorted(samples)[math.ceil(.95 * len(samples)) - 1],
                minimum_ms=min(samples), maximum_ms=max(samples),
                sample_stddev_ms=statistics.stdev(samples))


def paired(functions, seed):
    # Ten AB and ten BA trials, shuffled before timing. Each sample is one call.
    rng = random.Random(seed)
    orders = [(0, 1), (1, 0)] * 10
    rng.shuffle(orders)
    for order in ((0, 1), (1, 0)):
        for i in order:
            functions[i]()
    raw = []
    samples = [[], []]
    for order in orders:
        budget()
        times = [None, None]
        for i in order:
            tick = time.perf_counter_ns()
            functions[i]()
            times[i] = (time.perf_counter_ns() - tick) / 1e6
            samples[i].append(times[i])
        raw.append(dict(order=list(order), milliseconds=times))
    ratios = [r['milliseconds'][0] / r['milliseconds'][1] for r in raw]
    boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
    return dict(raw_ordered_pairs=raw, baseline=summary(samples[0]), candidate=summary(samples[1]),
                median_paired_speedup=statistics.median(ratios),
                ratio_of_medians=statistics.median(samples[0]) / statistics.median(samples[1]),
                descriptive_bootstrap_95=[boot[49], boot[1949]], warmup_per_variant=2)


def stages(a, ell):
    """Intrusive individual-stage wall times, not additive or speedup evidence."""
    samples = {key: [] for key in ('qr', 'small_svd', 'vh_recovery', 'shrink_output')}
    for iteration in range(22):
        budget()
        tick = time.perf_counter_ns()
        q, r = torch.linalg.qr(a.T, mode='reduced')
        t1 = time.perf_counter_ns()
        _, s, small_vh = torch.linalg.svd(r.T, full_matrices=False)
        t2 = time.perf_counter_ns()
        vh = small_vh @ q.T
        t3 = time.perf_counter_ns()
        scale = s[0].clamp_min(torch.finfo(s.dtype).tiny)
        normalized = s / scale
        s = scale * (normalized.square() - normalized[ell - 1].square()).clamp_min(0).sqrt()
        out = a.new_zeros((ell, a.shape[1]))
        out[:] = s[:ell, None] * vh[:ell]
        if not bool(torch.isfinite(out).all()):
            raise FloatingPointError('stage output')
        t4 = time.perf_counter_ns()
        if iteration >= 2:
            for key, duration in zip(samples, (t1-tick, t2-t1, t3-t2, t4-t3)):
                samples[key].append(duration / 1e6)
    return {key: summary(values) for key, values in samples.items()}


def benchmarks():
    records = []
    rng = torch.Generator().manual_seed(SEED)
    shapes = [(16, 128), (32, 512), (64, 1024), (128, 2048), (4, 8), (16, 4)]
    for dtype in (torch.float32, torch.float64):
        for m, d in shapes:
            ell = m // 2
            a = torch.randn(m, d, dtype=dtype, generator=rng)
            streams = [cls(ell, d, dtype=dtype) for cls in (FrequentDirections, QRSmallSVD)]
            # Independent full-Gram correctness gate on each timed factor input.
            gate = compare(a, [s._factor(a, shrink=True) for s in streams], ell)
            if not gate['passed']:
                raise AssertionError('benchmark factor gate')
            for shrink in (True, False):
                if not shrink and (m, d) not in ((4, 8), (16, 4)):
                    continue
                timing = paired([lambda s=s: s._factor(a, shrink=shrink) for s in streams],
                                SEED + len(records))
                record = dict(kind='factor', m=m, d=d, ell=ell, dtype=str(dtype),
                              shrink=shrink, gate=gate, timing=timing)
                if shrink and m <= d:
                    record['instrumented_nonadditive_stages'] = stages(a, ell)
                records.append(record)
            # Multiple shrinks; fresh state, input validation, chunk loop, snapshots included.
            for remainder in (ell - 1, ell + 1):
                n = m + 3 * (ell + 1) + remainder - (ell - 1)
                data = torch.randn(n, d, dtype=dtype, generator=rng)

                def run(cls):
                    s = cls(ell, d, dtype=dtype)
                    for chunk in data.split(max(1, ell // 2)):
                        s.update(chunk)
                    return s.sketch()

                outputs = [run(cls) for cls in (FrequentDirections, QRSmallSVD)]
                stream_gate = compare(data, outputs, ell)
                if not stream_gate['passed']:
                    raise AssertionError('benchmark streaming gate')
                timing = paired([lambda cls=cls: run(cls)
                                 for cls in (FrequentDirections, QRSmallSVD)], SEED + len(records))
                records.append(dict(kind='stream', n=n, d=d, ell=ell, dtype=str(dtype),
                                    intended_remainder=remainder, gate=stream_gate, timing=timing))
    return records


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
    if dirty:
        raise RuntimeError('clean source required')
    report = dict(source_sha=sha, dirty=False, seed=SEED, device='cpu',
                  gpu='NOT RUN (explicit scope)', python=sys.version, torch=torch.__version__,
                  platform=platform.platform(), cpu=subprocess.check_output(['lscpu'], text=True),
                  threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                  torch_build=torch.__config__.show(), deadline_seconds=DEADLINE,
                  method='2 warmups each; 20 balanced randomized AB/BA pairs; 2000 paired bootstrap draws',
                  predeclared_criterion='All math gates pass; promising only if every primary wide full-stream '
                  'workload has median paired speedup >1.05 and descriptive lower interval >1.0; '
                  'controls reported separately. No production integration in this issue.',
                  files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [Path('torchfrequentdirections/fd.py'), Path(__file__).relative_to(Path.cwd()),
                          Path('experiments/qr_svd/prototype.py'), Path('tests/test_qr_svd.py')]})
    destination = Path(sys.argv[1])
    try:
        report['correctness'] = validate()
        if not report['correctness']['passed']:
            raise AssertionError('correctness gate failed; benchmarks NOT RUN')
        report['benchmarks'] = benchmarks()
        primary = [r for r in report['benchmarks'] if r['kind'] == 'stream' and r['d'] >= 128]
        report['promising'] = all(r['timing']['median_paired_speedup'] > 1.05 and
                                  r['timing']['descriptive_bootstrap_95'][0] > 1 for r in primary)
        report['status'] = 'PASS'
    except Exception as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - START
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
