"""Issue #11 CPU-only staged study. Run each thread setting in a fresh process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

import torch

from experiments.qr_svd import driver
from experiments.qr_svd.prototype import QRSmallSVD
from torchfrequentdirections import FrequentDirections

SEED = 11020261021
SHAPES = [(16, d) for d in (32, 64, 128, 256, 512, 1024)] + [
    (ell, 1024) for ell in (8, 32, 64)]


def rows_for_shrinks(ell, d, final_count, shrinks=4):
    retained = min(ell - 1, d)
    if not retained <= final_count < 2 * ell:
        raise ValueError('unreachable final occupancy')
    return 2 * ell + (shrinks - 1) * (2 * ell - retained) + final_count - retained


def stream(a, ell, cls, chunk=None):
    state = cls(ell, a.shape[1], dtype=a.dtype)
    for part in a.split(chunk or max(1, ell // 2)):
        state.update(part)
        state.update(part[:0])
    return state


def run(a, ell, cls):
    return stream(a, ell, cls).sketch()


def path_counts(a, ell, cls):
    shapes = []
    original = cls._factor

    def factor(self, rows, *, shrink):
        shapes.append([len(rows), rows.shape[1], shrink])
        return original(self, rows, shrink=shrink)

    with patch.object(cls, '_factor', factor), \
         patch('torch.linalg.qr', wraps=torch.linalg.qr) as qr, \
         patch('torch.linalg.svd', wraps=torch.linalg.svd) as svd:
        state = stream(a, ell, cls)
        before = svd.call_count
        state.sketch()
    return dict(full_shrinks=sum(s[0] == 2 * ell for s in shapes),
                factor_shapes=shapes, svd_calls=svd.call_count, qr_calls=qr.call_count,
                snapshot_svd_calls=svd.call_count - before, final_count=state._count)


def gate(a, ell, kind):
    if kind == 'factor':
        outputs = [cls(ell, a.shape[1], dtype=a.dtype)._factor(a, shrink=True)
                   for cls in (FrequentDirections, QRSmallSVD)]
        paths = None
    else:
        outputs = [run(a, ell, cls) for cls in (FrequentDirections, QRSmallSVD)]
        paths = [path_counts(a, ell, cls) for cls in (FrequentDirections, QRSmallSVD)]
        if paths[0]['svd_calls'] != paths[1]['svd_calls']:
            raise AssertionError('unequal mathematical work')
        for cls, output in zip((FrequentDirections, QRSmallSVD), outputs):
            state = stream(a, ell, cls)
            saved, count = state._buffer.clone(), state._count
            snapshot = state.sketch()
            if snapshot.requires_grad or not torch.equal(snapshot, output):
                raise AssertionError('no-grad/repeatability')
            snapshot.fill_(123)
            if not torch.equal(state._buffer, saved) or state._count != count:
                raise AssertionError('snapshot mutation')
            if not torch.equal(state.sketch(), output):
                raise AssertionError('snapshot alias')
    result = driver.compare(a, outputs, ell)
    if not result['passed']:
        raise AssertionError({'ell': ell, 'shape': list(a.shape), 'oracle': result})
    return dict(math=result, paths=paths)


def workloads(shapes, controls=False):
    specs = []
    for dtype in (torch.float32, torch.float64):
        for ell, d in shapes:
            rng = torch.Generator().manual_seed(SEED + ell * 10000 + d)
            for kind, n in [('factor', 2 * ell),
                            ('four_short', rows_for_shrinks(ell, d, ell - 1)),
                            ('four_long', rows_for_shrinks(ell, d, ell + 1)),
                            ('constant_n', 257)]:
                a = torch.randn(n, d, generator=rng, dtype=dtype)
                specs.append((dict(kind=kind, ell=ell, d=d, n=n, dtype=str(dtype),
                                   rho=d / (2 * ell)), a))
        if controls:
            for kind, ell, d, n in [('tiny', 2, 8, 15), ('fallback', 8, 4, 54),
                                    ('no_factor', 16, 128, 15), ('empty', 16, 128, 0)]:
                rng = torch.Generator().manual_seed(SEED + n)
                specs.append((dict(kind=kind, ell=ell, d=d, n=n, dtype=str(dtype),
                                   rho=d/(2*ell)), torch.randn(n, d, dtype=dtype, generator=rng)))
    return specs


def spectral_workloads():
    specs = []
    for dtype in (torch.float32, torch.float64):
        for ell, d in ((16, 64), (16, 512)):
            n = rows_for_shrinks(ell, d, ell + 1)
            rng = torch.Generator().manual_seed(SEED + d)
            rank = 2 * ell
            q, _ = torch.linalg.qr(torch.randn(n, rank, dtype=torch.float64, generator=rng))
            v, _ = torch.linalg.qr(torch.randn(d, rank, dtype=torch.float64, generator=rng))
            spectra = dict(rank_two=torch.tensor([2., 1.] + [0.] * (rank - 2)),
                           decay_1e16=torch.logspace(0, -16, rank, dtype=torch.float64),
                           tie=torch.tensor([2.]*(ell-2) + [1.]*4 + [.1]*(rank-ell-2)))
            near = torch.linspace(2., .1, rank, dtype=torch.float64)
            near[ell-2:ell+2] = 1. + torch.tensor([8., 4., 0., -4.]) * torch.finfo(dtype).eps
            spectra['near_tie'] = near
            spectra['zero'] = torch.zeros(rank)
            for name, s in spectra.items():
                a = ((q * s.double()) @ v.T).to(dtype)
                # Preserve a noncontiguous view with identical logical contents.
                storage = torch.empty(n, 2*d, dtype=dtype)
                storage[:, ::2] = a
                a = storage[:, ::2]
                specs.append((dict(kind='spectrum_' + name, ell=ell, d=d, n=n,
                                   dtype=str(dtype), rho=d/(2*ell)), a))
    for dtype in (torch.float32, torch.float64):
        for ell, d in ((1, 16), (8, 3)):
            rng = torch.Generator().manual_seed(SEED + ell)
            specs.append((dict(kind='edge', ell=ell, d=d, n=47, dtype=str(dtype),
                               rho=d/(2*ell)), torch.randn(47, d, dtype=dtype, generator=rng)))
    return specs


def select_shapes(records):
    scored = []
    for shape in SHAPES:
        values = [r['timing']['median_paired_speedup'] for r in records
                  if (r['ell'], r['d']) == shape and r['kind'] in ('four_short','four_long')]
        scored.append((statistics.median(values), shape))
    losing = min(scored)[1]
    remaining = [(score, shape) for score, shape in scored if shape != losing]
    near = min(remaining, key=lambda item: abs(item[0] - 1))[1]
    winning = max(item for item in remaining if item[1] != near)[1]
    return dict(losing=list(losing), near=list(near), winning=list(winning),
                scores=[dict(ell=s[0], d=s[1], median_score=v) for v, s in scored])


def study(specs, repeat_batches=2):
    records = []
    # Complete correctness and intended-path gates before timing this stage.
    for spec, a in specs:
        driver.budget()
        result = gate(a, spec['ell'], spec['kind'])
        paths = result['paths']
        if paths:
            for p in paths:
                if spec['kind'].startswith('four_') and p['full_shrinks'] != 4:
                    raise AssertionError('wrong shrink count')
                if spec['kind'] in ('no_factor', 'empty') and (p['qr_calls'] or p['svd_calls']):
                    raise AssertionError('no-factor control decomposes')
            if spec['kind'] == 'fallback' and paths[1]['qr_calls']:
                raise AssertionError('fallback control uses QR')
        if spec['kind'].startswith('spectrum_') or spec['kind'] == 'edge':
            # Independent partitioning, continued updates and intermediate snapshots.
            consistency = []
            for cls in (FrequentDirections, QRSmallSVD):
                state = cls(spec['ell'], spec['d'], dtype=a.dtype)
                start = 0
                for stop in (min(3,len(a)), min(19,len(a)), len(a)):
                    state.update(a[start:stop]); state.update(a[stop:stop])
                    oracle = driver.oracle(a[:stop], state.sketch(), spec['ell'])
                    if not oracle['passed']:
                        raise AssertionError('spectral prefix bound')
                    consistency.append(oracle)
                    start = stop
                other = stream(a, spec['ell'], cls, chunk=1).sketch()
                direct = state.sketch()
                compared = driver.compare(a, [other, direct], spec['ell'])
                if not compared['passed']:
                    raise AssertionError('chunk dependence')
                consistency.append(compared)
            result['chunk_prefix_checks'] = consistency
        records.append(dict(**spec, gate=result))
    for batch in range(repeat_batches):
        # Reverse block order in batch two to expose order/thermal drift.
        indices = range(len(specs)) if batch == 0 else reversed(range(len(specs)))
        for index in indices:
            spec, a = specs[index]
            ell = spec['ell']
            if spec['kind'] == 'factor':
                instances = [cls(ell, a.shape[1], dtype=a.dtype)
                             for cls in (FrequentDirections, QRSmallSVD)]
                functions = [lambda state=state: state._factor(a, shrink=True) for state in instances]
            else:
                functions = [lambda cls=cls: run(a, ell, cls)
                             for cls in (FrequentDirections, QRSmallSVD)]
            tick = time.monotonic()
            seed = SEED + 1000 * batch + index
            timing = driver.paired(functions, seed)
            records[index].setdefault('batches', []).append(dict(batch=batch, seed=seed,
                elapsed_seconds=time.monotonic()-tick, timing=timing))
            if batch == 0:
                records[index]['timing'] = timing  # Stage-2 selection uses batch one only.
    return records


def placebos(shapes):
    records = []
    for spec, a in workloads(shapes):
        if spec['kind'] != 'four_long':
            continue
        result = gate(a, spec['ell'], 'placebo')
        batches = []
        for batch in range(2):
            fn = lambda: run(a, spec['ell'], FrequentDirections)
            seed = SEED + 8000 + batch * 100 + len(records)
            batches.append(dict(batch=batch, seed=seed, timing=driver.paired([fn, fn], seed)))
        records.append(dict(**spec, gate=result, batches=batches, comparison='direct/direct'))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['shape', 'threads'])
    parser.add_argument('output')
    parser.add_argument('--threads', type=int, choices=[1,2,4,8], default=1)
    parser.add_argument('--selection')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if subprocess.check_output(['git', 'status', '--porcelain']):
        raise RuntimeError('clean source required')
    report = dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        clean=True, phase=args.phase, seed=SEED, cpu=platform.processor(),
        platform=platform.platform(), host_frequency_turbo='uncontrolled; WSL exposes no frequency isolation',
        cpu_details=subprocess.check_output(['lscpu'],text=True), python=sys.version,
        torch=torch.__version__, torch_build=torch.__config__.show(),
        intra_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS','OMP_DYNAMIC','MKL_DYNAMIC')},
        affinity=sorted(os.sched_getaffinity(0)),
        cgroup_cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip()
            if Path('/sys/fs/cgroup/cpu.max').exists() else 'unavailable',
        gpu='NOT_RUN', production_backend='UNCHANGED', tiny_eigenvalue_relative_error='NOT_ESTABLISHED',
        screening='both batches median paired speedup >1.05 and descriptive lower >1; '
        'integration still requires stable placebos/controls and independent review',
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
               [Path('experiments/qr_svd/shape_threads.py'),Path('experiments/qr_svd/driver.py'),
                Path('experiments/qr_svd/prototype.py'),Path('torchfrequentdirections/fd.py'),
                Path('tests/test_shape_threads.py')]})
    tick = time.monotonic()
    try:
        if args.phase == 'shape':
            if args.threads != 1:
                raise ValueError('shape phase is single thread')
            report['records'] = study(workloads(SHAPES, controls=True))
            report['selection'] = select_shapes(report['records'])
            report['spectra'] = study(spectral_workloads())
            report['placebos'] = placebos([(16,32),(16,1024)])
            report['nonadditive_stages'] = []
            for dtype in (torch.float32,torch.float64):
                for ell,d in ((16,32),(16,256),(16,1024)):
                    a=torch.randn(2*ell,d,dtype=dtype,generator=torch.Generator().manual_seed(SEED))
                    report['nonadditive_stages'].append(dict(ell=ell,d=d,dtype=str(dtype),
                                                          stages=driver.stages(a,ell)))
        else:
            selection=json.loads(Path(args.selection).read_text())['selection']
            shapes=[tuple(selection[key]) for key in ('losing','near','winning')]
            report['selection']=selection
            report['records']=study(workloads(shapes,controls=True))
            report['placebos']=placebos(shapes)
        report['status']='PASS'
    except Exception as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-tick
        Path(args.output).write_text(json.dumps(report, separators=(',',':'), allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
