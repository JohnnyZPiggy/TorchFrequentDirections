"""Issue #3: exact-source install, independent CPU oracle, bounded GPU comparison."""
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

SEED = 20261020
SOURCE_FILES = {'LICENSE': '2f0819452dd5f7667e2c0e93b769238a41c8164e62f89f212646076f66765bde', 'pyproject.toml': '31ecd084cafb753e17912f93d98695bfb3f81ed4a23527ec0ca2ac59e1782a3c', 'torchfrequentdirections/__init__.py': 'e65e1bcff754fe3cc2b5e3fb383f7b6293a7a3a358757e8cc0861856e54306c7', 'torchfrequentdirections/fd.py': 'df1b7af6832574409856edaac2533c9c23e5fb9c8f0936826d383eff952cdff7', 'torchfrequentdirections/benchmarks.py': '832cdcab970a70465d79f8f73595f2bd3e50639d776e96bcfdc24a9240dd0020'}


def install_source(sha, destination, local=None):
    root = Path(local) if local else destination / 'source'
    if not local:
        root.mkdir()
        with urllib.request.urlopen('https://codeload.github.com/JohnnyZPiggy/TorchFrequentDirections/tar.gz/' + sha,
                                    timeout=30) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise RuntimeError('SOURCE_TOO_LARGE')
        found = set()
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
            for member in archive:
                name = member.name.partition('/')[2]
                if name not in SOURCE_FILES:
                    continue
                if not member.isfile() or name in found or member.size > 1024 * 1024:
                    raise RuntimeError('UNSAFE_SOURCE')
                data = archive.extractfile(member).read()
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                found.add(name)
        if found != set(SOURCE_FILES):
            raise RuntimeError('SOURCE_INCOMPLETE')
    if not SOURCE_FILES:
        raise RuntimeError('SOURCE_MANIFEST_EMPTY')
    for name, expected in SOURCE_FILES.items():
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('SOURCE_MISMATCH')
    if local:
        sys.path.insert(0, str(root))
    else:
        target = destination / 'installed'
        env = dict(os.environ, PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
                   TMPDIR=str(destination), PIP_CACHE_DIR=str(destination / 'pip-cache'))
        call = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
            '--no-build-isolation', '--target', str(target), str(root)], env=env,
            capture_output=True, timeout=90)
        if call.returncode:
            raise RuntimeError('INSTALL_FAILED')
        sys.path.insert(0, str(target))
    return {'status': 'PASS', 'source_sha': sha, 'verified_files': SOURCE_FILES,
            'mode': 'local verified source' if local else 'exact SHA archive + no-deps pip target'}


def oracle(a, b, ell):
    """Independent float64 CPU Gram/full-SVD check, never the FD shrink."""
    dtype = a.dtype
    a, b = a.cpu().double(), b.detach().cpu().double()
    energy = float(a.square().sum())
    tolerance = 32 * torch.finfo(dtype).eps * max(*a.shape, 2 * ell, 1) * energy
    difference = a.T @ a - b.T @ b
    values = torch.linalg.eigvalsh((difference + difference.T) / 2)
    minimum, spectral = float(values.min()), float(values.abs().max())
    singular = torch.linalg.svdvals(a)
    _, _, vh = torch.linalg.svd(b, full_matrices=False)
    checks = []
    for k in range(min(ell - 1, a.shape[1]) + 1):
        tail = float(singular[k:].square().sum())
        residual = float((a - (a @ vh[:k].T) @ vh[:k]).square().sum())
        checks.append({'k': k, 'tail_squared': tail, 'covariance_bound': tail / (ell-k),
            'projection_squared': residual, 'projection_bound': ell * tail / (ell-k),
            'pass': minimum >= -tolerance and spectral <= tail/(ell-k)+tolerance
                    and residual <= ell*tail/(ell-k)+tolerance})
    return {'status': 'PASS' if all(c['pass'] for c in checks) else 'FAIL',
            'tolerance': tolerance, 'energy': energy, 'minimum_eigenvalue': minimum,
            'spectral_error': spectral, 'checks': checks}


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def measure(function, device, warmup=1, repeats=5):
    for _ in range(warmup):
        function()
    sync(device)
    samples = []
    for _ in range(repeats):
        sync(device)
        tick = time.perf_counter()
        function()
        sync(device)
        samples.append(1000 * (time.perf_counter() - tick))
    return {'samples_ms': samples, 'median_ms': statistics.median(samples),
            'p95_ms': sorted(samples)[math.ceil(.95 * len(samples)) - 1],
            'warmup': warmup, 'repeats': repeats, 'p95_method': 'nearest-rank'}


def compare(a, ell, method, gpu):
    from torchfrequentdirections import fast_fd, reference_fd
    fn = fast_fd if method == 'fast' else reference_fd
    cpu = fn(a, ell)
    result = {'method': method, 'n': a.shape[0], 'd': a.shape[1], 'ell': ell,
              'dtype': str(a.dtype), 'cpu_oracle': oracle(a, cpu, ell)}
    if gpu:
        cuda = fn(a.to('cuda'), ell).cpu()
        result['gpu_oracle'] = oracle(a, cuda, ell)
        gap = float(torch.linalg.matrix_norm(cpu.double().T @ cpu.double() -
                    cuda.double().T @ cuda.double(), ord=2))
        tolerance = 4 * result['cpu_oracle']['tolerance']
        result['same_algorithm_covariance_gap'] = gap
        result['same_algorithm_covariance_gap_tolerance'] = tolerance
        result['same_algorithm_pass'] = gap <= tolerance
    result['status'] = 'PASS' if (result['cpu_oracle']['status'] == 'PASS' and
        (not gpu or result['gpu_oracle']['status'] == 'PASS' and result['same_algorithm_pass'])) else 'FAIL'
    return result


def correctness(gpu):
    rng = torch.Generator().manual_seed(SEED)
    q, _ = torch.linalg.qr(torch.randn(19, 7, dtype=torch.float64, generator=rng))
    v, _ = torch.linalg.qr(torch.randn(7, 7, dtype=torch.float64, generator=rng))
    fixtures = [ ('random_partial', torch.randn(23, 9, dtype=torch.float64, generator=rng), 4),
        ('ill_conditioned', (q * torch.logspace(0, -8, 7, dtype=torch.float64)) @ v.T, 4),
        ('near_tie', (q * torch.tensor([1.,1.,1.,.999999,.1,.01,0.], dtype=torch.float64)) @ v.T, 4),
        ('short', torch.randn(2, 7, dtype=torch.float64, generator=rng), 4),
        ('zero', torch.zeros(9, 7, dtype=torch.float64), 4),
        ('ell_one', torch.randn(5, 3, dtype=torch.float64, generator=rng), 1),
        ('ell_over_d', torch.randn(17, 3, dtype=torch.float64, generator=rng), 5)]
    results = []
    for dtype in (torch.float32, torch.float64):
        for name, original, ell in fixtures:
            for method in ('reference', 'fast'):
                result = compare(original.to(dtype), ell, method, gpu)
                result['case'] = name
                results.append(result)
    return results


def profile(fn, a, ell):
    """Intrusive diagnostic only; never used as the primary speedup timing."""
    from unittest.mock import patch
    from torchfrequentdirections import FrequentDirections
    device = a.device
    svd, validate = torch.linalg.svd, FrequentDirections._validate
    counters = {'svd_calls': 0, 'svd_host_wall_ms': 0., 'input_validation_host_wall_ms': 0.}
    def wrapped_svd(*args, **kwargs):
        sync(device)
        tick = time.perf_counter()
        output = svd(*args, **kwargs)
        sync(device)
        counters['svd_calls'] += 1
        counters['svd_host_wall_ms'] += 1000 * (time.perf_counter() - tick)
        return output
    def wrapped_validate(self, rows):
        sync(device)
        tick = time.perf_counter()
        output = validate(self, rows)
        sync(device)
        counters['input_validation_host_wall_ms'] += 1000 * (time.perf_counter() - tick)
        return output
    with patch.object(torch.linalg, 'svd', wrapped_svd), patch.object(FrequentDirections, '_validate', wrapped_validate):
        sync(device)
        tick = time.perf_counter()
        fn(a, ell)
        sync(device)
        counters['instrumented_total_host_wall_ms'] = 1000 * (time.perf_counter()-tick)
    stream = FrequentDirections(ell, a.shape[1], method='reference' if fn.__name__ == 'reference_fd' else 'fast',
                                dtype=a.dtype, device=device)
    sync(device)
    tick = time.perf_counter()
    stream.update(a)
    sync(device)
    counters['update_including_validation_and_svd_ms'] = 1000 * (time.perf_counter()-tick)
    counters['snapshot'] = measure(stream.sketch, device)
    counters['finite_check_calibration'] = measure(lambda: bool(torch.isfinite(a).all()), device)
    counters['scope'] = 'synchronized host wall; intrusive/nonadditive; finite calibration is standalone, not total output-check attribution; CUDA kernel-only NOT_RUN'
    return counters


def benchmarks(gpu):
    from torchfrequentdirections import fast_fd, reference_fd
    results = []
    for n,d,ell in ((64,32,4),(128,64,8),(256,128,16)):
        for dtype in (torch.float32, torch.float64):
            a = torch.randn(n,d,dtype=torch.float64,generator=torch.Generator().manual_seed(SEED)).to(dtype)
            row = {'n':n, 'd':d, 'ell':ell, 'dtype':str(dtype), 'methods':[]}
            if gpu:
                cuda = a.to('cuda')
                row['h2d_pageable'] = measure(lambda: a.to('cuda'), 'cuda')
                row['d2h_input_size'] = measure(lambda: cuda.cpu(), 'cuda')
            for name,fn in (('reference',reference_fd),('fast',fast_fd)):
                check = compare(a,ell,name,gpu)
                if check['status'] != 'PASS':
                    raise RuntimeError('BENCHMARK_CORRECTNESS_FAILED')
                item = {'method':name,'correctness':check,'cpu_resident':measure(lambda:fn(a,ell),'cpu'),
                        'cpu_profile':profile(fn,a,ell)}
                if gpu:
                    item['gpu_resident'] = measure(lambda:fn(cuda,ell),'cuda')
                    item['gpu_transfer_included'] = measure(lambda:fn(a.to('cuda'),ell).cpu(),'cuda')
                    item['gpu_profile'] = profile(fn,cuda,ell)
                    b = fn(cuda,ell)
                    item['d2h_sketch'] = measure(lambda:b.cpu(),'cuda')
                    item['cpu_over_gpu_resident'] = item['cpu_resident']['median_ms']/item['gpu_resident']['median_ms']
                    item['cpu_over_gpu_transfer_included'] = item['cpu_resident']['median_ms']/item['gpu_transfer_included']['median_ms']
                row['methods'].append(item)
            results.append(row)
    return results


def main():
    global torch
    sha = os.environ['SAE_CANDIDATE_SHA']
    if len(sha)!=40 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('INVALID_SHA')
    cpu_only = os.environ.get('SAE_CPU_ONLY') == '1'
    report = {'status':'FAIL','source_sha':sha,'classification':'SCIENTIFIC_FD_CPU_GPU',
              'utc_start':datetime.now(timezone.utc).isoformat(),'seed':SEED,
              'gpu':'NOT_RUN','tpu':'NOT_RUN','compilation':'NOT_RUN',
              'methodology': 'same runtime/input/dtype/algorithm, CPU threads=1; CPU float64 independent oracle; synchronized CUDA host wall; constructor+update+snapshot; generation excluded; H2D+D2H separately and included; fixed case order, one warmup/five repeats; p95 nearest-rank; tiny pilot not universal speedup'}
    tick = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='fd-issue3-') as temporary:
        try:
            report['install'] = install_source(sha,Path(temporary),os.environ.get('SAE_LOCAL_SOURCE') if cpu_only else None)
            import torch
            torch.set_num_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision('highest')
            report['environment'] = {'python':platform.python_version(),'torch':str(torch.__version__),
                'cuda_version':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
                'cpu_threads':torch.get_num_threads(),'tf32':False,
                'cpu_model':next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'UNKNOWN')}
            if not cpu_only:
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA_UNAVAILABLE')
                properties = torch.cuda.get_device_properties(0)
                report['device'] = {'requested':'T4','observed':properties.name,
                    'compute_capability':[properties.major,properties.minor],
                    'total_memory_bytes':properties.total_memory,'tensor_device':str(torch.empty(1,device='cuda').device)}
                probes = []
                for dtype in (torch.float32,torch.float64):
                    x = torch.eye(4,dtype=dtype,device='cuda')
                    u,s,vh = torch.linalg.svd(x,full_matrices=False)
                    probes.append({'dtype':str(dtype),'svd_reconstruction_error':float((u@torch.diag(s)@vh-x).abs().max())})
                report['gpu_probe'] = probes
            report['correctness'] = correctness(not cpu_only)
            if any(r['status']!='PASS' for r in report['correctness']):
                raise RuntimeError('NUMERICAL_CHECK_FAILED')
            report['benchmark'] = benchmarks(not cpu_only)
            report['gpu'] = 'NOT_RUN' if cpu_only else 'PASS'
            report['status'] = 'PASS'
        except Exception as exc:
            report['error_class'] = type(exc).__name__
            report['error_code'] = str(exc) if str(exc) in ('CUDA_UNAVAILABLE','SOURCE_MISMATCH',
                'INSTALL_FAILED','SOURCE_INCOMPLETE','SOURCE_TOO_LARGE','UNSAFE_SOURCE',
                'NUMERICAL_CHECK_FAILED','BENCHMARK_CORRECTNESS_FAILED') else 'SCIENTIFIC_OR_ENVIRONMENT_FAILURE'
    report.update(utc_end=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-tick)
    output = Path(os.environ.get('SAE_ARTIFACT_DIR','/content'))
    output.mkdir(parents=True,exist_ok=True)
    raw = (json.dumps(report,indent=2,allow_nan=False)+'\n').encode()
    (output/'fd-science.json').write_bytes(raw)
    print('ARTIFACT fd-science.json',len(raw),hashlib.sha256(raw).hexdigest(),flush=True)
    return report


if __name__ == '__main__':
    main()
