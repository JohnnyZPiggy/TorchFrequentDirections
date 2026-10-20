"""Issue #5: pinned old/new Fast FD, independent oracle and paired A/B timing.

No GitHub writes. The existing Issue #3 installer/oracle is hash-pinned and reused.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

BASE_SHA = '73203cb3415bc0c8733e7f4e7c0d1d63a431a03f'
BASE_FD_HASH = 'df1b7af6832574409856edaac2533c9c23e5fb9c8f0936826d383eff952cdff7'
HELPER_HASH = 'c3f0cb1dc62fadefe3aa3051a85c4e1ed1420879817a4671c20b986e9262ae86'
SOURCE_FILES = {
    'LICENSE': '2f0819452dd5f7667e2c0e93b769238a41c8164e62f89f212646076f66765bde',
    'pyproject.toml': '31ecd084cafb753e17912f93d98695bfb3f81ed4a23527ec0ca2ac59e1782a3c',
    'torchfrequentdirections/__init__.py': 'e65e1bcff754fe3cc2b5e3fb383f7b6293a7a3a358757e8cc0861856e54306c7',
    'torchfrequentdirections/fd.py': '18e84afa295d3bb75cb97f68bd98fa339fc3ee061e26e8071051021066d30e85',
    'torchfrequentdirections/benchmarks.py': '832cdcab970a70465d79f8f73595f2bd3e50639d776e96bcfdc24a9240dd0020'}
SEED, REPEATS, WARMUPS = 20261020, 20, 2


def pinned_file(sha, path, expected, local=None):
    if local:
        raw = subprocess.check_output(['git', '-C', str(local), 'show', sha+':'+path], timeout=10)
    else:
        url = 'https://raw.githubusercontent.com/JohnnyZPiggy/TorchFrequentDirections/'+sha+'/'+path
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read(1024*1024+1)
    if len(raw) > 1024*1024 or hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError('SOURCE_MISMATCH')
    return raw


def module_from_bytes(name, raw, temporary):
    path = temporary/(name+'.py')
    path.write_bytes(raw)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def paired(functions, device, seed=SEED, repeats=REPEATS):
    """Randomized paired order on one device; synchronize outside/inside bounds."""
    rng = random.Random(seed)
    samples = {name: [] for name in functions}
    orders = []
    for trial in range(-WARMUPS, repeats):
        names = list(functions)
        rng.shuffle(names)
        if trial >= 0:
            orders.append(names)
        for name in names:
            sync(device)
            tick = time.perf_counter()
            functions[name]()
            sync(device)
            if trial >= 0:
                samples[name].append(1000*(time.perf_counter()-tick))
    result = {'order': orders, 'warmups_per_variant': WARMUPS, 'repeats': repeats,
              'scope': 'synchronized host wall milliseconds; includes Python/launch/sync; not kernel-only'}
    for name, values in samples.items():
        result[name] = {'samples_ms': values, 'median_ms': statistics.median(values),
            'p95_ms': sorted(values)[math.ceil(.95*len(values))-1],
            'min_ms': min(values), 'max_ms': max(values), 'sample_stdev_ms': statistics.stdev(values)}
    if set(functions) == {'baseline', 'candidate'}:
        result['ratio_of_medians'] = result['baseline']['median_ms']/result['candidate']['median_ms']
        result['paired_ratios'] = [a/b for a,b in zip(samples['baseline'],samples['candidate'])]
        result['median_paired_ratio'] = statistics.median(result['paired_ratios'])
    return result


def instrument(function, device):
    """Count true SVD/finite calls, with explicitly intrusive synchronized wall."""
    from unittest.mock import patch
    svd, finite = torch.linalg.svd, torch.isfinite
    counters = {'svd_calls': 0, 'finite_calls': 0, 'svd_synchronized_host_ms': 0.}
    def wrapped_svd(*args, **kwargs):
        sync(device)
        tick = time.perf_counter()
        value = svd(*args, **kwargs)
        sync(device)
        counters['svd_calls'] += 1
        counters['svd_synchronized_host_ms'] += 1000*(time.perf_counter()-tick)
        return value
    def wrapped_finite(*args, **kwargs):
        counters['finite_calls'] += 1
        return finite(*args, **kwargs)
    with patch.object(torch.linalg,'svd',wrapped_svd), patch.object(torch,'isfinite',wrapped_finite):
        sync(device)
        tick = time.perf_counter()
        function()
        sync(device)
        counters['instrumented_host_ms'] = 1000*(time.perf_counter()-tick)
    counters['scope'] = 'intrusive/nonadditive; isfinite counts include source-audited all()+bool checks; no kernel claim'
    return counters


def evaluate(a, ell, device, classes, helper):
    data = a.to(device)
    streams = {name:cls(ell,a.shape[1],dtype=a.dtype,device=device).update(data)
               for name,cls in classes.items()}
    outputs = {name:stream.sketch() for name,stream in streams.items()}
    oracle = {name:helper.oracle(a,b,ell) for name,b in outputs.items()}
    old,new = (outputs[name].cpu().double() for name in ('baseline','candidate'))
    gap = float(torch.linalg.matrix_norm(old.T@old-new.T@new,ord=2))
    tolerance = 4*oracle['candidate']['tolerance']
    stream = streams['candidate']
    count = stream._count
    active = count <= ell
    if active:
        assert torch.equal(outputs['candidate'][:count], stream._buffer[:count])
        assert int(torch.count_nonzero(outputs['candidate'][count:])) == 0
    for name,s in streams.items():
        before = s._buffer.clone()
        saved = s.sketch()
        saved.fill_(123)
        assert torch.equal(s._buffer,before)
        again = s.sketch()
        assert torch.equal(again,outputs[name]) and not again.requires_grad
    counts = {name:instrument(s.sketch,device) for name,s in streams.items()}
    assert counts['baseline']['svd_calls'] == int(count > 0)
    assert counts['candidate']['svd_calls'] == int(count > ell)
    # The SVD is removed, never the finite output failure check.
    assert counts['baseline']['finite_calls'] == counts['candidate']['finite_calls']
    status = 'PASS' if all(o['status']=='PASS' for o in oracle.values()) and gap<=tolerance else 'FAIL'
    # Preserve all k-specific values compactly: bounds reconstruct from ell/k/tail.
    for value in oracle.values():
        value['k_tail_projection'] = [[c['k'],c['tail_squared'],c['projection_squared']]
                                      for c in value.pop('checks')]
    return streams, {'status':status,'device':device,'n':a.shape[0],'d':a.shape[1],
        'ell':ell,'dtype':str(a.dtype),'occupancy':count,'short_path':active,
        'oracle':oracle,'gram_gap':gap,'gram_gap_tolerance':tolerance,'snapshot_profile':counts}


def edge_checks(classes, helper, devices):
    results=[]
    for dtype in (torch.float32,torch.float64):
        for ell,d in ((1,3),(4,9),(5,2)):
            for n in sorted({0,1,ell-1,ell,ell+1}):
                for kind in ('random','rank_one','zero'):
                    a=torch.randn(n,d,dtype=dtype,generator=torch.Generator().manual_seed(SEED+n))
                    if kind=='rank_one': a[:]=a[:,:1].expand_as(a)
                    if kind=='zero': a.zero_()
                    for device in devices:
                        _,check=evaluate(a,ell,device,classes,helper)
                        check['case']=kind
                        results.append(check)
        # Several fills, repeated snapshots, alias isolation, then continued updates.
        a=torch.randn(57,9,dtype=dtype,generator=torch.Generator().manual_seed(SEED))
        for device in devices:
            data=a.to(device)
            for name,cls in classes.items():
                stream=cls(4,9,dtype=dtype,device=device)
                start=0
                for stop in (8,9,10,13,23,24,57):
                    stream.update(data[start:stop])
                    b=stream.sketch()
                    check=helper.oracle(a[:stop],b,4)
                    assert check['status']=='PASS'
                    b.fill_(999)
                    assert helper.oracle(a[:stop],stream.sketch(),4)['status']=='PASS'
                    start=stop
                one_shot=cls(4,9,dtype=dtype,device=device).update(data).sketch()
                assert torch.equal(stream.sketch(),one_shot)
    assert all(c['status']=='PASS' for c in results)
    return results


def workloads(classes, helper, devices):
    results=[]
    # Prior four shapes, each paired with a nearby n exercising the other branch.
    for n,d,ell in ((64,32,4),(65,32,4),(124,64,8),(128,64,8),
                    (253,128,16),(256,128,16),(98,512,32),(128,512,32)):
        for dtype in (torch.float32,torch.float64):
            a=torch.randn(n,d,dtype=torch.float64,generator=torch.Generator().manual_seed(SEED)).to(dtype)
            for device in devices:
                streams,check=evaluate(a,ell,device,classes,helper)
                assert check['status']=='PASS'
                data=a.to(device)
                full={name:(lambda cls=cls:cls(ell,d,dtype=dtype,device=device).update(data).sketch())
                      for name,cls in classes.items()}
                output=streams['candidate'].sketch()
                row={'correctness':check,'snapshot':paired({name:s.sketch for name,s in streams.items()},device),
                     'full_update_snapshot':paired(full,device,seed=SEED+1),
                     'full_profile':{name:instrument(fn,device) for name,fn in full.items()},
                     'finite_check_calibration':paired({
                         'input':lambda:bool(torch.isfinite(data).all()),
                         'output':lambda:bool(torch.isfinite(output).all())},device)}
                row['finite_check_calibration']['scope']='standalone input/output bool(isfinite(all)); complete API including host sync; nonadditive, not kernel-only'
                expected_saved=int(0<check['occupancy']<=ell)
                assert row['full_profile']['baseline']['svd_calls']-row['full_profile']['candidate']['svd_calls']==expected_saved
                results.append(row)
    return results


def main():
    global torch
    sha=os.environ['SAE_CANDIDATE_SHA']
    if len(sha)!=40 or any(c not in '0123456789abcdef' for c in sha): raise ValueError('INVALID_SHA')
    cpu_only=os.environ.get('SAE_CPU_ONLY')=='1'
    local=os.environ.get('SAE_LOCAL_SOURCE') if cpu_only else None
    report={'status':'FAIL','source_sha':sha,'baseline_sha':BASE_SHA,
        'utc_start':datetime.now(timezone.utc).isoformat(),'seed':SEED,'gpu':'NOT_RUN',
        'tpu':'NOT_RUN','compilation':'NOT_RUN','secondary_zero_clear':'NOT_RUN_NOT_CHANGED',
        'finite_check_behavior':'PRESERVED','methodology':'same runtime, resident identical input/dtype; one CPU thread; two warmups, 20 paired shuffled repeats per variant; CUDA synchronized host wall, not kernel-only; generation/transfer excluded from A/B; CPU and GPU measured separately'}
    tick=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='fd5-') as temp:
        temp=Path(temp)
        try:
            helper=module_from_bytes('fd5_helper',pinned_file(sha,'experiments/fd_gpu/driver.py',HELPER_HASH,local),temp)
            helper.SOURCE_FILES=dict(SOURCE_FILES)
            report['install']=helper.install_source(sha,temp,local)
            import torch
            helper.torch=torch
            torch.set_num_threads(1)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.set_float32_matmul_precision('highest')
            baseline=module_from_bytes('fd5_baseline',pinned_file(BASE_SHA,'torchfrequentdirections/fd.py',BASE_FD_HASH,local),temp)
            from torchfrequentdirections import FrequentDirections
            classes={'baseline':baseline.FrequentDirections,'candidate':FrequentDirections}
            report['environment']={'python':platform.python_version(),'torch':str(torch.__version__),
                'cuda':torch.version.cuda,'cpu_threads':torch.get_num_threads(),'tf32':False,
                'cpu_model':next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'UNKNOWN')}
            devices=['cpu']
            if not cpu_only:
                if not torch.cuda.is_available(): raise RuntimeError('CUDA_UNAVAILABLE')
                p=torch.cuda.get_device_properties(0)
                report['device']={'observed':p.name,'compute_capability':[p.major,p.minor],
                    'total_memory_bytes':p.total_memory,'tensor_device':str(torch.empty(1,device='cuda').device)}
                report['svd_probes']=[]
                for dtype in (torch.float32,torch.float64):
                    x=torch.eye(4,dtype=dtype,device='cuda')
                    u,s,vh=torch.linalg.svd(x,full_matrices=False)
                    error=float((u@torch.diag(s)@vh-x).abs().max())
                    assert error==0
                    report['svd_probes'].append({'dtype':str(dtype),'max_error':error})
                devices.append('cuda')
            report['edge_checks']=edge_checks(classes,helper,devices)
            report['workloads']=workloads(classes,helper,devices)
            report.update(status='PASS',gpu='NOT_RUN' if cpu_only else 'PASS')
        except Exception as exc:
            report['error_class']=type(exc).__name__
            report['error_code']=str(exc) if str(exc) in ('CUDA_UNAVAILABLE','SOURCE_MISMATCH','INSTALL_FAILED') else 'NUMERICAL_OR_ENVIRONMENT_FAILURE'
    report.update(utc_end=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-tick)
    out=Path(os.environ.get('SAE_ARTIFACT_DIR','/content'))
    out.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(report,indent=2,allow_nan=False)+'\n').encode()
    (out/'fd-snapshot.json').write_bytes(raw)
    print('ARTIFACT fd-snapshot.json',len(raw),hashlib.sha256(raw).hexdigest(),flush=True)
    # Structured FAIL evidence must still be downloaded; the pinned controller
    # rejects its status after preserving the artifact. Local runner checks status.
    return report


if __name__=='__main__':
    main()
