"""Issue #7: occupied-prefix safety plus balanced paired clear/FD measurements."""
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
from unittest.mock import patch

BASE_SHA='f6ef317edd2e9ad1a42d8bc37a9a2e6229f88448'
BASE_HASH='18e84afa295d3bb75cb97f68bd98fa339fc3ee061e26e8071051021066d30e85'
CANDIDATE_HASH='766f0981a371e2bb2b90fc9e2368d0aacbed14e15e9921508cc93f29a5e62172'
SUPPORT_HASH='c148c939434634f84a63c09038ec2514fb1b37f3768b0bca048c1d21ccb673bd'
INSTALLER_HASH='c3f0cb1dc62fadefe3aa3051a85c4e1ed1420879817a4671c20b986e9262ae86'
CHECKS_HASH='e574b55a29194bddcad3cf236c1669191c8d6a6a19bbddd661820b9e00787e0d'
SEED=20261020


def load_support(local, temporary):
    path='experiments/snapshot/driver.py'
    if local:
        raw=subprocess.check_output(['git','-C',str(local),'show',BASE_SHA+':'+path],timeout=10)
    else:
        with urllib.request.urlopen('https://raw.githubusercontent.com/JohnnyZPiggy/TorchFrequentDirections/'+BASE_SHA+'/'+path,timeout=30) as response:
            raw=response.read(1024*1024+1)
    if hashlib.sha256(raw).hexdigest()!=SUPPORT_HASH: raise RuntimeError('SOURCE_MISMATCH')
    target=temporary/'support.py'; target.write_bytes(raw)
    spec=importlib.util.spec_from_file_location('fd7_support',target)
    result=importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def paired(functions, device, seed=SEED, repeats=20):
    """Equal AB/BA trial counts, shuffled across paired blocks."""
    assert repeats>=20 and repeats%2==0
    names=list(functions)
    rng=random.Random(seed)
    orders=[names[:],names[::-1]]*(repeats//2)
    rng.shuffle(orders)
    samples={name:[] for name in names}
    for order in [names,names[::-1]]+orders:
        for name in order:
            sync(device); tick=time.perf_counter(); functions[name](); sync(device)
            samples[name].append(1000*(time.perf_counter()-tick))
    result={'order':orders,'warmups':2,'repeats':repeats}
    for name,values in samples.items():
        values=values[2:]
        result[name]={'samples_ms':values,'median_ms':statistics.median(values),
            'p95_ms':sorted(values)[math.ceil(.95*repeats)-1],
            'min_ms':min(values),'max_ms':max(values),'sample_sd_ms':statistics.stdev(values)}
    ratios=[a/b for a,b in zip(result['baseline']['samples_ms'],result['candidate']['samples_ms'])]
    result['paired_ratios']=ratios
    result['ratio_of_medians']=result['baseline']['median_ms']/result['candidate']['median_ms']
    result['median_paired_ratio']=statistics.median(ratios)
    rng=random.Random(seed+17)
    bootstrap=sorted(statistics.median(rng.choices(ratios,k=repeats)) for _ in range(2000))
    result['paired_median_bootstrap_95']=[bootstrap[49],bootstrap[1949]]
    result['interval_limit']='descriptive paired bootstrap, not independent sessions or multiple-comparison-adjusted inference'
    return result


def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize()


def profile(fn, device, support):
    zero=torch.Tensor.zero_
    extra={'clear_calls':0,'clear_intrusive_host_ms':0.}
    def wrapped(tensor,*args,**kwargs):
        sync(device); tick=time.perf_counter()
        result=zero(tensor,*args,**kwargs)
        sync(device)
        extra['clear_calls']+=1
        extra['clear_intrusive_host_ms']+=1000*(time.perf_counter()-tick)
        return result
    with patch.object(torch.Tensor,'zero_',wrapped): result=support.instrument(fn,device)
    result.update(extra)
    result['scope']='intrusive/nonadditive synchronized host wall; clear-only timing separate; not kernel-only'
    return result


def compact_oracle(report):
    report['k_tail_projection']=[[c['k'],c['tail_squared'],c['projection_squared']] for c in report.pop('checks')]
    return report


def run_checks(classes, checks, installer, devices):
    results=[]; faults=[]
    for dtype in (torch.float32,torch.float64):
        for ell,d in ((1,3),(4,7),(6,2)):
            n=8*ell+7
            for kind in ('random','rank_one','zero'):
                a=torch.randn(n,2*d,dtype=dtype,generator=torch.Generator().manual_seed(SEED))[:,::2]
                if kind=='rank_one': a[:]=a[:,:1].expand_as(a)
                if kind=='zero': a.zero_()
                patterns=([0,n],list(range(n+1)),sorted({0,1,ell-1,ell,ell+1,2*ell,2*ell+1,n//2,n-1,n}))
                for device in devices:
                    data=a.to(device)
                    # Preserve a genuinely noncontiguous CUDA input as well.
                    if device=='cuda': data=torch.stack((data,data),dim=2)[:,:,0]
                    for stops in patterns:
                        result=checks.check_stream(classes,data,ell,stops,installer.oracle)
                        result.update(device=device,dtype=str(dtype),ell=ell,d=d,case=kind,
                                      input_noncontiguous=not data.is_contiguous())
                        results.append(result)
        for device in devices:
            faults.append({'device':device,'dtype':str(dtype),'checks':checks.fault_checks(classes,device,dtype)})
    return results,faults


def measure(classes, installer, support, devices):
    results=[]; clears=[]
    shapes=((7,64,8),(15,64,8),(64,32,4),(512,64,8),(256,512,32),(258,1024,64))
    for n,d,ell in shapes:
        for dtype in (torch.float32,torch.float64):
            a=torch.randn(n,d,dtype=dtype,generator=torch.Generator().manual_seed(SEED))
            for device in devices:
                data=a.to(device)
                streams={name:cls(ell,d,dtype=dtype,device=device).update(data) for name,cls in classes.items()}
                outputs={name:s.sketch() for name,s in streams.items()}
                oracle={name:installer.oracle(a,b,ell) for name,b in outputs.items()}
                assert all(v['status']=='PASS' for v in oracle.values())
                old,new=(outputs[name].cpu().double() for name in ('baseline','candidate'))
                gap=float(torch.linalg.matrix_norm(old.T@old-new.T@new,ord=2))
                tolerance=4*oracle['candidate']['tolerance']
                assert gap<=tolerance
                functions={name:(lambda cls=cls:cls(ell,d,dtype=dtype,device=device).update(data).sketch()) for name,cls in classes.items()}
                timing=paired(functions,device,seed=SEED+n)
                profiles={name:profile(fn,device,support) for name,fn in functions.items()}
                assert profiles['baseline']['svd_calls']==profiles['candidate']['svd_calls']
                assert profiles['baseline']['finite_calls']==profiles['candidate']['finite_calls']
                expected=0 if n<2*ell else 1+(n-2*ell)//(2*ell-min(ell-1,d))
                assert profiles['baseline']['clear_calls']==expected and profiles['candidate']['clear_calls']==0
                results.append({'status':'PASS','n':n,'d':d,'ell':ell,'dtype':str(dtype),'device':device,
                    'no_full_shrink':expected==0,'full_shrinks':expected,'oracle':{k:compact_oracle(v) for k,v in oracle.items()},
                    'gram_gap':gap,'gram_gap_tolerance':tolerance,'timing':timing,'profiles':profiles})
    # Clear resident hot buffers only, without SVD or allocation in the timed call.
    for ell,d in ((4,32),(8,64),(32,512),(64,1024),(256,4096)):
        for dtype in (torch.float32,torch.float64):
            for device in devices:
                buffer=torch.ones(2*ell,d,dtype=dtype,device=device)
                clears.append({'ell':ell,'d':d,'dtype':str(dtype),'device':device,
                    'bytes':buffer.numel()*buffer.element_size(),
                    'timing':paired({'baseline':buffer.zero_,'candidate':lambda:None},device,repeats=40),
                    'scope':'resident hot-buffer zero_ versus Python no-op; synchronized host API latency, not additive FD savings or kernel-only'})
    return results,clears


def main():
    global torch
    sha=os.environ['SAE_CANDIDATE_SHA']
    if len(sha)!=40 or any(c not in '0123456789abcdef' for c in sha): raise ValueError('INVALID_SHA')
    cpu_only=os.environ.get('SAE_CPU_ONLY')=='1'
    local=os.environ.get('SAE_LOCAL_SOURCE') if cpu_only else None
    report={'status':'FAIL','source_sha':sha,'baseline_sha':BASE_SHA,'seed':SEED,
        'utc_start':datetime.now(timezone.utc).isoformat(),'gpu':'NOT_RUN','tpu':'NOT_RUN','compilation':'NOT_RUN',
        'performance_rule':'Report all cases; a full-workload median paired gain >=1% with descriptive 95% lower>1 is a candidate signal, not acceptance. If absent, recommend retaining baseline. No-clear controls diagnose variation.',
        'methodology':'same process/runtime/input/dtype; one CPU thread; two warmups, 20 balanced shuffled AB/BA trials; main synchronized host latency includes allocation/update/snapshot, excludes input generation/transfers/install; clear-only uses 40 pairs'}
    tick=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='fd7-') as temp:
        temp=Path(temp)
        try:
            support=load_support(local,temp)
            installer=support.module_from_bytes('fd7_installer',support.pinned_file(BASE_SHA,'experiments/fd_gpu/driver.py',INSTALLER_HASH,local),temp)
            installer.SOURCE_FILES=dict(support.SOURCE_FILES)
            installer.SOURCE_FILES['torchfrequentdirections/fd.py']=CANDIDATE_HASH
            report['install']=installer.install_source(sha,temp,local)
            import torch
            support.torch=installer.torch=torch
            torch.set_num_threads(1)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.set_float32_matmul_precision('highest')
            raw=support.pinned_file(BASE_SHA,'torchfrequentdirections/fd.py',BASE_HASH,local)
            assert hashlib.sha256(raw.replace(b'                self._buffer.zero_()\n',b'',1)).hexdigest()==CANDIDATE_HASH
            report['only_algorithm_change']='exact baseline source minus one self._buffer.zero_() line; finite checks byte-identical'
            baseline=support.module_from_bytes('fd7_baseline',raw,temp)
            checks=support.module_from_bytes('fd7_checks',support.pinned_file(sha,'experiments/zero_clear/checks.py',CHECKS_HASH,local),temp)
            from torchfrequentdirections import FrequentDirections
            classes={'baseline':baseline.FrequentDirections,'candidate':FrequentDirections}
            report['environment']={'python':platform.python_version(),'torch':str(torch.__version__),'cuda':torch.version.cuda,
                'cpu_threads':torch.get_num_threads(),'tf32':False,
                'cpu_model':next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'UNKNOWN')}
            devices=['cpu']
            if not cpu_only:
                if not torch.cuda.is_available(): raise RuntimeError('CUDA_UNAVAILABLE')
                p=torch.cuda.get_device_properties(0)
                report['device']={'observed':p.name,'compute_capability':[p.major,p.minor],
                    'total_memory_bytes':p.total_memory,'tensor_device':str(torch.empty(1,device='cuda').device)}
                devices.append('cuda')
            report['safety_checks'],report['fault_checks']=run_checks(classes,checks,installer,devices)
            report['workloads'],report['clear_only']=measure(classes,installer,support,devices)
            report.update(status='PASS',gpu='NOT_RUN' if cpu_only else 'PASS')
        except Exception as exc:
            report['error_class']=type(exc).__name__
            report['error_code']=str(exc) if str(exc) in ('SOURCE_MISMATCH','CUDA_UNAVAILABLE','INSTALL_FAILED') else 'NUMERICAL_OR_ENVIRONMENT_FAILURE'
    report.update(utc_end=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-tick)
    out=Path(os.environ.get('SAE_ARTIFACT_DIR','/content')); out.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(report,indent=2,allow_nan=False)+'\n').encode()
    (out/'fd-zero-clear.json').write_bytes(raw)
    print('ARTIFACT fd-zero-clear.json',len(raw),hashlib.sha256(raw).hexdigest(),flush=True)
    return report


if __name__=='__main__': main()
