"""Pinned standalone controller payload, CPU gate and one-session T4 study."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from unittest.mock import patch

SOURCE_FILES = {'LICENSE': '2f0819452dd5f7667e2c0e93b769238a41c8164e62f89f212646076f66765bde', 'torchfrequentdirections/__init__.py': 'e65e1bcff754fe3cc2b5e3fb383f7b6293a7a3a358757e8cc0861856e54306c7', 'torchfrequentdirections/fd.py': '18e84afa295d3bb75cb97f68bd98fa339fc3ee061e26e8071051021066d30e85', 'experiments/qr_svd/prototype.py': '58c10f6297588c79f7ddb00f58e9f0b607c8b690405c2f8e1f49bc778ae1ddb2', 'experiments/qr_svd/cuda_adapter.py': '2916edf326cd26522c1c92127ade2e57dee2394866b686823d08fbd2ddb004e1', 'experiments/qr_svd/driver.py': '65845b029b415e9214947433a76523380ae78f8e90006c11e304e7d9336480a4', 'experiments/zero_clear/driver.py': '0ffa14cbf6a56ea4fadd10f252a5bb6a56fa8ae70bf8d794e1c91d41958ce18e'}
SEED = 1320261021
START = time.monotonic()


def budget():
    if time.monotonic()-START > 340:
        raise TimeoutError('scientific deadline; cleanup reserved')


def install(sha, destination, local):
    if not re.fullmatch('[0-9a-f]{40}',sha) or not SOURCE_FILES:
        raise RuntimeError('SOURCE_IDENTITY_INVALID')
    if local and subprocess.check_output(['git','-C',local,'rev-parse','HEAD'],text=True).strip()!=sha:
        raise RuntimeError('SOURCE_SHA_MISMATCH')
    for name, expected in SOURCE_FILES.items():
        budget()
        if local:
            raw=subprocess.check_output(['git','-C',local,'show',sha+':'+name],timeout=10)
        else:
            with urllib.request.urlopen('https://raw.githubusercontent.com/JohnnyZPiggy/TorchFrequentDirections/'+sha+'/'+name,timeout=20) as response:
                raw=response.read(1024*1024+1)
        if hashlib.sha256(raw).hexdigest()!=expected:
            raise RuntimeError('SOURCE_HASH_MISMATCH')
        path=destination/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
    sys.path.insert(0,str(destination))
    return dict(status='PASS',source_sha=sha,files=SOURCE_FILES,mode='exact Git blobs + hash-checked module import; no deps')


def compact(result):
    # Keep every k-bound while avoiding repeated JSON keys in a ≤1 MiB artifact.
    for oracle in result['oracles']:
        oracle['k_tail_cov_projection_bound_pass']=[
            [c['k'],c['tail_squared'],c['covariance_bound'],c['projection_squared'],c['projection_bound'],c['passed']]
            for c in oracle.pop('checks')]
    return result


def check(a, ell, classes, oracle, pattern=True):
    outputs=[]; paths=[]; prefixes=[]
    original=a.detach().clone()
    for cls in classes:
        state=cls(ell,a.shape[1],dtype=a.dtype,device=a.device)
        stops=sorted({0,min(2,len(a)),min(7,len(a)),len(a)}) if pattern else [len(a)]
        start=0; factor_shapes=[]
        factor=state._factor
        def counted(rows,*,shrink):
            factor_shapes.append([len(rows),rows.shape[1],shrink]); return factor(rows,shrink=shrink)
        state._factor=counted
        for stop in stops:
            budget(); state.update(a[start:stop]); state.update(a[stop:stop]); start=stop
            snapshot=state.sketch(); again=state.sketch()
            saved=state._buffer.clone(); count=state._count
            if not torch.equal(snapshot,again) or snapshot.requires_grad:
                raise AssertionError('snapshot/no-grad')
            again.fill_(123)
            if not torch.equal(snapshot,state.sketch()) or not torch.equal(saved,state._buffer) or count!=state._count:
                raise AssertionError('snapshot alias/state')
            if pattern:
                r=oracle.oracle(a[:stop].cpu(),snapshot.cpu(),ell)
                if not r['passed']: raise AssertionError('prefix FD bound')
                prefixes.append(dict(stop=stop,minimum_eigenvalue=r['minimum_eigenvalue'],
                                     tolerance=r['tolerance'],k_count=len(r['checks']),passed=r['passed']))
        outputs.append(state.sketch().cpu()); paths.append(dict(final_count=state._count,factor_shapes=factor_shapes))
        before=state._buffer.clone()
        with torch.no_grad():
            invalid=torch.full((1,a.shape[1]),float('nan'),dtype=a.dtype,device=a.device)
        try: state.update(invalid)
        except ValueError: pass
        else: raise AssertionError('missing finite input check')
        if not torch.equal(before,state._buffer): raise AssertionError('invalid mutation')
    if not torch.equal(a.detach(),original): raise AssertionError('input mutation')
    result=oracle.compare(a.cpu(),outputs,ell)
    if not result['passed']: raise AssertionError('independent FD/Gram gate')
    return dict(math=compact(result),prefixes=prefixes,paths=paths)


def correctness(classes, oracle, device):
    records=[]
    for dtype in (torch.float32,torch.float64):
        for name,cpu in oracle.fixtures(dtype).items():
            if name=='long_drift': continue
            for ell in (1,4,cpu.shape[1]+2):
                a=cpu.to(device)
                # Real noncontiguous storage on the target device.
                storage=torch.empty(len(a),2*a.shape[1],dtype=dtype,device=device)
                storage[:,::2]=a; a=storage[:,::2]
                r=check(a,ell,classes,oracle)
                records.append(dict(case=name,dtype=str(dtype),ell=ell,n=len(a),d=a.shape[1],
                                    noncontiguous=not a.is_contiguous(),**r))
    # Exact threshold spectrum on the actual matrix being shrunk, separate from streams.
    for dtype in (torch.float32,torch.float64):
        s=torch.tensor([2.,1.,1.,1.,1.,.1,1e-8,1e-16],dtype=dtype,device=device)
        a=torch.cat([torch.diag(s),torch.zeros(8,9,dtype=dtype,device=device)],dim=1)
        for ell in (4,):
            outputs=[cls(ell,17,dtype=dtype,device=device)._factor(a,shrink=True).cpu() for cls in classes]
            r=oracle.compare(a.cpu(),outputs,ell)
            if not r['passed']: raise AssertionError('tied factor')
            records.append(dict(case='exact_local_threshold',dtype=str(dtype),ell=ell,n=8,d=17,math=compact(r)))
    faults=[]
    for cls in classes:
        for op in ('qr','svd'):
            if cls.__name__=='FrequentDirections' and op=='qr': continue
            with patch('torch.linalg.'+op,side_effect=RuntimeError('injected')):
                try: cls(4,17,dtype=torch.float64,device=device).update(torch.ones(8,17,dtype=torch.float64,device=device))
                except RuntimeError: faults.append(dict(algorithm=cls.__name__,operation=op,status='PASS'))
                else: raise AssertionError('backend failure swallowed')
    return dict(records=records,faults=faults,status='PASS')


def probe():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA_UNAVAILABLE')
    p=torch.cuda.get_device_properties(0)
    if p.name!='Tesla T4': raise RuntimeError('UNEXPECTED_DEVICE')
    checks=[]
    for dtype in (torch.float32,torch.float64):
        a=torch.arange(40,dtype=dtype,device='cuda:0').reshape(5,8)/40
        q,r=torch.linalg.qr(a.T,mode='reduced')
        u,s,vh=torch.linalg.svd(r.T,full_matrices=False)
        torch.cuda.synchronize()
        tolerance=64*torch.finfo(dtype).eps*float(torch.linalg.vector_norm(a))
        error=float(torch.linalg.vector_norm(a-(u*s)@vh@q.T))
        orth=float(torch.linalg.vector_norm(q.T@q-torch.eye(5,dtype=dtype,device='cuda:0')))
        if error>tolerance or orth>64*torch.finfo(dtype).eps*5:
            raise AssertionError('QR/SVD probe correctness')
        checks.append(dict(dtype=str(dtype),reconstruction_error=error,tolerance=tolerance,orthogonality_error=orth,status='PASS'))
    return dict(status='PASS',device=p.name,compute_capability=[p.major,p.minor],
        tensor_device=str(a.device),total_memory_bytes=p.total_memory,
        free_memory_bytes=torch.cuda.mem_get_info()[0],torch=torch.__version__,cuda=torch.version.cuda,
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
        probes=checks)


def stream(a,ell,cls):
    state=cls(ell,a.shape[1],dtype=a.dtype,device=a.device)
    for part in a.split(max(1,ell//2)):
        state.update(part)
    return state.sketch()


def diagnostics(a,ell):
    samples=[]
    for _ in range(5):
        torch.cuda.synchronize(); t0=time.perf_counter()
        q,r=torch.linalg.qr(a.T,mode='reduced'); torch.cuda.synchronize(); t1=time.perf_counter()
        _,s,v=torch.linalg.svd(r.T,full_matrices=False); torch.cuda.synchronize(); t2=time.perf_counter()
        vh=v@q.T; torch.cuda.synchronize(); t3=time.perf_counter()
        scale=s[0].clamp_min(torch.finfo(s.dtype).tiny); z=s/scale
        s=scale*(z.square()-z[ell-1].square()).clamp_min(0).sqrt()
        output=a.new_zeros((ell,a.shape[1])); output[:]=s[:ell,None]*vh[:ell]
        if not bool(torch.isfinite(output).all()): raise AssertionError('diagnostic nonfinite')
        torch.cuda.synchronize(); t4=time.perf_counter()
        samples.append([1000*(b-c) for b,c in ((t1,t0),(t2,t1),(t3,t2),(t4,t3))])
    return dict(columns=['qr_ms','small_svd_ms','vh_recovery_ms','shrink_output_ms'],
                samples=samples,scope='intrusive synchronized host wall; NONADDITIVE')


def benchmarks(classes, oracle, timing):
    specs=[]; skipped=[]
    for dtype in (torch.float32,torch.float64):
        for m,d,ell in ((32,64,16),(32,128,16),(32,512,16),(64,1024,32),(16,8,8),(128,2048,64)):
            for kind in ('factor','short','long'):
                retained=min(ell-1,d)
                n=m if kind=='factor' else m+3*(2*ell-retained)+(ell-1 if kind=='short' else ell+1)-retained
                a=torch.randn(n,d,dtype=dtype,generator=torch.Generator().manual_seed(SEED+n+d)).to('cuda:0')
                specs.append((dict(dtype=str(dtype),m=m,d=d,ell=ell,n=n,kind=kind),a))
        a=torch.randn(15,128,dtype=dtype,generator=torch.Generator().manual_seed(SEED)).to('cuda:0')
        specs.append((dict(dtype=str(dtype),m=32,d=128,ell=16,n=15,kind='no_factor'),a))
    records=[]
    # Oracle every measured configuration before any timing.
    for spec,a in specs:
        budget(); ell=spec['ell']
        if spec['kind']=='factor':
            b=[cls(ell,a.shape[1],dtype=a.dtype,device=a.device)._factor(a,shrink=True).cpu() for cls in classes]
            result=oracle.compare(a.cpu(),b,ell)
            if not result['passed']: raise AssertionError('factor benchmark gate')
            gate=dict(math=compact(result))
        else:
            gate=check(a,ell,classes,oracle,pattern=False)
        records.append(dict(**spec,gate=gate,status='READY'))
    for index,((spec,a),record) in enumerate(zip(specs,records)):
        if time.monotonic()-START > 285:
            record['status']='NOT_RUN'; record['reason']='cleanup reserve / scientific budget'; continue
        ell=spec['ell']
        if spec['kind']=='factor':
            instances=[cls(ell,a.shape[1],dtype=a.dtype,device=a.device) for cls in classes]
            functions={name:lambda s=s:s._factor(a,shrink=True) for name,s in zip(('baseline','candidate'),instances)}
        else:
            functions={name:lambda cls=cls:stream(a,ell,cls) for name,cls in zip(('baseline','candidate'),classes)}
        record['seed']=SEED+index
        record['timing']=timing.paired(functions,'cuda',seed=record['seed'])
        if spec['kind']=='factor' and spec['m']<=spec['d']:
            record['diagnostics']=diagnostics(a,ell)
        if spec['kind'] in ('no_factor',) or spec['kind']=='long' and spec['d']==512:
            fn=functions['baseline']; record['placebo_seed']=SEED+500+index
            record['AA']=timing.paired({'baseline':fn,'candidate':fn},'cuda',seed=record['placebo_seed'])
        record['status']='PASS'
    return records


def main():
    global torch
    sha=os.environ.get('SAE_CANDIDATE_SHA','')
    cpu_only=os.environ.get('SAE_CPU_ONLY')=='1'
    report=dict(source_sha=sha,status='FAIL',utc_start=datetime.now(timezone.utc).isoformat(),
        seed=SEED,classification='CPU_ONLY' if cpu_only else 'PROBE_ONLY',
        tolerance='32 eps(dtype) max(n,d,2ell,1) ||A||_F^2; Gram Frobenius gap same bound',
        screen='paired median direct/QR>1.05 and descriptive lower>1; noisy controls => INCONCLUSIVE',
        interval='exploratory descriptive paired bootstrap; no multiple-comparison correction',
        scope='GPU-resident synchronized complete update+sketch; H2D/D2H and data generation excluded',
        tiny_singular_relative_accuracy='NOT_ESTABLISHED')
    with tempfile.TemporaryDirectory(prefix='fd13-source-') as temp:
        try:
            report['install']=install(sha,Path(temp),os.environ.get('SAE_LOCAL_SOURCE'))
            import torch
            from experiments.qr_svd.cuda_adapter import ExperimentalQRSmallSVD
            from experiments.qr_svd import driver as oracle
            from experiments.zero_clear import driver as timing
            from torchfrequentdirections import FrequentDirections
            timing.torch=torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.set_float32_matmul_precision('highest')
            report['environment']=dict(python=platform.python_version(),torch=str(torch.__version__),
                cuda=torch.version.cuda,cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                tf32=False,float32_matmul_precision=torch.get_float32_matmul_precision())
            classes=(FrequentDirections,ExperimentalQRSmallSVD)
            if not cpu_only: report['probe']=probe()
            report['correctness']=correctness(classes,oracle,'cpu' if cpu_only else 'cuda:0')
            if not cpu_only: report['workloads']=benchmarks(classes,oracle,timing)
            report.update(status='PASS',classification='CPU_PASS' if cpu_only else 'VALIDATE_PASS')
        except Exception as error:
            report.update(error_class=type(error).__name__,error_code=str(error)[:100])
    report.update(utc_end=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-START)
    out=Path(os.environ.get('SAE_ARTIFACT_DIR','/content')); out.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(report,separators=(',',':'),allow_nan=False)+'\n').encode()
    (out/'fd-cuda-qr.json').write_bytes(raw)
    print('ARTIFACT fd-cuda-qr.json',len(raw),hashlib.sha256(raw).hexdigest(),flush=True)
    return report


if __name__=='__main__': main()
