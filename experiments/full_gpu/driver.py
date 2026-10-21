"""Pinned Issue #17 study. Allocation is exclusively the immutable controller's job."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

SOURCE_FILES = {'LICENSE': '2f0819452dd5f7667e2c0e93b769238a41c8164e62f89f212646076f66765bde', 'torchfrequentdirections/__init__.py': 'e65e1bcff754fe3cc2b5e3fb383f7b6293a7a3a358757e8cc0861856e54306c7', 'torchfrequentdirections/fd.py': '18e84afa295d3bb75cb97f68bd98fa339fc3ee061e26e8071051021066d30e85', 'experiments/qr_svd/driver.py': '65845b029b415e9214947433a76523380ae78f8e90006c11e304e7d9336480a4', 'experiments/qr_svd/cuda_driver.py': 'd8f919ac1d92a3f0ba17a577f84b4f98ff9b8361921b3ebf463fc66f7ef7b9b6', 'experiments/qr_svd/prototype.py': '58c10f6297588c79f7ddb00f58e9f0b607c8b690405c2f8e1f49bc778ae1ddb2', 'experiments/zero_clear/driver.py': '0ffa14cbf6a56ea4fadd10f252a5bb6a56fa8ae70bf8d794e1c91d41958ce18e', 'experiments/compile_fd/driver.py': '4bb2108c0f2150770643d1f7d9f943d570388eca7fe014124a39b0bb74a178c0', 'experiments/full_gpu/candidate.py': 'f37ad64cfb11a7bc1071566e7d196cd9a1756bbbdd76a20a1ece4a9e986c8a66'}
SEED=1720261021
START=time.monotonic()


def bootstrap(sha,directory):
    if len(sha)!=40 or any(c not in '0123456789abcdef' for c in sha): raise ValueError('bad SHA')
    local=os.environ.get('SAE_LOCAL_SOURCE')
    for name,digest in SOURCE_FILES.items():
        if local:
            assert subprocess.check_output(['git','-C',local,'rev-parse','HEAD'],text=True).strip()==sha
            raw=subprocess.check_output(['git','-C',local,'show',sha+':'+name],timeout=10)
        else:
            with urllib.request.urlopen('https://raw.githubusercontent.com/JohnnyZPiggy/TorchFrequentDirections/'+sha+'/'+name,timeout=20) as r: raw=r.read(1048577)
        assert hashlib.sha256(raw).hexdigest()==digest,'source mismatch'
        p=directory/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(raw)
    sys.path.insert(0,str(directory))


def error(exc):
    return dict(type=type(exc).__name__,message=str(exc)[:1200])


def bounded(fn,seconds=45):
    def expired(*_): raise TimeoutError('bounded scientific probe')
    old=signal.signal(signal.SIGALRM,expired); signal.setitimer(signal.ITIMER_REAL,seconds)
    try: return fn()
    finally: signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old)


def run(a,ell,cls):
    state=cls(ell,a.shape[1],dtype=a.dtype,device=a.device)
    for part in a.split(max(1,ell//2)): state.update(part)
    return state.sketch()


def gate(a,ell,oracle,support):
    states=[cls(ell,a.shape[1],dtype=a.dtype,device=a.device) for cls in (FD,Candidate)]
    counts=[]
    for part in a.split(max(1,ell//2)):
        for state in states: state.update(part); state.update(part[:0])
        assert states[0]._count==states[1]._count
        torch.testing.assert_close(states[0]._buffer,states[1]._buffer,rtol=0,atol=0)
        counts.append(states[0]._count)
    outputs=[s.sketch().cpu() for s in states]
    result=oracle.compare(a.cpu(),outputs,ell); assert result['passed'],'math gate'
    state=states[1]; before=state._buffer.clone(); count=state._count
    state.sketch().fill_(123)
    assert torch.equal(state._buffer,before) and state._count==count
    assert not state.sketch().requires_grad
    for value in (float('inf'),float('nan')):
        try: state.update(a.new_full((1,a.shape[1]),value))
        except ValueError: pass
        else: raise AssertionError('finite check bypass')
        assert torch.equal(state._buffer,before) and state._count==count
    return dict(math=support.compact(result),occupancies=counts,invalid_atomicity='PASS',snapshot='PASS')


def metrics():
    r=subprocess.run(['nvidia-smi','--query-gpu=name,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
    return dict(status='OBSERVED' if r.returncode==0 else 'UNAVAILABLE',fields='name,temp_C,sm_MHz,memory_MHz,power_W,utilization_pct',value=r.stdout.strip()[:250])


def graph_probe(a,ell):
    from experiments.full_gpu.candidate import graph_transition
    rows=a[:2*ell].contiguous().clone(); buffer=torch.empty_like(rows)
    report=dict(scope='NON_EQUIVALENT tensor transition only; strict finite predicates excluded',full_stream='FULL_STREAM_GRAPH_NOT_AVAILABLE',capture_attempts=1)
    stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2): graph_transition(rows,buffer,ell)
    torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph(); tick=time.monotonic()
    try:
        with torch.cuda.graph(graph): graph_transition(rows,buffer,ell)
        torch.cuda.synchronize(); report['capture_seconds']=time.monotonic()-tick
        samples=[]
        for index in range(5):
            # Copy outside capture; each replay zeroes all destination storage.
            fresh=rows.clone()*(1+index/10); rows.copy_(fresh)
            torch.cuda.synchronize(); tick=time.perf_counter(); graph.replay(); torch.cuda.synchronize()
            samples.append((time.perf_counter()-tick)*1000)
            expected=FD(ell,rows.shape[1],dtype=rows.dtype,device=rows.device).update(rows)
            torch.testing.assert_close(buffer,expected._buffer,rtol=64*torch.finfo(rows.dtype).eps,atol=64*torch.finfo(rows.dtype).eps)
        report.update(status='CAPTURE_REPLAY_PASS_PROBE_ONLY',replay_ms=samples,input_address=rows.data_ptr(),output_address=buffer.data_ptr(),recaptures=0)
    except Exception as exc:
        report.update(status='CAPTURE_FAILED_PROBE_ONLY',capture_seconds=time.monotonic()-tick,error=error(exc))
    return report


def paired_block(functions,seed,timing):
    return dict(seed=seed,timing=timing.paired(functions,'cuda',seed=seed,repeats=20))


def study(oracle,support,timing,profiler,directory):
    report=dict(records=[],block_order=[],controls_first=True,full_stream_graph='NOT_AVAILABLE; strict host finite predicates remain eager')
    # Mandatory H0 control before candidates, repeated later in each sandwich.
    control=torch.randn(95,64,dtype=torch.float32).cuda()
    control_fn=lambda:run(control,16,FD)
    report['initial_AA']=paired_block(dict(baseline=control_fn,candidate=control_fn),SEED,timing)
    specs=[]
    for ell,d in ((16,64),(16,512),(32,1024),(64,2048),(8,3),(2,8)):
        for dtype in (torch.float32,torch.float64):
            for final in (ell-1,ell+1): specs.append((ell,d,dtype,final,'stream'))
    for dtype in (torch.float32,torch.float64): specs.append((16,64,dtype,15,'no_factor'))
    random.Random(SEED).shuffle(specs)
    for index,(ell,d,dtype,final,kind) in enumerate(specs):
        record=dict(index=index,ell=ell,d=d,dtype=str(dtype),final_count=final,kind=kind)
        report['records'].append(record); report['block_order'].append(index)
        if time.monotonic()-START>285:
            record.update(status='NOT_RUN',reason='timing reserve for cleanup'); continue
        retained=min(ell-1,d); n=2*ell+3*(2*ell-retained)+final-retained if kind=='stream' else final
        cpu=torch.randn(n,d,dtype=dtype,generator=torch.Generator().manual_seed(SEED+index)); a=cpu.cuda()
        record['n']=n; record['device_before']=metrics()
        fns=dict(baseline=lambda:run(a,ell,FD),candidate=lambda:run(a,ell,Candidate))
        cold={}
        for name,fn in fns.items():
            torch.cuda.synchronize(); tick=time.perf_counter(); fn(); torch.cuda.synchronize(); cold[name]=(time.perf_counter()-tick)*1000
        record['first_call_ms']=cold
        record['gate']=bounded(lambda:gate(a,ell,oracle,support),45)
        cross=oracle.compare(cpu,[run(cpu,ell,FD),run(a,ell,FD).cpu()],ell)
        assert cross['passed'],'CPU CUDA math'; record['cpu_cuda_math']=support.compact(cross)
        # Entire context blocks are randomized; anchors bracket ALL intervening work.
        seed=SEED+100+index*10
        record['AA_before']=paired_block(dict(baseline=fns['baseline'],candidate=fns['baseline']),seed,timing)
        jobs=['cuda_candidate','cpu_cuda_context']; random.Random(seed).shuffle(jobs)
        record['context_order']=jobs; record['comparisons']={}
        for offset,name in enumerate(jobs):
            funcs=fns if name=='cuda_candidate' else dict(baseline=lambda:run(cpu,ell,FD),candidate=fns['baseline'])
            record['comparisons'][name]=paired_block(funcs,seed+1+offset,timing)
        record['AA_after']=paired_block(dict(baseline=fns['baseline'],candidate=fns['baseline']),seed+3,timing)
        before=record['AA_before']['timing']['baseline']['median_ms']; after=record['AA_after']['timing']['baseline']['median_ms']
        drift=after/before-1
        placebo=[]
        for tag in ('AA_before','AA_after'):
            lo,hi=record[tag]['timing']['paired_median_bootstrap_95']; placebo.append(lo>1.05 or hi<1/1.05)
        # Compare CUDA baseline inside CPU context to CUDA-only paired baseline.
        context=record['comparisons']['cpu_cuda_context']['timing']['candidate']['median_ms']/record['comparisons']['cuda_candidate']['timing']['baseline']['median_ms']-1
        pair=record['comparisons']['cuda_candidate']['timing']
        stable=abs(drift)<=.05 and abs(context)<=.05 and not any(placebo)
        initial=report['initial_AA']['timing']['paired_median_bootstrap_95']
        stable=stable and not (initial[0]>1.05 or initial[1]<1/1.05)
        record.update(status='PASS',baseline_drift_fraction=drift,cpu_context_cuda_drift_fraction=context,placebo_material=placebo,performance='INCONCLUSIVE' if not stable else 'SCREEN_PASS' if pair['median_paired_ratio']>1.05 and pair['paired_median_bootstrap_95'][0]>1 else 'SCREEN_NOT_MET',device_after=metrics())
        if d in (64,512) and final>ell and ell==16:
            record['profiles']={name:profiler.kernel_profile(fn,'cuda',directory,'full-'+name) for name,fn in fns.items()}
            record['factor_pair']=paired_block({name:lambda cls=cls:cls(ell,d,dtype=dtype,device='cuda')._factor(a[:2*ell],shrink=True) for name,cls in (('baseline',FD),('candidate',Candidate))},seed+4,timing)
    report['final_AA']=paired_block(dict(baseline=control_fn,candidate=control_fn),SEED+999,timing)
    return report


def main():
    global torch,FD,Candidate
    sha=os.environ.get('SAE_CANDIDATE_SHA','')
    report=dict(source_sha=sha,status='FAIL',seed=SEED,utc_start=datetime.now(timezone.utc).isoformat(),source_files=SOURCE_FILES,
        policy='strict E0/E1 identical update/sketch; graph E2 tensor transition diagnostic NON_EQUIVALENT; no production changes',
        timing_scope='synchronized host wall; fresh state; resident inputs; excludes generation and H2D; first calls separate; 20 balanced pairs and 2 warmups per block; no ratio multiplication')
    with tempfile.TemporaryDirectory(prefix='fd17-') as temporary:
        directory=Path(temporary)
        try:
            bootstrap(sha,directory/'source')
            import torch
            from torchfrequentdirections import FrequentDirections as FD
            from experiments.full_gpu.candidate import ScratchFD as Candidate
            from experiments.qr_svd import driver as oracle,cuda_driver as support
            from experiments.zero_clear import driver as timing
            from experiments.compile_fd import driver as profiler
            support.torch=timing.torch=profiler.torch=torch
            torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32=False; torch.set_float32_matmul_precision('highest')
            report['environment']=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,cpu_threads=1,interop_threads=1,tf32=False,cpu_affinity=sorted(os.sched_getaffinity(0)),torch_build=torch.__config__.show())
            report['cpu_or_gpu_fixture_gates']=[]
            device='cpu' if os.environ.get('SAE_CPU_ONLY')=='1' else 'cuda'
            if device=='cuda':
                report['probe']=support.probe(); report['device_initial']=metrics()
            for dtype in (torch.float32,torch.float64):
                for name,a in oracle.fixtures(dtype).items():
                    for ell in (1,4,a.shape[1]+2):
                        report['cpu_or_gpu_fixture_gates'].append(dict(name=name,dtype=str(dtype),ell=ell,**gate(a.to(device),ell,oracle,support)))
            if device=='cuda':
                report['graphs']=[]
                for dtype in (torch.float32,torch.float64):
                    a=torch.randn(32,64,dtype=dtype,device='cuda')
                    report['graphs'].append(dict(dtype=str(dtype),**bounded(lambda:graph_probe(a,16),30)))
                report['gpu']=study(oracle,support,timing,profiler,directory)
            report['status']='PASS'
        except Exception as exc: report['error']=error(exc)
    report.update(elapsed_seconds=time.monotonic()-START,utc_end=datetime.now(timezone.utc).isoformat())
    out=Path(os.environ.get('SAE_ARTIFACT_DIR','/content')); out.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(report,separators=(',',':'),allow_nan=False)+'\n').encode()
    assert len(raw)<=1048576,'artifact bound'
    (out/'fd-full-gpu.json').write_bytes(raw)
    print('ARTIFACT fd-full-gpu.json',len(raw),hashlib.sha256(raw).hexdigest(),flush=True)


if __name__=='__main__': main()
