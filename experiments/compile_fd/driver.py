"""Issue #15 pinned CPU/one-T4 compile study; no allocation in this module."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','1')
SOURCE_FILES = {'LICENSE': '2f0819452dd5f7667e2c0e93b769238a41c8164e62f89f212646076f66765bde', 'torchfrequentdirections/__init__.py': 'e65e1bcff754fe3cc2b5e3fb383f7b6293a7a3a358757e8cc0861856e54306c7', 'torchfrequentdirections/fd.py': '18e84afa295d3bb75cb97f68bd98fa339fc3ee061e26e8071051021066d30e85', 'experiments/qr_svd/driver.py': '65845b029b415e9214947433a76523380ae78f8e90006c11e304e7d9336480a4', 'experiments/qr_svd/cuda_driver.py': 'd8f919ac1d92a3f0ba17a577f84b4f98ff9b8361921b3ebf463fc66f7ef7b9b6', 'experiments/zero_clear/driver.py': '0ffa14cbf6a56ea4fadd10f252a5bb6a56fa8ae70bf8d794e1c91d41958ce18e', 'experiments/compile_fd/shrink.py': '53f5f97bc1013d5865a10de7fcb7b1b6832aa7a41fa6951e4144d1e1689501a2', 'experiments/qr_svd/prototype.py': '58c10f6297588c79f7ddb00f58e9f0b607c8b690405c2f8e1f49bc778ae1ddb2'}
SEED=1520261021
START=time.monotonic()
CPU_ONLY=os.environ.get('SAE_CPU_ONLY')=='1'


def bounded(fn,seconds=45):
    if not CPU_ONLY:
        seconds=min(seconds,350-(time.monotonic()-START))
        if seconds<=0: raise TimeoutError('scientific deadline; cleanup protected')
    def expired(*_): raise TimeoutError('bounded compile/probe timeout')
    previous=signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try: return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)


def bootstrap(sha,destination,local):
    if not re.fullmatch('[0-9a-f]{40}',sha) or not SOURCE_FILES: raise RuntimeError('SOURCE_ID_INVALID')
    if local and subprocess.check_output(['git','-C',local,'rev-parse','HEAD'],text=True).strip()!=sha:
        raise RuntimeError('SOURCE_SHA_MISMATCH')
    for name,expected in SOURCE_FILES.items():
        if local: raw=subprocess.check_output(['git','-C',local,'show',sha+':'+name],timeout=10)
        else:
            with urllib.request.urlopen('https://raw.githubusercontent.com/JohnnyZPiggy/TorchFrequentDirections/'+sha+'/'+name,timeout=20) as response:
                raw=response.read(1024*1024+1)
        if hashlib.sha256(raw).hexdigest()!=expected: raise RuntimeError('SOURCE_HASH_MISMATCH')
        p=destination/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(raw)
    sys.path.insert(0,str(destination))
    return dict(status='PASS',files=SOURCE_FILES,mode='hash-checked exact Git modules; no dependencies installed')


def error(exc):
    # No tracebacks/compiled caches or private runtime paths are published.
    return dict(type=type(exc).__name__,message=str(exc).splitlines()[0][:220])


def explain(fn,*args):
    try:
        report=torch._dynamo.explain(fn)(*args)
        return dict(status='PASS',graph_count=report.graph_count,graph_break_count=report.graph_break_count,
            reasons=[str(r.reason)[:450] for r in report.break_reasons],
            ops=[[str(op) for op in group] for group in report.ops_per_graph],
            guard_count=len(report.out_guards))
    except Exception as exc: return dict(status='FAIL',**error(exc))


def compilation(fn,backend='inductor'):
    from torch._dynamo.backends.registry import lookup_backend
    actual=lookup_backend(backend); graphs=[]
    def counted(gm,inputs):
        graphs.append(dict(ops=[str(n.target) for n in gm.graph.nodes if n.op.startswith('call_')],
                           tensor_inputs=[dict(shape=[str(x) for x in t.shape],dtype=str(t.dtype),device=str(t.device))
                                          for t in inputs if isinstance(t,torch.Tensor)]))
        return actual(gm,inputs)
    result=torch.compile(fn,backend=counted,fullgraph=True,dynamic=True,mode='default')
    return result,graphs


def run(a,ell,cls,fn=None):
    kwargs={} if fn is None else dict(shrink_fn=fn)
    state=cls(ell,a.shape[1],dtype=a.dtype,device=a.device,**kwargs)
    for part in a.split(max(1,ell//2)):
        state.update(part)
    return state.sketch()


def math_gate(a,ell,cls,fn,oracle):
    eager=run(a,ell,cls)
    state=cls(ell,a.shape[1],dtype=a.dtype,device=a.device,shrink_fn=fn)
    for part in a.split(max(1,ell//2)):
        state.update(part); state.update(part[:0])
    before=state._buffer.clone(); count=state._count
    output=state.sketch(); other=state.sketch(); other.fill_(123)
    if not torch.equal(before,state._buffer) or count!=state._count or output.requires_grad:
        raise AssertionError('snapshot state/no-grad')
    if not torch.equal(output,state.sketch()): raise AssertionError('snapshot alias')
    report=oracle.compare(a.cpu(),[eager.cpu(),output.cpu()],ell)
    if not report['passed']: raise AssertionError('independent compiled FD math failed')
    invalid=torch.full((1,a.shape[1]),float('inf'),device=a.device,dtype=a.dtype)
    try: state.update(invalid)
    except ValueError: pass
    else: raise AssertionError('finite check lost')
    if not torch.equal(before,state._buffer): raise AssertionError('invalid input mutation')
    return support.compact(report)


def cpu_diagnosis(module,oracle):
    a=torch.randn(8,17,dtype=torch.float64,generator=torch.Generator().manual_seed(SEED))
    _,s,v=torch.linalg.svd(a,full_matrices=False)
    report=dict(compiler_paths={n:shutil.which(n) for n in ('g++','gcc','clang++','clang')},
        pure_explain=explain(module.tensor_shrink,s,v,4,True),
        guarded_factor_explain=explain(module.guarded_factor,a,4),
        whole_stream_explain=explain(module.whole_stream,a,4),settings=dict(fullgraph=True,dynamic=True,mode='default'))
    report['fullgraph']={}
    for name,fn,args in [('guarded_factor',module.guarded_factor,(a,4)),('whole_stream',module.whole_stream,(a,4))]:
        try:
            bounded(lambda:torch.compile(fn,fullgraph=True,backend='aot_eager',dynamic=True)(*args))
            report['fullgraph'][name]=dict(status='PASS')
        except Exception as exc: report['fullgraph'][name]=dict(status='FAIL_EXPECTED_DIAGNOSTIC',**error(exc))
    for backend in ('aot_eager','inductor'):
        torch._dynamo.reset()
        compiled,graphs=compilation(module.tensor_shrink,backend)
        tick=time.monotonic(); entry=dict(backend=backend,graphs=graphs)
        try:
            actual=bounded(lambda:compiled(s,v,4,True),60)
            torch.testing.assert_close(actual,module.tensor_shrink(s,v,4,True))
            entry['cold_seconds']=time.monotonic()-tick
            entry['checks']=[]
            for dtype in (torch.float32,torch.float64):
                for d in (9,17):
                    data=torch.randn(23,d,dtype=dtype,generator=torch.Generator().manual_seed(SEED))
                    entry['checks'].append(math_gate(data,4,module.TensorShrinkFD,compiled,oracle))
            entry.update(status='PASS',compile_graph_count=len(graphs))
        except Exception as exc: entry.update(status='UNAVAILABLE_OR_FAIL',elapsed_seconds=time.monotonic()-tick,**error(exc))
        report[backend]=entry
    # Partial compilation is explicit; unsupported full FD is never silently retried as eager.
    report['full_stream_compiled']='NOT_AVAILABLE (finite Python bool/state entrypoints)' if any(
        x['status']!='PASS' for x in report['fullgraph'].values()) else 'CAPTURED_AOT_ONLY_NOT_INDUCTOR_VALIDATED'
    if report['aot_eager']['status']!='PASS': raise AssertionError('CPU captured math gate failed')
    return report


def kernel_profile(fn,device,directory,name):
    try:
        from torch.profiler import profile,ProfilerActivity
        if device=='cuda': torch.cuda.synchronize()
        activities=[ProfilerActivity.CPU]+([ProfilerActivity.CUDA] if device=='cuda' else [])
        with profile(activities=activities,record_shapes=False) as prof:
            fn()
            if device=='cuda': torch.cuda.synchronize()
        path=directory/(name+'.json'); prof.export_chrome_trace(str(path))
        trace=json.loads(path.read_text())['traceEvents']; path.unlink()
        kernels=[e for e in trace if e.get('cat')=='kernel' and e.get('ph')=='X']
        names={}
        for e in kernels:
            key=e['name']; names[key]=names.get(key,0)+1
        waits=[dict(name=e['name'],duration_us=e.get('dur')) for e in trace if e.get('cat')=='cuda_runtime'
               and any(t in e.get('name','') for t in ('Synchronize','GraphLaunch','LaunchKernel'))]
        return dict(status='PASS',kernel_launches=len(kernels),kernel_names=names,
            cuda_runtime_diagnostics=waits[:120],
            cpu_ops=[dict(name=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total) for e in prof.key_averages()
                     if any(t in e.key for t in ('svd','isfinite','item','Compiled','Dynamo'))],
            scope='intrusive one-call profiler; actual trace kernel category; nonadditive')
    except Exception as exc: return dict(status='NOT_AVAILABLE',**error(exc))


def events_and_memory(fn):
    values=[]; memory=[]
    for _ in range(20):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start=torch.cuda.Event(enable_timing=True); stop=torch.cuda.Event(enable_timing=True)
        start.record(); fn(); stop.record(); torch.cuda.synchronize()
        values.append(start.elapsed_time(stop)); memory.append(torch.cuda.max_memory_allocated())
    return dict(cuda_event_ms=values,peak_allocated_bytes=memory,p95_memory_bytes=sorted(memory)[18],
                scope='CUDA event timeline includes dispatch gaps; not pure kernel-only time')


def gpu_study(module,oracle,timing,directory):
    report=dict(full_stream_compiled='NOT_AVAILABLE; only pure tensor shrink is Inductor compiled',
                settings=dict(backend='counted_inductor',mode='default',dynamic=True,fullgraph=True),records=[])
    torch._dynamo.reset(); compiled,graphs=compilation(module.tensor_shrink)
    report['graphs']=graphs
    # Probe both dtypes before any timing; true CUDA Inductor and generated output.
    for dtype in (torch.float32,torch.float64):
        a=torch.randn(32,128,dtype=dtype,generator=torch.Generator().manual_seed(SEED)).to('cuda')
        _,s,v=torch.linalg.svd(a,full_matrices=False)
        tick=time.monotonic(); out=bounded(lambda:compiled(s,v,16,True),60); torch.cuda.synchronize()
        cold=time.monotonic()-tick
        expected=module.tensor_shrink(s,v,16,True)
        torch.testing.assert_close(out,expected,rtol=64*torch.finfo(dtype).eps,atol=64*torch.finfo(dtype).eps)
        report.setdefault('compile_probes',[]).append(dict(dtype=str(dtype),status='PASS',cold_seconds=cold,
                                                         graph_count=len(graphs)))
    specs=[]
    # All requested configurations are named, with optional large ones last.
    for ell,d in ((16,64),(16,128),(16,512),(8,4),(2,8),(32,1024),(64,2048)):
        for dtype in (torch.float32,torch.float64):
            for final in (ell-1,ell+1):
                retained=min(ell-1,d); n=2*ell+3*(2*ell-retained)+final-retained
                a=torch.randn(n,d,dtype=dtype,generator=torch.Generator().manual_seed(SEED+n+d))
                specs.append((ell,d,final,a,'stream'))
    for dtype in (torch.float32,torch.float64):
        specs.append((16,128,15,torch.randn(15,128,dtype=dtype,generator=torch.Generator().manual_seed(SEED)),'no_factor'))
    stopped=False
    for index,(ell,d,final,cpu,kind) in enumerate(specs):
        record=dict(ell=ell,d=d,n=len(cpu),dtype=str(cpu.dtype),final_count=final,kind=kind,
                    compile_scope='partial shrink only; SVD/validation/state eager')
        report['records'].append(record)
        if stopped or time.monotonic()-START>285:
            record.update(status='NOT_RUN',reason='compile/time budget; no silent eager fallback'); continue
        a=cpu.to('cuda'); before=len(graphs); tick=time.monotonic()
        try:
            record['cold_subgraph_calls']=[]
            if kind!='no_factor':
                for m in sorted({min(2*ell,len(a))} | ({final} if final>ell else set())):
                    _,s,v=torch.linalg.svd(a[:m],full_matrices=False); torch.cuda.synchronize()
                    count=len(graphs); cold_tick=time.monotonic()
                    bounded(lambda:compiled(s,v,ell,True)); torch.cuda.synchronize()
                    record['cold_subgraph_calls'].append(dict(m=m,seconds=time.monotonic()-cold_tick,new_graphs=len(graphs)-count))
            if time.monotonic()-START>300:
                record.update(status='NOT_RUN',reason='cold compile exhausted timing reserve'); stopped=True; continue
            record['math']=bounded(lambda:math_gate(a,ell,module.TensorShrinkFD,compiled,oracle),45)
            cpu_output=run(cpu,ell,module.TensorShrinkFD)
            eager_gpu=run(a,ell,module.TensorShrinkFD).cpu()
            cross=oracle.compare(cpu,[cpu_output,eager_gpu],ell)
            if not cross['passed']: raise AssertionError('CPU/CUDA oracle gap')
            record['cpu_cuda_math']=support.compact(cross)
        except AssertionError: raise
        except Exception as exc:
            record.update(status='COMPILE_NOT_AVAILABLE',**error(exc)); stopped=True; continue
        record.update(cold_or_guard_seconds=time.monotonic()-tick,new_graphs=len(graphs)-before)
        if time.monotonic()-START>300:
            record.update(status='NOT_RUN',reason='math gate consumed timing reserve'); stopped=True; continue
        functions=[lambda:run(cpu,ell,module.TensorShrinkFD),lambda:run(a,ell,module.TensorShrinkFD),
                   lambda:run(a,ell,module.TensorShrinkFD,compiled)]
        record['comparisons']={}
        for offset,(name,left,right) in enumerate([('cpu_eager/cuda_eager',0,1),
                ('cuda_eager/cuda_partial',1,2),('cpu_eager/cuda_partial',0,2)]):
            seed=SEED+index*10+offset
            record['comparisons'][name]=dict(seed=seed,timing=timing.paired(
                {'baseline':functions[left],'candidate':functions[right]},'cuda',seed=seed))
        record['steady_new_graphs']=len(graphs)-before-record['new_graphs']
        record['cuda_eager_events_memory']=events_and_memory(functions[1])
        record['cuda_partial_events_memory']=events_and_memory(functions[2])
        record['status']='PASS'
        if kind=='no_factor' or d==512 and final>ell:
            seed=SEED+index*10+7; record['AA']=dict(seed=seed,timing=timing.paired(
                {'baseline':functions[1],'candidate':functions[1]},'cuda',seed=seed))
        if final>ell and d in (128,512) and ell==16:
            record['full_profiles']={name:kernel_profile(fn,'cuda',directory,'full-'+name)
                                     for name,fn in zip(('eager','partial'),functions[1:])}
            _,s,v=torch.linalg.svd(a[:2*ell],full_matrices=False)
            pure=[lambda:module.tensor_shrink(s,v,ell,True),lambda:compiled(s,v,ell,True)]
            seed=SEED+index*10+8
            record['shrink_pair']=dict(seed=seed,timing=timing.paired(dict(zip(('baseline','candidate'),pure)),'cuda',seed=seed))
            record['shrink_profiles']={name:kernel_profile(fn,'cuda',directory,'pure-'+name)
                                       for name,fn in zip(('eager','partial'),pure)}
            factors=[module.TensorShrinkFD(ell,d,dtype=a.dtype,device=a.device,shrink_fn=fn)
                     for fn in (module.tensor_shrink,compiled)]
            seed=SEED+index*10+9
            record['factor_pair']=dict(seed=seed,timing=timing.paired({name:lambda state=state:state._factor(a[:2*ell],shrink=True)
                for name,state in zip(('baseline','candidate'),factors)},'cuda',seed=seed))
        record['new_graphs_after_all_measurements']=len(graphs)-before-record['new_graphs']
    from torch._inductor import metrics
    report['inductor_metrics']={key:getattr(metrics,key,None) for key in
        ('generated_kernel_count','generated_cpp_vec_kernel_count','ir_nodes_pre_fusion')}
    report['compile_graph_count']=len(graphs)
    return report


def main():
    global torch,support
    sha=os.environ.get('SAE_CANDIDATE_SHA',''); cpu_only=os.environ.get('SAE_CPU_ONLY')=='1'
    report=dict(source_sha=sha,status='FAIL',seed=SEED,utc_start=datetime.now(timezone.utc).isoformat(),
        scope='Direct algorithm; explicit partial tensor-shrink compile only; production unchanged',
        screen='math PASS; stable placebo; paired CUDA eager/partial median>1.05 and lower>1; exploratory only')
    with tempfile.TemporaryDirectory(prefix='fd15-') as temporary:
        directory=Path(temporary)
        os.environ['TORCHINDUCTOR_CACHE_DIR']=str(directory/'inductor')
        os.environ['TRITON_CACHE_DIR']=str(directory/'triton')
        try:
            report['install']=bootstrap(sha,directory/'source',os.environ.get('SAE_LOCAL_SOURCE'))
            import torch
            from experiments.compile_fd import shrink as module
            from experiments.qr_svd import driver as oracle
            from experiments.qr_svd import cuda_driver as support
            from experiments.zero_clear import driver as timing
            support.torch=timing.torch=torch
            torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_grad_enabled(False)
            torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
            torch.set_float32_matmul_precision('highest')
            report['environment']=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
                cpu_threads=1,interop_threads=1,tf32=False,compiler_threads=1,cold_cache='new isolated temp directories')
            report['environment'].update(cpu_model=next((line.split(':',1)[1].strip() for line in
                Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'UNKNOWN'),
                cpu_affinity=sorted(os.sched_getaffinity(0)),torch_build=torch.__config__.show(),
                thread_environment={key:os.environ.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
            if cpu_only:
                report['cpu']=cpu_diagnosis(module,oracle)
                report['status']='PASS'
            else:
                report['probe']=support.probe()
                report['gpu']=gpu_study(module,oracle,timing,directory)
                report['status']='PASS'
        except Exception as exc: report['error']=error(exc)
    report.update(utc_end=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-START)
    out=Path(os.environ.get('SAE_ARTIFACT_DIR','/content')); out.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(report,separators=(',',':'),allow_nan=False)+'\n').encode()
    (out/'fd-compile.json').write_bytes(raw)
    print('ARTIFACT fd-compile.json',len(raw),hashlib.sha256(raw).hexdigest(),flush=True)
    return report


if __name__=='__main__': main()
