"""Issue #17 follow-up: no CUDA imports/context in this orchestration process."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

SOURCE_FILES = {'LICENSE': '2f0819452dd5f7667e2c0e93b769238a41c8164e62f89f212646076f66765bde', 'torchfrequentdirections/__init__.py': 'e65e1bcff754fe3cc2b5e3fb383f7b6293a7a3a358757e8cc0861856e54306c7', 'torchfrequentdirections/fd.py': '18e84afa295d3bb75cb97f68bd98fa339fc3ee061e26e8071051021066d30e85', 'experiments/qr_svd/driver.py': '65845b029b415e9214947433a76523380ae78f8e90006c11e304e7d9336480a4', 'experiments/qr_svd/cuda_driver.py': 'd8f919ac1d92a3f0ba17a577f84b4f98ff9b8361921b3ebf463fc66f7ef7b9b6', 'experiments/qr_svd/prototype.py': '58c10f6297588c79f7ddb00f58e9f0b607c8b690405c2f8e1f49bc778ae1ddb2', 'experiments/zero_clear/driver.py': '0ffa14cbf6a56ea4fadd10f252a5bb6a56fa8ae70bf8d794e1c91d41958ce18e', 'experiments/compile_fd/driver.py': '4bb2108c0f2150770643d1f7d9f943d570388eca7fe014124a39b0bb74a178c0', 'experiments/full_gpu/candidate.py': 'f37ad64cfb11a7bc1071566e7d196cd9a1756bbbdd76a20a1ece4a9e986c8a66', 'experiments/full_gpu/driver.py': 'c1c0e7aa33f02cf3f0093fac8134fa9115b1db72e8b8876f51d2213972820191', 'experiments/full_gpu/graph_worker.py': '5c3cd913e2efc311c41cb87639e2001d2e4e4edc9a74b23772d6244517583ab5', 'experiments/full_gpu/benchmark_worker.py': '76662f9c156ddd7c625a03738c583bd25996d0cde28ddb98e61fe5f90e60f5bf'}


def invoke(command,path,seconds):
    """A failed/timed-out child never prevents invoking the next experiment."""
    tick=time.monotonic()
    try:
        child=subprocess.run(command,capture_output=True,text=True,timeout=seconds)
        transport=dict(exit_code=child.returncode,timeout=False)
    except subprocess.TimeoutExpired:
        transport=dict(exit_code=None,timeout=True)
    transport['wall_seconds']=time.monotonic()-tick
    try: result=json.loads(path.read_text()) if path.exists() else dict(status='NO_RESULT')
    except (OSError,ValueError): result=dict(status='INVALID_CHILD_RESULT')
    if not isinstance(result,dict): result=dict(status='INVALID_CHILD_RESULT')
    if transport['timeout']: result['status']='CHILD_TIMEOUT'
    elif transport['exit_code']!=0: result['status']='CHILD_EXIT_FAILED'
    return dict(transport=transport,result=result)


def main():
    # Never execute the historical driver's main, which used shared CUDA state.
    sha=os.environ['SAE_CANDIDATE_SHA']; cpu=os.environ.get('SAE_CPU_ONLY')=='1'
    start=time.monotonic(); report=dict(source_sha=sha,status='FAIL',source_files=SOURCE_FILES,
        utc_start=datetime.now(timezone.utc).isoformat(),mode='CPU' if cpu else 'CUDA',graphs=[])
    with tempfile.TemporaryDirectory(prefix='fd17-isolated-') as temporary:
        directory=Path(temporary)
        # Bootstrap is inlined to avoid requiring repo modules before hash checking.
        import urllib.request
        local=os.environ.get('SAE_LOCAL_SOURCE')
        if len(sha)!=40 or any(c not in '0123456789abcdef' for c in sha): raise ValueError('invalid source SHA')
        if local: assert subprocess.check_output(['git','-C',local,'rev-parse','HEAD'],text=True).strip()==sha
        for name,digest in SOURCE_FILES.items():
            if local: raw=subprocess.check_output(['git','-C',local,'show',sha+':'+name],timeout=10)
            else:
                with urllib.request.urlopen('https://raw.githubusercontent.com/JohnnyZPiggy/TorchFrequentDirections/'+sha+'/'+name,timeout=15) as r: raw=r.read(1048577)
            assert hashlib.sha256(raw).hexdigest()==digest,'source mismatch'
            p=directory/'source'/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(raw)
        worker=directory/'source/experiments/full_gpu/graph_worker.py'
        env=dict(os.environ,PYTHONPATH=str(directory/'source'),PYTHONDONTWRITEBYTECODE='1')
        # Sequential fresh execs, never fork an initialized CUDA process.
        previous=os.environ.copy(); os.environ.update(env)
        try:
            for stage in ('pointwise','svd','svd_shrink','buffer'):
                for dtype in ('float32','float64'):
                    path=directory/(stage+'-'+dtype+'.json')
                    if not cpu and time.monotonic()-start>90:
                        report['graphs'].append(dict(stage=stage,dtype=dtype,status='NOT_RUN_TIME_RESERVE')); continue
                    report['graphs'].append(dict(stage=stage,dtype=dtype,**invoke(
                        [sys.executable,str(worker),stage,dtype,str(path),'cpu' if cpu else 'cuda'],path,10 if not cpu else 15)))
            # A ninth clean process owns ALL strict validation and benchmark state.
            path=directory/'benchmark.json'; seconds=300 if not cpu else 60
            report['benchmark']=invoke([sys.executable,str(directory/'source/experiments/full_gpu/benchmark_worker.py'),str(path),'cpu' if cpu else 'cuda'],path,seconds)
            report['status']='PASS' if report['benchmark']['result']['status']=='PASS' else 'FAIL'
        finally:
            os.environ.clear(); os.environ.update(previous)
    report.update(elapsed_seconds=time.monotonic()-start,utc_end=datetime.now(timezone.utc).isoformat())
    out=Path(os.environ.get('SAE_ARTIFACT_DIR','/content')); out.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(report,separators=(',',':'),allow_nan=False)+'\n').encode(); assert len(raw)<=1048576
    (out/'fd-isolated.json').write_bytes(raw)
    print('ARTIFACT fd-isolated.json',len(raw),hashlib.sha256(raw).hexdigest(),flush=True)


if __name__=='__main__': main()
