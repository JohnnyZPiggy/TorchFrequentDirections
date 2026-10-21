"""Fresh CUDA process, independent of every graph-probe runtime/context."""
import json
from pathlib import Path
import sys
import time


def main(path,device='cuda'):
    import torch
    from torchfrequentdirections import FrequentDirections
    from experiments.full_gpu.candidate import ScratchFD
    from experiments.full_gpu import driver as base
    from experiments.qr_svd import driver as oracle,cuda_driver as support
    from experiments.zero_clear import driver as timing
    from experiments.compile_fd import driver as profiler
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False; torch.set_float32_matmul_precision('highest')
    torch.manual_seed(base.SEED)
    base.torch=support.torch=timing.torch=profiler.torch=torch
    base.FD=FrequentDirections; base.Candidate=ScratchFD
    result=dict(status='FAIL',fixture_gates=[],environment=dict(python=sys.version.split()[0],torch=str(torch.__version__),cuda=torch.version.cuda,
        cpu_threads=1,interop_threads=1,tf32=False),process_isolation='fresh exec; no graph capture calls in this process')
    tick=time.monotonic()
    def checkpoint(gpu):
        result['gpu']=gpu
        path.write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
    try:
        if device=='cuda':
            health=torch.randn(8,device='cuda'); assert bool(torch.isfinite(health).all()); torch.cuda.synchronize()
            result['post_probe_fresh_process_cuda_rng']='PASS'
            result['probe']=support.probe(); result['cold_cuda_probe_seconds']=time.monotonic()-tick
        for dtype in (torch.float32,torch.float64):
            for name,a in oracle.fixtures(dtype).items():
                for ell in (1,4,a.shape[1]+2):
                    result['fixture_gates'].append(dict(name=name,dtype=str(dtype),ell=ell,**base.gate(a.to(device),ell,oracle,support)))
        if device=='cuda': result['gpu']=base.study(oracle,support,timing,profiler,path.parent,deadline_seconds=245,checkpoint=checkpoint)
        result['status']='PASS'
    except Exception as exc: result['error']=base.error(exc)
    result['elapsed_seconds']=time.monotonic()-tick
    path.write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')


if __name__=='__main__': main(Path(sys.argv[1]),sys.argv[2] if len(sys.argv)>2 else 'cuda')
