"""Disposable process: exactly one stage/dtype; never reused after capture failure."""
import json
from pathlib import Path
import sys
import time


STAGES=('pointwise','svd','svd_shrink','buffer')


def shrink(s,vh,ell):
    scale=s[0].clamp_min(torch.finfo(s.dtype).tiny)
    z=s/scale
    values=scale*(z.square()-z[ell-1].square()).clamp_min(0).sqrt()
    return values[:ell,None]*vh[:ell]


def operation(stage,rows,work,ell):
    if stage=='pointwise':
        work['pointwise'].copy_(rows.square().clamp_min(0).sqrt())
        return work['pointwise']
    torch.linalg.svd(rows,full_matrices=False,out=(work['u'],work['s'],work['vh']))
    if stage=='svd': return work['s']
    work['factor'].copy_(shrink(work['s'],work['vh'],ell))
    if stage=='svd_shrink': return work['factor']
    work['buffer'].zero_()
    work['buffer'][:ell-1].copy_(work['factor'][:ell-1])
    return work['buffer']


def main(stage,dtype_name,path,device='cuda'):
    global torch
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    dtype=getattr(torch,dtype_name); ell=16; m,d=32,64
    result=dict(stage=stage,dtype=str(dtype),device=device,status='FAIL',phase='initialization',warmups=2,
        scope='PROBE_ONLY; tensor stages exclude mandatory host finite/error decisions; NOT full strict update graph',
        svd='preallocated out U/S/Vh; full_matrices=False',recaptures=0)
    def save(): path.write_text(json.dumps(result,separators=(',',':'))+'\n')
    def phase(name): result['phase']=name; save()
    def err(exc): return dict(type=type(exc).__name__,message=str(exc)[:1800])
    save()
    try:
        rows=torch.randn(m,d,dtype=dtype,generator=torch.Generator().manual_seed(1720261021)).to(device)
        work=dict(pointwise=torch.empty_like(rows),u=rows.new_empty((m,m)),s=rows.new_empty(m),
            vh=rows.new_empty((m,d)),factor=rows.new_empty((ell,d)),buffer=torch.empty_like(rows))
        addresses={key:value.data_ptr() for key,value in work.items()}; input_address=rows.data_ptr()
        if device=='cpu':
            phase('cpu_eager_operations'); output=operation(stage,rows,work,ell)
            assert bool(torch.isfinite(output).all())
            result.update(status='CPU_OPERATION_PASS',shape=list(output.shape)); save(); return
        phase('side_stream_warmup'); stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2): operation(stage,rows,work,ell)
        torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
        phase('capture'); graph=torch.cuda.CUDAGraph(); tick=time.monotonic()
        # Save the FIRST operation error before context exit can replace it.
        try:
            with torch.cuda.graph(graph,stream=stream):
                try: output=operation(stage,rows,work,ell)
                except Exception as exc:
                    result['operation_error']=err(exc); save(); raise
        except Exception as exc:
            result.update(status='CAPTURE_FAILED_PROBE_ONLY',capture_error=err(exc),capture_seconds=time.monotonic()-tick)
            save(); return  # No CUDA call, random generation or recovery in this process.
        torch.cuda.synchronize(); result['capture_seconds']=time.monotonic()-tick
        phase('replay'); samples=[]; comparisons=[]
        for index in range(5):
            fresh=torch.randn(m,d,dtype=dtype,generator=torch.Generator().manual_seed(1720261022+index)).to(device)
            rows.copy_(fresh); torch.cuda.synchronize(); tick=time.perf_counter()
            graph.replay(); torch.cuda.synchronize(); samples.append((time.perf_counter()-tick)*1000)
            actual=output.clone()
            if stage=='pointwise': expected=fresh.square().clamp_min(0).sqrt()
            else:
                from torchfrequentdirections import FrequentDirections
                _,s,vh=torch.linalg.svd(fresh,full_matrices=False)
                if stage=='svd': expected=s
                elif stage=='svd_shrink': expected=FrequentDirections(ell,d,dtype=dtype,device=device)._factor(fresh,shrink=True)
                else: expected=FrequentDirections(ell,d,dtype=dtype,device=device).update(fresh)._buffer
            torch.testing.assert_close(actual,expected,rtol=64*torch.finfo(dtype).eps,atol=64*torch.finfo(dtype).eps)
            comparisons.append(float((actual-expected).abs().max()))
        assert input_address==rows.data_ptr() and addresses=={k:v.data_ptr() for k,v in work.items()}
        result.update(status='CAPTURE_REPLAY_PASS_PROBE_ONLY',replay_ms=samples,max_abs_errors=comparisons,
            stable_preallocated_addresses=True,phase='complete')
    except Exception as exc: result.update(error=err(exc),status='PROBE_FAILED')
    save()


if __name__=='__main__': main(sys.argv[1],sys.argv[2],Path(sys.argv[3]),sys.argv[4] if len(sys.argv)>4 else 'cuda')
