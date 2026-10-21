"""Deterministic documentation figures/tables from immutable Git evidence; stdlib only."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[2]
BASE='f97070dddc1cb1788d4b1fab85c58da650a9f552'
HASHES={}
FOLDERS={3:'gpu/issue3-20261020',5:'gpu/issue5-20261020',7:'gpu/issue7-20261020',
    9:'cpu/issue9-20261021',11:'cpu/issue11-20261021',13:'gpu/issue13-20261021',
    15:'gpu/issue15-20261021',17:'gpu/issue17-20261021',19:'gpu/issue17-followup-20261021'}


def raw(path):
    data=subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':'+path])
    HASHES[path]=hashlib.sha256(data).hexdigest()
    return data


def report(issue,name): return 'reports/'+FOLDERS[issue]+'/'+name
def data(issue,name): return json.loads(raw(report(issue,name)))
def rows(issue): return list(csv.DictReader(io.StringIO(raw(report(issue,'benchmark.csv')).decode())))
def select(records,**criteria):
    found=[r for r in records if all(str(r.get(k))==str(v) for k,v in criteria.items())]
    assert len(found)==1,(criteria,len(found))
    return found[0]


def start(title,desc,height):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">'+escape(title)+'</title><desc id="desc">'+escape(desc)+'</desc>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#475569"/></marker></defs>',
        '<rect width="100%" height="100%" rx="16" fill="#f8fafc"/>',
        '<g font-family="Arial, sans-serif" fill="#0f172a">']


def text(parts,x,y,value,size=16,color='#0f172a',**attrs):
    extra=' '.join(f'{k.replace("_","-")}="{escape(str(v))}"' for k,v in attrs.items())
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" {extra}>{escape(value)}</text>')


def box(parts,x,y,w,h,title,lines):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#fff" stroke="#94a3b8"/>')
    text(parts,x+15,y+28,title,18)
    for i,line in enumerate(lines): text(parts,x+15,y+55+22*i,line,15,'#334155')


def arrow(parts,path): parts.append(f'<path d="{path}" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>')
def finish(parts): return '\n'.join(parts+['</g></svg>'])+'\n'


def algorithm():
    raw('torchfrequentdirections/fd.py')
    p=start('Buffered Fast Frequent Directions: sequential state and nonmutating snapshots',
        'Rows fill a 2ell buffer. A full buffer is factored, shrunk at sigma ell, cleared and reduced to min(ell-1,d) rows before appending more. Short snapshots copy; long snapshots shrink without changing state.',650)
    text(p,28,40,'Fast FD: one sequential state, two snapshot paths',26)
    text(p,28,69,'Dense finite real rows; ell >= 1; reduced Direct SVD; inference only',16,'#475569')
    box(p,28,112,150,112,'Rows arrive',['Preserve order','Validate chunk'])
    box(p,224,112,205,112,'Fill occupied prefix',['Capacity: 2ell x d','Track m explicitly'])
    box(p,478,112,180,112,'Full: m = 2ell',['C = U diag(s) Vh','Reduced SVD'])
    box(p,704,112,268,112,'Threshold + shrink',['delta = sigma_ell^2','w = sqrt(max(s^2 - delta, 0))'])
    arrow(p,'M178,167 H224'); arrow(p,'M429,167 H478'); arrow(p,'M658,167 H704')
    box(p,478,274,494,100,'Retain q = min(ell - 1, d) directions',['Zero the physical buffer, copy (w * Vh)[:q], then append.'])
    arrow(p,'M839,224 V274'); arrow(p,'M478,324 H328 V224')
    text(p,40,290,'Continue on the SAME state',16,'#0f766e')
    text(p,28,407,'Observe sketch(): independent ell x d output; no state mutation',20)
    box(p,28,432,420,106,'0 <= m <= ell',['Copy C = buffer[:m], then pad zeros.','No SVD; B^T B = C^T C; rank may equal ell.'])
    box(p,478,432,494,106,'ell < m < 2ell',['Factor the occupied prefix and apply the SAME shrink.','Never truncate pending rows to ell without shrink.'])
    text(p,28,574,'If ell > min(m,d), the missing threshold is zero: an exact-covariance extension.',15)
    text(p,28,599,'Code normalizes before squaring to limit overflow. For PCA, decompose B for its right subspace.',15)
    text(p,28,627,'Source: torchfrequentdirections/fd.py at documentation base f97070d; mathematical diagram, not timings.',13,'#475569')
    return finish(p)


def gpu_figure():
    table=rows(13)
    chosen=[select(table,dtype='torch.float32',ell=ell,d=d,kind='long',comparison='direct/QR') for ell,d in ((16,64),(16,128),(16,512),(32,1024),(64,2048))]
    p=start('T4 full update plus sketch: Direct SVD versus experimental QR-SVD',
        'All five long-final-buffer float32 compression cases from PR 14, one device/study/protocol. Points are median paired Direct/QR ratios; whiskers descriptive 95 percent paired bootstrap intervals, not cross-session confidence.',510)
    text(p,28,38,'T4 full update + sketch: local QR gains do not generalize',25)
    text(p,28,66,'PR #14 | 2026-10-21 | float32 | resident inputs | 20 balanced AB/BA pairs',16,'#475569')
    text(p,28,91,'Median paired Direct / QR time; >1 favors QR. Whiskers: descriptive 95% bootstrap.',15)
    x=lambda value:300+(value-.8)/.7*570
    for value in (.8,.9,1.,1.1,1.2,1.3,1.4,1.5):
        at=x(value); p.append(f'<path d="M{at:.2f},115 V390" stroke="'+('#334155' if value==1 else '#cbd5e1')+'" stroke-dasharray="4 4"/>')
        text(p,at-10,415,f'{value:.1f}',14)
    for index,r in enumerate(chosen):
        y=145+index*52; ratio=float(r['median_paired_speedup']); lo=float(r['descriptive_lower']); hi=float(r['descriptive_upper'])
        text(p,28,y+5,f"ell={r['ell']}, d={r['d']}, n={r['n']}",16)
        p.append(f'<g data-source="{report(13,"benchmark.csv")}" data-d="{r["d"]}" data-ratio="{ratio}" data-lower="{lo}" data-upper="{hi}">')
        p.append(f'<path d="M{x(lo):.2f},{y} H{x(hi):.2f} M{x(lo):.2f},{y-6} V{y+6} M{x(hi):.2f},{y-6} V{y+6}" stroke="#0f766e" stroke-width="2"/>')
        p.append(f'<circle cx="{x(ratio):.2f}" cy="{y}" r="6" fill="#0f766e"/>')
        text(p,900,y+5,f'{ratio:.3f}x',16); p.append('</g>')
    text(p,28,453,'Selection: every float32 long-final-buffer compression shape in this study; no factor-only or fallback rows.',14)
    text(p,28,478,'Different n/ell/d across rows. No CPU ratio, snapshot-only gain or cross-study ratio is mixed into this plot.',14,'#475569')
    return finish(p)


def kernel_figure():
    verification=data(15,'verification.json')
    fusion=select(verification['fusion'],ell=16,d=512,dtype='torch.float32')
    ratios=[float(r['paired_ratio']) for r in rows(15) if r['comparison']=='shrink_pair']
    p=start('Pointwise shrink fusion versus full-stream kernel counts',
        'PR16 T4 float32 ell16 d512 profiles: one pure shrink has ten eager kernel launches versus one compiled Triton launch; an independently profiled full stream has 709 versus 664 launches. Counts are not elapsed speedups.',440)
    text(p,28,40,'Fewer pointwise launches do not mean fused full FD',26)
    text(p,28,70,'PR #16 | T4 | float32 | ell=16, d=512 | separate intrusive profiler scopes',16,'#475569')
    for x,title,keys,maximum,scope in ((28,'One pure tensor shrink',('eager_launches','partial_launches'),10,'pure'),(522,'Full update + sketch',('full_eager_launches','full_partial_launches'),800,'full')):
        text(p,x,122,title,21)
        for index,(key,label) in enumerate(zip(keys,('Eager','Compiled shrink'))):
            value=fusion[key]; y=161+index*76
            text(p,x,y,label,15)
            p.append(f'<rect x="{x}" y="{y+12}" width="{value/maximum*345:.3f}" height="26" rx="3" fill="'+('#64748b' if index==0 else '#0f766e')+f'" data-scope="{scope}" data-variant="{index}" data-count="{value}"/>')
            text(p,x+365,y+32,str(value),19)
        text(p,x,310,f'Linear launch-count scale: 0 to {maximum}',14,'#475569')
    text(p,28,352,'10 -> 1 is a launch count, NOT a 10x latency gain. Full streams still call eager Direct SVD.',16)
    text(p,28,380,f'Pure shrink measured ~{min(ratios):.3f}-{max(ratios):.3f}x locally; {verification["full_workload_screen_count"]}/{verification["measured_configs"]} whole-workload performance screens passed.',15)
    text(p,28,413,'Source: reports/gpu/issue15-20261021/verification.json (fusion); no whole-FD fusion claim.',13,'#475569')
    return finish(p)


def measurements():
    lines=['# Selected measured observations (generated)','',
        'Generated by `python docs/tools/build_figures.py` from committed original records.',
        'Selected examples are not pooled estimates or a dispatch rule. Every section fixes its',
        'reference, runtime, protocol, transfer scope and exact **tested** SHA (not merge SHA).',
        'Times are median milliseconds; paired ratio is median of per-pair ratios, which can',
        'differ from ratio of medians. Interval columns are descriptive 95% paired bootstrap',
        'where present; they are not independent-session confidence or multiplicity-adjusted.',
        'All T4 studies use torch 2.11.0+cu130 / CUDA 13.0 / Python 3.13.15.', '']
    def section(issue,title,method):
        manifest=data(issue,'manifest.json'); sha=manifest.get('source_sha',manifest.get('candidate_sha'))
        lines.extend(['## '+title,'',method,'',f'Tested SHA: `{sha}`. [Original summary](../../{report(issue,"summary.md")}) · [CSV](../../{report(issue,"benchmark.csv")}) · [Manifest](../../{report(issue,"manifest.json")}).','',
            '| Device / timed scope | ell / d / n / dtype | Reference / candidate ms | Ratio [interval] | Interpretation |',
            '|---|---|---:|---|---|'])
    def entry(scope,r,left,right,ratio,interval,status):
        dims=' / '.join(str(r[k]).replace('torch.','') for k in ('ell','d','n','dtype'))
        ci='' if interval is None else f' [{float(interval[0]):.3f}, {float(interval[1]):.3f}]'
        lines.append(f'| {scope} | {dims} | {float(left):.4f} / {float(right):.4f} | {float(ratio):.3f}{ci} | {status} |')
    section(3,'Production Direct FD: CPU versus T4 (PR #4)',
        'Reference/candidate: one-thread Xeon CPU Fast FD / T4 Fast FD; 1 warmup + 5 fixed-order synchronized repeats, **not paired**. Both include fresh construction + full update + snapshot. Resident excludes transfers; transfer-included explicitly includes pageable H2D and sketch D2H. Ratio here is **ratio of medians**, no bootstrap; p95=max with five samples.')
    table=rows(3)
    for dtype in ('torch.float32','torch.float64'):
        left=select(table,n=128,d=512,ell=32,dtype=dtype,method='fast',scope='cpu_resident')
        for scope in ('gpu_resident','gpu_transfer_included'):
            right=select(table,n=128,d=512,ell=32,dtype=dtype,method='fast',scope=scope)
            entry('CPU / T4 '+scope,left,left['median_ms'],right['median_ms'],float(left['median_ms'])/float(right['median_ms']),None,'Pilot; no universal crossover')
    section(5,'Shipped short snapshot optimization (PR #6)',
        'Reference/candidate: exact prior Direct FD / prefix-copy short snapshot. T4; one CPU thread; 2 warmups + 20 shuffled pairs; resident, transfers excluded. Snapshot-only and fresh full update+snapshot are distinct. Ratio below is paired median, not the historical summary\'s ratio of medians.')
    table=rows(5)
    for scope in ('snapshot','full_update_snapshot'):
        a=select(table,device='cuda',n=98,d=512,ell=32,dtype='torch.float32',scope=scope,variant='baseline')
        b=select(table,device='cuda',n=98,d=512,ell=32,dtype='torch.float32',scope=scope,variant='candidate')
        entry('T4 '+scope,a,a['median_ms'],b['median_ms'],a['median_paired_ratio'],None,'Shipped; interval not in this CSV')
    section(7,'Rejected removal of physical clearing (PR #8)',
        'Reference/candidate: clearing / no-clear prototype. T4, one CPU thread, 2 warmups + 20 balanced shuffled pairs; resident full fresh update+sketch, transfers excluded. Candidate logical math passed, physical pickle tail differed; production clearing retained.')
    table=rows(7)
    a=select(table,device='cuda',n=256,d=512,ell=32,dtype='torch.float32',scope='workloads',variant='baseline')
    b=select(table,device='cuda',n=256,d=512,ell=32,dtype='torch.float32',scope='workloads',variant='candidate')
    entry('T4 full update+sketch',a,a['median_ms'],b['median_ms'],a['paired_median_ratio'],(a['bootstrap_lower'],a['bootstrap_upper']),'Rejected; small/mixed performance')
    section(11,'CPU QR-SVD shape and thread controls (PR #12)',
        'Reference/candidate: Direct / QR-Small-SVD prototype; Ryzen 7 4800H, WSL2 torch 2.11.0+cpu / Python 3.12.3. Each row identifies intra-op threads (inter-op=1). Resident CPU, 2 warmups + 20 balanced shuffled pairs in each of two reverse-order batches; selected batch 0/1 shown separately. Fresh full-stream calls include empty updates and clearing. No GPU inference from CPU results.')
    table=rows(11)
    for phase,threads,ell,d in (('shape',1,16,32),('shape',1,16,512),('threads',2,16,128),('threads',4,64,1024)):
        for batch in (0,1):
            r=select(table,phase=phase,threads=threads,ell=ell,d=d,kind='four_long',dtype='torch.float32',comparison='direct/QR',batch=batch)
            entry(f'CPU {threads} threads; full; batch {batch}',r,r['direct_median_ms'],r['other_median_ms'],r['paired_speedup'],(r['exploratory_lower'],r['exploratory_upper']),'Local observation; no dispatch accepted')
    section(13,'T4 QR-SVD complete streams (PR #14; figure selection)',
        'Reference/candidate: Direct / CUDA QR-Small-SVD adapter. T4, CPU oracle threads=1, 2 warmups + 20 balanced randomized AB/BA pairs. Resident inputs, generation/H2D/D2H excluded; full fresh update+sketch including strict checks. Selection includes all five float32 long-buffer compression configurations (final ell+1); float64 d512 added as a dtype contrast. A/A flags=0; no independent-session replication.')
    table=rows(13)
    for dtype,ell,d in [('torch.float32',e,dd) for e,dd in ((16,64),(16,128),(16,512),(32,1024),(64,2048))]+[('torch.float64',16,512)]:
        r=select(table,dtype=dtype,ell=ell,d=d,kind='long',comparison='direct/QR')
        entry('T4 full update+sketch',r,r['direct_median_ms'],r['other_median_ms'],r['median_paired_speedup'],(r['descriptive_lower'],r['descriptive_upper']),'Local screen '+r['screen']+'; research only')
    section(15,'CUDA Inductor shrink: component versus whole (PR #16)',
        'Reference/candidate: Direct eager wrapper / same wrapper with only pointwise shrink Inductor-compiled. T4, one Xeon CPU thread, 2 warmups + 20 balanced randomized pairs per block; resident inputs, transfers excluded. Pure helper, factor-only and full fresh stream are independent scopes; no multiplying ratios. 0/30 whole-workload screens, one A/A flag; 6.333s measured codegen/first-call setup separate from steady timing.')
    table=rows(15)
    for comparison,label in (('shrink_pair','pure pointwise shrink'),('factor_pair','factor only'),('cuda_eager/cuda_partial','full update+sketch')):
        r=select(table,ell=16,d=512,dtype='torch.float32',final_count=17,comparison=comparison)
        entry('T4 '+label,r,r['baseline_median_ms'],r['candidate_median_ms'],r['paired_ratio'],(r['descriptive_lower'],r['descriptive_upper']),'Partial compilation only')
    section(19,'Strict ScratchFD full streams (PR #19)',
        'Reference/candidate: original Direct eager / strict ScratchFD local storage reuse. T4, CPU threads=1; CPU model/affinity not recorded in this worker. 2 warmups + 20 balanced AB/BA pairs; randomized shapes/context order; A/A before/after and CPU/GPU pairing-context drift. Resident, transfers excluded; fresh construction+update+sketch. 0/26 screens, 11 control-inconclusive; no accepted acceleration.')
    table=rows(19)
    for ell,d,final in ((16,512,17),(64,2048,63),(64,2048,65)):
        r=select(table,ell=ell,d=d,dtype='torch.float32',final=final,comparison='cuda_candidate')
        entry('T4 full update+sketch',r,r['baseline_median_ms'],r['candidate_median_ms'],r['paired_median_ratio'],(r['lower95'],r['upper95']),r['performance'])
    lines+=['','PR #18 recorded **no timing samples**; it has no synthetic zero-latency row.',
        'See [integrated interpretation](README.md) for math, graph failures and limitations.','']
    return '\n'.join(lines)


def main():
    outputs={'docs/assets/fd-algorithm.svg':algorithm(),'docs/assets/gpu-benchmarks.svg':gpu_figure(),
        'docs/assets/kernel-fusion.svg':kernel_figure(),'docs/performance/measurements.md':measurements()}
    outputs['docs/assets/sources.json']=json.dumps(dict(documentation_base_sha=BASE,source_blob_sha256=HASHES,
        gpu_selection='PR14 float32 direct/QR kind=long; all five ell<=d configurations; no factor/fallback rows',
        kernel_selection='PR16 verification.fusion ell16 d512 float32; separate pure and full scopes',
        regeneration='python docs/tools/build_figures.py; --check compares bytes without writing'),indent=2)+'\n'
    for name,text_value in outputs.items():
        target=ROOT/name
        if '--check' in sys.argv: assert target.read_bytes()==text_value.encode(),name+' is stale'
        else: target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(text_value.encode())
    print(json.dumps(dict(status='PASS',mode='check' if '--check' in sys.argv else 'write',outputs=list(outputs),sources=len(HASHES))))


if __name__=='__main__': main()
