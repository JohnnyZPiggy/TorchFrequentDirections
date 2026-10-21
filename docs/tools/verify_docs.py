"""Offline docs/figure provenance checks; optional README smoke test needs CPU torch."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
BASE='f97070dddc1cb1788d4b1fab85c58da650a9f552'


def git(*args): return subprocess.check_output(['git','-C',str(ROOT),*args])
def original(path): return git('show',BASE+':'+path)


def main():
    subprocess.run([sys.executable,str(ROOT/'docs/tools/build_figures.py'),'--check'],check=True,capture_output=True)
    manifest=json.loads((ROOT/'docs/assets/sources.json').read_text())
    for path,digest in manifest['source_blob_sha256'].items():
        assert hashlib.sha256(original(path)).hexdigest()==digest,path
        assert git('show','HEAD:'+path)==original(path),'source changed: '+path
    # Files and same-file/cross-file fragment anchors for all current docs. Historical
    # docs are read as link targets but not rewritten or reclassified by this check.
    pages=[ROOT/'README.md',ROOT/'docs/README.md',ROOT/'docs/performance/README.md',
        ROOT/'docs/performance/measurements.md',ROOT/'docs/assets/README.md',ROOT/'docs/VERIFICATION.md']
    links=0
    for page in pages:
        for link in re.findall(r'\]\(([^)]+)\)',page.read_text()):
            if re.match(r'^[a-z]+:',link): continue
            filename,_,fragment=unquote(link).partition('#')
            target=page.parent/filename if filename else page
            assert target.exists(),(page.name,link); links+=1
            if fragment and target.suffix=='.md':
                headings=re.findall(r'^#{1,6}\s+(.+?)\s*$',target.read_text(),re.M)
                anchors=[re.sub(r'[^\w\- ]','',h.lower()).replace(' ','-') for h in headings]
                assert fragment in anchors,(str(target),fragment)
    ns={'s':'http://www.w3.org/2000/svg'}
    figures={name:ET.parse(ROOT/'docs/assets'/name).getroot() for name in ('fd-algorithm.svg','gpu-benchmarks.svg','kernel-fusion.svg')}
    for name,svg in figures.items():
        assert svg.find('s:title',ns).text and svg.find('s:desc',ns).text,name
        assert svg.attrib['role']=='img' and svg.attrib['viewBox'].startswith('0 0 ')
    table=list(csv.DictReader(io.StringIO(original('reports/gpu/issue13-20261021/benchmark.csv').decode())))
    points=[e for e in figures['gpu-benchmarks.svg'].iter() if 'data-ratio' in e.attrib]
    assert len(points)==5
    for point in points:
        r=next(r for r in table if r['d']==point.attrib['data-d'] and r['dtype']=='torch.float32' and r['kind']=='long' and r['comparison']=='direct/QR')
        for a,b in (('data-ratio','median_paired_speedup'),('data-lower','descriptive_lower'),('data-upper','descriptive_upper')):
            assert float(point.attrib[a])==float(r[b])
        dot=point.find('s:circle',ns)
        assert abs(float(dot.attrib['cx'])-(300+(float(r['median_paired_speedup'])-.8)/.7*570))<.006
    fusion=json.loads(original('reports/gpu/issue15-20261021/verification.json'))['fusion']
    row=next(r for r in fusion if r['ell']==16 and r['d']==512 and r['dtype']=='torch.float32')
    bars=[e for e in figures['kernel-fusion.svg'].iter() if 'data-count' in e.attrib]
    assert len(bars)==4
    for bar in bars:
        scope=bar.attrib['data-scope']; index=int(bar.attrib['data-variant'])
        key=('' if scope=='pure' else 'full_')+('eager_launches' if index==0 else 'partial_launches')
        count=int(bar.attrib['data-count']); assert count==row[key]
        assert abs(float(bar.attrib['width'])-count/(10 if scope=='pure' else 800)*345)<.001
    math_text=' '.join(figures['fd-algorithm.svg'].itertext())
    for label in ('m = 2ell','min(ell - 1, d)','0 <= m <= ell','ell < m < 2ell','sigma_ell^2','No SVD','No state mutation'):
        assert label.lower() in math_text.lower(),label
    example='NOT_RUN (use --run-example with CPU torch)'
    if '--run-example' in sys.argv:
        import torch
        assert str(torch.__version__).endswith('+cpu'),'Use CPU-only torch; no GPU access required'
        torch.set_num_threads(1); sys.path.insert(0,str(ROOT))
        code=re.findall(r'```python\n(.*?)```',(ROOT/'README.md').read_text(),re.S)
        assert len(code)==1; scope={}; exec(compile(code[0],'README.md','exec'),scope)
        a=scope['a']; ell=8; energy=float(a.square().sum()); tol=32*torch.finfo(a.dtype).eps*max(*a.shape,2*ell)*energy
        tails=torch.linalg.svdvals(a).square()
        for b in (scope['b'],scope['reference'],scope['snapshot']):
            gap=a.T@a-b.T@b; eig=torch.linalg.eigvalsh((gap+gap.T)/2)
            _,_,vh=torch.linalg.svd(b,full_matrices=False)
            assert float(eig.min())>=-tol
            for k in range(ell):
                bound=float(tails[k:].sum())/(ell-k)
                assert float(eig.abs().max())<=bound+tol
                residual=float((a-(a@vh[:k].T)@vh[:k]).square().sum())
                assert residual<=ell*bound+tol
        example=dict(status='PASS',torch=str(torch.__version__),device=str(a.device),outputs=3,per_k_checks=24)
    changed=git('diff',BASE,'--name-only').decode().splitlines()
    allowed={'README.md','docs/README.md','docs/VERIFICATION.md','docs/verification.json',
        'docs/performance/README.md','docs/performance/measurements.md',
        'docs/assets/README.md','docs/assets/sources.json','docs/assets/fd-algorithm.svg',
        'docs/assets/gpu-benchmarks.svg','docs/assets/kernel-fusion.svg',
        'docs/tools/build_figures.py','docs/tools/verify_docs.py'}
    assert set(changed)<=allowed,changed
    print(json.dumps(dict(status='PASS',documentation_base_sha=BASE,checked_head=git('rev-parse','HEAD').decode().strip(),
        relative_links=links,svg_files=3,benchmark_points_independently_matched=5,kernel_counts_independently_matched=4,
        source_hashes=len(manifest['source_blob_sha256']),historical_evidence_and_production_unchanged=True,readme_example=example)))


if __name__=='__main__': main()
