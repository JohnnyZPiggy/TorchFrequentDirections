"""Occupied-prefix safety checks; poison and raw trace tensors never leave here."""
from unittest.mock import patch

import torch


def check_stream(classes, a, ell, stops, oracle, *, poison=True):
    traces = {name: [] for name in classes}
    streams = {}
    for name, cls in classes.items():
        class Traced(cls):
            def _factor(self, rows, *, shrink):
                if not bool(torch.isfinite(rows).all()):
                    raise AssertionError('STALE_TAIL_REACHED_FACTOR')
                traces[self.trace_name].append(rows.detach().cpu().double().clone())
                result = super()._factor(rows, shrink=shrink)
                assert result.untyped_storage().data_ptr() != self._buffer.untyped_storage().data_ptr()
                return result
        s = Traced(ell,a.shape[1],dtype=a.dtype,device=a.device)
        s.trace_name = name
        assert int(torch.count_nonzero(s._buffer)) == 0
        streams[name] = s
    maximum = {'psd':0., 'covariance':0., 'projection':0., 'snapshot_gap':0., 'svd_input_gap':0.}
    occupancies=[]
    previous=0
    oracle_calls=0
    for stop in stops:
        for name,s in streams.items():
            s.update(a[previous:stop])
            if name=='candidate' and poison:
                s._buffer[s._count:].fill_(float('nan'))
        assert streams['baseline']._count==streams['candidate']._count
        count=streams['candidate']._count
        occupancies.append(count)
        snapshots={}
        for name,s in streams.items():
            prefix=s._buffer[:count].clone()
            b=s.sketch()
            again=s.sketch()
            torch.testing.assert_close(b,again,rtol=0,atol=0)
            assert b.shape==(ell,a.shape[1]) and b.dtype==a.dtype and b.device==a.device
            assert not b.requires_grad and bool(torch.isfinite(b).all())
            if count<=ell: assert int(torch.count_nonzero(b[count:]))==0
            snapshots[name]=b.cpu().double().clone()
            b.fill_(123456)
            torch.testing.assert_close(s._buffer[:count],prefix,rtol=0,atol=0)
            report=oracle(a[:stop],again,ell)
            assert report['status']=='PASS', 'INDEPENDENT_ORACLE_FAILED'
            tol=report['tolerance']
            def excess(value):
                return max(0.,value)/tol if tol else (0. if value<=0 else float('inf'))
            maximum['psd']=max(maximum['psd'],excess(-report['minimum_eigenvalue']))
            for c in report['checks']:
                maximum['covariance']=max(maximum['covariance'],excess(report['spectral_error']-c['covariance_bound']))
                maximum['projection']=max(maximum['projection'],excess(c['projection_squared']-c['projection_bound']))
            oracle_calls+=1
        old,new=(snapshots[k] for k in ('baseline','candidate'))
        energy=float(a[:stop].cpu().double().square().sum())
        tolerance=128*torch.finfo(a.dtype).eps*max(stop,a.shape[1],2*ell,1)*energy
        gap=float(torch.linalg.matrix_norm(old.T@old-new.T@new,ord=2))
        assert gap<=tolerance
        maximum['snapshot_gap']=max(maximum['snapshot_gap'],gap/tolerance if tolerance else 0.)
        previous=stop
    assert previous==a.shape[0]
    assert len(traces['baseline'])==len(traces['candidate'])
    tolerance=128*torch.finfo(a.dtype).eps*max(*a.shape,2*ell,1)*float(a.cpu().double().square().sum())
    for old,new in zip(traces['baseline'],traces['candidate']):
        assert old.shape==new.shape
        gap=float(torch.linalg.matrix_norm(old.T@old-new.T@new,ord=2))
        assert gap<=tolerance
        maximum['svd_input_gap']=max(maximum['svd_input_gap'],gap/tolerance if tolerance else 0.)
    return {'status':'PASS','poisoned_tail':poison,'updates':len(stops),
        'occupancies':occupancies,'factor_reads_checked':len(traces['candidate']),
        'independent_oracle_calls':oracle_calls,'max_excess_over_tolerance':maximum,
        'alias_isolation':'PASS','factor_storage_independent':'PASS'}


def fault_checks(classes, device, dtype):
    results=[]
    for name,cls in classes.items():
        s=cls(4,7,dtype=dtype,device=device).update(torch.ones(3,7,dtype=dtype,device=device))
        s._buffer[s._count:].fill_(float('nan'))
        before=s._buffer.clone()
        try:
            s.update(torch.full((2,7),float('nan'),dtype=dtype,device=device))
        except ValueError:
            torch.testing.assert_close(before,s._buffer,rtol=0,atol=0,equal_nan=True)
        else: raise AssertionError('INVALID_INPUT_NOT_REJECTED')
        for kind in ('svd_exception','nonfinite_factor'):
            s=cls(4,7,dtype=dtype,device=device).update(torch.ones(3,7,dtype=dtype,device=device))
            s._buffer[s._count:].fill_(float('nan'))
            rows=torch.ones(5,7,dtype=dtype,device=device)
            if kind=='svd_exception':
                options={'side_effect':RuntimeError('injected SVD failure')}
                expected=RuntimeError
            else:
                u,v,vh=torch.linalg.svd(torch.ones(8,7,dtype=dtype,device=device),full_matrices=False)
                v[0]=float('inf')
                options={'return_value':(u,v,vh)}
                expected=FloatingPointError
            with patch('torch.linalg.svd',**options):
                try: s.update(rows)
                except expected: pass
                else: raise AssertionError('BACKEND_FAILURE_NOT_PROPAGATED')
            assert s._count==8 and bool(torch.isfinite(s._buffer).all())
            # Failed backend instances are discarded, per the public contract.
        s=cls(4,7,dtype=dtype,device=device).update(torch.ones(3,7,dtype=dtype,device=device))
        s._buffer[0,0]=float('inf')
        try: s.sketch()
        except FloatingPointError: pass
        else: raise AssertionError('SNAPSHOT_FINITE_CHECK_REMOVED')
        results.append({'variant':name,'status':'PASS','invalid_update_atomic':'PASS',
            'svd_exception':'PASS','nonfinite_factor':'PASS','nonfinite_snapshot':'PASS'})
    return results
