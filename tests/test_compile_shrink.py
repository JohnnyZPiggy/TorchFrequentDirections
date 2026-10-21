import unittest
from unittest.mock import patch

import torch

from experiments.compile_fd.shrink import TensorShrinkFD,tensor_shrink
from experiments.qr_svd.driver import compare,fixtures
from torchfrequentdirections import FrequentDirections


class TensorShrinkTests(unittest.TestCase):
    def test_independent_stream_bounds_and_production_parity(self):
        for dtype in (torch.float32,torch.float64):
            for a in fixtures(dtype).values():
                for ell in (1,4,a.shape[1]+2):
                    outputs=[cls(ell,a.shape[1],dtype=dtype).update(a).sketch()
                             for cls in (FrequentDirections,TensorShrinkFD)]
                    self.assertTrue(compare(a,outputs,ell)['passed'])
                    torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)

    def test_pure_edges_alias_no_grad_and_overflow(self):
        for dtype in (torch.float32,torch.float64):
            for m,d,ell in ((0,7,4),(2,7,4),(8,3,5),(8,17,1),(8,17,4)):
                a=torch.randn(m,d,dtype=dtype)
                _,s,v=torch.linalg.svd(a,full_matrices=False)
                for shrink in (False,True):
                    old_s,old_v=s.clone(),v.clone()
                    out=tensor_shrink(s.requires_grad_(),v.requires_grad_(),ell,shrink)
                    expected=FrequentDirections(ell,d,dtype=dtype)._factor(a,shrink=shrink)
                    torch.testing.assert_close(out,expected,rtol=0,atol=0)
                    self.assertFalse(out.requires_grad); out.fill_(123)
                    torch.testing.assert_close(s,old_s,rtol=0,atol=0)
                    torch.testing.assert_close(v,old_v,rtol=0,atol=0)
        a=torch.diag(torch.tensor([4e20,3e20,2e20,1e20]))
        b=TensorShrinkFD(2,4,dtype=a.dtype).update(a).sketch()
        self.assertTrue(bool(torch.isfinite(b).all()))

    def test_fault_and_state_contracts(self):
        a=torch.randn(23,9,dtype=torch.float64,requires_grad=True)
        state=TensorShrinkFD(4,9).update(a[:7]); before=state._buffer.clone(); count=state._count
        snapshot=state.sketch(); snapshot.fill_(123)
        torch.testing.assert_close(before,state._buffer,rtol=0,atol=0)
        self.assertEqual(count,state._count)
        with self.assertRaises(ValueError): state.update(torch.full((2,9),float('inf'),dtype=a.dtype))
        torch.testing.assert_close(before,state._buffer,rtol=0,atol=0)
        state.update(a[7:]); self.assertFalse(state.sketch().requires_grad)
        with patch('torch.linalg.svd',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'): TensorShrinkFD(4,9).update(a)
        def nonfinite(s,v,ell,shrink): return v.new_full((ell,v.shape[1]),float('nan'))
        with self.assertRaises(FloatingPointError): TensorShrinkFD(4,9,shrink_fn=nonfinite).update(a)
