import pickle
import unittest
from unittest.mock import patch
import torch
from experiments.full_gpu.candidate import ScratchFD, graph_transition
from experiments.qr_svd.driver import compare, fixtures
from torchfrequentdirections import FrequentDirections


class ScratchTests(unittest.TestCase):
    def test_math_partitions_state_and_ownership(self):
        for dtype in (torch.float32, torch.float64):
            for name, a in fixtures(dtype).items():
                for ell in (1, 4, a.shape[1]+2):
                    for chunk in (1, 7, max(1, len(a))):
                        states = [cls(ell,a.shape[1],dtype=dtype) for cls in (FrequentDirections,ScratchFD)]
                        for part in a.split(chunk):
                            for state in states: state.update(part); state.update(part[:0])
                            self.assertEqual(states[0]._count,states[1]._count)
                            torch.testing.assert_close(states[0]._buffer,states[1]._buffer,rtol=0,atol=0)
                        outputs=[s.sketch() for s in states]
                        self.assertTrue(compare(a,outputs,ell)['passed'],(name,ell,chunk))
                        state=states[1]; before=state._buffer.clone(); count=state._count
                        outputs[1].fill_(123)
                        torch.testing.assert_close(state._buffer,before,rtol=0,atol=0)
                        self.assertEqual(count,state._count)
                        self.assertEqual(set(state.__dict__),set(states[0].__dict__))
                        restored=pickle.loads(pickle.dumps(state))
                        torch.testing.assert_close(restored._buffer,before,rtol=0,atol=0)
                        self.assertFalse(state.sketch().requires_grad)

    def test_faults_atomicity_and_no_grad(self):
        for dtype in (torch.float32,torch.float64):
            a=torch.randn(23,9,dtype=dtype,requires_grad=True)
            state=ScratchFD(4,9,dtype=dtype).update(a[:7]); before=state._buffer.clone()
            invalid=[None,torch.ones(9,dtype=dtype),torch.ones(1,8,dtype=dtype),torch.ones(1,9,dtype=torch.int64)]
            invalid += [torch.full((2,9),v,dtype=dtype) for v in (float('nan'),float('inf'),-float('inf'))]
            for value in invalid:
                with self.assertRaises((ValueError,TypeError)): state.update(value)
                torch.testing.assert_close(state._buffer,before,rtol=0,atol=0)
                self.assertEqual(state._count,7)
            state.update(a[7:]); self.assertFalse(state.sketch().requires_grad)
            with patch('torch.linalg.svd',side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError,'injected'): ScratchFD(4,9,dtype=dtype).update(a)
            actual=torch.linalg.svd(a.detach()[:8],full_matrices=False)
            for field in (1,2):
                broken=list(actual); broken[field]=torch.full_like(broken[field],float('nan'))
                with patch('torch.linalg.svd',return_value=tuple(broken)):
                    with self.assertRaises(FloatingPointError): ScratchFD(4,9,dtype=dtype).update(a)
            state=ScratchFD(4,9,dtype=dtype).update(a[:2]); state._buffer[0,0]=float('inf')
            with self.assertRaises(FloatingPointError): state.sketch()

    def test_graph_tensor_transition_and_extreme_scale(self):
        for dtype in (torch.float32,torch.float64):
            for ell,d in ((1,7),(4,17),(8,3)):
                a=torch.randn(2*ell,d,dtype=dtype)
                out=torch.empty_like(a); graph_transition(a,out,ell)
                state=FrequentDirections(ell,d,dtype=dtype).update(a)
                torch.testing.assert_close(out,state._buffer,rtol=0,atol=0)
                torch.testing.assert_close(ScratchFD(ell,d,dtype=dtype).update(a)._buffer,state._buffer,rtol=0,atol=0)
            a=torch.diag(torch.tensor([4e20,3e20,2e20,1e20],dtype=dtype))
            torch.testing.assert_close(ScratchFD(2,4,dtype=dtype).update(a).sketch(),FrequentDirections(2,4,dtype=dtype).update(a).sketch(),rtol=0,atol=0)
