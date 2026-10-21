import unittest
from unittest.mock import patch

import torch

from experiments.qr_svd.cuda_adapter import ExperimentalQRSmallSVD
from experiments.qr_svd.driver import compare, fixtures
from experiments.qr_svd.prototype import QRSmallSVD
from torchfrequentdirections import FrequentDirections


class CUDAAdapterCPUTests(unittest.TestCase):
    def test_cpu_independent_bounds_and_historic_parity(self):
        for dtype in (torch.float32, torch.float64):
            for a in fixtures(dtype).values():
                for ell in (1,4,a.shape[1]+2):
                    b=[cls(ell,a.shape[1],dtype=dtype).update(a).sketch()
                       for cls in (FrequentDirections,ExperimentalQRSmallSVD)]
                    self.assertTrue(compare(a,b,ell)['passed'])
                    old=QRSmallSVD(ell,a.shape[1],dtype=dtype).update(a).sketch()
                    torch.testing.assert_close(old,b[1],rtol=0,atol=0)

    def test_faults_atomicity_alias_and_no_grad(self):
        a=torch.randn(23,9,dtype=torch.float64,requires_grad=True)
        s=ExperimentalQRSmallSVD(4,9).update(a[:7])
        saved=s._buffer.clone(); count=s._count
        with self.assertRaises(ValueError):
            s.update(torch.full((2,9),float('nan'),dtype=a.dtype))
        torch.testing.assert_close(saved,s._buffer,rtol=0,atol=0)
        self.assertEqual(count,s._count)
        b=s.sketch(); b.zero_()
        torch.testing.assert_close(saved,s._buffer,rtol=0,atol=0)
        s.update(a[7:]); self.assertFalse(s.sketch().requires_grad)
        for op in ('qr','svd'):
            with patch('torch.linalg.'+op,side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    ExperimentalQRSmallSVD(4,9).update(a)
        original=torch.linalg.svd
        def corrupt(*args,**kwargs):
            u,s,v=original(*args,**kwargs); v.fill_(float('inf')); return u,s,v
        with patch('torch.linalg.svd',side_effect=corrupt):
            with self.assertRaises(FloatingPointError):
                ExperimentalQRSmallSVD(4,9).update(a)
        with self.assertRaises(ValueError):
            ExperimentalQRSmallSVD(4,9,device='meta')
