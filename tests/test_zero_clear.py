import pickle
import unittest

import torch

from torchfrequentdirections import FrequentDirections
from experiments.zero_clear.checks import check_stream, fault_checks
from test_gpu_driver import driver as oracle_helper


class ClearingBaseline(FrequentDirections):
    """Extra-clear comparator; pinned integration tests the actual no-clear source."""
    def _factor(self, rows, *, shrink):
        result=super()._factor(rows,shrink=shrink)
        if shrink and rows.shape[0]==2*self.ell:
            self._buffer.zero_()
        return result


class ZeroClearTests(unittest.TestCase):
    def test_retained_runtime_pickle_does_not_export_stale_unused_rows(self):
        stream=FrequentDirections(4,7).update(torch.arange(56,dtype=torch.float64).reshape(8,7))
        restored=pickle.loads(pickle.dumps(stream))
        self.assertEqual(stream._count,3)
        self.assertEqual(int(torch.count_nonzero(restored._buffer[restored._count:])),0)
        torch.testing.assert_close(restored.sketch(),stream.sketch(),rtol=0,atol=0)

    def test_poisoned_tail_never_enters_factor_or_snapshot(self):
        for dtype in (torch.float32,torch.float64):
            for ell,d in ((1,3),(4,7),(6,2)):
                n=8*ell+7
                for kind in ('random','rank_one','zero'):
                    a=torch.randn(n,2*d,dtype=dtype,generator=torch.Generator().manual_seed(7))[:,::2]
                    if kind=='rank_one': a[:]=a[:,:1].expand_as(a)
                    if kind=='zero': a.zero_()
                    patterns=([0,n],list(range(n+1)),sorted({0,1,ell-1,ell,ell+1,2*ell,2*ell+1,n//2,n-1,n}))
                    for stops in patterns:
                        with self.subTest(dtype=dtype,ell=ell,kind=kind,stops=len(stops)):
                            result=check_stream({'baseline':ClearingBaseline,'candidate':FrequentDirections},
                                a,ell,stops,oracle_helper.oracle)
                            self.assertEqual(result['status'],'PASS')
                            self.assertGreater(result['factor_reads_checked'],0)

    def test_failure_paths_preserve_checks_and_atomicity(self):
        for dtype in (torch.float32,torch.float64):
            result=fault_checks({'baseline':ClearingBaseline,'candidate':FrequentDirections},'cpu',dtype)
            self.assertTrue(all(c['status']=='PASS' for c in result))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
