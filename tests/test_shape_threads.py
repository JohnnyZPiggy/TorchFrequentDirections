"""Controls must execute the paths that their evidence labels describe."""
import unittest

import torch

from experiments.qr_svd.shape_threads import (
    SHAPES, gate, path_counts, rows_for_shrinks, select_shapes, stream)
from experiments.qr_svd.prototype import QRSmallSVD
from torchfrequentdirections import FrequentDirections


class ShapeThreadTests(unittest.TestCase):
    def test_fixed_axes_and_four_shrink_occupancies(self):
        self.assertEqual([d for ell,d in SHAPES if ell == 16], [32,64,128,256,512,1024])
        self.assertEqual(sorted(ell for ell,d in SHAPES if d == 1024), [8,16,32,64])
        for ell,d in ((2,8),(4,8),(8,3)):
            for final in (ell-1,ell+1):
                a=torch.randn(rows_for_shrinks(ell,d,final),d,dtype=torch.float64)
                counts=[path_counts(a,ell,cls) for cls in (FrequentDirections,QRSmallSVD)]
                for p in counts:
                    self.assertEqual(p['full_shrinks'],4)
                    self.assertEqual(p['final_count'],final)
                    self.assertEqual(p['snapshot_svd_calls'],int(final > ell))
                self.assertEqual(counts[0]['svd_calls'],counts[1]['svd_calls'])

    def test_no_factor_and_fallback_controls(self):
        for n in (0,3):
            p=path_counts(torch.randn(n,8,dtype=torch.float64),4,QRSmallSVD)
            self.assertEqual((p['svd_calls'],p['qr_calls']),(0,0))
        p=path_counts(torch.randn(54,4,dtype=torch.float64),8,QRSmallSVD)
        self.assertGreater(p['svd_calls'],0)
        self.assertEqual(p['qr_calls'],0)

    def test_no_grad_ownership_and_gate(self):
        a=torch.randn(23,8,dtype=torch.float64,requires_grad=True)
        original=a.detach().clone()
        state=stream(a,4,QRSmallSVD)
        self.assertFalse(state.sketch().requires_grad)
        torch.testing.assert_close(a.detach(),original,rtol=0,atol=0)
        self.assertTrue(gate(a.detach(),4,'stream')['math']['passed'])

    def test_selection_is_distinct_and_uses_measured_streams(self):
        records=[]
        for i,(ell,d) in enumerate(SHAPES):
            records.append(dict(ell=ell,d=d,kind='four_long',
                                timing=dict(median_paired_speedup=.6+.1*i)))
        result=select_shapes(records)
        self.assertEqual(result['losing'],list(SHAPES[0]))
        self.assertEqual(result['near'],list(SHAPES[4]))
        self.assertEqual(result['winning'],list(SHAPES[-1]))
