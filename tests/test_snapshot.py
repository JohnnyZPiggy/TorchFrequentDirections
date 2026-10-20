"""Lossless short snapshots: Gram/subspace contract, never SVD row orientation."""
import unittest
from unittest.mock import patch

import torch

from torchfrequentdirections import FrequentDirections
from test_fd import invariants


class SnapshotTests(unittest.TestCase):
    def test_occupancy_boundaries_match_old_snapshot_gram(self):
        for dtype in (torch.float32, torch.float64):
            for ell, d in ((1, 3), (4, 7), (5, 2)):
                for count in sorted({0, 1, ell - 1, ell, ell + 1}):
                    for kind in ('random', 'rank_one', 'zero'):
                        with self.subTest(dtype=dtype, ell=ell, d=d, count=count, kind=kind):
                            a = torch.randn(count, d, dtype=dtype,
                                            generator=torch.Generator().manual_seed(5))
                            if kind == 'rank_one':
                                a[:] = a[:, :1].expand_as(a)
                            elif kind == 'zero':
                                a.zero_()
                            stream = FrequentDirections(ell, d, dtype=dtype).update(a)
                            pending = stream._buffer[:stream._count]
                            old = stream._factor(pending, shrink=stream._count > ell)
                            before = stream._buffer.clone()
                            with patch('torch.linalg.svd', wraps=torch.linalg.svd) as svd:
                                new = stream.sketch()
                            self.assertEqual(svd.call_count, int(stream._count > ell))
                            torch.testing.assert_close(stream._buffer, before, rtol=0, atol=0)
                            if stream._count <= ell:
                                torch.testing.assert_close(new[:stream._count], pending, rtol=0, atol=0)
                                self.assertEqual(int(torch.count_nonzero(new[stream._count:])), 0)
                            torch.testing.assert_close(new.double().T @ new.double(),
                                old.double().T @ old.double(),
                                rtol=2e-5 if dtype == torch.float32 else 2e-13,
                                atol=2e-5 if dtype == torch.float32 else 2e-13)
                            self.assertEqual(new.shape, (ell, d))
                            self.assertEqual(new.dtype, dtype)
                            self.assertEqual(new.device, a.device)
                            self.assertFalse(new.requires_grad)
                            self.assertTrue(invariants(a, new, ell)['pass'])

    def test_repeated_snapshots_after_fills_and_continued_updates(self):
        for dtype in (torch.float32, torch.float64):
            a = torch.randn(57, 9, dtype=dtype, requires_grad=True,
                            generator=torch.Generator().manual_seed(17))
            stream = FrequentDirections(4, 9, dtype=dtype)
            untouched = FrequentDirections(4, 9, dtype=dtype)
            start = 0
            # After full-buffer shrink, occupancies alternate across ell.
            for stop in (8, 9, 10, 13, 23, 24, 57):
                stream.update(a[start:stop]); untouched.update(a[start:stop])
                before = stream._buffer.clone()
                saved = stream.sketch()
                expected = stream.sketch()
                saved.fill_(999)
                torch.testing.assert_close(stream.sketch(), expected, rtol=0, atol=0)
                torch.testing.assert_close(stream._buffer, before, rtol=0, atol=0)
                self.assertFalse(expected.requires_grad)
                self.assertTrue(invariants(a[:stop].detach(), expected, 4)['pass'])
                start = stop
            torch.testing.assert_close(stream.sketch(), untouched.sketch(), rtol=0, atol=0)

    def test_short_snapshot_retains_finite_output_failure_detection(self):
        stream = FrequentDirections(4, 3).update(torch.ones(2, 3, dtype=torch.float64))
        # Fault injection: do not silently hide corruption when bypassing SVD.
        stream._buffer[0, 0] = float('inf')
        with self.assertRaises(FloatingPointError):
            stream.sketch()


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
