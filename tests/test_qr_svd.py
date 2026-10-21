"""Fault, fallback and inherited state contracts for the opt-in experiment."""
import unittest
from unittest.mock import patch

import torch

from experiments.qr_svd.prototype import QRSmallSVD
from torchfrequentdirections import FrequentDirections


class QRTests(unittest.TestCase):
    def test_fallback_and_empty_do_not_use_qr(self):
        for rows in (torch.empty(0, 3), torch.randn(7, 3)):
            candidate = QRSmallSVD(5, 3, dtype=rows.dtype)
            with patch("torch.linalg.qr", side_effect=AssertionError("unexpected QR")):
                actual = candidate._factor(rows, shrink=True)
            expected = FrequentDirections(5, 3, dtype=rows.dtype)._factor(rows, shrink=True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_backend_failure_propagates(self):
        for name in ("qr", "svd"):
            with patch("torch.linalg." + name, side_effect=RuntimeError("injected")):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    QRSmallSVD(2, 5)._factor(torch.ones(4, 5, dtype=torch.float64), shrink=True)

    def test_nonfinite_factor_detected(self):
        original = torch.linalg.svd

        def corrupted(*args, **kwargs):
            u, s, vh = original(*args, **kwargs)
            vh.fill_(float("nan"))
            return u, s, vh

        with patch("torch.linalg.svd", side_effect=corrupted):
            with self.assertRaises(FloatingPointError):
                QRSmallSVD(2, 5)._factor(torch.eye(4, 5, dtype=torch.float64), shrink=True)

    def test_invalid_input_atomic_and_cpu_only(self):
        stream = QRSmallSVD(2, 5).update(torch.ones(3, 5, dtype=torch.float64))
        before = stream._buffer.clone()
        for value in (float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                stream.update(torch.full((2, 5), value, dtype=torch.float64))
            torch.testing.assert_close(before, stream._buffer, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            QRSmallSVD(2, 5, device="cuda")

    def test_overflow_guard_and_svd_count(self):
        rows = torch.diag(torch.tensor([4e20, 3e20, 2e20, 1e20])).repeat(1, 2)
        self.assertTrue(bool(torch.isfinite(QRSmallSVD(2, 8, dtype=rows.dtype)
                                           ._factor(rows, shrink=True)).all()))
        a = torch.randn(41, 13, dtype=torch.float64)
        counts = []
        for cls in (FrequentDirections, QRSmallSVD):
            with patch("torch.linalg.svd", wraps=torch.linalg.svd) as svd:
                cls(4, 13).update(a).sketch()
                counts.append(svd.call_count)
        self.assertEqual(counts[0], counts[1])
