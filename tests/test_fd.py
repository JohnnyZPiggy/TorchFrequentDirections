"""Independent Gram/eigenvalue/projection oracles; no FD shrink in the oracle."""

import unittest
from unittest.mock import patch

import torch

from torchfrequentdirections import FrequentDirections, fast_fd, reference_fd


def invariants(a, b, ell):
    original_dtype = a.dtype
    a, b = a.double(), b.double()
    energy = float(a.square().sum())
    tolerance = 32 * torch.finfo(original_dtype).eps * max(*a.shape, 2 * ell, 1) * energy
    gram_error = a.T @ a - b.T @ b
    eigen = torch.linalg.eigvalsh((gram_error + gram_error.T) / 2)
    minimum = float(eigen.min())
    spectral = float(eigen.abs().max())
    singular = torch.linalg.svdvals(a)
    _, _, vh = torch.linalg.svd(b, full_matrices=False)
    checks = []
    for k in range(min(ell - 1, a.shape[1]) + 1):
        tail = float(singular[k:].square().sum())
        projection = (a @ vh[:k].T) @ vh[:k]
        residual = float((a - projection).square().sum())
        covariance_bound = tail / (ell - k)
        projection_bound = ell * covariance_bound
        checks.append({"k": k, "tail_frobenius_squared": tail,
                       "covariance_bound": covariance_bound,
                       "projection_residual_squared": residual,
                       "projection_bound": projection_bound,
                       "pass": minimum >= -tolerance and
                       spectral <= covariance_bound + tolerance and
                       residual <= projection_bound + tolerance})
    return {"energy": energy, "tolerance": tolerance,
            "minimum_covariance_eigenvalue": minimum, "spectral_error": spectral,
            "checks": checks, "pass": all(c["pass"] for c in checks)}


def cases(dtype):
    generator = torch.Generator().manual_seed(1729)

    def random(n, d):
        return torch.randn(n, d, generator=generator, dtype=dtype)

    q, _ = torch.linalg.qr(random(19, 7), mode="reduced")
    v, _ = torch.linalg.qr(random(7, 7))
    decay = (q * torch.logspace(0, -5, 7, dtype=dtype)) @ v.T
    near = (q * torch.tensor([1., 1., 1., .999999, .5, .1, 0.], dtype=dtype)) @ v.T
    return {
        "random_tall": random(41, 9),
        "random_wide": random(7, 13),
        "exact_rank_two": random(23, 2) @ random(2, 8),
        "decaying": decay,
        "near_degenerate": near,
        "zero": torch.zeros(13, 7, dtype=dtype),
        "empty": torch.empty(0, 7, dtype=dtype),
        "short": random(2, 7),
        "one_feature": random(17, 1),
        "signed_cancelling_rows": torch.tensor([[1., -1., 0.], [2., -2., 0.],
                                                [0., 0., 0.], [0., 1., -1.]], dtype=dtype),
        "noncontiguous": random(12, 16)[:, ::2],
    }


class FrequentDirectionsTests(unittest.TestCase):
    def test_seeded_shape_sweep(self):
        for seed in range(12):
            generator = torch.Generator().manual_seed(seed)
            n, d, ell = 3 + seed * 2, 2 + seed % 7, 1 + seed % 5
            for dtype in (torch.float32, torch.float64):
                a = torch.randn(n, d, dtype=dtype, generator=generator)
                for fn in (reference_fd, fast_fd):
                    with self.subTest(seed=seed, dtype=dtype, fn=fn.__name__):
                        self.assertTrue(invariants(a, fn(a, ell), ell)["pass"])

    def test_independent_bounds(self):
        for dtype in (torch.float32, torch.float64):
            for name, a in cases(dtype).items():
                for ell in sorted({1, 2, 4, a.shape[1], a.shape[1] + 2}):
                    for fn in (reference_fd, fast_fd):
                        with self.subTest(dtype=dtype, case=name, ell=ell, fn=fn.__name__):
                            b = fn(a, ell)
                            self.assertEqual(b.shape, (ell, a.shape[1]))
                            self.assertEqual(b.dtype, a.dtype)
                            self.assertEqual(b.device, a.device)
                            report = invariants(a, b, ell)
                            self.assertTrue(report["pass"], report)

    def test_exact_rank_below_capacity(self):
        a = cases(torch.float64)["exact_rank_two"]
        for fn in (reference_fd, fast_fd):
            b = fn(a, 4)
            torch.testing.assert_close(b.T @ b, a.T @ a, rtol=2e-13, atol=2e-12)

    def test_ell_greater_than_features_exact_fallback(self):
        a = cases(torch.float64)["random_tall"]
        for fn in (reference_fd, fast_fd):
            b = fn(a, 12)
            torch.testing.assert_close(b.T @ b, a.T @ a, rtol=2e-13, atol=2e-12)

    def test_stream_chunking_and_nonmutating_snapshot(self):
        a = cases(torch.float64)["random_tall"]
        for method, fn in (("reference", reference_fd), ("fast", fast_fd)):
            stream = FrequentDirections(4, 9, method=method)
            for start, stop in ((0, 2), (2, 11), (11, 11), (11, 28), (28, 41)):
                stream.update(a[start:stop])
                saved = stream.sketch()
                saved.fill_(123)
                self.assertTrue(invariants(a[:stop], stream.sketch(), 4)["pass"])
            b, full = stream.sketch(), fn(a, 4)
            torch.testing.assert_close(b.T @ b, full.T @ full, rtol=1e-13, atol=1e-13)

    def test_partial_buffer_is_compressed_not_truncated(self):
        a = torch.tensor([[.01, 0., 0.], [.01, 0., 0.], [0., 10., 0.]], dtype=torch.float64)
        self.assertFalse(invariants(a, a[:2], 2)["pass"])
        self.assertTrue(invariants(a, fast_fd(a, 2), 2)["pass"])

    def test_one_indexed_threshold(self):
        a = torch.diag(torch.tensor([4., 3., 2., 1.], dtype=torch.float64))
        expected = torch.diag(torch.tensor([7., 0., 0., 0.], dtype=torch.float64))
        b = fast_fd(a, 2)
        torch.testing.assert_close(b.T @ b, expected)

    def test_reference_last_row_zero(self):
        for ell in (1, 4, 12):
            b = reference_fd(cases(torch.float64)["random_tall"], ell)
            self.assertEqual(float(b[-1].abs().sum()), 0.)

    def test_zero_and_empty_are_exact_zero(self):
        for fn in (reference_fd, fast_fd):
            for a in (torch.zeros(13, 7), torch.empty(0, 7)):
                self.assertEqual(int(torch.count_nonzero(fn(a, 4))), 0)

    def test_invalid_configuration(self):
        for value in (0, -1, True, 1.5, "2"):
            with self.assertRaises(ValueError):
                FrequentDirections(value, 3)
            with self.assertRaises(ValueError):
                FrequentDirections(2, value)
        with self.assertRaises(ValueError):
            FrequentDirections(2, 3, method="other")
        for dtype in (torch.int64, torch.complex64, torch.float16, torch.bfloat16):
            with self.assertRaises(ValueError):
                FrequentDirections(2, 3, dtype=dtype)

    def test_invalid_update_is_atomic(self):
        stream = FrequentDirections(2, 3).update(torch.ones(3, 3, dtype=torch.float64))
        before = stream.sketch()
        invalid = [torch.ones(3), torch.ones(2, 4, dtype=torch.float64),
                   torch.ones(2, 3, dtype=torch.float32),
                   torch.full((2, 3), float("nan"), dtype=torch.float64),
                   torch.full((2, 3), float("inf"), dtype=torch.float64),
                   torch.eye(3, dtype=torch.float64).to_sparse()]
        for rows in invalid:
            with self.assertRaises(ValueError):
                stream.update(rows)
            torch.testing.assert_close(stream.sketch(), before)
        with self.assertRaises(TypeError):
            stream.update([[1., 2., 3.]])
        for fn in (reference_fd, fast_fd):
            with self.assertRaises(TypeError):
                fn([], 2)
            with self.assertRaises(ValueError):
                fn(torch.ones(3), 2)
            with self.assertRaises(ValueError):
                fn(torch.empty(2, 0), 2)

    def test_input_ownership_and_no_autograd_graph(self):
        for fn in (reference_fd, fast_fd):
            a = torch.randn(11, 5, dtype=torch.float64, requires_grad=True)
            before = a.detach().clone()
            b = fn(a, 3)
            self.assertFalse(b.requires_grad)
            torch.testing.assert_close(a.detach(), before)
            b.zero_()
            torch.testing.assert_close(a.detach(), before)

    def test_repeatability(self):
        a = cases(torch.float64)["random_tall"]
        for fn in (reference_fd, fast_fd):
            first, second = fn(a, 4), fn(a, 4)
            torch.testing.assert_close(first.T @ first, second.T @ second, rtol=0, atol=0)

    def test_fast_reduces_svd_count(self):
        a = cases(torch.float64)["random_tall"]
        with patch("torch.linalg.svd", wraps=torch.linalg.svd) as svd:
            reference_fd(a, 4)
            reference_count = svd.call_count
        with patch("torch.linalg.svd", wraps=torch.linalg.svd) as svd:
            fast_fd(a, 4)
            fast_count = svd.call_count
        self.assertEqual(reference_count, a.shape[0])
        self.assertLess(fast_count, reference_count // 2)

    def test_merge_covariance_invariants(self):
        a = cases(torch.float64)["random_tall"]
        for fn in (reference_fd, fast_fd):
            merged = fn(torch.cat([fn(a[:20], 4), fn(a[20:], 4)]), 4)
            self.assertTrue(invariants(a, merged, 4)["pass"])

    def test_large_finite_values_do_not_overflow_shrink_square(self):
        for fn in (reference_fd, fast_fd):
            a = torch.diag(torch.tensor([4e20, 3e20, 2e20, 1e20], dtype=torch.float32))
            b = fn(a, 2)
            self.assertTrue(bool(torch.isfinite(b).all()))
            self.assertTrue(invariants(a, b, 2)["pass"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
