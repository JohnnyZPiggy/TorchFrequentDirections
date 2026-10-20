"""Algorithms 2.1 / 3.1 of arXiv:1501.01711v2, with fixed-size snapshots."""

import torch


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class FrequentDirections:
    """Stream real float32/float64 rows into an ell-by-n_features sketch.

    ``method='reference'`` shrinks after every row; ``'fast'`` uses a 2*ell
    temporary buffer. Updates detach inputs, never retain an autograd graph,
    and require the configured dtype/device. Snapshots do not change state.
    Missing singular values when ell > n_features are zero (exact fallback).
    """

    def __init__(self, ell, n_features, *, method="fast", dtype=torch.float64,
                 device="cpu"):
        _positive_int(ell, "ell")
        _positive_int(n_features, "n_features")
        if method not in ("reference", "fast"):
            raise ValueError("method must be 'reference' or 'fast'")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.ell = ell
        self.n_features = n_features
        self.method = method
        self._buffer = torch.zeros((ell if method == "reference" else 2 * ell,
                                    n_features), dtype=dtype, device=device)
        self._count = 0

    def _validate(self, rows):
        if not isinstance(rows, torch.Tensor):
            raise TypeError("rows must be a torch.Tensor")
        if rows.layout != torch.strided or rows.ndim != 2:
            raise ValueError("rows must be a dense two-dimensional tensor")
        if rows.shape[1] != self.n_features:
            raise ValueError("feature dimension does not match")
        if rows.dtype != self._buffer.dtype or rows.device != self._buffer.device:
            raise ValueError("rows must match the sketch dtype and device")
        if not bool(torch.isfinite(rows).all()):
            raise ValueError("rows must be finite")

    def _factor(self, rows, *, shrink):
        output = rows.new_zeros((self.ell, self.n_features))
        if rows.shape[0] == 0:
            return output
        _, singular, vh = torch.linalg.svd(rows, full_matrices=False)
        if shrink and singular.numel() >= self.ell:
            # Normalize before squaring: finite singular values may square to inf.
            scale = singular[0].clamp_min(torch.finfo(singular.dtype).tiny)
            normalized = singular / scale
            singular = scale * (normalized.square() -
                                normalized[self.ell - 1].square()).clamp_min(0).sqrt()
        keep = min(self.ell, singular.numel())
        output[:keep] = singular[:keep, None] * vh[:keep]
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("SVD produced a non-finite sketch")
        return output

    @torch.no_grad()
    def update(self, rows):
        """Consume rows in order. Invalid input is rejected before mutation.

        A backend SVD failure may interrupt an update; discard that instance.
        """
        self._validate(rows)
        rows = rows.detach()
        if self.method == "reference":
            for row in rows:
                self._buffer[-1] = row
                self._buffer.copy_(self._factor(self._buffer, shrink=True))
            return self
        offset = 0
        while offset < rows.shape[0]:
            take = min(self._buffer.shape[0] - self._count, rows.shape[0] - offset)
            self._buffer[self._count:self._count + take] = rows[offset:offset + take]
            self._count += take
            offset += take
            if self._count == self._buffer.shape[0]:
                factor = self._factor(self._buffer, shrink=True)
                self._buffer.zero_()
                # After sigma_ell shrink only ell-1 directions can survive.
                self._count = min(self.ell - 1, self.n_features)
                self._buffer[:self._count] = factor[:self._count]
        return self

    @torch.no_grad()
    def sketch(self):
        """Return an independent ell-by-d factor, finalizing a partial buffer.

        More than ell pending rows require one extra FD shrink; truncating them
        would violate the error bound. At most ell rows can be returned exactly.
        """
        if self.method == "reference":
            return self._buffer.clone()
        return self._factor(self._buffer[:self._count], shrink=self._count > self.ell)


def _sketch(rows, ell, method):
    if not isinstance(rows, torch.Tensor):
        raise TypeError("rows must be a torch.Tensor")
    if rows.ndim != 2:
        raise ValueError("rows must be two-dimensional")
    return FrequentDirections(ell, rows.shape[1], method=method, dtype=rows.dtype,
                              device=rows.device).update(rows).sketch()


def reference_fd(rows, ell):
    """Reference row-by-row FD; returns ell rows, including trailing zeros."""
    return _sketch(rows, ell, "reference")


def fast_fd(rows, ell):
    """Buffered FD; the temporary 2*ell rows are never exposed as the sketch."""
    return _sketch(rows, ell, "fast")
