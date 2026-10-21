"""CPU-only research candidate; never selected by the public package."""
import torch

from torchfrequentdirections import FrequentDirections


class QRSmallSVD(FrequentDirections):
    def __init__(self, *args, **kwargs):
        if torch.device(kwargs.get("device", "cpu")).type != "cpu":
            raise ValueError("QRSmallSVD is a CPU-only experiment")
        super().__init__(*args, **kwargs)

    def _factor(self, rows, *, shrink):
        if rows.shape[0] == 0 or rows.shape[0] > rows.shape[1]:
            return super()._factor(rows, shrink=shrink)
        q, r = torch.linalg.qr(rows.T, mode="reduced")
        _, singular, small_vh = torch.linalg.svd(r.T, full_matrices=False)
        vh = small_vh @ q.T
        output = rows.new_zeros((self.ell, self.n_features))
        if shrink and singular.numel() >= self.ell:
            scale = singular[0].clamp_min(torch.finfo(singular.dtype).tiny)
            normalized = singular / scale
            singular = scale * (normalized.square() -
                                normalized[self.ell - 1].square()).clamp_min(0).sqrt()
        keep = min(self.ell, singular.numel())
        output[:keep] = singular[:keep, None] * vh[:keep]
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("SVD produced a non-finite sketch")
        return output
