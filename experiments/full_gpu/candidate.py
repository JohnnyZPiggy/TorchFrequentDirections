"""Issue #17 local scratch reuse; production API and object state are inherited."""
import torch
from torchfrequentdirections import FrequentDirections


class ScratchFD(FrequentDirections):
    @torch.no_grad()
    def _factor(self, rows, *, shrink):
        output = rows.new_zeros((self.ell, self.n_features))
        if rows.shape[0] == 0:
            return output
        _, singular, vh = torch.linalg.svd(rows, full_matrices=False)
        if shrink and singular.numel() >= self.ell:
            scale = singular[0].clamp_min(torch.finfo(singular.dtype).tiny)
            normalized = singular / scale
            threshold = normalized[self.ell - 1].square()
            normalized.square_().sub_(threshold).clamp_min_(0).sqrt_().mul_(scale)
            singular = normalized
        keep = min(self.ell, singular.numel())
        torch.mul(singular[:keep, None], vh[:keep], out=output[:keep])
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("SVD produced a non-finite sketch")
        return output


@torch.no_grad()
def graph_transition(rows, buffer, ell):
    """Tensor-only diagnostic, NOT strict update: no host finite/atomicity checks."""
    _, s, vh = torch.linalg.svd(rows, full_matrices=False)
    scale = s[0].clamp_min(torch.finfo(s.dtype).tiny)
    normalized = s / scale
    if s.numel() >= ell:
        s = scale * (normalized.square() - normalized[ell-1].square()).clamp_min(0).sqrt()
    count = min(ell - 1, rows.shape[1])
    buffer.zero_()
    buffer[:count].copy_(s[:count, None] * vh[:count])
    return buffer
