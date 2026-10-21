"""Pure tensor subgraph and explicit partial-compile Direct FD experiment."""
import torch

from torchfrequentdirections import FrequentDirections


@torch.no_grad()
def tensor_shrink(singular, vh, ell, do_shrink=True):
    output=vh.new_zeros((ell,vh.shape[1]))
    if singular.numel()==0:
        return output
    if do_shrink and singular.numel()>=ell:
        scale=singular[0].clamp_min(torch.finfo(singular.dtype).tiny)
        normalized=singular/scale
        singular=scale*(normalized.square()-normalized[ell-1].square()).clamp_min(0).sqrt()
    keep=min(ell,singular.numel())
    output[:keep]=singular[:keep,None]*vh[:keep]
    return output


class TensorShrinkFD(FrequentDirections):
    """Only tensor_shrink may compile; validation/SVD/state remain eager."""
    def __init__(self,*args,shrink_fn=tensor_shrink,**kwargs):
        super().__init__(*args,**kwargs)
        self.shrink_fn=shrink_fn

    def _factor(self,rows,*,shrink):
        if len(rows)==0:
            return rows.new_zeros((self.ell,self.n_features))
        _,singular,vh=torch.linalg.svd(rows,full_matrices=False)
        output=self.shrink_fn(singular,vh,self.ell,shrink)
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError('SVD produced a non-finite sketch')
        return output


def guarded_factor(rows,ell):
    return TensorShrinkFD(ell,rows.shape[1],dtype=rows.dtype,device=rows.device)._factor(rows,shrink=True)


def whole_stream(rows,ell):
    return TensorShrinkFD(ell,rows.shape[1],dtype=rows.dtype,device=rows.device).update(rows).sketch()
