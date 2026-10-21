"""Explicit experimental CPU/CUDA adapter; production never dispatches here."""
import torch

from experiments.qr_svd.prototype import QRSmallSVD
from torchfrequentdirections import FrequentDirections


class ExperimentalQRSmallSVD(QRSmallSVD):
    def __init__(self, *args, **kwargs):
        if torch.device(kwargs.get('device', 'cpu')).type not in ('cpu', 'cuda'):
            raise ValueError('experiment supports only CPU or CUDA')
        # Reuse the pinned device-generic algebra; retain the historic CPU guard.
        FrequentDirections.__init__(self, *args, **kwargs)
