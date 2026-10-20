"""Small CPU benchmark CLI. No installs, networking, or report publication."""

import argparse
import json
import statistics
import time

import torch

from .fd import fast_fd, reference_fd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--ell", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if min(args.n, args.d, args.ell, args.repeat) < 1:
        parser.error("dimensions and repeat must be positive")
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(1729)
    a = torch.randn(args.n, args.d, dtype=getattr(torch, args.dtype), generator=generator)
    rows = []
    for fn in (reference_fd, fast_fd):
        for _ in range(2):
            fn(a, args.ell)
        samples = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            fn(a, args.ell)
            samples.append(time.perf_counter() - start)
        rows.append({"method": fn.__name__, "seconds": samples,
                     "median_seconds": statistics.median(samples)})
    print(json.dumps({"status": "PASS", "device": "cpu", "torch": torch.__version__,
                      "threads": 1, "seed": 1729, "warmup": 2,
                      "scope": "constructor + update + final snapshot; excludes input generation",
                      "parameters": vars(args), "measurements": rows}))


if __name__ == "__main__":
    main()
