"""Run the CPU suite and emit independent numerical observations to stdout."""

import io
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_fd import cases, invariants  # noqa: E402
from torchfrequentdirections import fast_fd, reference_fd  # noqa: E402


def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    source_sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip())
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    observations = []
    for dtype in (torch.float32, torch.float64):
        for name, a in cases(dtype).items():
            for ell in sorted({1, 2, 4, a.shape[1], a.shape[1] + 2}):
                for fn in (reference_fd, fast_fd):
                    report = invariants(a, fn(a, ell), ell)
                    # Each k is checked; retain worst excess rather than repetitive k tables.
                    observations.append({"case": name, "dtype": str(dtype), "n": a.shape[0],
                        "d": a.shape[1], "ell": ell, "method": fn.__name__,
                        "k_checked": len(report["checks"]), "tolerance": report["tolerance"],
                        "energy": report["energy"], "spectral_error": report["spectral_error"],
                        "minimum_covariance_eigenvalue": report["minimum_covariance_eigenvalue"],
                        "max_covariance_bound_excess": max(c["covariance_bound"] * -1 + report["spectral_error"] for c in report["checks"]),
                        "max_projection_bound_excess": max(c["projection_residual_squared"] - c["projection_bound"] for c in report["checks"]),
                        "pass": report["pass"]})
    passed = result.wasSuccessful() and all(o["pass"] for o in observations)
    print(json.dumps({"status": "PASS" if passed else "FAIL", "source_sha": source_sha,
        "working_tree_dirty": dirty, "python": platform.python_version(),
        "torch": torch.__version__, "platform": platform.system(), "machine": platform.machine(),
        "device": "cpu", "threads": 1, "cuda_available": torch.cuda.is_available(),
        "seed": 1729, "elapsed_seconds": time.perf_counter() - start,
        "tests_run": result.testsRun, "tests_failed": len(result.failures),
        "tests_errors": len(result.errors), "unittest_log": log.getvalue(),
        "oracle": "float64 full SVD tail + symmetric Gram eigvalsh + top-k right projection; no FD oracle",
        "tolerance": "32 * eps(input_dtype) * max(n,d,2*ell,1) * ||A||_F^2",
        "observations": observations}, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
