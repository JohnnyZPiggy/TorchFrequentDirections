import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fd_gpu_driver', ROOT/'experiments/fd_gpu/driver.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
driver.torch = torch


class GPUDriverTests(unittest.TestCase):
    def test_oracle_rejects_bad_partial_truncation(self):
        a = torch.tensor([[.01,0.,0.],[.01,0.,0.],[0.,10.,0.]],dtype=torch.float64)
        self.assertEqual(driver.oracle(a,a[:2],2)['status'],'FAIL')

    def test_install_rejects_source_mismatch_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'LICENSE').write_text('wrong')
            with patch.object(driver.urllib.request,'urlopen',side_effect=AssertionError('network')):
                with self.assertRaisesRegex(RuntimeError,'SOURCE_MISMATCH'):
                    driver.install_source('0'*40,root,local=root)

    def test_measurement_counts_and_quantiles(self):
        calls = []
        report = driver.measure(lambda:calls.append(1),'cpu',warmup=2,repeats=5)
        self.assertEqual(len(calls),7)
        self.assertEqual(report['p95_ms'],max(report['samples_ms']))
        self.assertGreaterEqual(report['median_ms'],0)

    def test_both_methods_profile_the_actual_update(self):
        from torchfrequentdirections import reference_fd,fast_fd
        a = torch.eye(4,dtype=torch.float64)
        # Fast also canonicalizes the one-row residual during snapshot.
        for fn,expected in ((reference_fd,4),(fast_fd,2)):
            report = driver.profile(fn,a,2)
            self.assertEqual(report['svd_calls'],expected)
            self.assertIn('standalone',report['scope'])


if __name__ == '__main__':
    unittest.main()
