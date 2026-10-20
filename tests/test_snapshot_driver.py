import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('snapshot_driver', ROOT/'experiments/snapshot/driver.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class SnapshotDriverTests(unittest.TestCase):
    def test_paired_trials_have_equal_work_and_reproducible_shuffled_order(self):
        calls = []
        functions = {name:(lambda name=name:calls.append(name)) for name in ('baseline','candidate')}
        first = driver.paired(functions,'cpu')
        second = driver.paired(functions,'cpu')
        self.assertEqual(first['order'],second['order'])
        self.assertEqual(len(first['order']),20)
        self.assertEqual({tuple(x) for x in first['order']},
                         {('baseline','candidate'),('candidate','baseline')})
        for name in functions:
            self.assertEqual(calls.count(name),44)
            self.assertEqual(len(first[name]['samples_ms']),20)
            self.assertEqual(first[name]['p95_ms'],sorted(first[name]['samples_ms'])[18])

    def test_pinned_baseline_rejects_wrong_bytes_before_import(self):
        with patch.object(driver.subprocess,'check_output',return_value=b'wrong source'), \
             patch.object(driver.urllib.request,'urlopen',side_effect=AssertionError('network')):
            with self.assertRaisesRegex(RuntimeError,'SOURCE_MISMATCH'):
                driver.pinned_file(driver.BASE_SHA,'torchfrequentdirections/fd.py',driver.BASE_FD_HASH,local=ROOT)


if __name__=='__main__':
    unittest.main()
