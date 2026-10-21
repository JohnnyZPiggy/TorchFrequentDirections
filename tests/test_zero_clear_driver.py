import importlib.util
from pathlib import Path
import unittest

import torch

from test_snapshot_driver import driver as support

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('clear_driver',ROOT/'experiments/zero_clear/driver.py')
driver=importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
driver.torch=support.torch=torch


class ZeroClearDriverTests(unittest.TestCase):
    def test_balanced_pairs_have_equal_work_and_raw_values(self):
        calls=[]
        result=driver.paired({name:(lambda name=name:calls.append(name)) for name in ('baseline','candidate')},'cpu')
        self.assertEqual(calls.count('baseline'),22)
        self.assertEqual(calls.count('candidate'),22)
        self.assertEqual(sum(order[0]=='baseline' for order in result['order']),10)
        self.assertEqual(sum(order[0]=='candidate' for order in result['order']),10)
        self.assertEqual(len(result['paired_ratios']),20)
        for name in ('baseline','candidate'):
            self.assertEqual(result[name]['p95_ms'],sorted(result[name]['samples_ms'])[18])

    def test_clear_counter_is_specific_and_does_not_count_new_zeros(self):
        value=torch.ones(8,7)
        actual=driver.profile(lambda:value.zero_(),'cpu',support)
        other=driver.profile(lambda:value.new_zeros(value.shape),'cpu',support)
        self.assertEqual(actual['clear_calls'],1)
        self.assertEqual(other['clear_calls'],0)
        self.assertEqual(actual['svd_calls'],0)
        self.assertEqual(actual['finite_calls'],0)


if __name__=='__main__': unittest.main()
