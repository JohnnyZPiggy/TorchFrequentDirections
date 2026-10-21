import json
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from experiments.full_gpu import graph_worker as graph
from experiments.full_gpu.isolated_driver import invoke
from torchfrequentdirections import FrequentDirections


class IsolationTests(unittest.TestCase):
    def test_failure_timeout_and_malformed_result_do_not_block_next_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'result.json'
            bad=invoke([sys.executable,'-c','raise RuntimeError("injected")'],p,5)
            self.assertEqual(bad['result']['status'],'CHILD_EXIT_FAILED')
            slow=invoke([sys.executable,'-c','import time; time.sleep(2)'],p,.05)
            self.assertEqual(slow['result']['status'],'CHILD_TIMEOUT')
            p.write_text('{')
            malformed=invoke([sys.executable,'-c','pass'],p,5)
            self.assertEqual(malformed['result']['status'],'INVALID_CHILD_RESULT')
            good=invoke([sys.executable,'-c','import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(\'{"status":"PASS"}\')',str(p)],p,5)
            self.assertEqual(good['result']['status'],'PASS')
            self.assertEqual(good['transport']['exit_code'],0)

    def test_all_preallocated_cpu_stages_and_addresses(self):
        graph.torch=torch
        for dtype in (torch.float32,torch.float64):
            a=torch.randn(32,64,dtype=dtype,generator=torch.Generator().manual_seed(1720261021))
            work=dict(pointwise=torch.empty_like(a),u=a.new_empty((32,32)),s=a.new_empty(32),
                vh=a.new_empty((32,64)),factor=a.new_empty((16,64)),buffer=torch.empty_like(a))
            addresses={k:v.data_ptr() for k,v in work.items()}
            for stage in graph.STAGES:
                actual=graph.operation(stage,a,work,16)
                if stage=='pointwise': expected=a.abs()
                elif stage=='svd': expected=torch.linalg.svdvals(a)
                elif stage=='svd_shrink': expected=FrequentDirections(16,64,dtype=dtype)._factor(a,shrink=True)
                else: expected=FrequentDirections(16,64,dtype=dtype).update(a)._buffer
                torch.testing.assert_close(actual,expected,rtol=64*torch.finfo(dtype).eps,atol=64*torch.finfo(dtype).eps)
                self.assertEqual(addresses,{k:v.data_ptr() for k,v in work.items()})
