# Create tests/test_nt8_server.py
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import grpc
from nt8_server import StrategyBrainServicer
import nt8_bridge_pb2 as pb2

class TestServerBuffer(unittest.TestCase):
    def test_buffer_requires_warmup(self):
        servicer = StrategyBrainServicer()
        request = pb2.CandleRequest(symbol="MGC JUN26", open=100, high=101, low=99, close=100, volume=10, time="2026-04-10T10:00:00Z")
        response = servicer.EvaluateCandle(request, None)
        # Vì chưa đủ nến warmup, server phải bắt buộc HOLD
        self.assertEqual(response.action, pb2.ActionResponse.HOLD)
        self.assertTrue("Warmup" in response.message)

if __name__ == '__main__':
    unittest.main()
