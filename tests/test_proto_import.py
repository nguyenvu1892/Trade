# Create tests/test_proto_import.py
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestProto(unittest.TestCase):
    def test_proto_compiled(self):
        try:
            import nt8_bridge_pb2 as nt8_pb2
            import nt8_bridge_pb2_grpc as nt8_pb2_grpc
            msg = nt8_pb2.ActionResponse(action=1, confidence=95.5, message="BUY")
            self.assertEqual(msg.message, "BUY")
        except ImportError:
            self.fail("Protobuf files not generated yet")

if __name__ == '__main__':
    unittest.main()
