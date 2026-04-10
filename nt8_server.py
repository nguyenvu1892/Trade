import pandas as pd
from collections import deque
import nt8_bridge_pb2 as pb2
import nt8_bridge_pb2_grpc as pb2_grpc
import grpc
from concurrent import futures
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("nt8_server")

class StrategyBrainServicer(pb2_grpc.StrategyBrainServicer):
    def __init__(self):
        self.window_size = 128
        self.warmup_bars = 14
        self.required_bars = self.window_size + self.warmup_bars
        self.candles_buffer = deque(maxlen=self.required_bars * 2)
    
    def EvaluateCandle(self, request, context):
        new_candle = {
            'time': request.time, 'open': request.open, 
            'high': request.high, 'low': request.low, 
            'close': request.close, 'tick_volume': request.volume
        }
        self.candles_buffer.append(new_candle)
        
        if len(self.candles_buffer) < self.required_bars:
            return pb2.ActionResponse(
                action=pb2.ActionResponse.HOLD,
                confidence=0.0,
                message=f"Warmup... {len(self.candles_buffer)}/{self.required_bars}"
            )
            
        # Model integration will happen in future subagents
        return pb2.ActionResponse(action=pb2.ActionResponse.HOLD, confidence=0.0, message="Hold")

def serve():
    server = grpc.server(futures.ThreadPoolAddress(max_workers=10))
    pb2_grpc.add_StrategyBrainServicer_to_server(StrategyBrainServicer(), server)
    server.add_insecure_port('[::]:50051')
    log.info("Listening on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
