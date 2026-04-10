import pandas as pd
import numpy as np
import torch
from collections import deque
from protos import nt8_bridge_pb2 as pb2
from protos import nt8_bridge_pb2_grpc as pb2_grpc
import grpc
from concurrent import futures
import sys
import logging
from pathlib import Path

# Thêm đường dẫn để import custom modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model.transformer import XAUTransformer
from src.data.data_processor import DataProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("nt8_server")

class StrategyBrainServicer(pb2_grpc.StrategyBrainServicer):
    def __init__(self):
        self.window_size = 256
        self.warmup_bars = 24
        self.required_bars = self.window_size + self.warmup_bars
        self.candles_buffer = deque(maxlen=self.required_bars * 2)
        
        self.device = torch.device("cpu")
        self.model = XAUTransformer(
            n_features=13,
            window_size=self.window_size,
            d_model=256,
            n_heads=8,
            n_layers=6,
        ).to(self.device)
        
        ckpt = torch.load("checkpoints/best_model_bc.pt", map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            self.model.load_state_dict(ckpt["model_state"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()
        
        self.processor = DataProcessor(atr_period=14)
        n_params = sum(p.numel() for p in self.model.parameters())
        log.info(f"🧠 Model loaded: best_model_bc.pt ({n_params:,} params)")
    
    def EvaluateCandle(self, request, context):
        new_candle = {
            'datetime': pd.to_datetime(request.time, utc=True) if request.time else pd.Timestamp.now(tz='UTC'), 
            'open': request.open, 
            'high': request.high, 
            'low': request.low, 
            'close': request.close, 
            'tick_volume': request.volume
        }
        self.candles_buffer.append(new_candle)
        
        if len(self.candles_buffer) < self.required_bars:
            if len(self.candles_buffer) % 50 == 0:
                log.info(f"Historical Injection: Nhận nến thứ {len(self.candles_buffer)}...")
            return pb2.ActionResponse(
                action=pb2.ActionResponse.HOLD,
                confidence=0.0,
                message=f"Warmup... {len(self.candles_buffer)}/{self.required_bars}"
            )
            
        if len(self.candles_buffer) == self.required_bars:
            log.info(f"✅ HOÀN TẤT WARMUP: Đã nhận đủ {self.required_bars} nến. Chờ dữ liệu Live!")
            # Trả về hold cho ngay nến kết thúc warmup (vẫn là historical bar thường)
            return pb2.ActionResponse(action=pb2.ActionResponse.HOLD, confidence=0.0, message="Warmup Complete")
            
        # --- MODEL INFERENCE ---
        df = pd.DataFrame(list(self.candles_buffer))
        df.set_index('datetime', inplace=True)
        # Sửa lỗi timezone: DataProcessor mặc định cần timezone UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        try:
            features_df = self.processor.compute_features(df)
            features_np = features_df.values.astype(np.float32)
        except Exception as e:
            log.error(f"❌ Feature computation error: {e}")
            return pb2.ActionResponse(action=pb2.ActionResponse.HOLD, confidence=0.0, message="Feature Error")

        if len(features_np) < self.window_size:
            return pb2.ActionResponse(action=pb2.ActionResponse.HOLD, confidence=0.0, message="Insufficient valid features")
            
        window = features_np[-self.window_size:]
        tensor = torch.tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            
        action_idx = int(np.argmax(probs))
        confidence = float(probs[action_idx])
        
        action_names = {0: "CLOSE_ALL", 1: "BUY", 2: "SELL"}
        log.info(f"🟢 [LIVE] Inference Close: {request.close:8.2f} | Out: {action_names.get(action_idx)} ({confidence*100:4.1f}%) | H:{probs[0]*100:3.0f}% B:{probs[1]*100:3.0f}% S:{probs[2]*100:3.0f}%")

        # Ánh xạ kết quả sang gRPC Action
        pb_action = pb2.ActionResponse.HOLD
        
        if action_idx == 0:
            pb_action = pb2.ActionResponse.CLOSE_ALL
        elif action_idx == 1 and confidence >= 0.45:
            pb_action = pb2.ActionResponse.BUY
        elif action_idx == 2 and confidence >= 0.45:
            pb_action = pb2.ActionResponse.SELL

        # --- SIGNAL BRIDGE TO EXNESS ---
        signal_data = {
            "timestamp": pd.Timestamp.now(tz='UTC').isoformat(),
            "candle_time": new_candle['datetime'].isoformat(),
            "close_price": request.close,
            "action_id": action_idx,
            "action_name": str(action_names.get(action_idx)),
            "confidence": confidence,
            "probs": {
                "HOLD": float(probs[0]),
                "BUY": float(probs[1]),
                "SELL": float(probs[2])
            }
        }
        
        signal_file = Path("logs/nt8_signal.json")
        try:
            with open(signal_file, "w", encoding="utf-8") as f:
                import json
                json.dump(signal_data, f, indent=2)
        except Exception as e:
            log.error(f"❌ Khong the ghi nt8_signal.json: {e}")

        return pb2.ActionResponse(
            action=pb_action, 
            confidence=confidence, 
            message=str(action_names.get(action_idx))
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_StrategyBrainServicer_to_server(StrategyBrainServicer(), server)
    server.add_insecure_port('0.0.0.0:50051')
    log.info("Listening on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
