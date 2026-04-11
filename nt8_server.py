import pandas as pd
import csv
import json
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
            n_features=15,
            window_size=self.window_size,
            d_model=256,
            n_heads=8,
            n_layers=6,
        ).to(self.device)
        
        try:
            ckpt = torch.load("checkpoints/best_model_cme_sniper_v2.pt", map_location=self.device, weights_only=False)
            log.info("🧠 Loaded Sniper Model: checkpoints/best_model_cme_sniper_v2.pt")
            state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        except FileNotFoundError:
            ckpt = torch.load("checkpoints/best_model_bc.pt", map_location=self.device, weights_only=False)
            log.warning("⚠️ Sniper model not found! Falling back to best_model_bc.pt and doing live surgery.")
            state_dict = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
            
            # On-the-fly Network Surgery to prevent crash
            if "input_projection.weight" in state_dict:
                old_weight = state_dict["input_projection.weight"]
                if old_weight.shape[1] == 13:
                    new_weight = self.model.input_projection.weight.data.clone()
                    new_weight[:, :13] = old_weight
                    state_dict["input_projection.weight"] = new_weight
                    
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        self.processor = DataProcessor(atr_period=14)
        n_params = sum(p.numel() for p in self.model.parameters())
        log.info(f"🧠 Model loaded: best_model_bc.pt ({n_params:,} params)")
        
        # --- DATA COLLECTOR: Lưu nến CME sạch để retrain ---
        self.data_file = Path("data/mgc_m5_cme.csv")
        self.data_file.parent.mkdir(exist_ok=True)
        self._existing_times = set()
        if self.data_file.exists():
            try:
                df_existing = pd.read_csv(self.data_file)
                self._existing_times = set(df_existing['datetime'].tolist())
                log.info(f"📦 Data Collector: Đã có {len(self._existing_times)} nến trong {self.data_file}")
            except: pass
        else:
            with open(self.data_file, 'w', newline='') as f:
                csv.writer(f).writerow(['datetime','open','high','low','close','tick_volume'])
            log.info(f"📦 Data Collector: Tạo file mới {self.data_file}")
        self._candles_saved = 0
        
        # Track NT8 position state để phát hiện close từ trailing stop C#
        self._last_nt8_position = 0
    
    def EvaluateCandle(self, request, context):
        new_candle = {
            'datetime': pd.to_datetime(request.time, utc=True) if request.time else pd.Timestamp.now(tz='UTC'), 
            'open': request.open, 
            'high': request.high, 
            'low': request.low, 
            'close': request.close, 
            'tick_volume': request.volume,
            'vwap_distance': request.vwap_distance if hasattr(request, 'vwap_distance') else 0.0,
            'volume_surge': request.volume_surge if hasattr(request, 'volume_surge') else 0.0
        }
        self.candles_buffer.append(new_candle)
        
        # Lưu nến vào CSV (chống trùng lặp bằng datetime)
        dt_str = str(new_candle['datetime'])
        if dt_str not in self._existing_times:
            self._existing_times.add(dt_str)
            with open(self.data_file, 'a', newline='') as f:
                csv.writer(f).writerow([dt_str, new_candle['open'], new_candle['high'], new_candle['low'], new_candle['close'], new_candle['tick_volume']])
            self._candles_saved += 1
            if self._candles_saved % 500 == 0:
                log.info(f"💾 Data Collector: Đã lưu {self._candles_saved} nến mới vào {self.data_file}")
        
        if len(self.candles_buffer) < self.required_bars:
            if len(self.candles_buffer) % 50 == 0:
                log.info(f"Historical Injection: Nhận nến thứ {len(self.candles_buffer)}...")
            return pb2.ActionResponse(
                action=pb2.ActionResponse.HOLD,
                confidence=0.0,
                message=f"Warmup... {len(self.candles_buffer)}/{self.required_bars}"
            )
            
        if len(self.candles_buffer) == self.required_bars:
            log.info(f"✅ HOÀN TẤT WARMUP: Đã nhận đủ {self.required_bars} nến.")
            self._warmup_done = True
            return pb2.ActionResponse(action=pb2.ActionResponse.HOLD, confidence=0.0, message="Warmup Complete")
            
        # --- Phân biệt Nến Historical vs Realtime cực kì an toàn bằng Timestamp Age ---
        # Nếu nến đang xét (dựa theo request.time) quá cũ (chênh lệch > 5 phút so với giờ hiện tại)
        # thì chắc chắn nó là nến Historical từ hệ thống Warmup của NinjaTrader.
        dt_obj_check = getattr(request, 'time', None)
        is_historical = False
        if dt_obj_check:
            try:
                c_time = pd.to_datetime(dt_obj_check)
                if c_time.tz is None:
                    c_time = c_time.tz_localize('UTC')
                # Nếu nến cũ hơn 10 phút -> Historical
                if (pd.Timestamp.now(tz='UTC') - c_time).total_seconds() > 600:
                    is_historical = True
            except:
                pass
                
        if is_historical:
            if len(self.candles_buffer) % 100 == 0:
                log.info(f"⏩ Historical processing: {len(self.candles_buffer)} nến (skip inference)")
            return pb2.ActionResponse(action=pb2.ActionResponse.HOLD, confidence=0.0, message="Historical Skip")
        
        # --- PHÁT HIỆN NT8 CLOSE ĐỘC LẬP (trailing stop, SL, v.v.) ---
        # Đã xóa: Lõi Python hiện đóng vai trò Provider, không bị động theo dõi lệnh từ NT8.
        self._last_nt8_position = request.current_position
        
        # --- MODEL INFERENCE (chỉ cho nến Realtime) ---
        df = pd.DataFrame(list(self.candles_buffer))
        df.set_index('datetime', inplace=True)
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
        
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        log.info(f"🟢 [LIVE] Inference Close: {request.close:8.2f} | Out: {action_names.get(action_idx)} ({confidence*100:4.1f}%) | H:{probs[0]*100:3.0f}% B:{probs[1]*100:3.0f}% S:{probs[2]*100:3.0f}%")

        # Ánh xạ kết quả sang gRPC Action
        pb_action = pb2.ActionResponse.HOLD
        
        if action_idx == 0:
            pb_action = pb2.ActionResponse.HOLD
        elif action_idx == 1 and confidence >= 0.40:
            pb_action = pb2.ActionResponse.BUY
        elif action_idx == 2 and confidence >= 0.40:
            pb_action = pb2.ActionResponse.SELL

        # --- SIGNAL BRIDGE TO EXNESS (dùng action ĐÃ LỌC, không dùng raw) ---
        # Map pb_action sang action_id cho Exness
        filtered_action_map = {
            pb2.ActionResponse.HOLD: (0, "HOLD"),
            pb2.ActionResponse.BUY: (1, "BUY"),
            pb2.ActionResponse.SELL: (2, "SELL"),
        }
        filtered_id, filtered_name = filtered_action_map.get(pb_action, (0, "HOLD"))
        
        signal_data = {
            "timestamp": pd.Timestamp.now(tz='UTC').isoformat(),
            "candle_time": new_candle['datetime'].isoformat(),
            "close_price": request.close,
            "action_id": filtered_id,
            "action_name": filtered_name,
            "confidence": confidence,
            "probs": {
                "HOLD": float(probs[0]),
                "BUY": float(probs[1]),
                "SELL": float(probs[2])
            }
        }
        
        try:
            with open(Path("logs/ai_signal.json"), "w", encoding="utf-8") as f:
                json.dump(signal_data, f, indent=2)
        except Exception as e:
            log.error(f"❌ Failed to write JSON signal: {e}")

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
