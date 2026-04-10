# Phase 4 NinjaTrader gRPC Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng hệ thống giao tiếp gRPC để NinjaTrader 8 ném nến sang Lõi AI Python và nhận lại lệnh (Buy/Sell/Hold), với hệ thống cắt lỗ Prop Firm tự động được nhúng thẳng vào phía C#.

**Architecture:** C# (NinjaScript Client) -> gRPC Socket (127.0.0.1:50051) -> Python (gRPC Server) -> Rolling Window Buffer (142 Nến) -> XAUTransformer Model -> C# Execution.

**Tech Stack:** Python 3.11, NinjaTrader 8.1.6.3 (C# 8.0), gRPC, Protobuf, PyTorch.

---

### Task 1: Định nghĩa giao thức Protobuf

**Files:**
- Create: `protos/nt8_bridge.proto`
- Create: `tests/test_proto_import.py`

- [ ] **Step 1: Viết test (Failing test)**

```python
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
```

- [ ] **Step 2: Chạy test và xác nhận lỗi**

Run: `python -m unittest tests/test_proto_import.py`
Expected: FAIL với lỗi Import.

- [ ] **Step 3: Viết Protobuf thuần (Minimal implementation)**

```proto
// Create protos/nt8_bridge.proto
syntax = "proto3";
package nt8bridge;

service StrategyBrain {
  rpc EvaluateCandle (CandleRequest) returns (ActionResponse) {}
}

message CandleRequest {
    string symbol = 1;
    double open = 2;
    double high = 3;
    double low = 4;
    double close = 5;
    double volume = 6;
    string time = 7;
    double current_position = 8; 
    double current_pnl = 9;
}

message ActionResponse {
    enum Action {
        HOLD = 0;
        BUY = 1;
        SELL = 2;
        CLOSE_ALL = 3;
    }
    Action action = 1;
    double confidence = 2;
    string message = 3;
}
```

- [ ] **Step 4: Cài đặt và Compile Protobuf cho Python (Sửa lỗi Absolute Import)**

Run:
```bash
pip install grpcio grpcio-tools protobuf
python -m grpc_tools.protoc -I protos --python_out=. --pyi_out=. --grpc_python_out=. protos/nt8_bridge.proto
```

- [ ] **Step 5: Chạy lại test Pass và Commit**

Run: `python -m unittest tests/test_proto_import.py`
Expected: PASS
Run: 
```bash
git add protos/nt8_bridge.proto tests/test_proto_import.py requirements.txt nt8_bridge_pb2.py nt8_bridge_pb2_grpc.py
git commit -m "feat: setup gRPC protobuf definitions root import"
```

---

### Task 2: Module Python gRPC Server & Rolling Window

**Files:**
- Create: `nt8_server.py`
- Create: `tests/test_nt8_server.py`

- [ ] **Step 1: Viết Failing test cho Warmup Buffer**

```python
# Create tests/test_nt8_server.py
import unittest
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
```

- [ ] **Step 2: Chạy test xác nhận lỗi**

Run: `python -m unittest tests/test_nt8_server.py`
Expected: FAIL.

- [ ] **Step 3: Viết code Python Server (Quy chuẩn Window Lõi AI)**

```python
# Create nt8_server.py
import pandas as pd
from collections import deque
import nt8_bridge_pb2 as pb2
import nt8_bridge_pb2_grpc as pb2_grpc

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
            
        # Tích hợp Model thật sẽ thực hiện tại subagent update
        return pb2.ActionResponse(action=pb2.ActionResponse.HOLD, confidence=0.0, message="Hold")
```

- [ ] **Step 4: Chạy test Pass**

Run: `python -m unittest tests/test_nt8_server.py`
Expected: PASS

- [ ] **Step 5: Commit**

Run:
```bash
git add nt8_server.py tests/test_nt8_server.py
git commit -m "feat: implement python gRPC server strict warmup window"
```

---

### Task 3: NinjaTrader C# Bridge Client (Phẫu thuật An toàn)

**Files:**
- Create: `ninjatrader/ScalpEx200_AI_Client.cs`

- [ ] **Step 1: Code C# Thread-safe & Quản trị Rủi ro 50K**

**Bảng tính rủi ro mỗi lệnh:**

| Thông số | Giá trị |
|---|---|
| Vốn quỹ | $50,000 |
| Risk/Trade (0.3%) | **$150** |
| MGC tick value | $1.00/tick (0.10 point) |
| → Hard SL (1 lot) | **150 ticks = $15.00 giá** |
| → Hard SL (2 lots) | **75 ticks = $7.50 giá** |
| TP (Take Profit) | **Không fix cứng** -- Trailing + AI quyết định |
| Trailing Stop | Kích hoạt khi lãi > 1.5 ATR, bám cách 0.5 ATR |
| Max lệnh thua liên tiếp trước khi chạm DD | 2500/150 = **~16 lệnh** |

```csharp
// Create ninjatrader/ScalpEx200_AI_Client.cs
// Require Copy to: Documents/NinjaTrader 8/bin/Custom/Strategies/
using NinjaTrader.NinjaScript.Strategies;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.Cbi;
using System.Threading.Tasks;
using System;
using Nt8Bridge; // Namespace created by protoc

public class ScalpEx200_AI_Client : Strategy {
    public bool EnforceCutoffTime { get; set; } = true;
    
    // === THONG SO QUY $50K ===
    public double AccountSize { get; set; } = 50000.0;
    public double MaxDrawdown { get; set; } = 2500.0;    // Fail thi
    public double TargetPass { get; set; } = 3000.0;     // Pass thi
    
    // === QUAN TRI RUI RO TUNG LENH ===
    public double RiskPerTradePercent { get; set; } = 0.3;  // 0.3% = $150/lenh
    public int MaxConcurrentTrades { get; set; } = 2;       // Max 2 vi the cung luc
    public int StopLossTicks { get; set; } = 150;           // $150 / $1.00 tick = 150 ticks MGC
    
    // === TRAILING STOP (Clone tu Exness live_bot.py) ===
    public double TrailingActivateATR { get; set; } = 1.5;  // Kich hoat khi lai > 1.5 ATR
    public double TrailingDistanceATR { get; set; } = 0.5;  // Trail bam cach gia 0.5 ATR
    public int ATRPeriod { get; set; } = 14;
    
    // gRPC Client
    private StrategyBrain.StrategyBrainClient grpcClient;
    private Channel channel;
    private ATR atrIndicator;
    
    protected override void OnStateChange() {
        if (State == State.SetDefaults) {
            Name = "ScalpEx200 AI Bridge";
            Description = "Ket noi AI Python qua gRPC cho Prop Firm $50K";
            Calculate = Calculate.OnBarClose;
            EntriesPerDirection = 1;
            IsExitOnSessionCloseStrategy = true;  // Prop firm: khong giu qua dem
            ExitOnSessionCloseSeconds = 30;
            StartBehavior = StartBehavior.WaitUntilFlat;
            RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
            BarsRequiredToTrade = 20; // Du cho ATR(14) khoi dong
            
            // KHOA CUNG Stop Loss ban dau (Trailing se ghi de khi co lai)
            SetStopLoss(CalculationMode.Ticks, StopLossTicks); // 150 ticks = $150
        }
        else if (State == State.DataLoaded) {
            atrIndicator = ATR(ATRPeriod); // Khoi tao ATR indicator
            try {
                Print("Connecting to Python AI at 127.0.0.1:50051...");
                channel = new Channel("127.0.0.1:50051", ChannelCredentials.Insecure);
                grpcClient = new StrategyBrain.StrategyBrainClient(channel);
                Print("Connected!");
            } catch (Exception ex) {
                Print("gRPC Connection Failed: " + ex.Message);
            }
        }
        else if (State == State.Terminated) {
            if (channel != null) channel.ShutdownAsync().Wait();
        }
    }
    
    protected override void OnBarUpdate() {
        if (CurrentBar < BarsRequiredToTrade || grpcClient == null) return;
        
        // 1. Tinh tong PnL bao gom ca trang thai tha noi (Unrealized)
        double currentFloating = Position.Account.Get(AccountItem.UnrealizedProfitLoss, Currency.UsDollar);
        double totalDailyPnL = SystemPerformance.AllTrades.TradesPerformance.Currency.CumProfit + currentFloating;
        
        // CIRCUIT BREAKER: Drawdown Max
        if (totalDailyPnL <= -MaxDrawdown) {
            Print("CHAM MAX DRAWDOWN $2500! CAT TOAN BO LENH.");
            if (Position.MarketPosition != MarketPosition.Flat) { ExitLong(); ExitShort(); }
            return;
        }
        
        // AUTO-PASS: Cham muc tieu
        if (totalDailyPnL >= TargetPass) {
            Print("DA PASS QUY 50K (+$3000)!!! BOT TU DONG.");
            if (Position.MarketPosition != MarketPosition.Flat) { ExitLong(); ExitShort(); }
            return;
        }
        
        // PROP FIRM CUT-OFF: 3:55 PM EST
        if (EnforceCutoffTime && ToTime(Time[0]) >= 155500 && ToTime(Time[0]) <= 160000) {
            Print("CUT-OFF 3:55 PM. Going Flat!");
            if (Position.MarketPosition != MarketPosition.Flat) { ExitLong(); ExitShort(); }
            return;
        }
        
        // TRAILING STOP LOGIC (giong Exness bot)
        if (Position.MarketPosition != MarketPosition.Flat) {
            double atrValue = atrIndicator[0];
            double unrealizedPnlPoints = 0;
            
            if (Position.MarketPosition == MarketPosition.Long)
                unrealizedPnlPoints = Close[0] - Position.AveragePrice;
            else if (Position.MarketPosition == MarketPosition.Short)
                unrealizedPnlPoints = Position.AveragePrice - Close[0];
            
            // Kich hoat trailing khi lai > 1.5 ATR
            if (unrealizedPnlPoints > atrValue * TrailingActivateATR) {
                double trailDistance = atrValue * TrailingDistanceATR;
                
                if (Position.MarketPosition == MarketPosition.Long) {
                    double newStop = Close[0] - trailDistance;
                    if (newStop > Position.AveragePrice)
                        SetStopLoss(CalculationMode.Price, newStop);
                }
                else if (Position.MarketPosition == MarketPosition.Short) {
                    double newStop = Close[0] + trailDistance;
                    if (newStop < Position.AveragePrice)
                        SetStopLoss(CalculationMode.Price, newStop);
                }
            }
        }
        
        // CHOT CHAN VI THE: Max 2 lenh cung luc
        if (Position.Quantity >= MaxConcurrentTrades) {
            return;
        }
        
        // Chuan bi du lieu nen cho AI (MGC JUN26)
        var request = new CandleRequest {
            Symbol = Instrument.FullName,
            Open = Open[0], High = High[0], Low = Low[0], Close = Close[0], Volume = Volume[0],
            Time = Time[0].ToString("o"),
            CurrentPosition = Position.MarketPosition == MarketPosition.Long ? 1 : 
                              (Position.MarketPosition == MarketPosition.Short ? -1 : 0),
            CurrentPnl = currentFloating
        };
        
        // 2. Gọi gRPC Bất đồng bộ (Tránh làm đơ Main Thread)
        Task.Run(async () => {
            try {
                var response = await grpcClient.EvaluateCandleAsync(request);
                Dispatcher.InvokeAsync(() => {
                    if (State != State.Realtime) return; // Bỏ qua bar lịch sử
                    
                    Print($"[{Time[0]:HH:mm}] AI: {response.Action} | Conf: {response.Confidence:F1}%");
                    
                    if (response.Action == ActionResponse.Types.Action.Buy) {
                        if (Position.MarketPosition == MarketPosition.Short) ExitShort();
                        if (Position.MarketPosition == MarketPosition.Flat) EnterLong();
                    }
                    else if (response.Action == ActionResponse.Types.Action.Sell) {
                        if (Position.MarketPosition == MarketPosition.Long) ExitLong();
                        if (Position.MarketPosition == MarketPosition.Flat) EnterShort();
                    }
                    else if (response.Action == ActionResponse.Types.Action.CloseAll) {
                        ExitLong(); ExitShort();
                    }
                });
            } catch (Exception e) {
                Print("⚠️ gRPC Error: " + e.Message);
            }
        });
    }
}
```

- [ ] **Step 2: Commit**

Run:
```bash
git add ninjatrader/ScalpEx200_AI_Client.cs
git commit -m "feat: complete thread-safe C# execution engine"
```
