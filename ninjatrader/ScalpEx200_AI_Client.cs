// Create ninjatrader/ScalpEx200_AI_Client.cs
// Require Copy to: Documents/NinjaTrader 8/bin/Custom/Strategies/
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.Cbi;
using System.Threading.Tasks;
using System;
using Nt8Bridge; // Namespace created by protoc
using Grpc.Core; // Chua cac object Channel, ChannelCredentials
using System.Collections.Generic;
using System.Linq;

namespace NinjaTrader.NinjaScript.Strategies
{
    public class ScalpEx200_AI_Client : Strategy {
        public bool EnforceCutoffTime { get; set; } = true;
        
        // === THONG SO QUY $50K ===
        public double AccountSize { get; set; } = 50000.0;
        public double MaxDrawdown { get; set; } = 2500.0;    // Fail thi
        public double TargetPass { get; set; } = 3000.0;     // Pass thi
        
        // === QUAN TRI RUI RO TUNG LENH ===
        public double RiskPerTradePercent { get; set; } = 0.3;  // 0.3% = $150/lenh
        public int MaxConcurrentTrades { get; set; } = 2;       // Max 2 vi the cung luc
        public int StopLossTicks { get; set; } = 150;           // (Fallback) $150 / $1.00 tick = 150 ticks MGC
        public double SLAtrMultiplier { get; set; } = 1.0;
        public int MinStopLossTicks { get; set; } = 20;
        public int MaxQuantity { get; set; } = 10;
        
        // === TRAILING STOP (Clone tu Exness live_bot.py) ===
        public double TrailingActivateATR { get; set; } = 1.5;  // Kich hoat khi lai > 1.5 ATR
        public double TrailingDistanceATR { get; set; } = 0.5;  // Trail bam cach gia 0.5 ATR
        public int ATRPeriod { get; set; } = 14;
        
        // gRPC Client
        private StrategyBrain.StrategyBrainClient grpcClient;
        private Channel channel;
        private ATR atrIndicator;
        
        // --- CUSTOM INDICATORS FOR AI ---
        private double cumVolume = 0;
        private double cumTypicalPriceVol = 0;
        private Queue<double> volQueue = new Queue<double>();
        
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
            }
            else if (State == State.Configure) {
                Log("ScalpEx200: Enter State.Configure", LogLevel.Information);
                // KHOA CUNG Stop Loss ban dau (Trailing se ghi de khi co lai)
                SetStopLoss(CalculationMode.Ticks, StopLossTicks); // 150 ticks = $150
            }
            else if (State == State.DataLoaded) {
                Log("ScalpEx200: Enter State.DataLoaded", LogLevel.Information);
                atrIndicator = ATR(ATRPeriod); // Khoi tao ATR indicator
                try {
                    Log("ScalpEx200: Connecting to python gRPC...", LogLevel.Information);
                    channel = new Channel("127.0.0.1:50051", ChannelCredentials.Insecure);
                    grpcClient = new StrategyBrain.StrategyBrainClient(channel);
                    Log("ScalpEx200: Connected to gRPC Client!", LogLevel.Information);
                } catch (Exception ex) {
                    Log("ScalpEx200: gRPC Exception: " + ex.ToString(), LogLevel.Error);
                }
            }
            else if (State == State.Terminated) {
                if (channel != null) channel.ShutdownAsync().Wait();
            }
        }
        
        protected override void OnBarUpdate() {
            if (CurrentBar == 0) {
                Log("ScalpEx200: First OnBarUpdate triggered. CurrentBar = 0. State = " + State, LogLevel.Information);
            }
            
            if (CurrentBar < BarsRequiredToTrade || grpcClient == null) {
                if (CurrentBar % 50 == 0 && CurrentBar > 0) {
                    Log($"ScalpEx200: Skipping Bar {CurrentBar}. Required: {BarsRequiredToTrade}", LogLevel.Information);
                }
                return;
            }

            if (CurrentBar == BarsRequiredToTrade) {
                Log("ScalpEx200: Minimum BarsRequired reached! Starting gRPC evaluations.", LogLevel.Information);
            }

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
            
            // CUTOFF: Dong tat ca lenh truoc 4h sang VN (= ~3:50 PM EST / 9:50 PM UTC)
            // Bao dam khong giu lenh qua dem theo gio Viet Nam
            if (EnforceCutoffTime && ToTime(Time[0]) >= 155000 && ToTime(Time[0]) <= 160500) {
                Print("CUT-OFF 3:50 PM EST (= ~4h sang VN). Going Flat!");
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
            
            // --- OFFLINE FEATURE ENGINEERING CHO AI ---
            // 1. VWAP Distance (Reset theo phien CME)
            if (Bars.IsFirstBarOfSession) {
                cumVolume = 0;
                cumTypicalPriceVol = 0;
            }
            double typicalPrice = (High[0] + Low[0] + Close[0]) / 3.0;
            cumVolume += Volume[0];
            cumTypicalPriceVol += typicalPrice * Volume[0];
            double currentVwap = cumVolume > 0 ? (cumTypicalPriceVol / cumVolume) : Close[0];
            double vwapDist = (Close[0] - currentVwap) / currentVwap;
            
            // 2. Volume Surge (Chong NaN bang Epsilon 1e-8)
            volQueue.Enqueue(Volume[0]);
            if (volQueue.Count > 20) volQueue.Dequeue();
            double volMean = volQueue.Average();
            double epsilon = 1e-8;
            double surge = Math.Log((Volume[0] + epsilon) / (volMean + epsilon));
            if (surge > 5.0) surge = 5.0;
            if (surge < -5.0) surge = -5.0;
            
            var request = new CandleRequest {
                Time = Time[0].ToString("o"),
                Open = Open[0],
                High = High[0],
                Low = Low[0],
                Close = Close[0],
                Volume = Volume[0],
                CurrentPnl = Position.GetUnrealizedProfitLoss(PerformanceUnit.Currency, Close[0]),
                CurrentPosition = Position.MarketPosition == MarketPosition.Long ? 1 :
                                  Position.MarketPosition == MarketPosition.Short ? -1 : 0,
                VwapDistance = vwapDist,
                VolumeSurge = surge
            };

            // [LỊCH SỬ] Gửi đồng bộ (Sync)
            if (State == State.Historical) {
                try {
                    grpcClient.EvaluateCandle(request); // Blocking call
                    if (CurrentBar % 50 == 0) {
                        Log($"ScalpEx200: Sent 50 historical bars. CurrentBar={CurrentBar}", LogLevel.Information);
                    }
                } catch (Exception ex) {
                    Log("ScalpEx200: Historical gRPC Error: " + ex.ToString(), LogLevel.Error);
                }
                return;
            }

            // [THỜI GIAN THỰC] Gửi đồng bộ để đảm bảo EnterLong chạy đúng trên luồng NinjaScript Core.
            // Vì gọi local gRPC siêu nhanh (~1-5ms) nên không lo đơ UI.
            if (State == State.Realtime) {
                try {
                    var response = grpcClient.EvaluateCandle(request);

                    if (response.Action == ActionResponse.Types.Action.CloseAll) {
                        if (Position.MarketPosition == MarketPosition.Long)
                            ExitLong("AI_Exit", "Entry");
                        else if (Position.MarketPosition == MarketPosition.Short)
                            ExitShort("AI_Exit", "Entry");
                        return;
                    }

                    if (response.Action == ActionResponse.Types.Action.Buy || response.Action == ActionResponse.Types.Action.Sell) {
                        bool isBuy = response.Action == ActionResponse.Types.Action.Buy;
                        Print($"ScalpEx200: AI Sent {(isBuy ? "BUY" : "SELL")}! MarketPosition={Position.MarketPosition}, Qty={Position.Quantity}");
                        
                        var expectedPosition = isBuy ? MarketPosition.Long : MarketPosition.Short;
                        if (Position.MarketPosition != expectedPosition && Position.Quantity < MaxConcurrentTrades) {
                            
                            // --- DYNAMIC LOT SIZING ---
                            double atrVal = atrIndicator[0];
                            double slPriceDist = atrVal * SLAtrMultiplier;
                            int slTicks = Math.Max(MinStopLossTicks, (int)(slPriceDist / Instrument.MasterInstrument.TickSize));
                            
                            double tickValue = Instrument.MasterInstrument.PointValue * Instrument.MasterInstrument.TickSize;
                            double riskPerContract = slTicks * tickValue;
                            double riskAmount = AccountSize * (RiskPerTradePercent / 100.0);
                            
                            int calculatedQty = riskPerContract > 0 ? (int)Math.Floor(riskAmount / riskPerContract) : 1;
                            int finalQty = Math.Max(1, Math.Min(MaxQuantity, calculatedQty));
                            
                            SetStopLoss("Entry", CalculationMode.Ticks, slTicks, false);
                            
                            if (isBuy) {
                                EnterLong(finalQty, "Entry");
                                Print($"ScalpEx200: Executed EnterLong with quantity {finalQty}. (SL Ticks: {slTicks})");
                            } else {
                                EnterShort(finalQty, "Entry");
                                Print($"ScalpEx200: Executed EnterShort with quantity {finalQty}. (SL Ticks: {slTicks})");
                            }
                        }
                    }
                } catch (Exception ex) {
                    Print("Realtime gRPC Eval Error: " + ex.Message);
                }
            }
        }
    }
}
