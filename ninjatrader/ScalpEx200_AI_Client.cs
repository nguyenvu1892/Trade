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
        
        // 2. Goi gRPC Bat dong bo (Tranh lam do Main Thread)
        Task.Run(async () => {
            try {
                var response = await grpcClient.EvaluateCandleAsync(request);
                Dispatcher.InvokeAsync(() => {
                    if (State != State.Realtime) return; // Bo qua bar lich su
                    
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
                Print("gRPC Error: " + e.Message);
            }
        });
    }
}
