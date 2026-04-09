"""
live_bot.py — ScalpEx200 M5 Sniper Live Trading Bot v2
=======================================================
PHASE 2 UPGRADES:
  ✅ Trailing Stop (ATR-based, check mỗi 30s)
  ✅ Dynamic Lot Sizing (Half-Kelly Criterion)

Chạy tự động trên Exness MT5 Pro Account.

Model:      ppo_best.pt (Sharpe 6.85)
Timeframe:  M5 (256-bar rolling window)
Confidence: ≥ 45% mới vào lệnh
Lot:        Dynamic (Kelly) hoặc Fixed 0.01
SL/TP:      Trailing Stop khi lãi > 1 ATR

Circuit Breaker:
  - Daily Loss > 15% equity → Tắt bot
  - Max Drawdown > 30% từ đỉnh → Tắt bot
  - 10 lệnh thua liên tiếp → Tắt bot

Cách chạy:
  1. Mở MT5 Exness Terminal, đăng nhập tài khoản Pro
  2. python live_bot.py
  3. Bot tự chạy 24/5, ngủ khi thị trường đóng cửa

Ctrl+C để dừng an toàn.
"""

import sys
import time
import csv
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import torch
import MetaTrader5 as mt5

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model.transformer import XAUTransformer
from src.data.data_processor import DataProcessor

# ═══════════════════════════════════════════════════════════════
# CẤU HÌNH — Thay đổi ở đây nếu cần
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    # ── Tài khoản ──
    "symbol":           "XAUUSD",
    "magic_number":     200500,

    # ── Model ──
    "checkpoint":       "checkpoints/ppo_best.pt",
    "window_size":      256,
    "d_model":          256,
    "n_heads":          8,
    "n_layers":         6,
    "confidence_min":   0.45,

    # ── Trailing Stop ──
    "trailing_activate_atr": 1.5,    # Kích hoạt trailing khi lãi > 1.5 ATR
    "trailing_distance_atr": 0.5,    # Trailing cách giá 0.5 ATR
    "trailing_check_interval": 30,   # Check giá mỗi 30 giây

    # ── Dynamic Lot Sizing (Kelly) ──
    "lot_mode":         "kelly",     # "fixed" hoặc "kelly"
    "lot_fixed":        0.01,        # Lot cố định (dùng khi mode=fixed hoặc chưa đủ data)
    "lot_min":          0.01,        # Lot tối thiểu
    "lot_max":          0.10,        # Lot tối đa (cap cứng)
    "kelly_fraction":   0.5,         # Half-Kelly (an toàn hơn Full Kelly)
    "kelly_min_trades": 30,          # Cần ít nhất 30 trades để tính Kelly
    "margin_per_lot":   1000.0,      # Margin ước tính cho 1 lot XAUUSD

    # ── Circuit Breaker ──
    "daily_loss_pct":   15.0,
    "max_dd_pct":       30.0,
    "max_consec_loss":  10,

    # ── Timing ──
    "candle_seconds":   300,
    "extra_wait":       5,
    "bars_to_fetch":    280,
}

# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "live_bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("ScalpEx200")

# ═══════════════════════════════════════════════════════════════
# TRADE JOURNAL
# ═══════════════════════════════════════════════════════════════

JOURNAL_PATH = LOG_DIR / "live_trade_journal.csv"
JOURNAL_FIELDS = [
    "timestamp", "trade_id", "direction", "entry_price", "exit_price",
    "pnl_usd", "duration_min", "confidence", "equity_before", "equity_after",
    "result", "close_reason", "lot_size",
]


def init_journal():
    if not JOURNAL_PATH.exists():
        with open(JOURNAL_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
            writer.writeheader()
        log.info(f"📝 Tạo Trade Journal: {JOURNAL_PATH}")


def write_journal(trade: dict):
    with open(JOURNAL_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        writer.writerow(trade)


# ═══════════════════════════════════════════════════════════════
# KELLY LOT CALCULATOR
# ═══════════════════════════════════════════════════════════════

class KellyLotCalculator:
    """Tính lot size dựa trên Half-Kelly Criterion từ trade history."""

    def __init__(self, config: dict):
        self.config = config
        self._cached_lot = config["lot_fixed"]
        self._last_calc_date = None

    def get_lot(self, equity: float) -> float:
        """Trả về lot size tối ưu. Tính lại mỗi ngày."""
        if self.config["lot_mode"] == "fixed":
            return self.config["lot_fixed"]

        today = datetime.now(timezone.utc).date()
        if self._last_calc_date == today:
            return self._cached_lot

        # Tính Kelly từ trade journal
        lot = self._calc_kelly(equity)
        self._cached_lot = lot
        self._last_calc_date = today
        return lot

    def _calc_kelly(self, equity: float) -> float:
        """Tính Half-Kelly lot size."""
        cfg = self.config

        if not JOURNAL_PATH.exists():
            log.info(f"💰 Kelly: Chưa có journal → dùng lot mặc định {cfg['lot_fixed']}")
            return cfg["lot_fixed"]

        try:
            df = pd.read_csv(JOURNAL_PATH)
        except Exception:
            return cfg["lot_fixed"]

        if len(df) < cfg["kelly_min_trades"]:
            log.info(f"💰 Kelly: Mới {len(df)} trades (cần {cfg['kelly_min_trades']}) → lot={cfg['lot_fixed']}")
            return cfg["lot_fixed"]

        # Lấy N trades gần nhất
        recent = df.tail(50)
        wins = recent[recent["pnl_usd"] > 0]["pnl_usd"]
        losses = recent[recent["pnl_usd"] <= 0]["pnl_usd"]

        if len(wins) == 0 or len(losses) == 0:
            return cfg["lot_fixed"]

        win_rate = len(wins) / len(recent)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())

        if avg_win <= 0:
            return cfg["lot_fixed"]

        # Kelly formula: K = (W × R - L) / R, where R = avg_win/avg_loss
        kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        half_kelly = kelly_pct * cfg["kelly_fraction"]

        if half_kelly <= 0:
            log.warning(f"💰 Kelly âm ({kelly_pct:.3f}) → WR={win_rate:.1%}, AvgW=${avg_win:.2f}, AvgL=${avg_loss:.2f}")
            log.warning(f"   Dùng lot tối thiểu {cfg['lot_min']}")
            return cfg["lot_min"]

        # Lot = Kelly% × Equity / Margin
        raw_lot = half_kelly * equity / cfg["margin_per_lot"]

        # Round về 0.01
        lot = round(max(cfg["lot_min"], min(cfg["lot_max"], raw_lot)), 2)

        log.info(
            f"💰 Kelly: WR={win_rate:.1%} AvgW=${avg_win:.2f} AvgL=${avg_loss:.2f} "
            f"→ K={kelly_pct:.3f} → HalfK={half_kelly:.3f} → Lot={lot}"
        )
        return lot


# ═══════════════════════════════════════════════════════════════
# TRAILING STOP
# ═══════════════════════════════════════════════════════════════

class TrailingStop:
    """Virtual Trailing Stop dựa trên ATR."""

    def __init__(self, config: dict):
        self.config = config
        self.active = False
        self.trail_price = 0.0
        self.atr = 0.0
        self.direction = 0  # 1=long, -1=short

    def reset(self, direction: int, atr: float):
        """Reset khi mở lệnh mới."""
        self.active = False
        self.trail_price = 0.0
        self.atr = atr
        self.direction = direction
        log.info(f"📐 Trailing Stop reset: ATR=${atr:.2f}, activate>${self.config['trailing_activate_atr']*atr:.2f}")

    def update(self, current_price: float, entry_price: float) -> bool:
        """
        Cập nhật trailing stop. Trả về True nếu cần đóng lệnh.

        Parameters
        ----------
        current_price : giá hiện tại (bid cho Buy, ask cho Sell)
        entry_price   : giá vào lệnh

        Returns
        -------
        True nếu trailing stop bị chạm → đóng lệnh
        """
        if self.atr <= 0:
            return False

        activate_dist = self.config["trailing_activate_atr"] * self.atr
        trail_dist = self.config["trailing_distance_atr"] * self.atr

        if self.direction == 1:  # LONG
            unrealized = current_price - entry_price
            if unrealized >= activate_dist:
                new_trail = current_price - trail_dist
                if not self.active:
                    self.active = True
                    self.trail_price = new_trail
                    log.info(f"🎯 Trailing ACTIVATED (Long): trail=${self.trail_price:.2f}")
                elif new_trail > self.trail_price:
                    self.trail_price = new_trail
                    log.debug(f"🎯 Trail updated: ${self.trail_price:.2f}")

                if current_price <= self.trail_price:
                    log.info(f"🔔 TRAILING STOP HIT! Price ${current_price:.2f} ≤ Trail ${self.trail_price:.2f}")
                    return True

        elif self.direction == -1:  # SHORT
            unrealized = entry_price - current_price
            if unrealized >= activate_dist:
                new_trail = current_price + trail_dist
                if not self.active:
                    self.active = True
                    self.trail_price = new_trail
                    log.info(f"🎯 Trailing ACTIVATED (Short): trail=${self.trail_price:.2f}")
                elif new_trail < self.trail_price:
                    self.trail_price = new_trail
                    log.debug(f"🎯 Trail updated: ${self.trail_price:.2f}")

                if current_price >= self.trail_price:
                    log.info(f"🔔 TRAILING STOP HIT! Price ${current_price:.2f} ≥ Trail ${self.trail_price:.2f}")
                    return True

        return False


# ═══════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════

class CircuitBreaker:
    def __init__(self, initial_equity: float, config: dict):
        self.peak_equity = initial_equity
        self.daily_start_equity = initial_equity
        self.daily_start_date = datetime.now(timezone.utc).date()
        self.consec_losses = 0
        self.config = config
        self.tripped = False
        self.trip_reason = ""

    def update_equity(self, equity: float):
        self.peak_equity = max(self.peak_equity, equity)

    def on_trade_result(self, pnl: float, equity: float):
        self.update_equity(equity)
        if pnl < 0:
            self.consec_losses += 1
        else:
            self.consec_losses = 0
        today = datetime.now(timezone.utc).date()
        if today != self.daily_start_date:
            self.daily_start_equity = equity
            self.daily_start_date = today

    def is_safe(self, equity: float) -> bool:
        if self.tripped:
            return False

        today = datetime.now(timezone.utc).date()
        if today != self.daily_start_date:
            self.daily_start_equity = equity
            self.daily_start_date = today

        daily_loss_pct = (self.daily_start_equity - equity) / self.daily_start_equity * 100
        if daily_loss_pct > self.config["daily_loss_pct"]:
            self.tripped = True
            self.trip_reason = f"⛔ DAILY LOSS: -{daily_loss_pct:.1f}% (limit {self.config['daily_loss_pct']}%)"
            log.critical(self.trip_reason)
            return False

        dd_pct = (self.peak_equity - equity) / self.peak_equity * 100
        if dd_pct > self.config["max_dd_pct"]:
            self.tripped = True
            self.trip_reason = f"⛔ MAX DD: -{dd_pct:.1f}% from ${self.peak_equity:.2f} (limit {self.config['max_dd_pct']}%)"
            log.critical(self.trip_reason)
            return False

        if self.consec_losses >= self.config["max_consec_loss"]:
            self.tripped = True
            self.trip_reason = f"⛔ CONSEC LOSS: {self.consec_losses} (limit {self.config['max_consec_loss']})"
            log.critical(self.trip_reason)
            return False

        return True


# ═══════════════════════════════════════════════════════════════
# MT5 HELPERS
# ═══════════════════════════════════════════════════════════════

def connect_mt5():
    if not mt5.initialize():
        log.error(f"❌ Không thể kết nối MT5: {mt5.last_error()}")
        return False
    account = mt5.account_info()
    log.info(f"✅ MT5 Connected")
    log.info(f"   Account: {account.login} ({account.server})")
    log.info(f"   Balance: ${account.balance:.2f}  |  Equity: ${account.equity:.2f}")
    log.info(f"   Leverage: 1:{account.leverage}")
    return True


def get_equity():
    info = mt5.account_info()
    return info.equity if info else 0.0


def get_current_price(symbol: str, direction: str) -> float:
    """Lấy giá hiện tại theo hướng (bid cho Long, ask cho Short)."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return 0.0
    return tick.bid if direction == "BUY" else tick.ask


def fetch_m5_bars(symbol: str, count: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, count)
    if rates is None or len(rates) == 0:
        log.error(f"❌ Không lấy được nến M5: {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.rename(columns={"time": "datetime"}, inplace=True)
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    return df


def calc_atr(bars_df: pd.DataFrame, period: int = 14) -> float:
    """Tính ATR từ DataFrame nến."""
    if bars_df is None or len(bars_df) < period + 1:
        return 0.0
    high = bars_df["high"].values
    low = bars_df["low"].values
    close = bars_df["close"].values
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    return float(np.mean(tr[-period:]))


def get_open_position(symbol: str, magic: int):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return None
    for pos in positions:
        if pos.magic == magic:
            return pos
    return None


def send_market_order(symbol: str, direction: str, lot: float, magic: int) -> bool:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error(f"❌ Không lấy được giá tick: {mt5.last_error()}")
        return False

    price = tick.ask if direction == "BUY" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "magic": magic,
        "comment": "ScalpEx200_v2",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        log.error(f"❌ order_send None: {mt5.last_error()}")
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"❌ Lệnh từ chối: {result.retcode} — {result.comment}")
        return False

    log.info(f"✅ {direction} {lot} lot @ {result.price:.2f} (#{result.order})")
    return True


def close_position(position, symbol: str) -> float:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return 0.0

    if position.type == mt5.ORDER_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": position.volume,
        "type": order_type,
        "price": price,
        "position": position.ticket,
        "magic": position.magic,
        "comment": "ScalpEx200_Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"❌ Đóng lệnh thất bại: {result}")
        return 0.0

    pnl = position.profit
    direction = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
    log.info(f"🔒 Đóng {direction} @ {result.price:.2f} | PnL: ${pnl:+.2f}")
    return pnl


# ═══════════════════════════════════════════════════════════════
# MODEL & INFERENCE
# ═══════════════════════════════════════════════════════════════

def load_model(config: dict, device: torch.device):
    n_features = 13
    model = XAUTransformer(
        n_features=n_features,
        window_size=config["window_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
    ).to(device)

    ckpt = torch.load(config["checkpoint"], map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"🧠 Model loaded: {config['checkpoint']} ({n_params:,} params)")
    return model


def predict(model, features_np, window_size, device):
    if len(features_np) < window_size:
        return 0, 0.0, np.array([1.0, 0.0, 0.0])

    window = features_np[-window_size:]
    tensor = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    action = int(np.argmax(probs))
    confidence = float(probs[action])
    return action, confidence, probs


# ═══════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def seconds_to_next_candle():
    """Tính số giây đến khi nến M5 tiếp theo đóng."""
    now = datetime.now(timezone.utc)
    secs_in = (now.minute % 5) * 60 + now.second
    wait = CONFIG["candle_seconds"] - secs_in + CONFIG["extra_wait"]
    if wait <= 0:
        wait += CONFIG["candle_seconds"]
    return wait


def close_and_log(position, symbol, trade_count, total_pnl, entry_time,
                  entry_equity, entry_confidence, cb, close_reason, current_lot):
    """Đóng lệnh + ghi journal + cập nhật circuit breaker."""
    pnl = close_position(position, symbol)
    trade_count += 1
    total_pnl += pnl

    equity_after = get_equity()
    result = "WIN" if pnl > 0 else "LOSS"
    duration = 0
    if entry_time:
        duration = int((datetime.now(timezone.utc) - entry_time).total_seconds() / 60)

    pos_dir = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
    log.info(
        f"{'✅' if pnl > 0 else '❌'} Trade #{trade_count}: {pos_dir} {current_lot}lot | "
        f"PnL: ${pnl:+.2f} | {duration}m | Reason: {close_reason} | Total: ${total_pnl:+.2f}"
    )

    write_journal({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trade_id": trade_count,
        "direction": pos_dir,
        "entry_price": position.price_open,
        "exit_price": position.price_current,
        "pnl_usd": round(pnl, 2),
        "duration_min": duration,
        "confidence": round(entry_confidence, 4),
        "equity_before": round(entry_equity, 2),
        "equity_after": round(equity_after, 2),
        "result": result,
        "close_reason": close_reason,
        "lot_size": current_lot,
    })

    cb.on_trade_result(pnl, equity_after)
    return trade_count, total_pnl, pnl


def main():
    print("=" * 60)
    print("🤖 SCALPEX200 — LIVE TRADING BOT v2")
    print("   Model: PPO Best M5 Sniper (w256)")
    print(f"   Confidence Filter: ≥ {CONFIG['confidence_min']*100:.0f}%")
    print(f"   Trailing Stop: {CONFIG['trailing_activate_atr']} ATR activate, {CONFIG['trailing_distance_atr']} ATR trail")
    print(f"   Lot Mode: {CONFIG['lot_mode'].upper()}")
    print("=" * 60)

    # 1. Kết nối MT5
    if not connect_mt5():
        log.critical("Không thể kết nối MT5. Thoát.")
        return

    symbol = CONFIG["symbol"]
    if not mt5.symbol_select(symbol, True):
        log.critical(f"Symbol {symbol} không khả dụng!")
        mt5.shutdown()
        return

    # 2. Load Model
    device = torch.device("cpu")
    model = load_model(CONFIG, device)
    processor = DataProcessor(atr_period=14)

    # 3. Khởi tạo modules
    initial_equity = get_equity()
    cb = CircuitBreaker(initial_equity, CONFIG)
    trailing = TrailingStop(CONFIG)
    kelly = KellyLotCalculator(CONFIG)

    log.info(f"🛡️ Circuit Breaker: Daily {CONFIG['daily_loss_pct']}% | MaxDD {CONFIG['max_dd_pct']}% | Consec {CONFIG['max_consec_loss']}")
    log.info(f"🎯 Trailing Stop: Activate >{CONFIG['trailing_activate_atr']} ATR, Trail {CONFIG['trailing_distance_atr']} ATR")

    # 4. Khởi tạo Trade Journal
    init_journal()

    # 5. Tracking state
    trade_count = 0
    total_pnl = 0.0
    entry_time = None
    entry_equity = 0.0
    entry_confidence = 0.0
    current_lot = CONFIG["lot_fixed"]
    current_atr = 0.0

    log.info(f"\n🚀 Bot v2 bắt đầu! Equity: ${initial_equity:.2f}")
    log.info(f"   Symbol: {symbol} | Magic: {CONFIG['magic_number']}")

    try:
        while True:
            # ── Chờ nến M5 mới, NHƯNG check trailing stop mỗi 30s ──
            wait_total = seconds_to_next_candle()
            next_candle = datetime.now(timezone.utc) + timedelta(seconds=wait_total)
            log.info(f"⏳ Nến tiếp: {next_candle.strftime('%H:%M:%S')} UTC ({wait_total}s)")

            position = get_open_position(symbol, CONFIG["magic_number"])
            check_interval = CONFIG["trailing_check_interval"]

            while wait_total > 0:
                sleep_time = min(check_interval, wait_total)
                time.sleep(sleep_time)
                wait_total -= sleep_time

                # Kiểm tra trailing stop nếu đang có vị thế
                if position is not None and trailing.atr > 0:
                    pos_dir = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
                    price = get_current_price(symbol, pos_dir)

                    if price > 0 and trailing.update(price, position.price_open):
                        # TRAILING STOP HIT!
                        trade_count, total_pnl, _ = close_and_log(
                            position, symbol, trade_count, total_pnl,
                            entry_time, entry_equity, entry_confidence,
                            cb, "TRAILING_STOP", current_lot
                        )
                        entry_time = None
                        trailing.active = False
                        position = None  # Đã đóng
                        break

                    # Refresh position (có thể bị đóng bởi MT5 SL/TP bên ngoài)
                    position = get_open_position(symbol, CONFIG["magic_number"])
                    if position is None:
                        trailing.active = False

            # ── Kiểm tra Circuit Breaker ──
            equity = get_equity()
            if not cb.is_safe(equity):
                log.critical(f"🚨 CIRCUIT BREAKER: {cb.trip_reason}")
                break

            # ── Lấy nến M5 & tính features ──
            bars_df = fetch_m5_bars(symbol, CONFIG["bars_to_fetch"])
            if bars_df is None or len(bars_df) < CONFIG["window_size"] + 20:
                log.warning("⚠️ Không đủ dữ liệu nến")
                continue

            try:
                features_df = processor.compute_features(bars_df)
                features_np = features_df.values.astype(np.float32)
            except Exception as e:
                log.error(f"❌ Lỗi features: {e}")
                continue

            current_atr = calc_atr(bars_df)

            # ── Model Inference ──
            action, confidence, probs = predict(model, features_np, CONFIG["window_size"], device)
            action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
            current_price = float(bars_df["close"].iloc[-1])

            # ── Tính lot cho ngày hôm nay ──
            current_lot = kelly.get_lot(equity)

            log.info(
                f"📊 {symbol} ${current_price:.2f} | "
                f"{action_names[action]} ({confidence*100:.1f}%) | "
                f"H:{probs[0]*100:.0f}% B:{probs[1]*100:.0f}% S:{probs[2]*100:.0f}% | "
                f"Eq:${equity:.2f} | Lot:{current_lot} | ATR:${current_atr:.2f}"
            )

            # ── Kiểm tra vị thế ──
            position = get_open_position(symbol, CONFIG["magic_number"])

            if position is not None:
                pos_dir = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
                pos_pnl = position.profit

                need_close = False
                close_reason = ""

                if action == 0:
                    need_close = True
                    close_reason = "BOT_CLOSE"
                elif (pos_dir == "BUY" and action == 2) or (pos_dir == "SELL" and action == 1):
                    need_close = True
                    close_reason = "REVERSAL"

                if need_close:
                    trade_count, total_pnl, _ = close_and_log(
                        position, symbol, trade_count, total_pnl,
                        entry_time, entry_equity, entry_confidence,
                        cb, close_reason, current_lot
                    )
                    entry_time = None
                    trailing.active = False

                    # Reversal: mở lệnh mới
                    if close_reason == "REVERSAL" and cb.is_safe(get_equity()):
                        new_dir = "BUY" if action == 1 else "SELL"
                        if confidence >= CONFIG["confidence_min"]:
                            if send_market_order(symbol, new_dir, current_lot, CONFIG["magic_number"]):
                                entry_time = datetime.now(timezone.utc)
                                entry_equity = get_equity()
                                entry_confidence = confidence
                                dir_int = 1 if new_dir == "BUY" else -1
                                trailing.reset(dir_int, current_atr)
                        else:
                            log.info(f"🚫 Reversal filtered: {confidence*100:.1f}% < 45%")
                else:
                    ts_status = f"Trail=${trailing.trail_price:.2f}" if trailing.active else "Inactive"
                    log.info(f"   Giữ {pos_dir} | PnL: ${pos_pnl:+.2f} | TS: {ts_status}")

            else:
                # Không có vị thế — xét mở lệnh mới
                if action in (1, 2):
                    if confidence >= CONFIG["confidence_min"]:
                        direction = "BUY" if action == 1 else "SELL"
                        if send_market_order(symbol, direction, current_lot, CONFIG["magic_number"]):
                            entry_time = datetime.now(timezone.utc)
                            entry_equity = equity
                            entry_confidence = confidence
                            dir_int = 1 if direction == "BUY" else -1
                            trailing.reset(dir_int, current_atr)
                    else:
                        log.info(
                            f"🚫 Filtered: {action_names[action]} {confidence*100:.1f}% < 45%"
                        )

    except KeyboardInterrupt:
        log.info("\n⏹️ Bot dừng bởi Ctrl+C")
    except Exception as e:
        log.critical(f"💥 Lỗi: {e}", exc_info=True)
    finally:
        position = get_open_position(symbol, CONFIG["magic_number"])
        if position is not None:
            log.info("🔒 Đóng lệnh trước khi thoát...")
            close_position(position, symbol)

        mt5.shutdown()
        log.info(f"\n📊 KẾT QUẢ:")
        log.info(f"   Trades: {trade_count} | PnL: ${total_pnl:+.2f}")
        log.info(f"   Journal: {JOURNAL_PATH}")
        log.info("👋 Bot v2 tắt an toàn.")


if __name__ == "__main__":
    main()
