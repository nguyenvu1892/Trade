"""
live_bot.py — Master-Slave Exness Sync Bot v3
=======================================================
Vận hành dựa trên tín hiệu gRPC từ NinjaTrader (Master) xuất ra qua JSON.
Không còn chạy Pytorch bên trong tiến trình này để giảm thiểu delay.

Tính năng:
  - Trailing Stop: Độc lập
  - Circuit Breaker: Độc lập
  - Lot Sizing (Kelly): Độc lập
  - AI Inference: Đọc từ nt8_signal.json
"""

import sys
import time
import csv
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# Cấu hình
CONFIG = {
    "symbol": "XAUUSD",
    "magic_number": 200500,
    "confidence_min": 0.45,
    "trailing_activate_atr": 1.5,
    "trailing_distance_atr": 0.5,
    "trailing_check_interval": 1, 
    "lot_mode": "kelly",
    "lot_fixed": 0.01,
    "lot_min": 0.01,
    "lot_max": 0.10,
    "kelly_fraction": 0.5,
    "kelly_min_trades": 30,
    "margin_per_lot": 1000.0,
    "daily_loss_pct": 15.0,
    "max_dd_pct": 30.0,
    "max_consec_loss": 10,
}

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "live_bot.log", encoding="utf-8")],
)
log = logging.getLogger("ExnessSync")

JOURNAL_PATH = LOG_DIR / "live_trade_journal.csv"
JOURNAL_FIELDS = ["timestamp", "trade_id", "direction", "entry_price", "exit_price", "pnl_usd", "duration_min", "confidence", "equity_before", "equity_after", "result", "close_reason", "lot_size"]

def init_journal():
    if not JOURNAL_PATH.exists():
        with open(JOURNAL_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=JOURNAL_FIELDS).writeheader()

def write_journal(trade: dict):
    with open(JOURNAL_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=JOURNAL_FIELDS).writerow(trade)

class KellyLotCalculator:
    def __init__(self, config):
        self.config = config
        self._cached_lot = config["lot_fixed"]
        self._last_calc_date = None

    def get_lot(self, equity: float) -> float:
        if self.config["lot_mode"] == "fixed": return self.config["lot_fixed"]
        today = datetime.now(timezone.utc).date()
        if self._last_calc_date == today: return self._cached_lot
        lot = self._calc_kelly(equity)
        self._cached_lot, self._last_calc_date = lot, today
        return lot

    def _calc_kelly(self, equity: float) -> float:
        cfg = self.config
        if not JOURNAL_PATH.exists(): return cfg["lot_fixed"]
        try: df = pd.read_csv(JOURNAL_PATH)
        except: return cfg["lot_fixed"]
        if len(df) < cfg["kelly_min_trades"]: return cfg["lot_fixed"]
        recent = df.tail(50)
        wins, losses = recent[recent["pnl_usd"] > 0]["pnl_usd"], recent[recent["pnl_usd"] <= 0]["pnl_usd"]
        if len(wins) == 0 or len(losses) == 0: return cfg["lot_fixed"]
        win_rate = len(wins) / len(recent)
        avg_win, avg_loss = wins.mean(), abs(losses.mean())
        if avg_win <= 0: return cfg["lot_fixed"]
        kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        half_kelly = kelly_pct * cfg["kelly_fraction"]
        if half_kelly <= 0: return cfg["lot_min"]
        raw_lot = half_kelly * equity / cfg["margin_per_lot"]
        return round(max(cfg["lot_min"], min(cfg["lot_max"], raw_lot)), 2)

class TrailingStop:
    def __init__(self, config):
        self.config = config
        self.active, self.trail_price, self.atr, self.direction = False, 0.0, 0.0, 0

    def reset(self, direction, atr):
        self.active, self.trail_price, self.atr, self.direction = False, 0.0, atr, direction

    def update(self, current_price, entry_price) -> bool:
        if self.atr <= 0: return False
        activate_dist, trail_dist = self.config["trailing_activate_atr"] * self.atr, self.config["trailing_distance_atr"] * self.atr
        if self.direction == 1:
            if current_price - entry_price >= activate_dist:
                new_trail = current_price - trail_dist
                if not self.active: self.active, self.trail_price = True, new_trail
                elif new_trail > self.trail_price: self.trail_price = new_trail
                if current_price <= self.trail_price: return True
        elif self.direction == -1:
            if entry_price - current_price >= activate_dist:
                new_trail = current_price + trail_dist
                if not self.active: self.active, self.trail_price = True, new_trail
                elif new_trail < self.trail_price: self.trail_price = new_trail
                if current_price >= self.trail_price: return True
        return False

class CircuitBreaker:
    def __init__(self, initial_equity, config):
        self.peak_equity = self.daily_start_equity = initial_equity
        self.daily_start_date = datetime.now(timezone.utc).date()
        self.consec_losses, self.config, self.tripped, self.trip_reason = 0, config, False, ""

    def on_trade_result(self, pnl, equity):
        self.peak_equity = max(self.peak_equity, equity)
        self.consec_losses = self.consec_losses + 1 if pnl < 0 else 0
        today = datetime.now(timezone.utc).date()
        if today != self.daily_start_date: self.daily_start_equity, self.daily_start_date = equity, today

    def is_safe(self, equity) -> bool:
        if self.tripped: return False
        today = datetime.now(timezone.utc).date()
        if today != self.daily_start_date: self.daily_start_equity, self.daily_start_date = equity, today
        daily_loss_pct = (self.daily_start_equity - equity) / self.daily_start_equity * 100
        if daily_loss_pct > self.config["daily_loss_pct"]:
            self.tripped, self.trip_reason = True, f"DAILY LOSS LIMIT"
            return False
        dd_pct = (self.peak_equity - equity) / self.peak_equity * 100
        if dd_pct > self.config["max_dd_pct"]:
            self.tripped, self.trip_reason = True, f"MAX DD LIMIT"
            return False
        if self.consec_losses >= self.config["max_consec_loss"]:
            self.tripped, self.trip_reason = True, f"CONSEC LOSS LIMIT"
            return False
        return True

def get_equity():
    info = mt5.account_info()
    return info.equity if info else 0.0

def get_current_price(symbol, direction):
    tick = mt5.symbol_info_tick(symbol)
    return tick.bid if direction == "BUY" else tick.ask if tick else 0.0

def fetch_m5_bars(symbol, count):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, count)
    if rates is None or len(rates) == 0: return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.rename(columns={"time": "datetime"}, inplace=True)
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    return df

def calc_atr(bars_df, period=14):
    if bars_df is None or len(bars_df) < period + 1: return 0.0
    high, low, close = bars_df["high"].values, bars_df["low"].values, bars_df["close"].values
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    return float(np.mean(tr[-period:]))

def get_open_position(symbol, magic):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None: return None
    for pos in positions:
        if pos.magic == magic: return pos
    return None

def send_market_order(symbol, direction, lot, magic):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None: return False
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": tick.ask if direction == "BUY" else tick.bid,
        "magic": magic,
        "comment": "ScalpEx200_v3",
        "type_time": mt5.ORDER_TIME_GTC,
    }
    for filling in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
        request["type_filling"] = filling
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"✅ Executed {direction} {lot} lot @ {result.price:.2f}")
            return True
    log.error("❌ Order rejected.")
    return False

def close_position(position, symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None: return 0.0
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "price": tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask,
        "position": position.ticket,
        "magic": position.magic,
        "comment": "ScalpEx200_Close",
        "type_time": mt5.ORDER_TIME_GTC,
    }
    for filling in [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
        request["type_filling"] = filling
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE: return position.profit
    return 0.0

def close_and_log(position, symbol, trade_count, total_pnl, entry_time, entry_equity, entry_confidence, cb, close_reason, current_lot):
    pnl = close_position(position, symbol)
    trade_count += 1
    total_pnl += pnl
    equity_after = get_equity()
    duration = int((datetime.now(timezone.utc) - entry_time).total_seconds() / 60) if entry_time else 0
    pos_dir = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
    log.info(f"{'✅' if pnl>0 else '❌'} Closed {pos_dir} | PnL: ${pnl:+.2f} | Reason: {close_reason} | Total PnL: ${total_pnl:+.2f}")
    write_journal({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trade_id": trade_count, "direction": pos_dir,
        "entry_price": position.price_open, "exit_price": position.price_current,
        "pnl_usd": round(pnl, 2), "duration_min": duration, "confidence": round(entry_confidence, 4),
        "equity_before": round(entry_equity, 2), "equity_after": round(equity_after, 2),
        "result": "WIN" if pnl > 0 else "LOSS", "close_reason": close_reason, "lot_size": current_lot
    })
    cb.on_trade_result(pnl, equity_after)
    return trade_count, total_pnl

class SignalBridge:
    def __init__(self, filepath="logs/nt8_signal.json"):
        self.filepath = Path(filepath)
        self.last_ts = None

    def get_signal(self):
        if not self.filepath.exists(): return None
        try:
            with open(self.filepath, "r", encoding="utf-8") as f: data = json.load(f)
            ts = data.get("timestamp")
            if ts != self.last_ts:
                self.last_ts = ts
                return data
        except: return None
        return None

def main():
    print("=" * 50)
    print("🤖 EXNESS SLAVE BOT (Synced with NinjaTrader)")
    print("=" * 50)

    if not mt5.initialize(): return log.error("❌ Kết nối MT5 thất bại")
    mt5.symbol_select(CONFIG["symbol"], True)

    initial_equity = get_equity()
    cb = CircuitBreaker(initial_equity, CONFIG)
    trailing = TrailingStop(CONFIG)
    kelly = KellyLotCalculator(CONFIG)
    bridge = SignalBridge()
    init_journal()

    trade_count, total_pnl, entry_time, entry_equity = 0, 0.0, None, 0.0
    entry_confidence, current_lot, current_atr = 0.0, CONFIG["lot_fixed"], 0.0

    log.info(f"🚀 Slave Bot v3 bắt đầu (Eq: ${initial_equity:.2f})")
    try:
        while True:
            time.sleep(CONFIG["trailing_check_interval"])
            position = get_open_position(CONFIG["symbol"], CONFIG["magic_number"])
            
            # --- TRAILING STOP LOGIC ---
            if position and trailing.atr > 0:
                pos_dir = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
                price = get_current_price(CONFIG["symbol"], pos_dir)
                if price > 0 and trailing.update(price, position.price_open):
                    trade_count, total_pnl = close_and_log(position, CONFIG["symbol"], trade_count, total_pnl, entry_time, entry_equity, entry_confidence, cb, "TRAILING_STOP", current_lot)
                    trailing.active = False; position = None

            if not cb.is_safe(get_equity()): break

            # --- SIGNAL PIPELINE ---
            signal = bridge.get_signal()
            if signal:
                action, confidence = signal.get("action_id", 0), signal.get("confidence", 0.0)
                log.info(f"📶 Nhận tín hiệu NT8: Giá {signal.get('close_price')} | {signal.get('action_name')} ({confidence*100:.1f}%)")

                equity = get_equity()
                current_lot = kelly.get_lot(equity)
                bars = fetch_m5_bars(CONFIG["symbol"], 30)
                if bars is not None: current_atr = calc_atr(bars)

                if position is not None:
                    pos_dir = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
                    need_close, close_reason = False, ""
                    if action == 0: need_close, close_reason = True, "BOT_CLOSE"
                    elif (pos_dir == "BUY" and action == 2) or (pos_dir == "SELL" and action == 1):
                        need_close, close_reason = True, "REVERSAL"

                    if need_close:
                        trade_count, total_pnl = close_and_log(position, CONFIG["symbol"], trade_count, total_pnl, entry_time, entry_equity, entry_confidence, cb, close_reason, current_lot)
                        trailing.active = False
                        if close_reason == "REVERSAL" and cb.is_safe(get_equity()) and confidence >= CONFIG["confidence_min"]:
                            new_dir = "BUY" if action == 1 else "SELL"
                            if send_market_order(CONFIG["symbol"], new_dir, current_lot, CONFIG["magic_number"]):
                                entry_time, entry_equity, entry_confidence = datetime.now(timezone.utc), get_equity(), confidence
                                trailing.reset(1 if new_dir == "BUY" else -1, current_atr)

                else:
                    if action in (1, 2) and confidence >= CONFIG["confidence_min"]:
                        direction = "BUY" if action == 1 else "SELL"
                        if send_market_order(CONFIG["symbol"], direction, current_lot, CONFIG["magic_number"]):
                            entry_time, entry_equity, entry_confidence = datetime.now(timezone.utc), equity, confidence
                            trailing.reset(1 if direction == "BUY" else -1, current_atr)

    except KeyboardInterrupt: log.info("⏹️ Dừng bởi người dùng")
    except Exception as e: log.error(f"Lỗi: {e}")
    finally:
        pos = get_open_position(CONFIG["symbol"], CONFIG["magic_number"])
        if pos: close_position(pos, CONFIG["symbol"])
        mt5.shutdown()
        log.info(f"Tổng kết: {trade_count} trades, PnL: ${total_pnl:+.2f}")

if __name__ == "__main__": main()
