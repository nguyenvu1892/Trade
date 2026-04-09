"""
live_bot.py — ScalpEx200 M5 Sniper Live Trading Bot
=====================================================
Chạy tự động trên Exness MT5 Pro Account.

Model:      ppo_best.pt (Sharpe 6.85)
Timeframe:  M5 (256-bar rolling window)
Confidence: ≥ 45% mới vào lệnh
Lot:        0.01 cố định
SL/TP:      Không — Bot tự quyết đóng/mở lệnh

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
    "lot_size":         0.01,
    "magic_number":     200500,     # ID riêng để nhận biết lệnh của bot

    # ── Model ──
    "checkpoint":       "checkpoints/ppo_best.pt",
    "window_size":      256,
    "d_model":          256,
    "n_heads":          8,
    "n_layers":         6,
    "confidence_min":   0.45,       # Ngưỡng confidence 45%

    # ── Circuit Breaker ──
    "daily_loss_pct":   15.0,       # Dừng nếu lỗ > 15% equity trong ngày
    "max_dd_pct":       30.0,       # Dừng nếu equity giảm > 30% từ đỉnh
    "max_consec_loss":  10,         # Dừng nếu thua 10 lệnh liên tiếp

    # ── Timing ──
    "candle_seconds":   300,        # M5 = 300 giây
    "extra_wait":       5,          # Chờ thêm 5s sau khi nến đóng để MT5 cập nhật
    "bars_to_fetch":    280,        # Lấy dư nến để DataProcessor tính ATR warmup
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
    "result", "close_reason",
]


def init_journal():
    """Tạo file CSV header nếu chưa có."""
    if not JOURNAL_PATH.exists():
        with open(JOURNAL_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
            writer.writeheader()
        log.info(f"📝 Tạo Trade Journal: {JOURNAL_PATH}")


def write_journal(trade: dict):
    """Ghi 1 dòng trade vào CSV."""
    with open(JOURNAL_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        writer.writerow(trade)


# ═══════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════

class CircuitBreaker:
    """Rào chắn sinh tồn — tự động tắt bot khi phát hiện nguy hiểm."""

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
        """Gọi sau mỗi lệnh đóng."""
        self.update_equity(equity)

        if pnl < 0:
            self.consec_losses += 1
        else:
            self.consec_losses = 0

        # Reset daily tracking nếu sang ngày mới
        today = datetime.now(timezone.utc).date()
        if today != self.daily_start_date:
            self.daily_start_equity = equity
            self.daily_start_date = today

    def is_safe(self, equity: float) -> bool:
        """Kiểm tra tất cả điều kiện an toàn."""
        if self.tripped:
            return False

        # 1. Daily Loss
        today = datetime.now(timezone.utc).date()
        if today != self.daily_start_date:
            self.daily_start_equity = equity
            self.daily_start_date = today

        daily_loss_pct = (self.daily_start_equity - equity) / self.daily_start_equity * 100
        if daily_loss_pct > self.config["daily_loss_pct"]:
            self.tripped = True
            self.trip_reason = f"⛔ DAILY LOSS LIMIT: Lỗ {daily_loss_pct:.1f}% trong ngày (giới hạn: {self.config['daily_loss_pct']}%)"
            log.critical(self.trip_reason)
            return False

        # 2. Max Drawdown
        dd_pct = (self.peak_equity - equity) / self.peak_equity * 100
        if dd_pct > self.config["max_dd_pct"]:
            self.tripped = True
            self.trip_reason = f"⛔ MAX DRAWDOWN: Giảm {dd_pct:.1f}% từ đỉnh ${self.peak_equity:.2f} (giới hạn: {self.config['max_dd_pct']}%)"
            log.critical(self.trip_reason)
            return False

        # 3. Consecutive Losses
        if self.consec_losses >= self.config["max_consec_loss"]:
            self.tripped = True
            self.trip_reason = f"⛔ CONSECUTIVE LOSSES: Thua {self.consec_losses} lệnh liên tiếp (giới hạn: {self.config['max_consec_loss']})"
            log.critical(self.trip_reason)
            return False

        return True


# ═══════════════════════════════════════════════════════════════
# MT5 HELPERS
# ═══════════════════════════════════════════════════════════════

def connect_mt5():
    """Kết nối MT5 Terminal."""
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
    """Lấy equity hiện tại."""
    info = mt5.account_info()
    return info.equity if info else 0.0


def get_balance():
    """Lấy balance hiện tại."""
    info = mt5.account_info()
    return info.balance if info else 0.0


def fetch_m5_bars(symbol: str, count: int) -> pd.DataFrame:
    """Lấy nến M5 từ MT5, trả về DataFrame chuẩn."""
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


def get_open_position(symbol: str, magic: int):
    """Tìm vị thế đang mở của bot (theo magic number)."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return None
    for pos in positions:
        if pos.magic == magic:
            return pos
    return None


def send_market_order(symbol: str, direction: str, lot: float, magic: int) -> bool:
    """Gửi lệnh Market Order."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error(f"❌ Không lấy được giá tick: {mt5.last_error()}")
        return False

    if direction == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "magic": magic,
        "comment": "ScalpEx200",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    if result is None:
        log.error(f"❌ order_send trả về None: {mt5.last_error()}")
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"❌ Lệnh bị từ chối: {result.retcode} — {result.comment}")
        return False

    log.info(f"✅ {direction} {lot} lot @ {result.price:.2f} (Order #{result.order})")
    return True


def close_position(position, symbol: str) -> float:
    """Đóng vị thế, trả về PnL."""
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
        "type_filling": mt5.ORDER_FILLING_FOK,
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
    """Load model PPO best."""
    # Đọc n_features từ DataProcessor output
    # DataProcessor tạo 13 features: is_gap, log_open/high/low/close, log_volume,
    # atr_norm, hour_sin/cos, dow_sin/cos, is_us_session, is_weekend
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


def predict(model, features_np: np.ndarray, window_size: int, device: torch.device):
    """
    Chạy inference, trả về (action, confidence, probs).

    Parameters
    ----------
    features_np : np.ndarray shape (T, 13) — features đã tính
    window_size : int — cửa sổ nhìn (256)

    Returns
    -------
    action     : int (0=Hold, 1=Buy, 2=Sell)
    confidence : float (xác suất của action được chọn)
    probs      : np.ndarray [hold_prob, buy_prob, sell_prob]
    """
    if len(features_np) < window_size:
        log.warning(f"⚠️ Chỉ có {len(features_np)} bars (cần {window_size})")
        return 0, 0.0, np.array([1.0, 0.0, 0.0])

    # Lấy đúng window_size bars cuối cùng
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

def wait_for_next_candle():
    """Chờ nến M5 tiếp theo đóng."""
    now = datetime.now(timezone.utc)
    seconds_in_candle = (now.minute % 5) * 60 + now.second
    wait = CONFIG["candle_seconds"] - seconds_in_candle + CONFIG["extra_wait"]
    if wait <= 0:
        wait += CONFIG["candle_seconds"]

    next_candle = now + timedelta(seconds=wait)
    log.info(f"⏳ Chờ nến M5 tiếp theo: {next_candle.strftime('%H:%M:%S')} UTC ({wait}s)")
    time.sleep(wait)


def main():
    print("=" * 60)
    print("🤖 SCALPEX200 — LIVE TRADING BOT")
    print("   Model: PPO Best M5 Sniper (w256)")
    print(f"   Confidence Filter: ≥ {CONFIG['confidence_min']*100:.0f}%")
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
    device = torch.device("cpu")  # Live dùng CPU cho ổn định
    model = load_model(CONFIG, device)
    processor = DataProcessor(atr_period=14)

    # 3. Khởi tạo Circuit Breaker
    initial_equity = get_equity()
    cb = CircuitBreaker(initial_equity, CONFIG)
    log.info(f"🛡️ Circuit Breaker: Daily Loss {CONFIG['daily_loss_pct']}% | "
             f"Max DD {CONFIG['max_dd_pct']}% | Consec Loss {CONFIG['max_consec_loss']}")

    # 4. Khởi tạo Trade Journal
    init_journal()

    # 5. Tracking
    trade_count = 0
    total_pnl = 0.0
    entry_time = None
    entry_equity = 0.0
    entry_confidence = 0.0

    log.info(f"\n🚀 Bot bắt đầu hoạt động! Equity: ${initial_equity:.2f}")
    log.info(f"   Symbol: {symbol} | Lot: {CONFIG['lot_size']} | Magic: {CONFIG['magic_number']}")

    try:
        while True:
            # ── Chờ nến M5 mới ──
            wait_for_next_candle()

            # ── Kiểm tra Circuit Breaker ──
            equity = get_equity()
            if not cb.is_safe(equity):
                log.critical(f"🚨 CIRCUIT BREAKER TRIPPED: {cb.trip_reason}")
                log.critical(f"   Bot đã dừng. Equity: ${equity:.2f}")
                break

            # ── Lấy nến M5 ──
            bars_df = fetch_m5_bars(symbol, CONFIG["bars_to_fetch"])
            if bars_df is None or len(bars_df) < CONFIG["window_size"] + 20:
                log.warning("⚠️ Không đủ dữ liệu nến, bỏ qua chu kỳ này.")
                continue

            # ── Tính Features ──
            try:
                features_df = processor.compute_features(bars_df)
                features_np = features_df.values.astype(np.float32)
            except Exception as e:
                log.error(f"❌ Lỗi tính features: {e}")
                continue

            # ── Model Inference ──
            action, confidence, probs = predict(model, features_np, CONFIG["window_size"], device)
            action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}

            current_price = float(bars_df["close"].iloc[-1])
            log.info(
                f"📊 {symbol} ${current_price:.2f} | "
                f"Dự đoán: {action_names[action]} (Conf: {confidence*100:.1f}%) | "
                f"H:{probs[0]*100:.0f}% B:{probs[1]*100:.0f}% S:{probs[2]*100:.0f}% | "
                f"Equity: ${equity:.2f}"
            )

            # ── Kiểm tra vị thế hiện tại ──
            position = get_open_position(symbol, CONFIG["magic_number"])

            if position is not None:
                # Đang có vị thế
                pos_dir = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
                pos_pnl = position.profit

                need_close = False
                close_reason = ""

                if action == 0:
                    # Hold = đóng lệnh
                    need_close = True
                    close_reason = "BOT_CLOSE"
                elif (pos_dir == "BUY" and action == 2) or (pos_dir == "SELL" and action == 1):
                    # Đảo chiều
                    need_close = True
                    close_reason = "REVERSAL"

                if need_close:
                    # Đóng lệnh
                    pnl = close_position(position, symbol)
                    trade_count += 1
                    total_pnl += pnl

                    equity_after = get_equity()
                    result = "WIN" if pnl > 0 else "LOSS"
                    duration = 0
                    if entry_time:
                        duration = int((datetime.now(timezone.utc) - entry_time).total_seconds() / 60)

                    log.info(
                        f"{'✅' if pnl > 0 else '❌'} Trade #{trade_count}: {pos_dir} | "
                        f"PnL: ${pnl:+.2f} | Duration: {duration}m | "
                        f"Total PnL: ${total_pnl:+.2f}"
                    )

                    # Ghi journal
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
                    })

                    # Cập nhật Circuit Breaker
                    cb.on_trade_result(pnl, equity_after)
                    entry_time = None

                    # Nếu reversal, mở lệnh mới ngay
                    if close_reason == "REVERSAL" and cb.is_safe(equity_after):
                        new_dir = "BUY" if action == 1 else "SELL"
                        if confidence >= CONFIG["confidence_min"]:
                            if send_market_order(symbol, new_dir, CONFIG["lot_size"], CONFIG["magic_number"]):
                                entry_time = datetime.now(timezone.utc)
                                entry_equity = equity_after
                                entry_confidence = confidence
                        else:
                            log.info(f"🚫 Reversal bị chặn: Conf {confidence*100:.1f}% < {CONFIG['confidence_min']*100:.0f}%")
                else:
                    # Giữ lệnh tiếp
                    log.info(f"   Giữ {pos_dir} | Unrealized PnL: ${pos_pnl:+.2f}")

            else:
                # Không có vị thế — xét mở lệnh mới
                if action in (1, 2):
                    if confidence >= CONFIG["confidence_min"]:
                        direction = "BUY" if action == 1 else "SELL"
                        if send_market_order(symbol, direction, CONFIG["lot_size"], CONFIG["magic_number"]):
                            entry_time = datetime.now(timezone.utc)
                            entry_equity = equity
                            entry_confidence = confidence
                    else:
                        log.info(
                            f"🚫 Bị lọc: {action_names[action]} conf {confidence*100:.1f}% "
                            f"< {CONFIG['confidence_min']*100:.0f}%"
                        )

    except KeyboardInterrupt:
        log.info("\n⏹️ Bot dừng bởi người dùng (Ctrl+C)")
    except Exception as e:
        log.critical(f"💥 Lỗi không mong muốn: {e}", exc_info=True)
    finally:
        # Đóng tất cả lệnh nếu còn
        position = get_open_position(symbol, CONFIG["magic_number"])
        if position is not None:
            log.info("🔒 Đóng lệnh đang mở trước khi thoát...")
            close_position(position, symbol)

        mt5.shutdown()
        log.info(f"\n📊 KẾT QUẢ PHIÊN GIAO DỊCH:")
        log.info(f"   Tổng lệnh:    {trade_count}")
        log.info(f"   Tổng PnL:     ${total_pnl:+.2f}")
        log.info(f"   Journal:      {JOURNAL_PATH}")
        log.info("👋 Bot đã tắt an toàn.")


if __name__ == "__main__":
    main()
