"""
download_mt5.py
---------------
Pull 10 năm dữ liệu nến XAUUSD (M15 và H1) trực tiếp từ Exness MT5 Terminal
thông qua gói MetaTrader5 chính thức.

Yêu cầu:
  - MT5 Exness Terminal đang chạy và đã đăng nhập trên máy Windows.
  - pip install MetaTrader5 pandas

Cách dùng:
  python src/data/download_mt5.py
  python src/data/download_mt5.py --symbol XAUUSD --timeframes M15 H1 --years 10
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

# ─── Cấu hình mặc định ────────────────────────────────────────────────

SYMBOL        = "XAUUSD"
YEARS_BACK    = 10
OUTPUT_DIR    = Path(__file__).resolve().parents[2] / "data" / "raw"

TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

# ─── Logging ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Core Functions ───────────────────────────────────────────────────

def connect_mt5() -> bool:
    """Khởi tạo kết nối tới MT5 terminal đang chạy."""
    if not mt5.initialize():
        log.error(
            "Không thể kết nối MT5 terminal. "
            "Hãy chắc chắn Exness MT5 đang mở và đã đăng nhập. "
            f"Lỗi: {mt5.last_error()}"
        )
        return False

    info = mt5.terminal_info()
    account = mt5.account_info()
    log.info(f"✅ Kết nối MT5 thành công.")
    log.info(f"   Terminal: {info.name}  |  Build: {info.build}")
    log.info(f"   Account : {account.login} ({account.server})")
    return True


def validate_symbol(symbol: str) -> bool:
    """Kiểm tra symbol có tồn tại và được phép trade trên terminal."""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        log.error(f"Symbol '{symbol}' không tìm thấy trong MT5.")
        return False
    if not symbol_info.visible:
        log.warning(f"Symbol '{symbol}' chưa visible — đang enable...")
        if not mt5.symbol_select(symbol, True):
            log.error(f"Không thể enable symbol '{symbol}': {mt5.last_error()}")
            return False
    log.info(f"✅ Symbol '{symbol}' hợp lệ. Digits: {symbol_info.digits}  "
             f"Spread: {symbol_info.spread} points")
    return True


def download_bars(
    symbol: str,
    tf_name: str,
    tf_const: int,
    years: int,
    output_dir: Path,
) -> Path | None:
    """
    Tải toàn bộ lịch sử nến cho một timeframe, lưu ra CSV.

    Parameters
    ----------
    symbol     : Tên symbol (ví dụ "XAUUSD")
    tf_name    : Tên timeframe dạng string (ví dụ "M15")
    tf_const   : Hằng số timeframe của MT5
    years      : Số năm nhìn lùi
    output_dir : Thư mục lưu file CSV

    Returns
    -------
    Path của file CSV đã lưu, hoặc None nếu thất bại.
    """
    utc_now   = datetime.now(timezone.utc)
    utc_from  = datetime(utc_now.year - years, 1, 1, tzinfo=timezone.utc)

    log.info(f"⬇️  Đang tải {symbol} {tf_name} từ {utc_from.date()} → {utc_now.date()} ...")

    rates = mt5.copy_rates_range(symbol, tf_const, utc_from, utc_now)

    if rates is None or len(rates) == 0:
        log.error(
            f"Không lấy được dữ liệu {symbol} {tf_name}. "
            f"Lỗi MT5: {mt5.last_error()}"
        )
        return None

    # ── Chuyển sang DataFrame ──────────────────────────────────────────
    df = pd.DataFrame(rates)

    # Chuyển timestamp Unix → datetime UTC
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.rename(columns={
        "time":         "datetime",
        "open":         "open",
        "high":         "high",
        "low":          "low",
        "close":        "close",
        "tick_volume":  "tick_volume",
        "spread":       "spread",
        "real_volume":  "real_volume",
    }, inplace=True)
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    # ── Kiểm tra nhanh tính toàn vẹn ──────────────────────────────────
    n_bars    = len(df)
    n_null    = df[["open","high","low","close"]].isnull().sum().sum()
    n_hl_err  = (df["high"] < df["low"]).sum()   # high < low = lỗi nghiêm trọng

    log.info(f"   Bars tải về : {n_bars:,}")
    log.info(f"   Khoảng thời gian: {df.index[0]}  →  {df.index[-1]}")

    if n_null > 0:
        log.warning(f"   ⚠️  Phát hiện {n_null} giá trị NULL trong OHLC!")
    if n_hl_err > 0:
        log.error(f"   ❌ {n_hl_err} nến có high < low — dữ liệu có vấn đề!")

    # ── Lưu CSV ───────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    filename  = f"{symbol}_{tf_name}_{utc_from.strftime('%Y%m%d')}_{utc_now.strftime('%Y%m%d')}.csv"
    out_path  = output_dir / filename

    df.to_csv(out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    log.info(f"   ✅ Lưu: {out_path}  ({size_mb:.2f} MB)")

    return out_path


def print_summary(results: dict[str, Path | None]) -> None:
    """In bảng tổng kết kết quả tải."""
    log.info("")
    log.info("=" * 60)
    log.info("  KẾT QUẢ TẢI DỮ LIỆU")
    log.info("=" * 60)
    for tf, path in results.items():
        status = f"✅ {path}" if path else "❌ THẤT BẠI"
        log.info(f"  {tf:>5}: {status}")
    log.info("=" * 60)


# ─── CLI Entry Point ──────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tải dữ liệu lịch sử từ Exness MT5 Terminal"
    )
    parser.add_argument(
        "--symbol",
        default=SYMBOL,
        help=f"Symbol cần tải (mặc định: {SYMBOL})",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["M15", "H1"],
        choices=list(TIMEFRAME_MAP.keys()),
        help="Danh sách timeframes cần tải (mặc định: M15 H1)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=YEARS_BACK,
        help=f"Số năm lịch sử cần kéo về (mặc định: {YEARS_BACK})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Thư mục lưu file CSV (mặc định: {OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    log.info("🚀 Bắt đầu tải dữ liệu XAUUSD từ Exness MT5")
    log.info(f"   Symbol     : {args.symbol}")
    log.info(f"   Timeframes : {', '.join(args.timeframes)}")
    log.info(f"   Số năm     : {args.years}")
    log.info(f"   Output dir : {args.output_dir}")
    log.info("")

    # Bước 1: Kết nối MT5
    if not connect_mt5():
        return 1

    # Bước 2: Kiểm tra symbol
    if not validate_symbol(args.symbol):
        mt5.shutdown()
        return 1

    # Bước 3: Tải từng timeframe
    results: dict[str, Path | None] = {}
    try:
        for tf_name in args.timeframes:
            tf_const = TIMEFRAME_MAP[tf_name]
            results[tf_name] = download_bars(
                symbol     = args.symbol,
                tf_name    = tf_name,
                tf_const   = tf_const,
                years      = args.years,
                output_dir = args.output_dir,
            )
    finally:
        mt5.shutdown()
        log.info("MT5 terminal connection đã được đóng.")

    # Bước 4: Tổng kết
    print_summary(results)

    # Exit code 1 nếu có bất kỳ timeframe nào thất bại
    failed = [tf for tf, path in results.items() if path is None]
    if failed:
        log.error(f"Các timeframe thất bại: {failed}")
        return 1

    log.info("🎉 Hoàn tất! Dữ liệu sẵn sàng cho Sprint 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
