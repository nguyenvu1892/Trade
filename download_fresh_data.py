"""
download_fresh_data.py
Download 6 tháng data M5 XAUUSD gần nhất từ MT5.
"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

Path("data/raw").mkdir(parents=True, exist_ok=True)

mt5.initialize()

# Download 6 tháng gần nhất (~35K bars)
start = datetime(2025, 10, 1, tzinfo=timezone.utc)
end = datetime.now(timezone.utc)

print(f"Downloading XAUUSD M5 from {start.date()} to {end.date()}...")
rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M5, start, end)
mt5.shutdown()

if rates is None or len(rates) == 0:
    print("ERROR: No data!")
    exit(1)

df = pd.DataFrame(rates)
df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
df = df[["datetime", "open", "high", "low", "close", "tick_volume"]]
df.set_index("datetime", inplace=True)
df.sort_index(inplace=True)

# Save
out = "data/raw/XAUUSD_M5_6months.csv"
df.to_csv(out)

print(f"Done! {len(df):,} bars saved to {out}")
print(f"Price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Monthly summary
for name, grp in df.groupby(df.index.to_period("M")):
    s = grp["close"].iloc[0]
    e = grp["close"].iloc[-1]
    ch = (e - s) / s * 100
    d = "UP" if ch > 0.5 else ("DOWN" if ch < -0.5 else "FLAT")
    print(f"  {name}: ${s:.0f}->${e:.0f} ({ch:+.1f}% {d}) {len(grp)} bars")
