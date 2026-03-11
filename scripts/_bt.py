"""Quick validation backtest."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hogan_bot.storage import load_candles, get_connection
from hogan_bot.backtest import run_backtest_on_candles

conn = get_connection("data/hogan.db")
with open("models/opt_BTC-USD_1h.json") as f:
    bc = json.load(f)["best_config"]

candles = load_candles(conn, "BTC/USD", "1h", limit=10000)
r = run_backtest_on_candles(
    candles, symbol="BTC/USD", starting_balance_usd=10000,
    aggressive_allocation=0.12, max_risk_per_trade=0.015,
    max_drawdown=0.20, short_ma_window=bc["short_ma_window"],
    long_ma_window=bc["long_ma_window"], volume_window=20,
    volume_threshold=bc["volume_threshold"], fee_rate=0.0026,
    atr_stop_multiplier=bc["atr_stop_multiplier"],
    use_ema_clouds=bc["use_ema_clouds"], signal_mode=bc["signal_mode"],
    trailing_stop_pct=bc["trailing_stop_pct"],
    take_profit_pct=bc["take_profit_pct"],
    use_ict=bc.get("use_ict", False), timeframe="1h",
    execution_mode="next_open",
)
print(f"Trades: {r.trades} | WR: {r.win_rate:.1%} | Return: {r.total_return_pct:.2f}% | MaxDD: {r.max_drawdown_pct:.2f}% | Sharpe: {r.sharpe_ratio:.3f} | Calmar: {r.calmar_ratio:.3f}")
conn.close()
