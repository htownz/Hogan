"""Run full backtest via CLI on all available data."""
import sys
sys.argv = [
    "backtest_cli",
    "--profile", "canonical",
    "--db", "data/hogan.db",
    "--bars", "5000",
]
from hogan_bot.backtest_cli import main
main()
