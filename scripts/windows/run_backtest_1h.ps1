$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$RepoRoot\.venv\Scripts\python.exe" -m hogan_bot.backtest_cli --symbol BTC/USD --timeframe 1h --limit 3000 --from-db
