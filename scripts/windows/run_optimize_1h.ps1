$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$RepoRoot\.venv\Scripts\python.exe" -m hogan_bot.optimize --from-db --symbol BTC/USD --timeframe 1h --limit 8000 --trials 50 --metric sharpe --engine optuna
