$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$RepoRoot\.venv\Scripts\python.exe" -m hogan_bot.retrain --symbol BTC/USD --timeframe 1h
