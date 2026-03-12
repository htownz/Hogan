# Backtest 1m, 15m, and 45m timeframes for BTC/USD and ETH/USD
# 1. Fetches 1m, 15m, 45m data from Alpaca
# 2. Runs backtests with ML filter
#
# Usage: .\scripts\backtest_1m_15m_45m.ps1
# Or:    .\scripts\backtest_1m_15m_45m.ps1 -Limit 5000

param(
    [int]$Limit = 10000,
    [string]$Db = "data/hogan.db"
)

$ErrorActionPreference = "Stop"
$HOGAN_ROOT = Split-Path -Parent $PSScriptRoot
if (-not $HOGAN_ROOT) { $HOGAN_ROOT = (Get-Location).Path }
$VENV_PYTHON = Join-Path $HOGAN_ROOT ".venv\Scripts\python.exe"

if (-not (Test-Path $VENV_PYTHON)) {
    Write-Host "ERROR: venv not found at $VENV_PYTHON" -ForegroundColor Red
    exit 1
}

Set-Location $HOGAN_ROOT
$DbPath = Join-Path $HOGAN_ROOT $Db

Write-Host ""
Write-Host "=== Hogan 1m, 15m & 45m Backtest ===" -ForegroundColor Cyan
Write-Host "Limit: $Limit bars per symbol/timeframe" -ForegroundColor Gray
Write-Host ""

# Step 1: Fetch 1m candles (Alpaca free tier: max 30 days for 1m)
Write-Host "[1/6] Fetching 1m candles (Alpaca, 30 days)..." -ForegroundColor Yellow
& $VENV_PYTHON -m hogan_bot.fetch_alpaca `
    --crypto-bars `
    --timeframe 1Min `
    --crypto-days 30 `
    --symbols "BTC/USD,ETH/USD" `
    --db $DbPath
if ($LASTEXITCODE -ne 0) { Write-Host "Fetch 1m failed" -ForegroundColor Red; exit $LASTEXITCODE }

# Step 2: Fetch 15m candles
Write-Host ""
Write-Host "[2/6] Fetching 15m candles (Alpaca, 90 days)..." -ForegroundColor Yellow
& $VENV_PYTHON -m hogan_bot.fetch_alpaca `
    --crypto-bars `
    --timeframe 15Min `
    --crypto-days 90 `
    --symbols "BTC/USD,ETH/USD" `
    --db $DbPath
if ($LASTEXITCODE -ne 0) { Write-Host "Fetch 15m failed" -ForegroundColor Red; exit $LASTEXITCODE }

# Step 3: Fetch 45m candles
Write-Host ""
Write-Host "[3/6] Fetching 45m candles (Alpaca, 90 days)..." -ForegroundColor Yellow
& $VENV_PYTHON -m hogan_bot.fetch_alpaca `
    --crypto-bars `
    --timeframe 45Min `
    --crypto-days 90 `
    --symbols "BTC/USD,ETH/USD" `
    --db $DbPath
if ($LASTEXITCODE -ne 0) { Write-Host "Fetch 45m failed" -ForegroundColor Red; exit $LASTEXITCODE }

Write-Host ""
Write-Host "[4/6] Running backtests..." -ForegroundColor Yellow
Write-Host ""

$results = @()

# BTC 1m
Write-Host "--- BTC/USD 1m ---" -ForegroundColor Cyan
$out = & $VENV_PYTHON -m hogan_bot.backtest_cli --from-db --db $DbPath --symbol BTC/USD --timeframe 1m --limit $Limit --use-ml 2>&1
$out | ForEach-Object { Write-Host $_ }
$results += @{ Symbol = "BTC/USD"; TF = "1m"; Output = $out }

# BTC 15m
Write-Host ""
Write-Host "--- BTC/USD 15m ---" -ForegroundColor Cyan
$out = & $VENV_PYTHON -m hogan_bot.backtest_cli --from-db --db $DbPath --symbol BTC/USD --timeframe 15m --limit $Limit --use-ml 2>&1
$out | ForEach-Object { Write-Host $_ }
$results += @{ Symbol = "BTC/USD"; TF = "15m"; Output = $out }

# BTC 45m
Write-Host ""
Write-Host "--- BTC/USD 45m ---" -ForegroundColor Cyan
$out = & $VENV_PYTHON -m hogan_bot.backtest_cli --from-db --db $DbPath --symbol BTC/USD --timeframe 45m --limit $Limit --use-ml 2>&1
$out | ForEach-Object { Write-Host $_ }
$results += @{ Symbol = "BTC/USD"; TF = "45m"; Output = $out }

# ETH 1m
Write-Host ""
Write-Host "--- ETH/USD 1m ---" -ForegroundColor Cyan
$out = & $VENV_PYTHON -m hogan_bot.backtest_cli --from-db --db $DbPath --symbol ETH/USD --timeframe 1m --limit $Limit --use-ml 2>&1
$out | ForEach-Object { Write-Host $_ }
$results += @{ Symbol = "ETH/USD"; TF = "1m"; Output = $out }

# ETH 15m
Write-Host ""
Write-Host "--- ETH/USD 15m ---" -ForegroundColor Cyan
$out = & $VENV_PYTHON -m hogan_bot.backtest_cli --from-db --db $DbPath --symbol ETH/USD --timeframe 15m --limit $Limit --use-ml 2>&1
$out | ForEach-Object { Write-Host $_ }
$results += @{ Symbol = "ETH/USD"; TF = "15m"; Output = $out }

# ETH 45m
Write-Host ""
Write-Host "--- ETH/USD 45m ---" -ForegroundColor Cyan
$out = & $VENV_PYTHON -m hogan_bot.backtest_cli --from-db --db $DbPath --symbol ETH/USD --timeframe 45m --limit $Limit --use-ml 2>&1
$out | ForEach-Object { Write-Host $_ }
$results += @{ Symbol = "ETH/USD"; TF = "45m"; Output = $out }

Write-Host ""
Write-Host "[5/6] Candle summary:" -ForegroundColor Yellow
& $VENV_PYTHON scripts/list_candles.py --db $DbPath 2>&1 | ForEach-Object { Write-Host $_ }

Write-Host ""
Write-Host "[6/6] Backtest-learn: retrain with backtest labels..." -ForegroundColor Yellow
& $VENV_PYTHON -u -m hogan_bot.retrain --from-db --db $DbPath --symbol BTC/USD --use-backtest-labels --window-bars 5000 2>&1 | ForEach-Object { Write-Host $_ }

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
