# Backtest 10m and 1h timeframes for BTC/USD and ETH/USD
# 1. Fetches 10m and 1h data from Alpaca (Kraken doesn't support 10m)
# 2. Runs backtests with ML filter
#
# Usage: .\scripts\backtest_10m_1h.ps1
# Or:    .\scripts\backtest_10m_1h.ps1 -Limit 5000

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
Write-Host "=== Hogan 10m & 1h Backtest ===" -ForegroundColor Cyan
Write-Host "Limit: $Limit bars per symbol/timeframe" -ForegroundColor Gray
Write-Host ""

# Step 1: Fetch 10m and 1h data from Alpaca (BTC + ETH)
Write-Host "[1/4] Fetching 10m candles (Alpaca, 90 days)..." -ForegroundColor Yellow
& $VENV_PYTHON -m hogan_bot.fetch_alpaca `
    --crypto-bars `
    --timeframe 10Min `
    --crypto-days 90 `
    --symbols "BTC/USD,ETH/USD" `
    --db $DbPath
if ($LASTEXITCODE -ne 0) { Write-Host "Fetch 10m failed" -ForegroundColor Red; exit $LASTEXITCODE }

Write-Host ""
Write-Host "[2/4] Fetching 1h candles (Alpaca, 90 days)..." -ForegroundColor Yellow
& $VENV_PYTHON -m hogan_bot.fetch_alpaca `
    --crypto-bars `
    --timeframe 1Hour `
    --crypto-days 90 `
    --symbols "BTC/USD,ETH/USD" `
    --db $DbPath
if ($LASTEXITCODE -ne 0) { Write-Host "Fetch 1h failed" -ForegroundColor Red; exit $LASTEXITCODE }

Write-Host ""
Write-Host "[3/4] Running backtests..." -ForegroundColor Yellow
Write-Host ""

$results = @()

# BTC 10m
Write-Host "--- BTC/USD 10m ---" -ForegroundColor Cyan
$out = & $VENV_PYTHON -m hogan_bot.backtest_cli --from-db --db $DbPath --symbol BTC/USD --timeframe 10m --limit $Limit --use-ml 2>&1
$out | ForEach-Object { Write-Host $_ }
$lines = $out | Where-Object { $_ -match "total_return_pct|max_drawdown|win_rate|trades" }
$results += @{ Symbol = "BTC/USD"; TF = "10m"; Output = $out }

# BTC 1h
Write-Host ""
Write-Host "--- BTC/USD 1h ---" -ForegroundColor Cyan
$out = & $VENV_PYTHON -m hogan_bot.backtest_cli --from-db --db $DbPath --symbol BTC/USD --timeframe 1h --limit $Limit --use-ml 2>&1
$out | ForEach-Object { Write-Host $_ }
$results += @{ Symbol = "BTC/USD"; TF = "1h"; Output = $out }

# ETH 10m
Write-Host ""
Write-Host "--- ETH/USD 10m ---" -ForegroundColor Cyan
$out = & $VENV_PYTHON -m hogan_bot.backtest_cli --from-db --db $DbPath --symbol ETH/USD --timeframe 10m --limit $Limit --use-ml 2>&1
$out | ForEach-Object { Write-Host $_ }
$results += @{ Symbol = "ETH/USD"; TF = "10m"; Output = $out }

# ETH 1h
Write-Host ""
Write-Host "--- ETH/USD 1h ---" -ForegroundColor Cyan
$out = & $VENV_PYTHON -m hogan_bot.backtest_cli --from-db --db $DbPath --symbol ETH/USD --timeframe 1h --limit $Limit --use-ml 2>&1
$out | ForEach-Object { Write-Host $_ }
$results += @{ Symbol = "ETH/USD"; TF = "1h"; Output = $out }

Write-Host ""
Write-Host "[4/4] Candle summary:" -ForegroundColor Yellow
& $VENV_PYTHON scripts/list_candles.py --db $DbPath 2>&1 | ForEach-Object { Write-Host $_ }

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
