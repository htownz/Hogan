<#
.SYNOPSIS
    Backfill maximum OHLCV data for Hogan using Kraken + Yahoo Finance.

.DESCRIPTION
    Fetches as much 1h and 30m data as possible:
    1. Kraken backfill (paginated) - extends backward when exchange allows
    2. Yahoo Finance - 2 years of 1h for BTC/ETH/SOL (no API key)

    Use this when you need deep history for OOS validation, retraining, etc.

.EXAMPLE
    .\scripts\backfill_kraken_yahoo.ps1
#>

param([string]$Db = "data/hogan.db")

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$PythonExe = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

Write-Host ""
Write-Host "=== Hogan Data Backfill (Kraken + Yahoo) ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Kraken backfill (paginates in 720-bar chunks)
Write-Host "Step 1 — Kraken backfill (1h + 30m)..." -ForegroundColor Yellow
& $PythonExe -m hogan_bot.fetch_data --symbol BTC/USD ETH/USD SOL/USD --timeframe 1h --backfill --target-bars 20000 --db $Db
& $PythonExe -m hogan_bot.fetch_data --symbol BTC/USD ETH/USD SOL/USD --timeframe 30m --backfill --target-bars 40000 --db $Db

# Step 2: Yahoo Finance 1h (2 years — more history than Kraken for 1h)
Write-Host ""
Write-Host "Step 2 — Yahoo Finance 1h (2y for BTC/ETH/SOL)..." -ForegroundColor Yellow
& $PythonExe -m hogan_bot.backfill --symbol BTC/USD ETH/USD SOL/USD --timeframe 1h --period 2y --db $Db

Write-Host ""
Write-Host "Backfill complete. Candle summary:" -ForegroundColor Green
& $PythonExe scripts\list_candles.py --db $Db
