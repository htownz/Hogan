<#
.SYNOPSIS
    Full multi-timeframe (MTF) historical candle backfill for Hogan.

.DESCRIPTION
    Fetches OHLCV bars for ALL configured symbols and timeframes and writes
    them into the local SQLite database.

    Symbols backfilled:
        Crypto : BTC/USD, ETH/USD, SOL/USD
        Stocks : SPY, QQQ, GLD, TLT  (daily candles only — most reliable on free Alpaca tier)

    Timeframes backfilled for crypto:
        10m, 30m, 1h, 1d

    After backfill, the database will contain enough data to:
        1. Run multi-timeframe backtests
        2. Train on BTC + ETH + SOL jointly (more data = better model)
        3. Use 10m + 30m features in the RL agent (extended_mtf mode)

.PARAMETER Days
    How many calendar days of history to fetch.
    Default: 365 (approx 1 year).
    Alpaca free tier provides several years of crypto and stock history.

.PARAMETER Db
    Path to the SQLite database file.
    Default: data/hogan.db (relative to repo root).

.EXAMPLE
    # Standard 1-year backfill (run once from the repo root):
    .\scripts\backfill_mtf.ps1

.EXAMPLE
    # 2-year backfill:
    .\scripts\backfill_mtf.ps1 -Days 730

.EXAMPLE
    # Quick 30-day test:
    .\scripts\backfill_mtf.ps1 -Days 30

.NOTES
    Requirements:
        - ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env or environment.
        - Run from the Hogan repo root directory.
        - Estimated time: 1–3 minutes for 365 days (Alpaca rate limits apply).

    After backfill, retrain the model on the expanded dataset:
        python -m hogan_bot.retrain --from-db --symbols BTC/USD,ETH/USD,SOL/USD --force-promote

    To enable extended MTF features (10m + 30m) for RL training:
        python -m hogan_bot.rl_train --ext-features --from-db
#>

param(
    [int]$Days = 365,
    [string]$Db = "data/hogan.db"
)

$ErrorActionPreference = "Stop"

# ── Resolve repo root ────────────────────────────────────────────────────────
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
Write-Host ""
Write-Host "=== Hogan MTF Historical Backfill ===" -ForegroundColor Cyan
Write-Host "  Repo   : $RepoRoot"
Write-Host "  DB     : $Db"
Write-Host "  Days   : $Days"
Write-Host ""

# ── Load .env if present ─────────────────────────────────────────────────────
$EnvFile = Join-Path $RepoRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match "^\s*([^#=]+)=(.*)$") {
            $k = $Matches[1].Trim()
            $v = $Matches[2].Trim().Trim('"').Trim("'")
            if (-not [System.Environment]::GetEnvironmentVariable($k)) {
                [System.Environment]::SetEnvironmentVariable($k, $v, "Process")
            }
        }
    }
    Write-Host "Loaded .env" -ForegroundColor DarkGray
}

# ── Resolve Python executable (prefer .venv) ────────────────────────────────
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
Write-Host "  Python : $PythonExe" -ForegroundColor DarkGray
Write-Host ""

# ── Check Alpaca keys ────────────────────────────────────────────────────────
if (-not $env:ALPACA_API_KEY -or -not $env:ALPACA_SECRET_KEY) {
    Write-Error "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env or environment."
    exit 1
}

# ── Run bulk backfill ────────────────────────────────────────────────────────
Write-Host "Starting bulk backfill (this may take 1-3 minutes)..." -ForegroundColor Yellow
$StartTime = Get-Date

& $PythonExe -m hogan_bot.fetch_alpaca --backfill-all --days $Days --db $Db
$ExitCode = $LASTEXITCODE
if ($ExitCode -ne 0) {
    Write-Host "Backfill failed with exit code $ExitCode" -ForegroundColor Red
    exit $ExitCode
}

$Elapsed = [math]::Round(((Get-Date) - $StartTime).TotalSeconds, 1)
Write-Host ""
Write-Host "Backfill complete in ${Elapsed}s" -ForegroundColor Green
Write-Host ""

# ── Show DB summary ──────────────────────────────────────────────────────────
Write-Host "Database candle summary:" -ForegroundColor Cyan
& $PythonExe scripts\list_candles.py --db $Db

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Retrain on expanded multi-symbol dataset:"
Write-Host "       $PythonExe -m hogan_bot.retrain --from-db --symbols BTC/USD,ETH/USD,SOL/USD --force-promote" -ForegroundColor White
Write-Host ""
Write-Host "  2. (Optional) Enable extended MTF features in .env after retraining:"
Write-Host "       HOGAN_USE_MTF_EXTENDED=true" -ForegroundColor White
Write-Host "       HOGAN_TRAINING_SYMBOLS=BTC/USD,ETH/USD,SOL/USD" -ForegroundColor White
Write-Host ""
Write-Host "  3. Restart the bot:"
Write-Host "       $PythonExe -m hogan_bot.main" -ForegroundColor White
Write-Host ""
