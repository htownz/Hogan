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

# ── Check Alpaca keys ────────────────────────────────────────────────────────
if (-not $env:ALPACA_API_KEY -or -not $env:ALPACA_SECRET_KEY) {
    Write-Error "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env or environment."
    exit 1
}

# ── Run bulk backfill ────────────────────────────────────────────────────────
Write-Host "Starting bulk backfill (this may take 1-3 minutes)..." -ForegroundColor Yellow
$StartTime = Get-Date

$Result = python -m hogan_bot.fetch_alpaca --backfill-all --days $Days --db $Db 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Backfill failed with exit code $LASTEXITCODE:`n$Result"
    exit $LASTEXITCODE
}

$Elapsed = [math]::Round(((Get-Date) - $StartTime).TotalSeconds, 1)
Write-Host ""
Write-Host "Backfill complete in ${Elapsed}s" -ForegroundColor Green
Write-Host ""

# ── Show DB summary ──────────────────────────────────────────────────────────
Write-Host "Database candle summary:" -ForegroundColor Cyan
python -c "
import sqlite3, os
db = r'$Db'
conn = sqlite3.connect(db)
rows = conn.execute('''
    SELECT symbol, timeframe, COUNT(*) as n,
           datetime(MIN(ts_ms)/1000, ''unixepoch'') as oldest,
           datetime(MAX(ts_ms)/1000, ''unixepoch'') as newest
    FROM candles
    GROUP BY symbol, timeframe
    ORDER BY symbol, timeframe
''').fetchall()
print(f'{\"Symbol\":<14} {\"TF\":<6} {\"Bars\":>8}  {\"Oldest\":<20} {\"Newest\"}')
print('-' * 70)
for sym, tf, n, old, new in rows:
    print(f'{sym:<14} {tf:<6} {n:>8}  {str(old):<20} {new}')
conn.close()
"

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Retrain on expanded multi-symbol dataset:"
Write-Host "       python -m hogan_bot.retrain --from-db --symbols BTC/USD,ETH/USD,SOL/USD --force-promote" -ForegroundColor White
Write-Host ""
Write-Host "  2. (Optional) Enable extended MTF features in .env after retraining:"
Write-Host "       HOGAN_USE_MTF_EXTENDED=true" -ForegroundColor White
Write-Host "       HOGAN_TRAINING_SYMBOLS=BTC/USD,ETH/USD,SOL/USD" -ForegroundColor White
Write-Host ""
Write-Host "  3. Restart the bot:"
Write-Host "       python -m hogan_bot.main" -ForegroundColor White
Write-Host ""
