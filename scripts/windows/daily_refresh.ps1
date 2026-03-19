# Hogan Daily Data Refresh — runs every morning via Windows Task Scheduler
# Refreshes all external data sources: Fear & Greed, CMC, CoinGecko, FRED,
# Alpaca, blockchain on-chain, DeFi Llama, Kraken futures, GPR, and more.
#
# Schedule: Daily at 7:00 AM
# Log:       C:\Users\15125\Documents\Hogan\Hogan\logs\daily_refresh.log

$HOGAN_ROOT = "C:\Users\15125\Documents\Hogan\Hogan"
$VENV_PYTHON = "$HOGAN_ROOT\.venv\Scripts\python.exe"
$LOG_DIR     = "$HOGAN_ROOT\logs"
$LOG_FILE    = "$LOG_DIR\daily_refresh.log"
$MAX_LOG_MB  = 10   # rotate when log exceeds 10 MB

# ── ensure log directory exists ──────────────────────────────────────────────
if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR | Out-Null }

# ── rotate log if too large ───────────────────────────────────────────────────
if (Test-Path $LOG_FILE) {
    $size_mb = (Get-Item $LOG_FILE).Length / 1MB
    if ($size_mb -gt $MAX_LOG_MB) {
        Move-Item $LOG_FILE "$LOG_FILE.bak" -Force
    }
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content $LOG_FILE ""
Add-Content $LOG_FILE "═══════════════════════════════════════════════════════════"
Add-Content $LOG_FILE "  Hogan Daily Refresh — $timestamp"
Add-Content $LOG_FILE "═══════════════════════════════════════════════════════════"

# ── activate venv and run refresh ────────────────────────────────────────────
Set-Location $HOGAN_ROOT

try {
    & $VENV_PYTHON scripts\data\refresh_daily.py 2>&1 | Tee-Object -Append -FilePath $LOG_FILE
    $exit_code = $LASTEXITCODE
} catch {
    Add-Content $LOG_FILE "ERROR: $_"
    $exit_code = 1
}

$end_time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content $LOG_FILE ""
Add-Content $LOG_FILE "Completed at $end_time — exit code: $exit_code"

exit $exit_code
