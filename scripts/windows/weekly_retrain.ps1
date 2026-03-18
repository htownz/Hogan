# Hogan Weekly Model Retrain — runs every Sunday via Windows Task Scheduler
# 1. Backfills latest 30 days of Alpaca 5m candles (catches any gaps)
# 2. Retrains XGBoost on the full 100k+ bar history
# 3. Promotes the new model only if ROC-AUC improves by >= 0.005
#
# Schedule: Every Sunday at 3:00 AM (after daily_refresh at 7:00 AM Saturday)
# Log:       C:\Users\15125\Documents\Hogan\Hogan\logs\weekly_retrain.log

$HOGAN_ROOT  = "C:\Users\15125\Documents\Hogan\Hogan"
$VENV_PYTHON = "$HOGAN_ROOT\.venv\Scripts\python.exe"
$LOG_DIR     = "$HOGAN_ROOT\logs"
$LOG_FILE    = "$LOG_DIR\weekly_retrain.log"
$MAX_LOG_MB  = 20

if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR | Out-Null }
if (Test-Path $LOG_FILE) {
    $size_mb = (Get-Item $LOG_FILE).Length / 1MB
    if ($size_mb -gt $MAX_LOG_MB) { Move-Item $LOG_FILE "$LOG_FILE.bak" -Force }
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content $LOG_FILE ""
Add-Content $LOG_FILE "═══════════════════════════════════════════════════════════"
Add-Content $LOG_FILE "  Hogan Weekly Retrain — $timestamp"
Add-Content $LOG_FILE "═══════════════════════════════════════════════════════════"

Set-Location $HOGAN_ROOT

# ── Step 1: backfill last 30 days of 5m candles from Alpaca ─────────────────
# NOTE: --stock-only was WRONG (skips crypto). Use --crypto-bars without it.
Add-Content $LOG_FILE ""
Add-Content $LOG_FILE "[1/3] Backfilling Alpaca 5m candles (30 days)..."
& $VENV_PYTHON -m hogan_bot.fetch_alpaca --crypto-bars --timeframe 5Min --crypto-days 30 --symbols "BTC/USD,ETH/USD" 2>&1 |
    Tee-Object -Append -FilePath $LOG_FILE

# ── Step 2: daily data refresh (ensure latest macro signals) ─────────────────
Add-Content $LOG_FILE ""
Add-Content $LOG_FILE "[2/3] Running daily data refresh..."
& $VENV_PYTHON scripts\data\refresh_daily.py 2>&1 | Tee-Object -Append -FilePath $LOG_FILE

# ── Step 3: retrain XGBoost on full history ───────────────────────────────────
# Multi-symbol + paper labels (when 5+ closed trades exist)
Add-Content $LOG_FILE ""
Add-Content $LOG_FILE "[3/3] Retraining XGBoost model (100k bars, 1h horizon)..."
& $VENV_PYTHON -m hogan_bot.retrain `
    --from-db `
    --symbols "BTC/USD,ETH/USD" `
    --window-bars 100000 `
    --model-type xgboost `
    --horizon-bars 12 `
    --use-paper-labels `
    2>&1 | Tee-Object -Append -FilePath $LOG_FILE

$exit_code = $LASTEXITCODE
$end_time  = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content $LOG_FILE ""
Add-Content $LOG_FILE "Retrain complete at $end_time — exit code: $exit_code"

exit $exit_code
