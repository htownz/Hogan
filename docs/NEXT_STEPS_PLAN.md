# Hogan Enhancement Plan — Take It to the Next Level

A prioritized roadmap for immediate improvements and medium-term upgrades. Based on current state: bot is trading (data-gathering mode), backtest ~-20%, ML AUC ~0.53, 101k+ candles in DB.

---

## Phase 1: Quick Wins (This Week)

### 1.1 Fix Weekly Retrain Script (Critical Bug)
**Problem:** `scripts/weekly_retrain.ps1` uses `--stock-only` which **skips crypto bars**. The retrain gets no new BTC/ETH data.

**Fix:** Remove `--stock-only` from the fetch_alpaca call. Use:
```powershell
& $VENV_PYTHON -m hogan_bot.fetch_alpaca --crypto-bars --timeframe 5Min --crypto-days 30 2>&1
```
Or use `--backfill-all` for full MTF + macro refresh.

**Also:** Add `--symbols "BTC/USD,ETH/USD"` and `--use-paper-labels` to the retrain command so paper-trade feedback is used. (Note: paper labels are currently skipped for multi-symbol — see 2.3.)

---

### 1.2 Ensure ETH Data Is Fetched
**Problem:** Your fetch loop only ran `BTC/USD`. The bot trades both BTC and ETH. ETH needs candles too.

**Fix:** Either:
- Add ETH to your fetch loop: `--symbol ETH/USD` in a second loop, or
- Use `fetch_data.py` with both symbols, or
- Run `backfill_mtf.ps1` periodically to keep both symbols + MTF + macro fresh.

---

### 1.3 Add Trade Explanations to Dashboard
**Problem:** Mistral generates explanations for every trade, but the dashboard doesn't show them.

**Fix:** In `dashboard.py`, join `trade_explanations` to the trade journal table and add an "Explanation" column to the Trade Journal section. Query:
```sql
SELECT fill_id, explanation, model_used FROM trade_explanations
```

---

### 1.4 Verify Task Scheduler
**Action:** Run `scripts\register_tasks.ps1` as Administrator (if not already). Confirm in `taskschd.msc` that:
- `Hogan_DailyRefresh` runs daily at 7:00 AM
- `Hogan_WeeklyRetrain` runs Sunday at 3:00 AM

---

## Phase 2: Strategy & ML Improvements (2–4 Weeks)

### 2.1 Test 1h Timeframe
**Why:** 5m moves are often 0.05–0.30%; round-trip fee is 0.52%. You need larger moves to overcome fees. 1h typical moves: 0.5–2%.

**How:**
1. Backfill 1h candles: `python -m hogan_bot.fetch_data --symbol BTC/USD --timeframe 1h --limit 5000`
2. Retrain on 1h: `python -m hogan_bot.retrain --from-db --symbol BTC/USD --timeframe 1h --window-bars 2000`
3. Backtest: `python -m hogan_bot.backtest_cli --from-db --symbol BTC/USD --timeframe 1h --limit 2000`
4. If backtest improves, set `HOGAN_TIMEFRAME=1h` in `.env` and retrain for all symbols.

---

### 2.2 Enable Extended MTF (10m + 30m Features)
**What:** Use 10m and 30m candle features in addition to 5m. More context = better signals.

**Steps:**
1. Ensure 10m/30m data exists: `scripts\backfill_mtf.ps1` (already run)
2. Retrain with extended MTF:
   ```powershell
   python -m hogan_bot.retrain --from-db --symbols "BTC/USD,ETH/USD" --use-extended-mtf --force-promote
   ```
3. Add to `.env`: `HOGAN_USE_MTF_EXTENDED=true`
4. Restart bot.

---

### 2.3 Add Paper Labels to Multi-Symbol Retrain
**Problem:** `HOGAN_RETRAIN_USE_PAPER_LABELS=true` but retrain with `--symbols "BTC/USD,ETH/USD"` skips paper labels (not yet implemented for multi-symbol).

**Fix:** Extend `retrain.py` to support paper labels for multi-symbol: for each symbol, call `make_paper_trade_labels(db_path, candles_sym, symbol)`, concatenate the extra (X_extra, y_extra) rows, and pass to training. Medium effort, high value for feedback loop.

---

### 2.4 Wire Online Learner
**What:** `hogan_bot.online_learner` exists but isn't called. It can incrementally update the model from new trade outcomes.

**How:**
- Option A: Run as a separate process: `python -m hogan_bot.online_learner --db data/hogan.db --interval 3600`
- Option B: Call `OnlineLearner(db_path).update(symbol)` from `main.py` after every N closed trades (e.g. 20). Requires careful integration so it doesn't block the trade loop.

---

### 2.5 Tune Horizon and Model Type
**Experiments:**
- `--horizon-bars 6` (30 min) vs `3` (15 min) vs `12` (1h) — longer horizon may reduce noise
- Try `--model-type lightgbm` — often faster and comparable to XGBoost
- Run `--tune` with Optuna for hyperparameter search (already supported in retrain)

---

## Phase 3: Observability & Automation (Ongoing)

### 3.1 Run Dashboard Regularly
**Action:** Start the dashboard when you're monitoring:
```powershell
streamlit run dashboard.py
```
Or add a Task Scheduler task to keep it running. Auto-refresh (30s) is already built in.

---

### 3.2 Add Trade Review to Your Routine
**Script:** `scripts\trade_review.py` — run weekly to see win rate, P&L by symbol/side, close reasons.

```powershell
python scripts\trade_review.py
```

---

### 3.3 Shorter Retrain Interval
**Current:** Weekly retrain. **Proposal:** Retrain every 24–48 hours while gathering data. Use:
```powershell
python -m hogan_bot.retrain --from-db --symbols "BTC/USD,ETH/USD" --schedule 24
```
Run in a separate terminal or as a scheduled task.

---

## Phase 4: Advanced (1–2 Months)

### 4.1 RL Agent as Additional Vote
**What:** PPO policy can vote alongside MA, EMA cloud, ICT. Requires trained policy.

**Steps:**
1. Train RL policy: `python -m hogan_bot.rl_train --symbol BTC/USD --timeframe 5m --steps 50000`
2. Set `HOGAN_USE_RL_AGENT=true` in `.env`
3. Ensure policy path is correct in config.

---

### 4.2 Triple-Barrier Labeling
**What:** Label trades by profit target, stop loss, and max hold time — not just "price up/down in N bars." Produces cleaner training labels.

**Status:** Check if `labeler.py` or `ml_advanced.py` has triple-barrier; integrate into `build_training_set` if not.

---

### 4.3 Champion/Challenger
**What:** Run two models in parallel; promote challenger only when it outperforms champion on a rolling window. Reduces regret from bad promotions.

---

### 4.4 Fee-Aware Backtest
**What:** Add a "min move" filter in backtest: only count a trade as profitable if price move > 2× round-trip fee. Surfaces whether the strategy has edge after costs.

---

## Summary Checklist

| Priority | Task | Effort | Impact |
|---------|------|--------|--------|
| P0 | Fix weekly_retrain.ps1 (remove --stock-only) | 5 min | High |
| P0 | Ensure ETH data fetched | 10 min | High |
| P1 | Add trade explanations to dashboard | 30 min | Medium |
| P1 | Verify Task Scheduler | 5 min | Medium |
| P2 | Test 1h timeframe | 1–2 hrs | High |
| P2 | Enable extended MTF | 30 min | Medium |
| P2 | Paper labels for multi-symbol | 2–3 hrs | High |
| P2 | Wire online learner | 1 hr | Medium |
| P3 | Run dashboard regularly | 5 min | Medium |
| P3 | Shorter retrain interval | 15 min | Medium |
| P4 | RL agent | 4+ hrs | High |
| P4 | Triple-barrier labeling | 2–4 hrs | High |

---

*Generated from Hogan codebase analysis. Adjust priorities based on your goals.*
