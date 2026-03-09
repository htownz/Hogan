START UP
cd C:\Users\15125\Documents\Hogan\Hogan
.venv\Scripts\Activate
python -m hogan_bot.trader_service



Step 1 — Refresh today's macro data (run this now):

python refresh_daily.py
Step 2 — Retrain with the full 100k+ bars using the current 43-feature set:

python -m hogan_bot.retrain --from-db --window-bars 100000 --model-type xgboost --horizon-bars 12 --force-promote

Step 3 — Restart the bot with all the overnight fixes active:

python -m hogan_bot.main

Command	What it shows
!balance	Equity, cash, invested, unrealized P&L, total return
!positions	Open trades: entry price, qty, current price, P&L %
!pnl	Realized P&L from fill history
!fills	Last 10 executed trades
!status	Mode, model, ICT/EMA cloud enabled, ML thresholds
!market	Fear & Greed, BTC dominance, yield curve, DeFi TVL, funding rate
!signals	Live ML up-probability for BTC/USD and ETH/USD
!help	Full command list

Quick-reference cheat sheet
What	Command
Activate venv	.\.venv\Scripts\Activate.ps1
Fetch latest candles	python -m hogan_bot.fetch_data --symbol BTC/USD --timeframe 5m --limit 720
Daily data refresh	python refresh_daily.py
Retrain ML	python -m hogan_bot.retrain
Train RL (full)	python -m hogan_bot.rl_train --symbol BTC/USD --timeframe 5m --from-db --timesteps 500000
Backtest	python -m hogan_bot.backtest_cli --symbol BTC/USD --limit 5000
Run the bot	python -m hogan_bot.main
Dashboard	streamlit run dashboard.py
Run tests	pytest tests/ -q
Check Oanda account	python -m hogan_bot.fetch_oanda --account-summary
Recommended terminal layout: open 3 terminals — one for the data loop (Step 2), one for the bot (Step 6), and one free for commands.

**Daily Data Refresh Workflow**

# Run everything at once (recommended):

python refresh_daily.py

# Or run individual free sources:

python -m hogan_bot.fetch_feargreed         # no key

python -m hogan_bot.fetch_coingecko         # COINGECKO_KEY

python -m hogan_bot.fetch_gpr               # no key (cached 30 days)

python -m hogan_bot.fetch_derivatives       # Kraken Futures, no key

python -m hogan_bot.fetch_news_sentiment    # CRYPTOPANIC_KEY (free token)

# Your active paid/keyed sources:

python -m hogan_bot.fetch_messari           # MESSARI_KEY (on-chain: NVT, realized cap)

python -m hogan_bot.fetch_dune              # DUNE_API_KEY (BTC exchange flow, whales)

python -m hogan_bot.fetch_oanda             # OANDA_ACCESS_TOKEN (BTC/ETH/XAU/EUR prices)

# Paid (run daily after subscribing):

python -m hogan_bot.fetch_glassnode         # GLASSNODE_KEY

python -m hogan_bot.fetch_santiment         # SANTIMENT_KEY





---

**STEP 0 — Open a terminal every time (Windows PowerShell)**

cd C:\Users\15125\Documents\Hogan\Hogan

.\.venv\Scripts\Activate.ps1

# You should see (.venv) at the start of your prompt. If not, run:

python -m venv .venv

pip install -r requirements.txt

---

**Ubuntu / WSL Terminal Commands**

source ~/hogan-venv/bin/activate

cd /mnt/c/Users/15125/Documents/Hogan/Hogan

# Ubuntu Training Command (heavy RL — run here for GPU speed):

python3 -m hogan_bot.rl_train --symbol BTC/USD --timeframe 5m --from-db --timesteps 500000

---

All commands below assume you're in the project directory with the venv active:



**Data accumulation (run continuously)**

while ($true) {

&nbsp;   python -m hogan_bot.fetch_data --symbol BTC/USD --timeframe 5m --limit 720

&nbsp;   Start-Sleep -Seconds 300

}

Fetches the latest candles every 5 minutes and upserts into data/hogan.db. Leave this running in its own terminal.



**RL training**

# Smoke test (fast, just verifies pipeline works)

python -m hogan_bot.rl_train --symbol BTC/USD --timeframe 5m --from-db --timesteps 5000 --verbose 0

# Real training (~3,000+ bars recommended)



# Quick re-train as more data accumulates (cheaper update)

python -m hogan_bot.rl_train --symbol BTC/USD --timeframe 5m --from-db --timesteps 200000

Backtesting



**# Standard backtest**

python -m hogan_bot.backtest_cli --symbol BTC/USD --limit 5000

# With RL vote enabled (requires trained policy)

python -m hogan_bot.backtest_cli --symbol BTC/USD --limit 5000 --use-rl

# Compare 5 signal configs side-by-side

python -m hogan_bot.backtest_cli --symbol BTC/USD --limit 5000 --compare

# With ML filter

python -m hogan_bot.backtest_cli --symbol BTC/USD --limit 5000 --use-ml

# With ICT signal

python -m hogan_bot.backtest_cli --symbol BTC/USD --limit 5000 --use-ict



**ML model retraining**

python -m hogan_bot.retrain

Live / paper bot

python -m hogan_bot.main



**Tests**

$env:PYTHONPATH = "c:\\Users\\15125\\Documents\\Hogan\\Hogan"

$env:PATH = "C:\\Users\\15125\\AppData\\Roaming\\Python\\Python311\\Scripts;" + $env:PATH

pytest tests/ -q

Check bar count in DB

python -c "

import sqlite3

conn = sqlite3.connect('data/hogan.db')

cur = conn.cursor()

cur.execute('SELECT symbol, timeframe, COUNT(\*), MIN(ts_ms), MAX(ts_ms) FROM candles GROUP BY symbol, timeframe')

for row in cur.fetchall(): print(row)

conn.close()

"



OPTIMIZATION COMMAND OPTIONS CURRENT - 



python -m hogan_bot.optimize --symbol BTC/USD --timeframe 5m --limit 20000 --trials 200 --metric sharpe --max-drawdown 20



python -m hogan_bot.optimize --symbol BTC/USD --timeframe 5m --limit 20000 `

&nbsp;   --trials 200 --metric sharpe --max-drawdown 20



python -m hogan_bot.optimize --symbol BTC/USD --timeframe 5m --limit 5000 --trials 200 --metric sharpe --max-drawdown 20 --seed 42



**TRAINING WORKFLOW - CURRENT 1:34 3/4**

# 1. Accumulate data (keep running in background)

python -m hogan_bot.fetch_data --symbol BTC/USD --timeframe 5m --limit 720



# 2. Train (run once you have 3000+ bars for meaningful results)

python -m hogan_bot.rl_train --symbol BTC/USD --timeframe 5m --from-db --timesteps 200000



# 3. Backtest with RL vote enabled

python -m hogan_bot.backtest_cli --symbol BTC/USD --limit 5000 --use-rl



**POLICY TRAINING-**

# Train from the accumulated DB (once you have ~300+ 5m bars)

python -m hogan_bot.rl_train --symbol BTC/USD --timeframe 5m --from-db --timesteps 500000



# Backtest with the RL vote added

python -m hogan_bot.backtest_cli --symbol BTC/USD --limit 5000 --use-rl

