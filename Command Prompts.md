**Daily Data Refresh Workflow**

\# Run everything at once (recommended):

python refresh\_daily.py

\# Or run individual free sources:

python -m hogan\_bot.fetch\_feargreed         # no key

python -m hogan\_bot.fetch\_coingecko         # COINGECKO\_KEY

python -m hogan\_bot.fetch\_gpr               # no key (cached 30 days)

python -m hogan\_bot.fetch\_derivatives       # Kraken Futures, no key

python -m hogan\_bot.fetch\_news\_sentiment    # CRYPTOPANIC\_KEY (free token)

\# Your active paid/keyed sources:

python -m hogan\_bot.fetch\_messari           # MESSARI\_KEY (on-chain: NVT, realized cap)

python -m hogan\_bot.fetch\_dune              # DUNE\_API\_KEY (BTC exchange flow, whales)

python -m hogan\_bot.fetch\_oanda             # OANDA\_ACCESS\_TOKEN (BTC/ETH/XAU/EUR prices)

\# Paid (run daily after subscribing):

python -m hogan\_bot.fetch\_glassnode         # GLASSNODE\_KEY

python -m hogan\_bot.fetch\_santiment         # SANTIMENT\_KEY





---

**STEP 0 — Open a terminal every time (Windows PowerShell)**

cd C:\Users\15125\Documents\Hogan\Hogan

.\.venv\Scripts\Activate.ps1

\# You should see (.venv) at the start of your prompt. If not, run:

python -m venv .venv

pip install -r requirements.txt

---

**Ubuntu / WSL Terminal Commands**

source ~/hogan-venv/bin/activate

cd /mnt/c/Users/15125/Documents/Hogan/Hogan

\# Ubuntu Training Command (heavy RL — run here for GPU speed):

python3 -m hogan\_bot.rl\_train --symbol BTC/USD --timeframe 5m --from-db --timesteps 500000

---

All commands below assume you're in the project directory with the venv active:



**Data accumulation (run continuously)**

while ($true) {

&nbsp;   python -m hogan\_bot.fetch\_data --symbol BTC/USD --timeframe 5m --limit 720

&nbsp;   Start-Sleep -Seconds 300

}

Fetches the latest candles every 5 minutes and upserts into data/hogan.db. Leave this running in its own terminal.



**RL training**

\# Smoke test (fast, just verifies pipeline works)

python -m hogan\_bot.rl\_train --symbol BTC/USD --timeframe 5m --from-db --timesteps 5000 --verbose 0

\# Real training (~3,000+ bars recommended)

python -m hogan\_bot.rl\_train --symbol BTC/USD --timeframe 5m --from-db --timesteps 500000

\# Quick re-train as more data accumulates (cheaper update)

python -m hogan\_bot.rl\_train --symbol BTC/USD --timeframe 5m --from-db --timesteps 200000

Backtesting



**# Standard backtest**

python -m hogan\_bot.backtest\_cli --symbol BTC/USD --limit 5000

\# With RL vote enabled (requires trained policy)

python -m hogan\_bot.backtest\_cli --symbol BTC/USD --limit 5000 --use-rl

\# Compare 5 signal configs side-by-side

python -m hogan\_bot.backtest\_cli --symbol BTC/USD --limit 5000 --compare

\# With ML filter

python -m hogan\_bot.backtest\_cli --symbol BTC/USD --limit 5000 --use-ml

\# With ICT signal

python -m hogan\_bot.backtest\_cli --symbol BTC/USD --limit 5000 --use-ict



**ML model retraining**

python -m hogan\_bot.retrain

Live / paper bot

python -m hogan\_bot.main



**Tests**

$env:PYTHONPATH = "c:\\Users\\15125\\Documents\\Hogan\\Hogan"

$env:PATH = "C:\\Users\\15125\\AppData\\Roaming\\Python\\Python311\\Scripts;" + $env:PATH

pytest tests/ -q

Check bar count in DB

python -c "

import sqlite3

conn = sqlite3.connect('data/hogan.db')

cur = conn.cursor()

cur.execute('SELECT symbol, timeframe, COUNT(\*), MIN(ts\_ms), MAX(ts\_ms) FROM candles GROUP BY symbol, timeframe')

for row in cur.fetchall(): print(row)

conn.close()

"



OPTIMIZATION COMMAND OPTIONS CURRENT - 



python -m hogan\_bot.optimize --symbol BTC/USD --timeframe 5m --limit 20000 --trials 200 --metric sharpe --max-drawdown 20



python -m hogan\_bot.optimize --symbol BTC/USD --timeframe 5m --limit 20000 `

&nbsp;   --trials 200 --metric sharpe --max-drawdown 20



python -m hogan\_bot.optimize --symbol BTC/USD --timeframe 5m --limit 5000 --trials 200 --metric sharpe --max-drawdown 20 --seed 42



**TRAINING WORKFLOW - CURRENT 1:34 3/4**

\# 1. Accumulate data (keep running in background)

python -m hogan\_bot.fetch\_data --symbol BTC/USD --timeframe 5m --limit 720



\# 2. Train (run once you have 3000+ bars for meaningful results)

python -m hogan\_bot.rl\_train --symbol BTC/USD --timeframe 5m --from-db --timesteps 200000



\# 3. Backtest with RL vote enabled

python -m hogan\_bot.backtest\_cli --symbol BTC/USD --limit 5000 --use-rl



**POLICY TRAINING-**

\# Train from the accumulated DB (once you have ~300+ 5m bars)

python -m hogan\_bot.rl\_train --symbol BTC/USD --timeframe 5m --from-db --timesteps 500000



\# Backtest with the RL vote added

python -m hogan\_bot.backtest\_cli --symbol BTC/USD --limit 5000 --use-rl

