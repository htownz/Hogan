# Execution Runbook — Live Execution vs Policy

Operator reference for when execution reality diverges from policy intent.
The gap between "what the strategy decided" and "what actually happened" is
where autonomy breaks down.

## Quick health check

```bash
# Prometheus endpoint (default port 8000)
curl -s http://localhost:8000/metrics | grep -E "hogan_(orders|order_fail|fills|slippage|rest_fallback|dead_man|gap_guard|order_failure_circuit)"
```

Key gauges (1 = active, 0 = normal):
- `hogan_rest_fallback_active` — WebSocket down, using REST polling
- `hogan_order_failure_circuit_open` — 5+ consecutive order failures
- `hogan_gap_guard_active` — candle data >4h stale, trading paused

Key counters:
- `hogan_orders_total` — every order attempt (by side, mode, exchange)
- `hogan_order_fail_total` — failed orders (same labels)
- `hogan_fills_total` — successful fills
- `hogan_slippage_bps` — histogram of realized slippage vs decision price
- `hogan_dead_man_alerts_total` — no candle data for >15 minutes

## Incident: WebSocket failures / REST fallback

**Symptoms**: `hogan_ws_reconnects_total` climbing, then `hogan_rest_fallback_active = 1`. Log lines: `WS error (...); reconnecting`.

**What happens**: After `HOGAN_WS_FAIL_THRESHOLD` (default 5) consecutive WS failures, `LiveDataEngine` auto-switches to REST polling. Trading continues but with higher latency (polling interval vs real-time).

**Env vars**:
| Var | Default | Purpose |
|-----|---------|---------|
| `HOGAN_USE_REST_DATA` | `false` | Force REST mode from startup (skip WS) |
| `HOGAN_WS_FAIL_THRESHOLD` | `5` | WS failures before auto-switch to REST |

**Operator action**:
1. Check exchange status page (Kraken, Binance, etc.)
2. If maintenance: set `HOGAN_USE_REST_DATA=1` until resolved, restart bot
3. If resolved: restart bot with `HOGAN_USE_REST_DATA` unset to restore WS
4. REST mode is safe — same data, just higher latency (~30s polling)

## Incident: Order failures

**Symptoms**: `hogan_order_fail_total` increasing. Log lines: `BUY_FAILED`, `SELL_FAILED`, `SHORT_FAILED`, `COVER_SHORT_FAILED`, `AUTO_EXIT_*_FAILED`.

**What happens**: The executor returns `ExecResult(ok=False)` or the call throws. The event loop logs the failure and skips the trade. No retry at the event-loop level (SmartExecution has its own reprice loop for limit orders).

**Circuit breaker**: After 5 consecutive failures, `hogan_order_failure_circuit_open = 1`. This is a **warning signal** — the bot does not auto-halt on order failures (only on drawdown breach). Operator should investigate immediately.

**Common causes**:
- Exchange maintenance → check status page
- Insufficient funds / margin → check exchange balance
- Rate limiting → reduce `HOGAN_TRADE_INTERVAL` or check API key permissions
- Network issues → check connectivity to exchange API

**Operator action**:
1. Check `hogan_order_fail_total` labels for pattern (all sides? one exchange?)
2. Verify exchange connectivity: `python -c "from hogan_bot.exchange import ExchangeClient; c = ExchangeClient(); print(c.fetch_ticker('BTC/USD'))"`
3. If exchange is down: wait for recovery; bot will retry on next signal
4. If persistent: restart bot, check API key validity

## Incident: Stale data / Dead-man switch

**Symptoms**: `hogan_dead_man_alerts_total` incrementing, `hogan_data_lag_seconds` > 900, `hogan_gap_guard_active = 1`.

**Two thresholds**:
1. **Dead-man (15 min)**: No candle received for >15 minutes. Bot logs warning and notifies webhook. **Trading continues** (this is a warning, not a halt).
2. **GAP_GUARD (4h)**: Candle timestamp >4h old. Bot **skips trading** for that iteration. At >8h, sends webhook notification.

**Operator action**:
1. Check data engine: is WS connected? Is REST polling returning data?
2. Check `hogan_candles_received_total` — is it flat?
3. If exchange is returning data but candles are stale: possible timezone/timestamp parsing bug
4. If no data at all: check exchange status, network, API keys

## Incident: Drawdown circuit breaker

**Symptoms**: Bot stops trading. Log: `Drawdown limit hit: equity=X peak=Y`. `hogan_drawdown` gauge at or above `max_drawdown`.

**What happens**: `DrawdownGuard.update_and_check()` returns `False`. Bot calls `emergency_flatten()` for all open positions, then **breaks out of the main loop** (clean shutdown).

**This is the only automatic trading halt.** All other issues (dead-man, order failures, stale data) are warnings that require operator judgment.

**Env var**: `HOGAN_MAX_DRAWDOWN` (default in config; e.g., 0.15 = 15%)

**Operator action**:
1. Review trade log — was the drawdown from a single bad trade or accumulation?
2. Check if emergency_flatten succeeded (log: `FLATTEN ... ok=True/False`)
3. If flatten failed: manually close positions on exchange
4. Do NOT restart bot until root cause is understood
5. Consider adjusting `max_drawdown` only after walk-forward validation

## Incident: High slippage

**Symptoms**: `hogan_slippage_bps` histogram showing values >25 bps.

**What happens**: The `execution_health` module computes slippage as `|fill_price - decision_price| / decision_price * 10000`. For paper mode, slippage is simulated via `HOGAN_SLIPPAGE_BPS`.

**Env vars**:
| Var | Default | Purpose |
|-----|---------|---------|
| `HOGAN_SLIPPAGE_BPS` | `5.0` | Simulated slippage for paper mode |
| `HOGAN_SPREAD_HALF_BPS` | `3.0` | Half-spread for realistic paper fills |
| `HOGAN_REALISTIC_PAPER` | `false` | Enable `RealisticPaperExecution` |
| `HOGAN_SMART_EXEC` | `false` | Use limit order execution with repricing |

**Operator action**:
1. Compare paper slippage settings against live observed slippage
2. If live slippage >> paper: increase `HOGAN_SLIPPAGE_BPS` for more realistic backtests
3. Consider `HOGAN_SMART_EXEC=true` for limit order execution (lower slippage, risk of non-fill)

## Incident: Kill switch / emergency halt

Hogan does not have a single "kill switch" env var. To stop trading:

1. **Graceful**: Send SIGTERM to the process. The event loop catches it and exits cleanly.
2. **Immediate**: Send SIGKILL / `Stop-Process -Id <pid> -Force`
3. **Prevent restart**: Set `HOGAN_LIVE_MODE=false` and `HOGAN_LIVE_ACK=""` in `.env`
4. **Flatten only**: The drawdown circuit handles this automatically. For manual flatten, use the exchange UI or API directly.

## Metrics alignment — what operators should watch

| Metric | What it tells you | Alert threshold |
|--------|-------------------|-----------------|
| `hogan_orders_total` | Trading activity | Flat for >2h when market is open = stall |
| `hogan_order_fail_total` | Execution reliability | Any increase needs investigation |
| `hogan_fills_total` | Confirmed executions | Should track `orders - fails` |
| `hogan_slippage_bps` | Execution quality | p95 > 50 bps = review execution path |
| `hogan_rest_fallback_active` | Data feed degradation | 1 = check WS status |
| `hogan_order_failure_circuit_open` | Repeated failures | 1 = immediate investigation |
| `hogan_gap_guard_active` | Data staleness | 1 = trading paused |
| `hogan_dead_man_alerts_total` | Data feed health | Any increase = check engine |
| `hogan_drawdown` | Risk state | >80% of max = prepare for circuit |
| `hogan_equity` | Account health | Unexpected drops = check fills |

## Architecture reference

- `hogan_bot/execution_health.py` — `ExecutionHealthState`, `record_exec_outcome()`
- `hogan_bot/event_loop.py` — wires health state into every `_safe_exec` call
- `hogan_bot/execution.py` — `ExecResult`, `LiveExecution`, `SmartExecution`
- `hogan_bot/data_engine.py` — `LiveDataEngine`, WS/REST fallback, `check_dead_man()`
- `hogan_bot/metrics.py` — all Prometheus counter/gauge/histogram definitions
- `hogan_bot/risk.py` — `DrawdownGuard`
