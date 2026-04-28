# Polymarket Alpha Lab

Hogan's Polymarket Alpha Lab is an analysis-only research layer for finding
prediction-market opportunities. It uses public market data, stores ranked
candidate snapshots, evaluates after-cost expected value, and tracks shadow
trades before any live-trading consideration.

## Guardrails

- Do not use VPNs or other methods to bypass geographic restrictions.
- Do not store private keys, wallet seed phrases, relayer keys, or API secrets
  in the repository.
- Do not place real Polymarket orders from this phase.
- Do not use non-public information, manipulation, wash trading, or reward abuse.
- Treat every candidate as research until it passes shadow-trade evidence and
  settlement-rule review.

## Public Data Scan

Run a public scan without authenticated trading:

```bash
python -m hogan_bot.fetch_polymarket --scan --limit 100 --no-clob --btc-prob 0.60 --eth-prob 0.55
```

Use CLOB midpoint and spread enrichment when public endpoints are reachable:

```bash
python -m hogan_bot.fetch_polymarket --scan --limit 100 --clob-limit 12 --btc-prob 0.60 --eth-prob 0.55
```

The scanner writes compact metrics to `onchain_metrics` and stores ranked
candidate snapshots in `polymarket_opportunities`.

## Opportunity Review

The scanner ranks candidates by:

- Hogan-vs-crowd disagreement when BTC/ETH probabilities are supplied.
- Liquidity score.
- Spread score.
- Catalyst relevance.
- Crowd probability and directional semantics.

Before shadow-trading a candidate, review:

- The exact question and resolution source.
- Whether `YES` is bullish, bearish, or ambiguous.
- Whether related markets contradict the candidate.
- Whether spreads and liquidity are realistic enough to trade.
- Whether the market is legal and eligible for the user.

## Shadow Ledger

Shadow trades are hypothetical positions only. Use them to measure:

- Hit rate.
- Mark-to-market PnL.
- Average PnL.
- Opportunity decay.
- Calibration and rule ambiguity failures.

Promotion to live-ready design requires sustained positive evidence, not a
single good scan.

## Standalone Program

Run Polymarket separately from the Hogan exchange-trading event loop:

```powershell
py -3.11 -m hogan_bot.polymarket_service --mode scan --db data/hogan.db --clob-limit 50
```

Use the no-write recommendation path when you only want machine reasoning:

```powershell
py -3.11 -m hogan_bot.polymarket_service --mode recommendations-only --db data/hogan.db --clob-limit 50
```

Run as a scheduled daemon loop:

```powershell
py -3.11 -m hogan_bot.polymarket_service --mode daemon --interval-minutes 30 --authority-mode shadow --clob-limit 50
```

Service defaults can be set with environment variables:

- `HOGAN_POLYMARKET_MODE`
- `HOGAN_POLYMARKET_AUTHORITY_MODE`
- `HOGAN_POLYMARKET_INTERVAL_MINUTES`
- `HOGAN_POLYMARKET_CLOB_LIMIT`
- `HOGAN_POLYMARKET_REPORT_DIR`
- `HOGAN_POLYMARKET_MAX_OPEN_SHADOW_TRADES`
- `HOGAN_POLYMARKET_MAX_OPEN_SHADOW_EXPOSURE`
- `HOGAN_POLYMARKET_WATCHLIST_EV_MARGIN`

The standalone service still uses the shared Hogan DB for BTC candles, latest
ML probability, Polymarket shadow evidence, and promotion metrics. It does not
load wallet credentials or place real Polymarket orders.

## Promotion Gate

The current promotion gate requires:

- A minimum number of closed shadow trades.
- Positive total and average PnL.
- Minimum win rate.
- No live-trading code path unless explicitly added later with separate safety
  controls.

Any future live implementation must add geoblock checks, explicit opt-in flags,
capital limits, kill switches, and credential isolation.
