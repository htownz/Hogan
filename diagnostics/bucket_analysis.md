# Regime x Confidence Bucket Analysis

Date: 2026-03-16
Dataset: 17,587 candles (BTC/USD, 1h)
Config: canonical profile with volatile shorts enabled

## Long Side (10 trades)

| Bucket | N | Win% | Avg PnL% | Total PnL% |
|---|---|---|---|---|
| ranging\|medium | 10 | 50.0% | +1.08 | +10.84 |

**Finding:** All 10 longs are in a single bucket (`ranging|medium`), and that
bucket is profitable.  No other regime/confidence combination produced any
long entries.  The ML filter (91.4% kill rate on buys) is effectively doing
the pruning: only signals the model confidently endorses in ranging markets
survive to execution.

**Decision:** No long-side pruning changes.  Revisit when trade count exceeds
30 (requires either model improvement or ML threshold relaxation).

## Short Side (26 trades)

| Bucket | N | Win% | Avg PnL% | Total PnL% |
|---|---|---|---|---|
| trending_down\|high | 1 | 100.0% | +5.96 | +5.96 |
| volatile\|high | 24 | 29.2% | -0.06 | -1.36 |
| volatile\|medium | 1 | 100.0% | +1.69 | +1.69 |

**Finding:** `volatile|high` shorts (24 trades) are near-breakeven at
-0.06% per trade, producing a small cumulative drag of -1.36%.  The
`trending_down|high` short was highly profitable but n=1.

**Decision:** Keep volatile shorts enabled at `short_size_scale=0.50`.
The 24 trades add statistical volume for learning without significant
capital destruction (-1.36% total over 2 years is acceptable research
cost).  The `trending_down` short shows the concept works when the
regime aligns.

## Remaining Regime Blocks (20 signals)

Of the 52 sell signals passing all gates, 20 remain blocked by regime
(all in ranging, which has `allow_shorts=False`).  Ranging shorts are
deliberately excluded due to whipsaw risk in 44.6% of the market.

## Next Steps

1. Revisit long pruning when ML model is retrained with more signal
2. Monitor volatile|high shorts — if they persist as a drag after model
   improvement, tighten `quality_final_mult` for volatile from 1.20 to 1.40
3. Consider `trending_up|short` enablement only if data shows divergence
   shorts (shorting overbought uptrends) have positive expectancy
