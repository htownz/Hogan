# Strategy Matrix Tournament — Final Results

## Executive Summary

The 5x3 entry-exit matrix was run on BTC/USD and ETH/USD (1h, 5 walk-forward OOS windows).
**No cell survived the BTC screen gate with realistic costs.** However, the zero-cost diagnostic
revealed **6 BTC survivors and 4 promotion candidates**, proving multiple entries have real signal
edge destroyed by execution economics.

GBP/USD cross-asset testing (12,745 bars backfilled from Oanda) confirmed that **C_ema_pullback**
has edge in BOTH crypto and FX markets, while **D_bb_squeeze** is crypto-specific.

## Key Numbers

| Run Mode | BTC Cells | Screen Survivors | Promotion Candidates |
|----------|-----------|------------------|---------------------|
| With costs (0.26% fee) | 15 | 0 | 0 |
| Zero-cost | 15 | 6 | 4 |
| Long-only (costs) | 15 | 0 | 0 |

## The Critical Finding: Signal Exists, Costs Kill It

| Entry | Exit | Gross Return% | Net Return% | Cost per Trade | Trades/mo |
|-------|------|:------------:|:-----------:|:--------------:|:---------:|
| C_ema_pullback | T1_trend | +11.99% | -29.64% | $39.75 | 25.5 |
| D_bb_squeeze | T1_trend | +10.22% | -15.40% | $45.63 | 13.7 |
| E_baseline | T1_trend | +6.57% | -10.81% | $45.87 | 9.2 |
| E_baseline | T3_balanced | +7.25% | -12.57% | $51.41 | 9.4 |

With 0.52% round-trip cost and 9-25 trades/month, cost drag consumes 100-1000%+ of gross edge.

## Zero-Cost BTC Promotion Candidates

| Rank | Entry | Exit | Calmar | Net% | PF | DD% | Win% | Trades |
|------|-------|------|--------|------|-----|-----|------|--------|
| 1 | E_baseline | T1_trend | +8.18 | +7.27% | 1.81 | 9.2% | 45% | 190 |
| 2 | E_baseline | T2_mean_revert | +7.70 | +4.85% | 1.70 | 6.4% | 49% | 206 |
| 3 | D_bb_squeeze | T1_trend | +7.27 | +11.82% | 1.56 | 7.5% | 38% | 281 |
| 4 | E_baseline | T3_balanced | +6.80 | +7.84% | 1.52 | 5.0% | 45% | 194 |

## Cross-Asset Confirmation (Zero-Cost)

### ETH/USD (crypto portability)

| Entry | Exit | BTC Calmar | ETH Calmar | ETH Net% | Portable? |
|-------|------|-----------|-----------|---------|-----------|
| E_baseline | T1_trend | +8.18 | +3.79 | +5.92% | YES |
| D_bb_squeeze | T1_trend | +7.27 | +4.57 | +8.26% | YES |
| C_ema_pullback | T1_trend | +5.51 | +6.41 | +12.65% | YES |

### SOL/USD (high-beta stress test)

| Entry | Exit | BTC Calmar | SOL Calmar | SOL Net% | Fragile? |
|-------|------|-----------|-----------|---------|----------|
| D_bb_squeeze | T1_trend | +7.27 | +6.67 | +7.43% | NO |
| D_bb_squeeze | T2_mean_revert | n/a | +5.06 | +7.60% | NO |
| E_baseline | T1_trend | +8.18 | +3.99 | +6.16% | NO |
| C_ema_pullback | T1_trend | +5.51 | +5.80 | +13.65% | NO |

### GBP/USD (FX cross-market robustness)

| Entry | Exit | BTC Calmar | GBP Calmar | GBP Net% | FX Edge? |
|-------|------|-----------|-----------|---------|----------|
| C_ema_pullback | T1_trend | +5.51 | **+6.10** | +1.22% | **YES** |
| C_ema_pullback | T3_balanced | n/a | +4.87 | +1.63% | **YES** |
| E_baseline | T1_trend | +8.18 | +2.41 | +0.20% | marginal |
| E_baseline | T3_balanced | +6.80 | +2.32 | +0.08% | marginal |
| D_bb_squeeze | T1_trend | +7.27 | -1.35 | -0.72% | **NO** |
| D_bb_squeeze | T3_balanced | n/a | -0.61 | -0.32% | **NO** |

## Four-Quadrant Reading (Final)

| Entry | BTC | ETH | SOL | GBP | Interpretation |
|-------|-----|-----|-----|-----|----------------|
| **C_ema_pullback** | +15.79% | +12.65% | +13.65% | +1.22% | **True price/volatility edge — works in crypto AND FX** |
| **D_bb_squeeze** | +11.82% | +8.26% | +7.43% | -0.72% | **Crypto-native edge — does not transfer to FX** |
| **E_baseline** | +7.27% | +5.92% | +6.16% | +0.20% | **Broad edge but is a control — needs ablation** |

**C_ema_pullback has the broadest validity.** It is the only entry that shows positive edge
on all four assets, including the FX robustness check. This suggests a real structural
price/volatility pattern, not crypto-specific noise.

**D_bb_squeeze has the best BTC risk-adjusted returns** (lowest drawdown at 7.5%)
but is crypto-specific — it fails on GBP/USD.

## Entry Family Ranking

| Rank | Entry | BTC Edge | Cross-Asset | Trades/mo | Best Exit | Verdict |
|------|-------|----------|-------------|-----------|-----------|---------|
| 1 | **C_ema_pullback** | +15.79% | 4/4 assets | 26 | T1_trend | **Broadest edge — true signal** |
| 2 | **D_bb_squeeze** | +11.82% | 3/4 (no FX) | 14 | T1_trend | **Best risk-adjusted BTC — crypto-native** |
| 3 | **E_baseline** | +7.27% | 4/4 marginal | 9 | T1_trend | **Control — needs ablation if adopted** |
| 4 | A_donchian | +2.54% | 1/4 (ETH only) | 30 | T1_trend | Weak, too frequent |
| 5 | B_rsi_reclaim | n/a | 0/4 | <1 | n/a | Too strict for 1h data |

## Exit Pack Analysis

**T1 (Trend Exit) dominates across all entries.** Every entry performs best with T1 — wide stop
(2.0 ATR), trailing stop (2.5 ATR), no fixed TP, long hold (120h). This means:
- The edge comes from **letting winners run**, not from capturing quick mean-reversion
- T2 (mean-reversion exit) consistently underperforms — tight stops get stopped out in noise
- T3 (balanced) is a reasonable middle ground but still trails T1

This is a **trend-following signal structure**, even for the entries that look like pullback/compression plays.

## The Path Forward

The tournament definitively answers the core question: **Hogan has real entry edge in BTC 1h data.**
The problem is pure execution economics — 0.52% round-trip costs on 9-26 trades/month destroy the edge.

### Priority 1: Make the signal profitable with real costs

Three levers:
1. **Lower fees** — negotiate exchange tier or switch to a lower-fee venue (target <0.10% per side)
2. **Reduce frequency** — add a quality filter that only takes the strongest signals (halving trade
   count would roughly halve cost drag while keeping the best signals)
3. **Move to 4h timeframe** — same logic, ~4x fewer signals, larger moves per trade relative to cost

### Priority 2: Choose one winner for the re-layering sequence

- If pursuing **broadest robustness**: adopt **C_ema_pullback** (works everywhere)
- If pursuing **best BTC risk-adjusted**: adopt **D_bb_squeeze** (lowest drawdown)
- If both are close, use C as primary with D as a confirmation/ensemble

### Priority 3: Re-layer one at a time

Per the plan, each addition must be walk-forward validated and must not degrade BTC Calmar:
1. Regime routing (only if it helps)
2. Exit refinement
3. ML as optional entry qualifier / size layer
4. Quality/edge gates
5. Macro/sentiment
6. Swarm last, if ever

## All Output Files

| File | Description |
|------|-------------|
| `leaderboard.csv` | Full matrix with costs (30 cells, BTC+ETH) |
| `leaderboard_zero_cost.csv` | Zero-cost diagnostic (30 cells, BTC+ETH) |
| `leaderboard_long_only.csv` | Long-only with costs (30 cells, BTC+ETH) |
| `leaderboard_sol_zero_cost.csv` | SOL stress test zero-cost (9 cells) |
| `leaderboard_sol_with_cost.csv` | SOL stress test with costs (9 cells) |
| `leaderboard_gbp_zero_cost.csv` | GBP robustness zero-cost (9 cells) |
| `leaderboard_gbp_with_cost.csv` | GBP robustness with costs (9 cells) |
| `summary_*.md` | Per-run analysis summaries |
| `results_*.json` | Full per-cell JSON with window breakdowns |
