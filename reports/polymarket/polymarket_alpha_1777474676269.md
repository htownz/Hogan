# Polymarket Alpha Lab Report

- Timestamp ms: `1777474676269`
- Candidates reviewed: `9`
- Arbitrage alerts: `18`
- Shadow marked open: `0`
- Shadow closed: `0`
- Shadow open exposure: `$0.00`
- Shadow unrealized PnL: `$0.00`
- Closed shadow PnL: `$0.00`
- Closed shadow win rate: `0.00%`
- Max drawdown: `$0.00`
- Shadow category coverage: `0`
- Fair-value source coverage: `0`
- Calibrated fair-value trades: `0`
- Hogan BTC probability: `0.3522`
- Hogan ETH probability: `n/a`
- Promotion approved: `False`
- Authority mode: `research`
- Promotion blockers: `insufficient_shadow_trades:0<50, total_pnl_below_gate:0.00<=0.00, avg_pnl_below_gate:0.00<0.10, win_rate_below_gate:0.00<0.55, market_type_coverage_below_gate:0<1, quality_weighted_pnl_below_gate:0.00<=0.00`

## Data Quality

- Average data quality: `0.6589`
- Shadow candidates: `0`
- Watchlist near-misses: `1`
- Category coverage: `crypto_policy:n=3,q=0.61,watch=0,shadow=0; crypto_price_target:n=3,q=0.76,watch=1,shadow=0; crypto_treasury:n=3,q=0.61,watch=0,shadow=0`
- Research-only: `7`
- Avoid: `0`

## Category Coverage

- `crypto_policy` count=`3` avg_quality=`0.6079` watchlist=`0` shadow_candidates=`0`
- `crypto_price_target` count=`3` avg_quality=`0.7612` watchlist=`1` shadow_candidates=`0`
- `crypto_treasury` count=`3` avg_quality=`0.6075` watchlist=`0` shadow_candidates=`0`

## Machine Recommendations

### 1. MicroStrategy sells any Bitcoin by June 30, 2026?
- Recommendation: `research`
- Evidence score: `0.7192`
- Confidence: `0.3740`
- Recommended size: `$0.00`
- Fair-value source: `market_implied_only`
- Evidence source: `crypto_treasury_public_context` confidence=`0.3000`
- CLOB diagnostic: `skipped_limit` - CLOB enrichment limit reached (50)
- Risk flags: `clob_skipped_limit, gamma_price_only, missing_clob_spread, non_positive_after_cost_ev`
- Thesis: Informational market; prediction-market crowd probability 0.03.

### 2. MicroStrategy sells any Bitcoin by December 31, 2026?
- Recommendation: `research`
- Evidence score: `0.7015`
- Confidence: `0.3648`
- Recommended size: `$0.00`
- Fair-value source: `market_implied_only`
- Evidence source: `crypto_treasury_public_context` confidence=`0.3000`
- CLOB diagnostic: `skipped_limit` - CLOB enrichment limit reached (50)
- Risk flags: `clob_skipped_limit, gamma_price_only, missing_clob_spread, non_positive_after_cost_ev`
- Thesis: Informational market; prediction-market crowd probability 0.10.

### 3. Trump eliminates capital gains tax on crypto before 2027?
- Recommendation: `research`
- Evidence score: `0.7003`
- Confidence: `0.3642`
- Recommended size: `$0.00`
- Fair-value source: `market_implied_only`
- Evidence source: `crypto_policy_public_context` confidence=`0.2500`
- CLOB diagnostic: `skipped_limit` - CLOB enrichment limit reached (50)
- Risk flags: `clob_skipped_limit, gamma_price_only, missing_clob_spread, non_positive_after_cost_ev`
- Thesis: Informational market; prediction-market crowd probability 0.04.

### 4. Will Bitcoin hit $150k by June 30, 2026?
- Recommendation: `monitor`
- Evidence score: `0.6979`
- Confidence: `0.6979`
- Recommended size: `$0.00`
- Fair-value source: `calibrated_long_horizon`
- Evidence source: `crypto_price_history` confidence=`0.6500`
- CLOB diagnostic: `ok` - CLOB midpoint and spread fetched
- Risk flags: `none`
- Thesis: Model/crowd disagreement with after-cost EV 0.0222; Hogan 0.00 vs crowd 0.03.
- Long-horizon model: `prob=0.0000 spot=$71,261 target=$150,000 days=62 drift=-18.38% vol=44.84% n=376`

### 5. SCOTUS accepts sports event contract case by July 31, 2026?     
- Recommendation: `research`
- Evidence score: `0.6956`
- Confidence: `0.3617`
- Recommended size: `$0.00`
- Fair-value source: `market_implied_only`
- Evidence source: `crypto_policy_public_context` confidence=`0.2500`
- CLOB diagnostic: `skipped_limit` - CLOB enrichment limit reached (50)
- Risk flags: `clob_skipped_limit, gamma_price_only, missing_clob_spread, non_positive_after_cost_ev`
- Thesis: Informational market; prediction-market crowd probability 0.14.

### 6. Will El Salvador hold $1b+ of BTC by December 31, 2026?
- Recommendation: `research`
- Evidence score: `0.6406`
- Confidence: `0.3331`
- Recommended size: `$0.00`
- Fair-value source: `market_implied_only`
- Evidence source: `crypto_treasury_public_context` confidence=`0.3000`
- CLOB diagnostic: `skipped_limit` - CLOB enrichment limit reached (50)
- Risk flags: `clob_skipped_limit, gamma_price_only, missing_clob_spread, non_positive_after_cost_ev`
- Thesis: Informational market; prediction-market crowd probability 0.23.

### 7. SCOTUS accepts sports event contract case by December 31, 2026?
- Recommendation: `research`
- Evidence score: `0.6020`
- Confidence: `0.3131`
- Recommended size: `$0.00`
- Fair-value source: `market_implied_only`
- Evidence source: `crypto_policy_public_context` confidence=`0.2500`
- CLOB diagnostic: `skipped_limit` - CLOB enrichment limit reached (50)
- Risk flags: `clob_skipped_limit, gamma_price_only, missing_clob_spread, non_positive_after_cost_ev`
- Thesis: Informational market; prediction-market crowd probability 0.27.

### 8. Will Bitcoin hit $150k by December 31, 2026?
- Recommendation: `monitor`
- Evidence score: `0.5397`
- Confidence: `0.3454`
- Recommended size: `$0.00`
- Fair-value source: `calibrated_long_horizon`
- Evidence source: `crypto_price_history` confidence=`0.6500`
- CLOB diagnostic: `skipped_limit` - CLOB enrichment limit reached (50)
- Risk flags: `clob_skipped_limit, gamma_price_only, missing_clob_spread`
- Thesis: Model/crowd disagreement with after-cost EV 0.0470; Hogan 0.01 vs crowd 0.10.
- Watchlist trigger: `near_shadow_ev_threshold; needs_ev=+0.0030; trigger_NO<=0.9020`
- Long-horizon model: `prob=0.0055 spot=$71,261 target=$150,000 days=246 drift=-18.38% vol=44.84% n=376`

### 9. Will bitcoin hit $1m before GTA VI?
- Recommendation: `research`
- Evidence score: `0.4870`
- Confidence: `0.1364`
- Recommended size: `$0.00`
- Fair-value source: `unavailable`
- Evidence source: `crypto_price_history` confidence=`0.6500`
- CLOB diagnostic: `skipped_limit` - CLOB enrichment limit reached (50)
- Risk flags: `clob_skipped_limit, fair_value_unavailable, gamma_price_only, long_horizon_price_target_requires_calibrated_fair_value, missing_clob_spread, non_positive_after_cost_ev`
- Thesis: Research-only: long_horizon_price_target_requires_calibrated_fair_value.


## Watchlist

### 1. Will Bitcoin hit $150k by December 31, 2026?
- Recommendation: `monitor`
- After-cost EV: `0.0470`
- Trigger: `near_shadow_ev_threshold; needs_ev=+0.0030; trigger_NO<=0.9020`
- Fair-value source: `calibrated_long_horizon`


## Shadow Positions

- `#10` `cancelled` `buy_no` entry=`0.4895` current=`0.4895` size=`$24.35` pnl=`$0.00` will-bitcoin-hit-1m-before-gta-vi-872
- `#8` `cancelled` `buy_yes` entry=`0.0135` current=`0.0135` size=`$25.00` pnl=`$0.00` will-bitcoin-hit-150k-by-june-30-2026
- `#9` `cancelled` `buy_yes` entry=`0.0950` current=`0.0950` size=`$18.67` pnl=`$0.00` will-bitcoin-hit-150k-by-december-31-2026
- `#1` `cancelled` `buy_yes` entry=`0.0005` current=`0.0005` size=`$25.00` pnl=`$0.00` will-beth-van-duyne-win-the-2026-republican-primary
- `#2` `cancelled` `buy_yes` entry=`0.0355` current=`0.0355` size=`$25.00` pnl=`$0.00` will-netherlands-win-the-2026-fifa-world-cup-739
- `#3` `cancelled` `buy_yes` entry=`0.0005` current=`0.0005` size=`$24.59` pnl=`$0.00` will-club-brugge-win-the-202526-champions-league
- `#4` `cancelled` `buy_yes` entry=`0.0005` current=`0.0005` size=`$24.13` pnl=`$0.00` will-manchester-united-win-the-202526-english-premier-league
- `#5` `cancelled` `buy_yes` entry=`0.0365` current=`0.0365` size=`$17.43` pnl=`$0.00` trump-eliminates-capital-gains-tax-on-crypto-before-2027
- `#6` `cancelled` `buy_yes` entry=`0.1155` current=`0.1155` size=`$22.87` pnl=`$0.00` will-atletico-madrid-win-the-202526-champions-league
- `#7` `cancelled` `buy_yes` entry=`0.1350` current=`0.1350` size=`$16.71` pnl=`$0.00` scotus-accepts-sports-event-contract-case-by-july-31-2026

## Top Candidates

### 1. MicroStrategy sells any Bitcoin by June 30, 2026?
- Side: `research`
- Decision: `research`
- Recommendation: `research`
- Category id: `crypto_treasury`
- Market type: `crypto_treasury` / `long_term`
- Evidence source: `crypto_treasury_public_context` confidence=`0.3000`
- Data quality: `0.6182`
- CLOB diagnostic: `skipped_limit` - CLOB enrichment limit reached (50)
- Confidence: `0.3740`
- Recommended size: `$0.00`
- Total score: `0.7705`
- After-cost EV: `-0.0425`
- Crowd probability: `0.0260`
- Hogan probability: `n/a`
- Rationale: prediction-market crowd probability 0.03
- Reject reasons: `research_only_side, non_positive_ev`

### 2. MicroStrategy sells any Bitcoin by December 31, 2026?
- Side: `research`
- Decision: `research`
- Recommendation: `research`
- Category id: `crypto_treasury`
- Market type: `crypto_treasury` / `long_term`
- Evidence source: `crypto_treasury_public_context` confidence=`0.3000`
- Data quality: `0.6005`
- CLOB diagnostic: `skipped_limit` - CLOB enrichment limit reached (50)
- Confidence: `0.3648`
- Recommended size: `$0.00`
- Total score: `0.7564`
- After-cost EV: `-0.0425`
- Crowd probability: `0.0950`
- Hogan probability: `n/a`
- Rationale: prediction-market crowd probability 0.10
- Reject reasons: `research_only_side, non_positive_ev`

### 3. Trump eliminates capital gains tax on crypto before 2027?
- Side: `research`
- Decision: `research`
- Recommendation: `research`
- Category id: `crypto_policy`
- Market type: `crypto_policy` / `long_term`
- Evidence source: `crypto_policy_public_context` confidence=`0.2500`
- Data quality: `0.5993`
- CLOB diagnostic: `skipped_limit` - CLOB enrichment limit reached (50)
- Confidence: `0.3642`
- Recommended size: `$0.00`
- Total score: `0.7555`
- After-cost EV: `-0.0425`
- Crowd probability: `0.0365`
- Hogan probability: `n/a`
- Rationale: prediction-market crowd probability 0.04
- Reject reasons: `research_only_side, non_positive_ev`

### 4. SCOTUS accepts sports event contract case by July 31, 2026?     
- Side: `research`
- Decision: `research`
- Recommendation: `research`
- Category id: `crypto_policy`
- Market type: `crypto_policy` / `long_term`
- Evidence source: `crypto_policy_public_context` confidence=`0.2500`
- Data quality: `0.6096`
- CLOB diagnostic: `skipped_limit` - CLOB enrichment limit reached (50)
- Confidence: `0.3617`
- Recommended size: `$0.00`
- Total score: `0.7487`
- After-cost EV: `-0.0425`
- Crowd probability: `0.1350`
- Hogan probability: `n/a`
- Rationale: prediction-market crowd probability 0.14
- Reject reasons: `research_only_side, non_positive_ev`

### 5. Will El Salvador hold $1b+ of BTC by December 31, 2026?
- Side: `research`
- Decision: `research`
- Recommendation: `research`
- Category id: `crypto_treasury`
- Market type: `crypto_treasury` / `long_term`
- Evidence source: `crypto_treasury_public_context` confidence=`0.3000`
- Data quality: `0.6039`
- CLOB diagnostic: `skipped_limit` - CLOB enrichment limit reached (50)
- Confidence: `0.3331`
- Recommended size: `$0.00`
- Total score: `0.6627`
- After-cost EV: `-0.0425`
- Crowd probability: `0.2250`
- Hogan probability: `n/a`
- Rationale: prediction-market crowd probability 0.23
- Target price: `$1`
- Reject reasons: `research_only_side, non_positive_ev`

### 6. SCOTUS accepts sports event contract case by December 31, 2026?
- Side: `research`
- Decision: `research`
- Recommendation: `research`
- Category id: `crypto_policy`
- Market type: `crypto_policy` / `long_term`
- Evidence source: `crypto_policy_public_context` confidence=`0.2500`
- Data quality: `0.6146`
- CLOB diagnostic: `skipped_limit` - CLOB enrichment limit reached (50)
- Confidence: `0.3131`
- Recommended size: `$0.00`
- Total score: `0.6048`
- After-cost EV: `-0.0425`
- Crowd probability: `0.2650`
- Hogan probability: `n/a`
- Rationale: prediction-market crowd probability 0.27
- Reject reasons: `research_only_side, non_positive_ev`

### 7. Will Bitcoin hit $150k by June 30, 2026?
- Side: `buy_no`
- Decision: `research`
- Recommendation: `monitor`
- Category id: `crypto_price_target`
- Market type: `price_target` / `long_term`
- Evidence source: `crypto_price_history` confidence=`0.6500`
- Data quality: `0.9970`
- CLOB diagnostic: `ok` - CLOB midpoint and spread fetched
- Confidence: `0.6979`
- Recommended size: `$0.00`
- Total score: `0.5607`
- After-cost EV: `0.0222`
- Crowd probability: `0.0255`
- Hogan probability: `0.0000`
- Rationale: Hogan 0.00 vs crowd 0.03
- Target price: `$150,000`
- Long-horizon model: `prob=0.0000 spot=$71,261 target=$150,000 days=62 drift=-18.38% vol=44.84% n=376`

### 8. Will Bitcoin hit $150k by December 31, 2026?
- Side: `buy_no`
- Decision: `research`
- Recommendation: `monitor`
- Category id: `crypto_price_target`
- Market type: `price_target` / `long_term`
- Evidence source: `crypto_price_history` confidence=`0.6500`
- Data quality: `0.6292`
- CLOB diagnostic: `skipped_limit` - CLOB enrichment limit reached (50)
- Confidence: `0.3454`
- Recommended size: `$0.00`
- Total score: `0.4836`
- After-cost EV: `0.0470`
- Crowd probability: `0.0950`
- Hogan probability: `0.0055`
- Rationale: Hogan 0.01 vs crowd 0.10
- Target price: `$150,000`
- Long-horizon model: `prob=0.0055 spot=$71,261 target=$150,000 days=246 drift=-18.38% vol=44.84% n=376`
- Watchlist trigger: `near_shadow_ev_threshold; needs_ev=+0.0030; trigger_NO<=0.9020`

### 9. Will bitcoin hit $1m before GTA VI?
- Side: `research`
- Decision: `research`
- Recommendation: `research`
- Category id: `crypto_price_target`
- Market type: `price_target` / `long_term`
- Evidence source: `crypto_price_history` confidence=`0.6500`
- Data quality: `0.6574`
- CLOB diagnostic: `skipped_limit` - CLOB enrichment limit reached (50)
- Confidence: `0.1364`
- Recommended size: `$0.00`
- Total score: `0.3848`
- After-cost EV: `-0.0425`
- Crowd probability: `0.4900`
- Hogan probability: `n/a`
- Rationale: long_horizon_price_target_requires_calibrated_fair_value; crowd probability 0.49
- Target price: `$1,000,000`
- Safety note: `long_horizon_price_target_requires_calibrated_fair_value`
- Reject reasons: `research_only_side, non_positive_ev`


## Arbitrage Alerts

### 1. mutually_exclusive_overpricing
- Severity: `1.0000`
- Market IDs: `965261, 556062, 1299187, 1373744, 1541748`
- Message: megaeth-market-cap-fdv-one-day-after-launch YES probabilities sum to 3.88

### 2. mutually_exclusive_overpricing
- Severity: `1.0000`
- Market IDs: `540819, 540844, 540820, 540816, 573647`
- Message: what-will-happen-before-gta-vi YES probabilities sum to 5.26

### 3. mutually_exclusive_overpricing
- Severity: `1.0000`
- Market IDs: `572847, 572860, 572845, 572855, 572849`
- Message: epl-which-clubs-get-relegated YES probabilities sum to 2.98

### 4. mutually_exclusive_overpricing
- Severity: `1.0000`
- Market IDs: `582138, 582134, 582135, 582133, 582145`
- Message: english-premier-league-top-4-finish YES probabilities sum to 4.05

### 5. mutually_exclusive_overpricing
- Severity: `1.0000`
- Market IDs: `582155, 582157, 582158, 582169, 582166`
- Message: laliga-top-4-finish YES probabilities sum to 4.01

### 6. mutually_exclusive_overpricing
- Severity: `1.0000`
- Market IDs: `579359, 579357, 579370, 579362, 579363`
- Message: serie-a-top-4-finish YES probabilities sum to 3.71

### 7. mutually_exclusive_overpricing
- Severity: `1.0000`
- Market IDs: `582321, 582316, 582320, 582317, 582315`
- Message: ligue-1-top-4-finish YES probabilities sum to 4.06

### 8. mutually_exclusive_overpricing
- Severity: `1.0000`
- Market IDs: `582174, 582178, 582177, 582184, 582175`
- Message: bundesliga-top-4-finish YES probabilities sum to 3.87

### 9. mutually_exclusive_overpricing
- Severity: `1.0000`
- Market IDs: `578405, 578399, 578404, 578400, 578398`
- Message: serie-a-which-clubs-get-relegated YES probabilities sum to 3.03

### 10. mutually_exclusive_overpricing
- Severity: `0.7955`
- Market IDs: `576815, 576813, 576810, 576814, 576812`
- Message: laliga-which-clubs-get-relegated YES probabilities sum to 1.80

### 11. mutually_exclusive_overpricing
- Severity: `0.5900`
- Market IDs: `573825, 676828, 676829, 1333258, 573826`
- Message: gpt-6-released-by YES probabilities sum to 1.59

### 12. mutually_exclusive_overpricing
- Severity: `0.5535`
- Market IDs: `576805, 576807, 576806, 576804, 576801`
- Message: bundesliga-which-clubs-get-relegated YES probabilities sum to 1.55

### 13. mutually_exclusive_overpricing
- Severity: `0.5100`
- Market IDs: `569373, 569360, 569362, 569364, 569365`
- Message: colombia-presidential-election YES probabilities sum to 1.51

### 14. crypto_ladder_monotonicity
- Severity: `0.4900`
- Market IDs: `573654, 540844`
- Message: Higher target $1,000,000 priced above lower target $150,000: 0.49 > 0.00

### 15. mutually_exclusive_overpricing
- Severity: `0.4885`
- Market IDs: `561999, 561990, 561982, 561976, 561986`
- Message: republican-presidential-nominee-2028 YES probabilities sum to 1.49


## Next Action

Authority is `research`; review recommendations without opening new shadow trades.