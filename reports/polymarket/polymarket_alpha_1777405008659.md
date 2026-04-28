# Polymarket Alpha Lab Report

- Timestamp ms: `1777405008659`
- Candidates reviewed: `7`
- Arbitrage alerts: `22`
- Shadow marked open: `3`
- Shadow closed: `0`
- Shadow open exposure: `$68.03`
- Shadow unrealized PnL: `$0.00`
- Hogan BTC probability: `0.0886`
- Hogan ETH probability: `n/a`
- Promotion approved: `False`
- Promotion blockers: `insufficient_shadow_trades:0<50, total_pnl_below_gate:0.00<=0.00, avg_pnl_below_gate:0.00<0.10, win_rate_below_gate:0.00<0.55`

## Top Candidates

### 1. Will Bitcoin hit $150k by June 30, 2026?
- Side: `research`
- Decision: `research`
- Market type: `price_target` / `long_term`
- Total score: `0.8500`
- After-cost EV: `-0.0425`
- Crowd probability: `0.0135`
- Hogan probability: `n/a`
- Rationale: long_horizon_price_target_requires_calibrated_fair_value; crowd probability 0.01
- Target price: `$150,000`
- Safety note: `long_horizon_price_target_requires_calibrated_fair_value`
- Reject reasons: `research_only_side, non_positive_ev`

### 2. Will Bitcoin hit $150k by December 31, 2026?
- Side: `research`
- Decision: `research`
- Market type: `price_target` / `long_term`
- Total score: `0.7994`
- After-cost EV: `-0.0425`
- Crowd probability: `0.0950`
- Hogan probability: `n/a`
- Rationale: long_horizon_price_target_requires_calibrated_fair_value; crowd probability 0.10
- Target price: `$150,000`
- Safety note: `long_horizon_price_target_requires_calibrated_fair_value`
- Reject reasons: `research_only_side, non_positive_ev`

### 3. MicroStrategy sells any Bitcoin by June 30, 2026?
- Side: `research`
- Decision: `research`
- Market type: `macro_crypto` / `long_term`
- Total score: `0.7822`
- After-cost EV: `-0.0425`
- Crowd probability: `0.0265`
- Hogan probability: `n/a`
- Rationale: prediction-market crowd probability 0.03
- Reject reasons: `research_only_side, non_positive_ev`

### 4. Trump eliminates capital gains tax on crypto before 2027?
- Side: `research`
- Decision: `research`
- Market type: `macro_crypto` / `long_term`
- Total score: `0.7770`
- After-cost EV: `-0.0425`
- Crowd probability: `0.0365`
- Hogan probability: `n/a`
- Rationale: prediction-market crowd probability 0.04
- Reject reasons: `research_only_side, non_positive_ev`

### 5. MicroStrategy sells any Bitcoin by December 31, 2026?
- Side: `research`
- Decision: `research`
- Market type: `macro_crypto` / `long_term`
- Total score: `0.7679`
- After-cost EV: `-0.0425`
- Crowd probability: `0.0950`
- Hogan probability: `n/a`
- Rationale: prediction-market crowd probability 0.10
- Reject reasons: `research_only_side, non_positive_ev`

### 6. Will El Salvador hold $1b+ of BTC by December 31, 2026?
- Side: `research`
- Decision: `research`
- Market type: `price_target` / `long_term`
- Total score: `0.5538`
- After-cost EV: `-0.0425`
- Crowd probability: `0.3100`
- Hogan probability: `n/a`
- Rationale: prediction-market crowd probability 0.31
- Target price: `$1`
- Reject reasons: `research_only_side, non_positive_ev`

### 7. Will bitcoin hit $1m before GTA VI?
- Side: `research`
- Decision: `research`
- Market type: `price_target` / `long_term`
- Total score: `0.4080`
- After-cost EV: `-0.0425`
- Crowd probability: `0.4895`
- Hogan probability: `n/a`
- Rationale: long_horizon_price_target_requires_calibrated_fair_value; crowd probability 0.49
- Target price: `$1,000,000`
- Safety note: `long_horizon_price_target_requires_calibrated_fair_value`
- Reject reasons: `research_only_side, non_positive_ev`


## Arbitrage Alerts

### 1. mutually_exclusive_overpricing
- Severity: `1.0000`
- Market IDs: `540819, 540844, 540820, 540816, 540843`
- Message: what-will-happen-before-gta-vi YES probabilities sum to 5.35

### 2. mutually_exclusive_overpricing
- Severity: `1.0000`
- Market IDs: `1541748, 1373744, 666614, 556062, 965261`
- Message: megaeth-market-cap-fdv-one-day-after-launch YES probabilities sum to 3.81

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
- Market IDs: `579384, 579396, 579390, 579400, 579381`
- Message: serie-a-top-goalscorer YES probabilities sum to 2.34

### 10. mutually_exclusive_overpricing
- Severity: `1.0000`
- Market IDs: `578405, 578399, 578404, 578400, 578398`
- Message: serie-a-which-clubs-get-relegated YES probabilities sum to 3.03

### 11. mutually_exclusive_overpricing
- Severity: `0.7955`
- Market IDs: `576815, 576813, 576810, 576814, 576812`
- Message: laliga-which-clubs-get-relegated YES probabilities sum to 1.80

### 12. mutually_exclusive_overpricing
- Severity: `0.6805`
- Market IDs: `578432, 578442, 578415, 578411, 578436`
- Message: ligue-1-top-goalscorer YES probabilities sum to 1.68

### 13. mutually_exclusive_overpricing
- Severity: `0.6500`
- Market IDs: `573825, 676828, 676829, 1333258, 573826`
- Message: gpt-6-released-by YES probabilities sum to 1.65

### 14. mutually_exclusive_overpricing
- Severity: `0.6475`
- Market IDs: `544093, 544094, 544095, 544092, 544096`
- Message: harvey-weinstein-prison-time YES probabilities sum to 1.65

### 15. mutually_exclusive_overpricing
- Severity: `0.5535`
- Market IDs: `576805, 576807, 576806, 576804, 576801`
- Message: bundesliga-which-clubs-get-relegated YES probabilities sum to 1.55
