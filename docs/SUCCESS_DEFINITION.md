# Defining trading success (Hogan)

## North star

**Ultimate success is profitable trading -- as much sustainable profit as the strategy and market can support, measured month to month in live or paper accounts.**

That is the *objective*. Hogan turns it into something you can **measure, gate, and improve** without fooling yourself.

---

## Why "green every month" needs a second layer

Monthly P&L is **noisy**. A good process can have red months; a bad one can get lucky. "As profitable as possible month to month" therefore means:

1. **Maximize the *expected* trajectory** of equity (after fees, slippage, and realistic execution).
2. **Constrain tail risk** so one bad regime doesn't wipe many months of gains.
3. **Judge changes** on **out-of-sample** evidence, not a single calendar month or one backtest window.

So we use **two time scales**:

| Horizon | Role |
|--------|------|
| **Bar / trade** | Execution quality, fills, slippage, gate behavior |
| **Month** | How you experience "success" in the account |
| **Multi-month + OOS splits** | How we know a change is real, not luck |

---

## What "success" means in Hogan (layered)

### Layer A -- Research & shipping (before you trust new logic)

**Question:** *Does this version add edge **out of sample**, without blowing up in the worst period?*

Use **walk-forward** (and optional certification) as the primary gate:

- **Mean OOS return** across splits (after costs, as modeled in the harness).
- **Stability:** fraction of splits with acceptable behavior (e.g. not all edge in one split).
- **Drawdown discipline:** worst-split max drawdown within your cap.
- **Calmar:** mean return / max DD across windows -- the primary metric.
- **Activity:** trade count not collapsed to zero and not exploded into overtrading.

Commands and archiving: [`PROMOTION_CHECKLIST.md`](PROMOTION_CHECKLIST.md), `scripts/run_validation_battery.py`, `python -m hogan_bot.walk_forward ...`.

**Success here:** you only promote configs that pass *your* written thresholds on Layer A. That is "success" for **process and safety**.

### Layer B -- Paper / constrained live (month-to-month experience)

**Question:** *Is the account making money **on average** while staying inside risk limits?*

Operational metrics (define in your journal / spreadsheet / broker statements):

- **Monthly net P&L** (after fees; use the same accounting always).
- **Rolling 6-month Calmar** (return / max DD) -- must stay above threshold before increasing size.
- **Max drawdown** vs a hard limit you set for the experiment (pause or resize if exceeded).

**Success here:** positive **expected** monthly trajectory with drawdowns you can tolerate -- not necessarily every single month green.

### Layer C -- Long horizon (ultimate profitability)

**Question:** *Over a year or more, is the system compounding in line with Layer A and B?*

- Majority of **rolling 3-month windows** positive, *or* annual return vs agreed risk budget.
- No "silent" drift: regime changes, exchange issues, or swarm/quarantine state don't go unnoticed.

---

## KPI stack

| KPI | What it catches | Your target |
|-----|------------------|-------------|
| **Calmar ratio** (return / max DD) | Profitable *as possible* per unit pain | **Primary metric -- higher is better** |
| **Net P&L (monthly & YTD)** | Aligns with your north star | Positive majority of months |
| **Max drawdown (peak-to-trough)** | Survival and sleep-at-night | **10-15% target, 20% hard stop** |
| **Walk-forward mean OOS & worst split** | Is the *idea* sound before you size up | mean Calmar > 0, worst DD <= 15% |
| **Trade count & fee drag** | Overtrading vs dead system | >= 5 trades / WF window |

**Calmar is the north-star metric.** It directly measures "how much profit did I get for the drawdown I endured?" -- the ratio that most tightly maps to "as profitable as possible month to month while staying alive."

Hogan's architecture note still applies: **ML AUC ~0.52** means "success" usually comes from **sizing, gates, macro sitout, exits**, not from picking direction perfectly -- so Layer A must include those components, not a single-window backtest.

---

## Promotion thresholds (your numbers)

These define "good enough to scale autonomy or size":

### Layer A -- Walk-forward gate (research / shipping)

- **Worst-split max DD <= 15%** (hard limit; 20% = absolute ceiling).
- **Mean Calmar across splits > 0** (positive expected profit-per-drawdown).
- **>= 3 / 5 splits above zero** (stability -- not all edge in one window).
- **>= 5 trades per window** (system is actually trading, not dead).

CLI:

```bash
python -m hogan_bot.walk_forward --db data/hogan.db --n-splits 5 --max-dd 15 --min-calmar 0.0
```

### Layer B -- Paper pilot (month-to-month)

- **Rolling 6-month Calmar >= 0.5** before increasing notional.
- **Max drawdown in paper account <= 15%** -- if hit, halt and review.
- Majority of rolling 3-month windows should be net positive.

### Layer C -- Live / scaled (circuit breakers)

- **Hard stop: drawdown from peak >= 20%** -- pause, halve size, review.
- **3 consecutive months negative** -- review strategy; do not auto-increase.
- Any walk-forward re-run (after changes) must pass Layer A before resuming.

---

## Summary

- **Your goal:** profitable trading, as strong as possible **month to month**.
- **Your primary metric:** Calmar ratio (return / max drawdown).
- **Your risk envelope:** 10-15% max drawdown target, 20% hard stop.
- **Hogan's definition of success:** same goal, plus **explicit OOS validation**, **drawdown limits**, and **Calmar-gated** review so "monthly profit" is optimized as an **expected path**, not a single lucky streak.

Related: [`PROMOTION_CHECKLIST.md`](PROMOTION_CHECKLIST.md), [`STRATEGY_CHANGE_GATE.md`](STRATEGY_CHANGE_GATE.md), [`AGENTS.md`](../AGENTS.md) (validation commands).
