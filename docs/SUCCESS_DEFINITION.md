# Defining trading success (Hogan)

## North star

**Ultimate success is profitable trading— as much sustainable profit as the strategy and market can support, measured month to month in live or paper accounts.**

That is the *objective*. Hogan turns it into something you can **measure, gate, and improve** without fooling yourself.

---

## Why “green every month” needs a second layer

Monthly P&L is **noisy**. A good process can have red months; a bad one can get lucky. “As profitable as possible month to month” therefore means:

1. **Maximize the *expected* trajectory** of equity (after fees, slippage, and realistic execution).
2. **Constrain tail risk** so one bad regime doesn’t wipe many months of gains.
3. **Judge changes** on **out-of-sample** evidence, not a single calendar month or one backtest window.

So we use **two time scales**:

| Horizon | Role |
|--------|------|
| **Bar / trade** | Execution quality, fills, slippage, gate behavior |
| **Month** | How you experience “success” in the account |
| **Multi-month + OOS splits** | How we know a change is real, not luck |

---

## What “success” means in Hogan (layered)

### Layer A — Research & shipping (before you trust new logic)

**Question:** *Does this version add edge **out of sample**, without blowing up in the worst period?*

Use **walk-forward** (and optional certification) as the primary gate:

- **Mean OOS return** across splits (after costs, as modeled in the harness).
- **Stability:** fraction of splits with acceptable behavior (e.g. not all edge in one split).
- **Drawdown discipline:** worst-split max drawdown (or similar) within a cap you choose.
- **Activity:** trade count not collapsed to zero (unless that’s intentional) and not exploded into overtrading.

Commands and archiving: [`PROMOTION_CHECKLIST.md`](PROMOTION_CHECKLIST.md), `scripts/run_validation_battery.py`, `python -m hogan_bot.walk_forward ...`.

**Success here:** you only promote configs that pass *your* written thresholds on Layer A. That is “success” for **process and safety**.

### Layer B — Paper / constrained live (month-to-month experience)

**Question:** *Is the account making money **on average** while staying inside risk limits?*

Operational metrics (define in your journal / spreadsheet / broker statements):

- **Monthly net P&L** (after fees; use the same accounting always).
- **Rolling risk-adjusted** view over **3–6 months** (e.g. Calmar, or return / max DD) so one lucky month doesn’t define “success.”
- **Max drawdown** vs a hard limit you set for the experiment (e.g. pause or resize if exceeded).

**Success here:** positive **expected** monthly trajectory with drawdowns you can tolerate—not necessarily every single month green.

### Layer C — Long horizon (ultimate profitability)

**Question:** *Over a year or more, is the system compounding in line with Layer A and B?*

- Majority of **rolling 3-month windows** positive, *or* annual return vs agreed risk budget.
- No “silent” drift: regime changes, exchange issues, or swarm/quarantine state don’t go unnoticed.

---

## Suggested KPI stack (pick weights that match your risk tolerance)

Use **more than one** number; optimizing only “last month P&L” overfits behavior to noise.

| KPI | What it catches |
|-----|------------------|
| **Net P&L (monthly & YTD)** | Aligns with your north star |
| **Max drawdown (month & peak-to-trough)** | Survival and sleep-at-night |
| **Return / max DD (Calmar-like)** | “Profitable *as possible*” per unit pain |
| **Walk-forward mean OOS & worst split** | Is the *idea* sound before you size up |
| **Trade count & fee drag** | Overtrading vs dead system |

Hogan’s architecture note still applies: **ML AUC ~0.52** means “success” usually comes from **sizing, gates, macro sitout, exits**, not from picking direction perfectly—so Layer A must include those components, not a single-window backtest.

---

## Write your own promotion numbers

Fill these in and treat them as the definition of “good enough to scale autonomy or size”:

- Walk-forward: mean OOS return ≥ **___** , worst-split max DD ≤ **___** , ≥ **___** / N splits above zero.
- Paper pilot: rolling **6-month** return / max DD ≥ **___** before increasing notional.
- Live: hard stop if drawdown from peak ≥ **___** % or **N** consecutive months below **___** (optional circuit breaker).

Until these are filled, default to **conservative** thresholds and [`STRATEGY_CHANGE_GATE.md`](STRATEGY_CHANGE_GATE.md).

---

## Summary

- **Your goal:** profitable trading, as strong as possible **month to month**.
- **Hogan’s definition of success:** same goal, plus **explicit OOS validation**, **drawdown limits**, and **multi-metric** review so “monthly profit” is optimized as an **expected path**, not a single lucky streak.

Related: [`PROMOTION_CHECKLIST.md`](PROMOTION_CHECKLIST.md), [`STRATEGY_CHANGE_GATE.md`](STRATEGY_CHANGE_GATE.md), [`AGENTS.md`](../AGENTS.md) (validation commands).
