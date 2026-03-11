# Hogan 2.0 — Quarantine vs Keep

## QUARANTINE (remove from champion path, keep under experimental)

### ICT Signal System

| Item | File | Action |
|------|------|--------|
| `ict_setup_signal()` in live path | `strategy.py` L171-189 | Remove from default `generate_signal()` |
| ICT params in optimizer | `optimize.py` L72-92 | Move to `_EXPERIMENTAL_SPACE` |
| ICT config defaults | `config.py` L84-96 | Keep fields, add to `_EXPERIMENTAL` set |
| ICT env vars in .env | `.env.example` L76-88 | Move to `# EXPERIMENTAL` section |
| ICT ML features (7) | `ml.py` L289-336 | Move to `_EXPERIMENTAL_FEATURES` |
| ICT params in backtest | `backtest.py` L257-267 | Gate behind `--experimental` |
| ICT params in backtest CLI | `backtest_cli.py` L89-107 | `--use-ict` → `--experimental-ict` |
| ICT params in main.py | `main.py` L181-191 | Remove from default signal call |
| ICT params in trader_service | `trader_service.py` L552-562 | Remove from default signal call |
| `--use-ict` in signal_smoke | `scripts/signal_smoke.py` | Mark experimental |
| ICT in README examples | `README.md` L124-158 | Move to "Experimental" section |
| ICT in Command Prompts | `Command Prompts.md` | Mark experimental |

### MA Gatekeeper

| Item | File | Action |
|------|------|--------|
| MA-must-fire gate | `strategy.py` L136-141 | Remove. MA becomes one vote among many. |
| `signal_mode="ma_only"` | `strategy.py` L213-237 | Keep as option but not default |

### 5m Default Timeframe

| Item | File | Action |
|------|------|--------|
| `timeframe: str = "5m"` | `config.py` L24 | Change to `"1h"` |
| 5m in README examples | `README.md` | Update to 1h |
| 5m in Command Prompts | `Command Prompts.md` | Update to 1h |

---

## KEEP (active in champion path)

### Core Trading Infrastructure

| Item | File | Status |
|------|------|--------|
| `ExchangeClient` | `exchange.py` | Keep — exchange-agnostic |
| `PaperPortfolio` | `paper.py` | Keep — portfolio mechanics |
| `PaperExecution` / `LiveExecution` | `execution.py` | Keep — execution layer |
| `DrawdownGuard` | `risk.py` | Keep — risk management |
| SQLite storage | `storage.py` | Keep — data persistence |
| Timeframe utilities | `timeframe_utils.py` | Keep — correct annualization |

### ML Pipeline

| Item | File | Status |
|------|------|--------|
| Feature engineering (minus ICT 7) | `ml.py` | Keep 59 of 66 features |
| Model training/inference | `ml.py` | Keep |
| Walk-forward retrain | `retrain.py` | Keep |
| Model registry | `registry.py` | Keep |
| Advanced ensemble | `ml_advanced.py` | Keep |
| Online learner | `online_learner.py` | Keep |
| Trade labeler | `labeler.py` | Keep |

### RL Pipeline

| Item | File | Status |
|------|------|--------|
| Trading environment | `rl_env.py` | Keep |
| PPO training | `rl_train.py` | Keep |
| Optuna RL tuning | `rl_tune.py` | Keep |
| RL agent inference | `rl_agent.py` | Keep |

### Indicators

| Item | File | Status |
|------|------|--------|
| ATR | `indicators.py` | Keep |
| EMA clouds | `indicators.py` | Keep |
| FVG detection | `indicators.py` | Keep (price-derived, not ICT-specific) |
| Regime detection | `regime.py` | Keep — becomes regime model seed |
| Macro filter | `macro_filter.py` | Keep — becomes macro layer seed |
| MTF ensemble | `mtf_ensemble.py` | Keep — multi-timeframe logic |

### Data Fetchers

| Item | File | Status |
|------|------|--------|
| All `fetch_*.py` | Various | Keep all — data sources are independent |
| `backfill.py` | `backfill.py` | Keep |
| `refresh_daily.py` | `refresh_daily.py` | Keep |

### Agent Pipeline (seed for swarm)

| Item | File | Status |
|------|------|--------|
| `TechnicalAgent` | `agent_pipeline.py` | Keep — seed for research swarm |
| `SentimentAgent` | `agent_pipeline.py` | Keep |
| `MacroAgent` | `agent_pipeline.py` | Keep |
| `MetaWeigher` | `agent_pipeline.py` | Keep — seed for policy layer |

### Backtesting

| Item | File | Status |
|------|------|--------|
| `run_backtest_on_candles` | `backtest.py` | Keep |
| `BacktestResult` | `backtest.py` | Keep |
| OOS validation | `scripts/validate_oos.py` | Keep |

### Infrastructure

| Item | File | Status |
|------|------|--------|
| Dashboard | `dashboard.py` | Keep |
| Notifier | `notifier.py` | Keep |
| Discord commands | `discord_commands.py` | Keep |
| Trade explainer | `trade_explainer.py` | Keep |
| MCP server | `mcp_server.py` | Keep |
| Security | `security.py` | Keep |
| Metrics | `metrics.py` | Keep |
| Lookahead checker | `lookahead_check.py` | Keep |
| RAG knowledge | `rag_knowledge.py` | Keep |

---

## DO NOT DELETE

The following are explicitly preserved:

- `hogan_bot/ict.py` — Full ICT implementation stays in the codebase
- `tests/test_ict.py` — ICT tests stay passing
- ICT config fields in `BotConfig` — Stay defined, just not in default search
- All ICT env vars — Stay documented under "Experimental"

The quarantine means: ICT code exists, tests pass, you can opt in with `--experimental` or `HOGAN_USE_ICT=true`, but the champion path, default optimizer, and default config don't use it.
