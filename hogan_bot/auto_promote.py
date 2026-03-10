"""Auto-promotion pipeline for Optuna optimisation results.

After an Optuna run finishes, this module compares the new result against the
incumbent config for the same symbol/timeframe and promotes only when the new
result beats the threshold.

Promotion rules
---------------
1. New Sharpe must exceed a configurable minimum (default 2.0).
2. New Sharpe must beat the incumbent by at least *min_improvement* (default 0.5).
3. New result must have >= *min_trades* (default 10) to avoid overfitting on
   a handful of lucky trades.
4. Max drawdown must be <= *max_drawdown_pct* (default 15%).

When promoted, the incumbent file is backed up to ``models/archive/`` and the
new result takes its place.  The per-symbol config cache is cleared so the
live bot picks up the new parameters on the next iteration.

Usage
-----
    from hogan_bot.auto_promote import evaluate_and_promote
    result = evaluate_and_promote("BTC/USD", "1h")
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PromotionResult:
    """Outcome of an auto-promotion evaluation."""
    promoted: bool
    reason: str
    symbol: str
    timeframe: str
    new_score: float
    incumbent_score: float
    new_file: str
    backup_file: str | None = None


def _opt_path(symbol: str, timeframe: str, models_dir: str = "models") -> Path:
    slug = symbol.replace("/", "-")
    return Path(models_dir) / f"opt_{slug}_{timeframe}.json"


def _read_opt_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def evaluate_and_promote(
    symbol: str,
    timeframe: str,
    *,
    candidate_path: str | Path | None = None,
    models_dir: str = "models",
    min_sharpe: float = 2.0,
    min_improvement: float = 0.5,
    min_trades: int = 10,
    max_drawdown_pct: float = 15.0,
) -> PromotionResult:
    """Compare a candidate Optuna result against the incumbent and promote if better.

    Parameters
    ----------
    symbol, timeframe : str
        Identifies which opt_*.json to evaluate.
    candidate_path : str or Path, optional
        Path to the candidate JSON.  Defaults to the standard location
        ``models/opt_{SYMBOL}_{TF}.json`` (same file serves as both
        candidate and incumbent when running sequential optimisations).
    models_dir : str
        Directory containing the Optuna JSON files.
    min_sharpe : float
        Absolute minimum Sharpe to promote.
    min_improvement : float
        Minimum Sharpe improvement over the incumbent.
    min_trades : int
        Minimum number of trades in the backtest.
    max_drawdown_pct : float
        Maximum allowed drawdown percentage.

    Returns
    -------
    PromotionResult
    """
    incumbent_path = _opt_path(symbol, timeframe, models_dir)
    cand_path = Path(candidate_path) if candidate_path else incumbent_path

    candidate = _read_opt_json(cand_path)
    if candidate is None:
        return PromotionResult(
            promoted=False,
            reason=f"Candidate file not found: {cand_path}",
            symbol=symbol,
            timeframe=timeframe,
            new_score=0.0,
            incumbent_score=0.0,
            new_file=str(cand_path),
        )

    new_score = candidate.get("best_score", 0.0)
    leaderboard = candidate.get("leaderboard", [])
    best_entry = leaderboard[0] if leaderboard else {}
    trades = best_entry.get("trades", 0)
    max_dd = best_entry.get("max_drawdown_pct", 100.0)

    # Read incumbent (might be the same file for sequential runs)
    incumbent = _read_opt_json(incumbent_path) if incumbent_path != cand_path else None
    incumbent_score = incumbent.get("best_score", 0.0) if incumbent else 0.0

    # Gate checks
    if new_score < min_sharpe:
        return PromotionResult(
            promoted=False,
            reason=f"Sharpe {new_score:.2f} < minimum {min_sharpe:.2f}",
            symbol=symbol, timeframe=timeframe,
            new_score=new_score, incumbent_score=incumbent_score,
            new_file=str(cand_path),
        )

    if trades < min_trades:
        return PromotionResult(
            promoted=False,
            reason=f"Only {trades} trades < minimum {min_trades}",
            symbol=symbol, timeframe=timeframe,
            new_score=new_score, incumbent_score=incumbent_score,
            new_file=str(cand_path),
        )

    if max_dd > max_drawdown_pct:
        return PromotionResult(
            promoted=False,
            reason=f"Max drawdown {max_dd:.2f}% > limit {max_drawdown_pct:.2f}%",
            symbol=symbol, timeframe=timeframe,
            new_score=new_score, incumbent_score=incumbent_score,
            new_file=str(cand_path),
        )

    improvement = new_score - incumbent_score
    if incumbent and improvement < min_improvement:
        return PromotionResult(
            promoted=False,
            reason=f"Improvement {improvement:.2f} < minimum {min_improvement:.2f} "
                   f"(new={new_score:.2f}, incumbent={incumbent_score:.2f})",
            symbol=symbol, timeframe=timeframe,
            new_score=new_score, incumbent_score=incumbent_score,
            new_file=str(cand_path),
        )

    # Promotion: archive incumbent, install candidate
    backup_path = None
    if incumbent and incumbent_path.exists() and incumbent_path != cand_path:
        archive_dir = Path(models_dir) / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = symbol.replace("/", "-")
        backup_name = f"opt_{slug}_{timeframe}_{ts}.json"
        backup_path = archive_dir / backup_name
        shutil.copy2(incumbent_path, backup_path)
        logger.info("Archived incumbent to %s", backup_path)

    if cand_path != incumbent_path:
        shutil.copy2(cand_path, incumbent_path)

    # Clear the config cache so the bot picks up new params
    try:
        from hogan_bot.config import reload_symbol_configs
        reload_symbol_configs()
    except Exception:
        pass

    logger.info(
        "PROMOTED %s/%s: Sharpe %.2f → %.2f (improvement=%.2f, trades=%d, dd=%.2f%%)",
        symbol, timeframe, incumbent_score, new_score, improvement, trades, max_dd,
    )

    return PromotionResult(
        promoted=True,
        reason=f"Sharpe {new_score:.2f} beats incumbent {incumbent_score:.2f} by {improvement:.2f}",
        symbol=symbol, timeframe=timeframe,
        new_score=new_score, incumbent_score=incumbent_score,
        new_file=str(incumbent_path),
        backup_file=str(backup_path) if backup_path else None,
    )


def promote_all(
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    models_dir: str = "models",
    **kwargs,
) -> list[PromotionResult]:
    """Evaluate and promote all symbol/timeframe combos found in models_dir.

    If *symbols* or *timeframes* are ``None``, discovers them from existing
    ``opt_*.json`` files in the directory.
    """
    results = []
    models = Path(models_dir)
    if not models.exists():
        return results

    if symbols is None or timeframes is None:
        for f in sorted(models.glob("opt_*.json")):
            parts = f.stem.split("_")
            if len(parts) >= 3:
                sym = parts[1].replace("-", "/")
                tf = parts[2]
                if symbols is None or sym in symbols:
                    if timeframes is None or tf in timeframes:
                        r = evaluate_and_promote(sym, tf, models_dir=models_dir, **kwargs)
                        results.append(r)
    else:
        for sym in symbols:
            for tf in timeframes:
                r = evaluate_and_promote(sym, tf, models_dir=models_dir, **kwargs)
                results.append(r)

    return results
