"""Central feature metadata registry for Hogan.

Every feature used by the ML pipeline is registered here with its source,
latency class, fill policy, and staleness threshold.  This enables:

- Staleness detection at inference time (are macro features current?)
- Post-hoc audit of which features were stale when a trade was taken
- Foundation for the DataCustodian agent concept

Usage::

    from hogan_bot.feature_registry import FEATURE_REGISTRY, check_staleness

    stale = check_staleness(feature_values, current_ts_ms, db_conn)
"""
from __future__ import annotations

from dataclasses import dataclass


# Decision relevance: which decision does this feature help?
# "entry_edge" | "regime" | "stop" | "sizing" | "veto" | "execution_timing" | "experimental"
DECISION_ENTRY_EDGE = "entry_edge"
DECISION_REGIME = "regime"
DECISION_STOP = "stop"
DECISION_SIZING = "sizing"
DECISION_VETO = "veto"
DECISION_EXECUTION = "execution_timing"
DECISION_EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class FeatureMeta:
    """Metadata for a single ML feature."""
    name: str
    source: str          # "candle", "macro_db", "onchain_db", "sentiment_db", "derivatives_db", "intermarket_db", "derived"
    latency_class: str   # "realtime", "hourly", "daily", "weekly"
    fill_policy: str     # "forward_fill", "zero", "drop_row", "nan_flag"
    staleness_limit_hours: float  # feature is stale if data is older than this
    decision_relevance: str = ""  # entry_edge, regime, stop, sizing, veto, execution_timing, experimental


def _candle(name: str) -> FeatureMeta:
    return FeatureMeta(
        name=name, source="candle", latency_class="realtime",
        fill_policy="drop_row", staleness_limit_hours=2.0,
    )


def _derived(name: str) -> FeatureMeta:
    return FeatureMeta(
        name=name, source="derived", latency_class="realtime",
        fill_policy="zero", staleness_limit_hours=2.0,
    )


def _macro(name: str) -> FeatureMeta:
    return FeatureMeta(
        name=name, source="macro_db", latency_class="daily",
        fill_policy="forward_fill", staleness_limit_hours=48.0,
    )


def _onchain(name: str) -> FeatureMeta:
    return FeatureMeta(
        name=name, source="onchain_db", latency_class="daily",
        fill_policy="forward_fill", staleness_limit_hours=72.0,
    )


def _sentiment(name: str) -> FeatureMeta:
    return FeatureMeta(
        name=name, source="sentiment_db", latency_class="daily",
        fill_policy="forward_fill", staleness_limit_hours=48.0,
    )


def _derivatives(name: str) -> FeatureMeta:
    return FeatureMeta(
        name=name, source="derivatives_db", latency_class="hourly",
        fill_policy="zero", staleness_limit_hours=12.0,
    )


def _intermarket(name: str) -> FeatureMeta:
    return FeatureMeta(
        name=name, source="intermarket_db", latency_class="daily",
        fill_policy="forward_fill", staleness_limit_hours=48.0,
    )


# ---------------------------------------------------------------------------
# Registry: one entry per _FEATURE_COLUMNS member (59 total)
# ---------------------------------------------------------------------------

FEATURE_REGISTRY: dict[str, FeatureMeta] = {m.name: m for m in [
    # Momentum (4)
    _candle("ret_1"),
    _candle("ret_3"),
    _candle("ret_6"),
    _candle("ret_12"),
    # Trend (1)
    _candle("ma_spread"),
    # Volatility / oscillators (3)
    _candle("volatility_20"),
    _candle("rsi_14"),
    _candle("atr_pct"),
    # MACD (1)
    _candle("macd_hist_pct"),
    # Bollinger (1)
    _candle("bb_pct_b"),
    # Regime proxy (1)
    _derived("vol_regime"),
    # Candle microstructure (4)
    _candle("range_pct"),
    _candle("candle_body_pct"),
    _candle("upper_wick_pct"),
    _candle("lower_wick_pct"),
    # Volume (2)
    _candle("vol_ratio"),
    _candle("vol_spike"),
    # EMA cloud (3)
    _derived("cloud_bull"),
    _derived("cloud_bear"),
    _derived("cloud_width_pct"),
    # FVG (4)
    _derived("fvg_bull_active"),
    _derived("fvg_bear_active"),
    _derived("in_bull_fvg"),
    _derived("in_bear_fvg"),
    # ADX (3)
    _candle("adx_14"),
    _candle("plus_di"),
    _candle("minus_di"),
    # Stochastic RSI (2)
    _candle("stoch_rsi_k"),
    _candle("stoch_rsi_d"),
    # OBV z-score (1)
    _candle("obv_norm"),
    # VWAP distance (1)
    _candle("vwap_dist"),
    # Keltner channel position (1)
    _candle("keltner_pos"),
    # CCI, MFI, CMF, ROC (4)
    _candle("cci_20"),
    _candle("mfi_14"),
    _candle("cmf_20"),
    _candle("roc_10"),
    # Macro-asset context (10)
    _macro("macro_spy_trend"),
    _macro("macro_spy_ret"),
    _macro("macro_vix_norm"),
    _macro("macro_vix_high"),
    _macro("macro_gld_trend"),
    _macro("macro_tlt_ret"),
    _macro("macro_uup_trend"),
    _macro("macro_tnx_norm"),
    _macro("macro_risk_off"),
    _macro("macro_qqq_spy_rel"),
    # On-chain features (4)
    _onchain("onchain_hashrate_trend"),
    _onchain("onchain_addr_trend"),
    _onchain("onchain_mempool_norm"),
    _onchain("onchain_fee_norm"),
    # Sentiment features (4)
    _sentiment("sent_fear_greed_norm"),
    _sentiment("sent_btc_dominance"),
    _sentiment("sent_defi_tvl_change"),
    _sentiment("sent_stablecoin_norm"),
    # Derivatives features (2)
    _derivatives("deriv_funding_rate"),
    _derivatives("deriv_oi_change"),
    # Inter-market features (3)
    _intermarket("intermarket_dxy_trend"),
    _intermarket("intermarket_spy_btc_corr"),
    _intermarket("intermarket_gold_btc_rel"),
]}

assert len(FEATURE_REGISTRY) == 59, f"Expected 59 features, got {len(FEATURE_REGISTRY)}"


# ---------------------------------------------------------------------------
# Champion feature subset (12–20 core features)
# ---------------------------------------------------------------------------
# Point-in-time, low-missingness, decision-relevant features only.
# When champion mode is on, training and inference use this subset.
# Every feature declares what decision it helps.

CHAMPION_FEATURE_COLUMNS: list[str] = [
    # Multi-horizon returns (entry edge)
    "ret_1", "ret_3", "ret_6", "ret_12",
    # Trend / distance from trend (entry edge, regime)
    "ma_spread",
    # Realized volatility, oscillators (stop, sizing, regime)
    "volatility_20", "rsi_14", "atr_pct",
    # Volume participation (entry edge, veto)
    "vol_ratio",
    # Range position / breakout pressure (entry edge)
    "bb_pct_b", "range_pct",
    # Regime strength (regime, veto)
    "adx_14",
    # Momentum confirmation (entry edge)
    "macd_hist_pct",
    # Higher-timeframe alignment (entry edge, veto)
    "macro_spy_trend", "macro_vix_norm",
]

CHAMPION_FEATURE_DECISIONS: dict[str, str] = {
    "ret_1": DECISION_ENTRY_EDGE,
    "ret_3": DECISION_ENTRY_EDGE,
    "ret_6": DECISION_ENTRY_EDGE,
    "ret_12": DECISION_ENTRY_EDGE,
    "ma_spread": DECISION_ENTRY_EDGE,
    "volatility_20": DECISION_STOP,
    "rsi_14": DECISION_ENTRY_EDGE,
    "atr_pct": DECISION_STOP,
    "vol_ratio": DECISION_VETO,
    "bb_pct_b": DECISION_ENTRY_EDGE,
    "range_pct": DECISION_ENTRY_EDGE,
    "adx_14": DECISION_REGIME,
    "macd_hist_pct": DECISION_ENTRY_EDGE,
    "macro_spy_trend": DECISION_ENTRY_EDGE,
    "macro_vix_norm": DECISION_VETO,
}

assert len(CHAMPION_FEATURE_COLUMNS) == 15
assert all(c in FEATURE_REGISTRY for c in CHAMPION_FEATURE_COLUMNS)


# Full 59 in canonical order (matches ml._FEATURE_COLUMNS)
_FULL_FEATURE_COLUMNS: list[str] = list(FEATURE_REGISTRY.keys())


def get_feature_columns(use_champion: bool | None = None) -> list[str]:
    """Return the feature column list for training/inference.

    When *use_champion* is True or when HOGAN_CHAMPION_MODE is set,
    returns the 16-feature champion subset. Otherwise returns the full 59.
    """
    if use_champion is None:
        try:
            from hogan_bot.champion import is_champion_mode
            use_champion = is_champion_mode()
        except Exception:
            use_champion = False
    return list(CHAMPION_FEATURE_COLUMNS) if use_champion else list(_FULL_FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Staleness checking
# ---------------------------------------------------------------------------

_CRITICAL_SOURCES = frozenset({"candle", "derived", "derivatives_db"})


@dataclass
class FeatureResult:
    """Feature vector with staleness metadata."""
    values: list[float]
    stale_features: list[str]
    missing_features: list[str]

    @property
    def has_stale(self) -> bool:
        return len(self.stale_features) > 0

    @property
    def critical_stale_count(self) -> int:
        count = 0
        for name in self.stale_features:
            meta = FEATURE_REGISTRY.get(name)
            if meta and meta.source in _CRITICAL_SOURCES:
                count += 1
        return count

    @property
    def freshness_summary(self) -> dict:
        """Compact dict suitable for JSON logging and QualityComponents."""
        out: dict[str, str] = {}
        for name in self.stale_features:
            out[name] = "stale"
        for name in self.missing_features:
            out[name] = "missing"
        return {
            **out,
            "stale_count": len(self.stale_features),
            "critical_stale_count": self.critical_stale_count,
        }


def check_staleness(
    feature_names: list[str],
    feature_values: list[float],
    data_ages_hours: dict[str, float] | None = None,
) -> FeatureResult:
    """Check feature values against registry staleness limits.

    Parameters
    ----------
    feature_names : list[str]
        Column names in the same order as *feature_values*.
    feature_values : list[float]
        The raw feature vector.
    data_ages_hours : dict[str, float] | None
        Optional mapping of source -> hours since last update.
        Keys should be source names like ``"macro_db"``, ``"sentiment_db"``, etc.
        When provided, features from stale sources are flagged.

    Returns
    -------
    FeatureResult
        The original values plus lists of stale and missing features.
    """
    import math

    stale: list[str] = []
    missing: list[str] = []
    ages = data_ages_hours or {}

    for name, val in zip(feature_names, feature_values):
        meta = FEATURE_REGISTRY.get(name)
        if meta is None:
            continue

        if math.isnan(val):
            missing.append(name)
            continue

        source_age = ages.get(meta.source)
        if source_age is not None and source_age > meta.staleness_limit_hours:
            stale.append(name)

    return FeatureResult(
        values=list(feature_values),
        stale_features=stale,
        missing_features=missing,
    )
