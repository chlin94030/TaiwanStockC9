"""Frozen strategy configuration for Alpha Radar V16.3.

Architecture contract
---------------------
1. Selection score ranks *which stock* is attractive for a horizon.
2. Intraday score measures *current session strength* only.
3. Entry timing decides *whether the current price is attractive enough to buy*.
4. Market regime controls *position size*, not stock quality/ranking.

Keep changes to strategy thresholds in this file.  The lower-level feature engine
(return_first_model.py) is intentionally frozen in V16.3 to avoid accidental
model drift while live validation is running.
"""
from __future__ import annotations

ARCHITECTURE_VERSION = "v16.3.0-architecture-freeze"
HORIZONS = ("short", "mid", "long")

# Daily stock-selection layer.  These are the high-level evidence weights.
SELECTION_WEIGHTS = {
    "short": {"technical": 0.52, "empirical": 0.20, "fundamental": 0.08, "flow": 0.20},
    "mid": {"technical": 0.42, "empirical": 0.22, "fundamental": 0.22, "flow": 0.14},
    "long": {"technical": 0.34, "empirical": 0.22, "fundamental": 0.34, "flow": 0.10},
}

# Intraday-strength layer.  No chase/overheat penalty belongs here; price
# extension is an execution issue handled by ENTRY_POLICY below.
INTRADAY_FACTOR_WEIGHTS = {
    "relative_market": 0.30,
    "vwap": 0.20,
    "volume": 0.18,
    "day_position": 0.12,
    "open_move": 0.08,
    "bid_pressure": 0.05,
    "turnover": 0.07,
}
INTRADAY_FACTOR_RANGES = {
    "relative_market": (-2.0, 4.0),
    "vwap": (-1.5, 2.0),
    "volume": (0.55, 2.50),
    "day_position": (0.18, 0.88),
    "open_move": (-2.5, 4.0),
    "bid_pressure": (0.35, 0.65),
    "turnover_twd": (30_000_000.0, 2_500_000_000.0),
}
INTRADAY_SPREAD_FREE_PCT = 0.60
INTRADAY_SPREAD_PENALTY_PER_PCT = 6.0

# Intraday signal should not dominate immediately after the open.  The weights
# mature as more of the session is observed.  Values are maximum blend weights
# with the complete daily model.  Key = minutes elapsed since 09:00.
INTRADAY_BLEND_SCHEDULE = {
    "short": ((0, 0.22), (20, 0.28), (60, 0.35), (150, 0.41), (225, 0.45)),
    "mid": ((0, 0.06), (20, 0.08), (60, 0.10), (150, 0.13), (225, 0.15)),
    "long": ((0, 0.02), (20, 0.025), (60, 0.03), (150, 0.04), (225, 0.05)),
}

# Entry/execution layer.  These rules NEVER change the selection score.
ENTRY_POLICY = {
    "base_score": 78.0,
    "extension": {
        "below_breakout": -0.60,
        "ideal_max": 0.45,
        "acceptable_max": 0.90,
        "warm_max": 1.50,
        "hot_max": 2.00,
    },
    "ma5_extension": {"warm": 1.20, "hot": 2.00},
    "one_day_move": {"watch": 3.0, "hot": 6.0, "very_hot": 8.0, "near_limit": 9.5},
    "market_surge": 2.0,
    "relative_strength_bonus": 2.0,
    "relative_weakness": -1.0,
    "volume": {"healthy_low": 1.10, "healthy_high": 2.50, "extreme": 3.50},
    "day_position": {"weak": 0.28, "strong": 0.72},
    "action": {"good": 82.0, "small": 70.0, "wait": 58.0, "watch": 42.0},
    "risk": {"medium": 15.0, "high": 34.0, "extreme": 55.0},
}

# Regime is a risk-budget layer only.  1.0 means 100% of the user's normal
# planned position, not 100% of portfolio capital.
REGIME_POSITION_MULTIPLIER = {
    "BULL": 1.00,
    "NEUTRAL": 0.70,
    "BEAR": 0.35,
    "UNKNOWN": 0.50,
}
ENTRY_POSITION_MULTIPLIER = {
    "good": 1.00,      # entry score >= 82
    "small": 0.55,     # >= 70
    "wait": 0.25,      # >= 58
    "avoid": 0.00,     # < 58
}
CHASE_POSITION_CAP = {
    "低": 1.00,
    "中": 0.55,
    "高": 0.25,
    "極高": 0.00,
    "待確認": 0.25,
}

# Data/scan defaults remain complete.  Faster operation comes from caching,
# batching and feature reuse, not shrinking the candidate universe.
SCAN_DEFAULTS = {
    "reference_size": 600,
    "candidate_size": 1000,
    "research_pool_per_horizon": 10,
    "history_period": "5y",
    "min_price": 10.0,
    "min_avg_turnover": 10_000_000.0,
}

TRADE_PLAN = {
    "short": {"trigger_mult": 1.005, "zone_low_mult": 0.98, "zone_high_mult": 1.01, "chase_mult": 1.03, "invalidation_atr": 2.0, "use_low20_floor": True, "entry_mode": "breakout_confirmed"},
    "mid": {"trigger_mult": 1.01, "zone_low_mult": 0.96, "zone_high_mult": 1.015, "chase_mult": 1.04, "invalidation_atr": 2.5, "use_low20_floor": False, "entry_mode": "zone_confirmed"},
    "long": {"trigger_mult": 1.02, "zone_low_mult": 0.94, "zone_high_mult": 1.02, "chase_mult": 1.05, "invalidation_atr": 3.0, "use_low20_floor": False, "entry_mode": "zone_confirmed"},
}

INTRADAY_STATE = {
    "relative_weak": -1.5,
    "vwap_weak": -1.2,
    "day_position_weak": 0.35,
    "strong_score": 70.0,
    "strong_relative": 0.8,
    "strong_volume": 1.05,
    "stable_score": 60.0,
    "stable_vwap": -0.2,
    "volume_watch": 1.20,
    "volume_watch_score": 50.0,
    "reason_relative": 1.0,
    "reason_volume": 1.25,
    "reason_vwap_up": 0.4,
    "reason_vwap_down": -0.5,
}
