"""Execution policy for Alpha Radar V16.3 Architecture Freeze.

This module owns *buy timing, position sizing and trade-plan rules*.  It must not
change the stock-selection score.  A strong stock can therefore remain ranked
#1 while the execution policy says "do not chase".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_config import (
    ARCHITECTURE_VERSION,
    CHASE_POSITION_CAP,
    ENTRY_POLICY,
    ENTRY_POSITION_MULTIPLIER,
    HORIZONS as CONFIG_HORIZONS,
    REGIME_POSITION_MULTIPLIER,
    TRADE_PLAN,
)

ENGINE_VERSION = ARCHITECTURE_VERSION
HORIZONS = list(CONFIG_HORIZONS)


def _num(value, default=None):
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    prev = close.shift(1)
    return pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def generate_trade_plan(df: pd.DataFrame, horizon: str) -> dict | None:
    """Build structural reference levels from the latest complete daily bars."""
    if df is None or df.empty or len(df) < 20:
        return None

    horizon = horizon if horizon in HORIZONS else "short"
    cfg = TRADE_PLAN[horizon]
    x = df.sort_index()
    close = pd.to_numeric(x["Close"], errors="coerce")
    p_close = float(close.iloc[-1])
    high_20 = float(pd.to_numeric(x["High"], errors="coerce").iloc[-20:].max())
    low_20 = float(pd.to_numeric(x["Low"], errors="coerce").iloc[-20:].min())
    prev_high_20 = pd.to_numeric(x["High"], errors="coerce").shift(1).iloc[-20:].max()
    breakout_level = float(prev_high_20) if np.isfinite(prev_high_20) else high_20
    tr = _true_range(x)
    atr = float(tr.iloc[-14:].mean()) if len(tr) >= 14 else p_close * 0.03
    if not np.isfinite(atr) or atr <= 0:
        atr = p_close * 0.03
    ma5 = float(close.iloc[-5:].mean()) if len(close) >= 5 else p_close
    ma20 = float(close.iloc[-20:].mean()) if len(close) >= 20 else p_close

    trigger = high_20 * float(cfg["trigger_mult"])
    zone_low = p_close * float(cfg["zone_low_mult"])
    zone_high = p_close * float(cfg["zone_high_mult"])
    chase_limit = p_close * float(cfg["chase_mult"])
    invalidation = p_close - float(cfg["invalidation_atr"]) * atr
    if cfg.get("use_low20_floor"):
        invalidation = max(low_20, invalidation)

    return {
        "trigger": round(trigger, 2),
        "zone_low": round(zone_low, 2),
        "zone_high": round(zone_high, 2),
        "chase_limit": round(chase_limit, 2),
        "invalidation": round(invalidation, 2),
        "entry_mode": str(cfg["entry_mode"]),
        "reference_close": round(p_close, 4),
        "breakout_level": round(breakout_level, 4),
        "atr14": round(atr, 4),
        "ma5": round(ma5, 4),
        "ma20": round(ma20, 4),
    }


def evaluate_entry_state(df: pd.DataFrame, plan: dict | None) -> str:
    """Legacy-compatible coarse state used by the presentation layer."""
    if not plan or df is None or df.empty:
        return "NO_RETURN_ESTIMATE"
    p_close = float(df["Close"].iloc[-1])
    if p_close <= plan["invalidation"]:
        return "INVALIDATED"
    if p_close > plan["chase_limit"]:
        return "DO_NOT_CHASE"
    if plan["zone_low"] <= p_close <= plan["zone_high"]:
        return "CONDITIONS_MET_NOT_FILLED"
    if p_close < plan["zone_low"]:
        return "WAIT_ENTRY_ZONE"
    return "WAIT_BREAKOUT"


def setup_from_df(df: pd.DataFrame) -> dict:
    """Compact technical snapshot retained for dashboard/backward compatibility."""
    if df is None or df.empty or "Close" not in df.columns:
        return {}
    x = df.sort_index()
    close = pd.to_numeric(x["Close"], errors="coerce")
    high = pd.to_numeric(x["High"], errors="coerce") if "High" in x.columns else close
    vol = pd.to_numeric(x["Volume"], errors="coerce") if "Volume" in x.columns else pd.Series(index=x.index, dtype=float)
    p = _num(close.iloc[-1])
    if p is None:
        return {}
    out = {"price": round(p, 4)}
    for n in [5, 20, 60, 120, 240]:
        if len(close) >= n:
            v = _num(close.iloc[-n:].mean())
            if v is not None:
                out[f"ma{n}"] = round(v, 4)
    tr = _true_range(x) if {"High", "Low", "Close"}.issubset(x.columns) else pd.Series(dtype=float)
    if len(tr) >= 14:
        atr = _num(tr.iloc[-14:].mean())
        if atr is not None:
            out["atr14"] = round(atr, 4)
            out["atr_pct"] = round(atr / p, 5) if p else None
    if len(high) >= 21:
        prev_high = _num(high.shift(1).iloc[-20:].max())
        if prev_high is not None and prev_high > 0:
            out["breakout20"] = round(p / prev_high - 1.0, 5)
            out["breakout_level"] = round(prev_high, 4)
    if len(vol) >= 21:
        base = _num(vol.iloc[-21:-1].mean())
        cur = _num(vol.iloc[-1])
        if base not in (None, 0) and cur is not None:
            out["volume_ratio"] = round(cur / base, 3)
    return out


def evaluate_entry_timing(
    price: float | None,
    plan: dict | None,
    horizon: str = "short",
    *,
    change_pct: float | None = None,
    market_change_pct: float | None = None,
    volume_ratio: float | None = None,
    relative_market_pct_pt: float | None = None,
    day_position: float | None = None,
) -> dict:
    """Score execution timing from 0-100 without changing selection ranking."""
    p = _num(price)
    if p is None or p <= 0 or not plan:
        return {
            "score": None,
            "action": "買點資料不足",
            "chase_risk": "待確認",
            "reasons": ["無法確認目前價格與防守區間"],
            "overnight_risk": False,
        }

    horizon = horizon if horizon in HORIZONS else "short"
    policy = ENTRY_POLICY
    ext = policy["extension"]
    ma5_cfg = policy["ma5_extension"]
    move = policy["one_day_move"]
    vol_cfg = policy["volume"]
    dpos_cfg = policy["day_position"]
    action_cfg = policy["action"]
    risk_cfg = policy["risk"]

    ref = _num(plan.get("reference_close"), p) or p
    atr = _num(plan.get("atr14"), max(ref * 0.03, 0.01)) or max(ref * 0.03, 0.01)
    atr = max(atr, ref * 0.008, 0.01)
    breakout = _num(plan.get("breakout_level"), _num(plan.get("trigger"), ref)) or ref
    ma5 = _num(plan.get("ma5"))
    invalidation = _num(plan.get("invalidation"), ref - 2.0 * atr) or (ref - 2.0 * atr)

    chg = _num(change_pct, 0.0) or 0.0
    mkt = _num(market_change_pct, 0.0) or 0.0
    rel = _num(relative_market_pct_pt, chg - mkt)
    rel = (chg - mkt) if rel is None else rel
    vr = _num(volume_ratio)
    dpos = _num(day_position)

    extension_atr = (p - breakout) / atr
    ma5_extension_atr = ((p - ma5) / atr) if ma5 is not None and ma5 > 0 else None
    score = float(policy["base_score"])
    risk_points = 0.0
    reasons: list[str] = []
    caps: list[float] = []

    if p <= invalidation:
        return {
            "score": 12.0,
            "action": "跌破防守，暫不進場",
            "chase_risk": "高",
            "extension_atr": round(extension_atr, 2),
            "breakout_level": round(breakout, 2),
            "atr14": round(atr, 2),
            "change_pct": round(float(chg), 2),
            "market_change_pct": round(float(mkt), 2),
            "relative_market_pct_pt": round(float(rel), 2),
            "reasons": ["目前價格已跌破原防守結構"],
            "overnight_risk": False,
        }

    # 1) Structural distance: main chase-risk anchor.
    if extension_atr < ext["below_breakout"]:
        score -= 13.0
        caps.append(68.0)
        reasons.append("尚未回到突破區，先等走勢確認")
    elif extension_atr <= ext["ideal_max"]:
        score += 9.0
        reasons.append("價格貼近突破區，風險報酬較佳")
    elif extension_atr <= ext["acceptable_max"]:
        score += 3.0
        risk_points += 4.0
    elif extension_atr <= ext["warm_max"]:
        score -= 7.0
        risk_points += 12.0
        reasons.append(f"已高於突破區 {extension_atr:.1f} ATR")
    elif extension_atr <= ext["hot_max"]:
        score -= 19.0
        risk_points += 24.0
        reasons.append(f"已高於突破區 {extension_atr:.1f} ATR，偏追高")
    else:
        score -= 32.0
        risk_points += 38.0
        caps.append(52.0 if horizon == "short" else 60.0)
        reasons.append(f"離突破區約 {extension_atr:.1f} ATR，延伸過大")

    if ma5_extension_atr is not None:
        if ma5_extension_atr > ma5_cfg["hot"]:
            score -= 10.0
            risk_points += 12.0
            reasons.append("價格離 5 日均線偏遠")
        elif ma5_extension_atr > ma5_cfg["warm"]:
            score -= 4.0
            risk_points += 5.0

    # 2) One-day acceleration.  This is deliberately *not* in intraday ranking.
    near_limit = chg >= move["near_limit"]
    if near_limit:
        score -= 40.0
        risk_points += 58.0
        caps.append(28.0)
        reasons.append("今日已接近漲停，隔夜回吐風險明顯上升")
    elif chg >= move["very_hot"]:
        score -= 25.0
        risk_points += 36.0
        caps.append(42.0 if horizon == "short" else 50.0)
        reasons.append("單日漲幅已大，不適合用強勢代替買點")
    elif chg >= move["hot"]:
        score -= 15.0
        risk_points += 22.0
        caps.append(55.0 if horizon == "short" else 62.0)
        reasons.append("今日已明顯噴出，建議等拉回")
    elif chg >= move["watch"]:
        score -= 5.0
        risk_points += 8.0
    elif chg <= -5.0:
        score -= 12.0
        risk_points += 10.0
        reasons.append("今日價格明顯轉弱")

    # 3) Broad-market context affects execution, not stock selection.
    if mkt >= policy["market_surge"] and chg >= move["hot"]:
        score -= 6.0
        risk_points += 10.0
        reasons.append("大盤與個股同步大漲，需留意隔日風險偏好降溫")
    if rel >= policy["relative_strength_bonus"]:
        score += 4.0
    elif rel <= policy["relative_weakness"]:
        score -= 8.0
        risk_points += 8.0
        reasons.append("相對大盤並不強，不宜只看絕對漲幅")

    if vr is not None:
        if vol_cfg["healthy_low"] <= vr <= vol_cfg["healthy_high"] and chg < move["hot"]:
            score += 4.0
        elif vr >= vol_cfg["extreme"] and chg >= move["hot"]:
            score -= 5.0
            risk_points += 8.0
            reasons.append("量能過度放大，需留意短線高檔換手")
    if dpos is not None:
        if dpos < dpos_cfg["weak"]:
            score -= 8.0
            risk_points += 6.0
        elif dpos >= dpos_cfg["strong"] and chg < move["hot"]:
            score += 2.0

    if horizon == "mid" and not near_limit:
        score += 2.0
    elif horizon == "long" and not near_limit:
        score += 4.0

    if caps:
        score = min(score, min(caps))
    score = float(np.clip(score, 0.0, 100.0))

    if risk_points >= risk_cfg["extreme"]:
        risk = "極高"
    elif risk_points >= risk_cfg["high"]:
        risk = "高"
    elif risk_points >= risk_cfg["medium"]:
        risk = "中"
    else:
        risk = "低"

    if near_limit:
        action = "漲停/近漲停：強勢但不追"
    elif chg >= move["very_hot"] or extension_atr > ext["hot_max"]:
        action = "強勢但不追，等拉回"
    elif score >= action_cfg["good"]:
        action = "買點佳，可分批"
    elif score >= action_cfg["small"]:
        action = "可小量試單"
    elif score >= action_cfg["wait"]:
        action = "等拉回/等確認"
    elif score >= action_cfg["watch"]:
        action = "強勢標的，現在不追"
    else:
        action = "暫不進場"

    overnight = bool(
        chg >= move["hot"]
        or (mkt >= policy["market_surge"] and chg >= 4.0)
        or extension_atr > ext["warm_max"]
    )
    if not reasons:
        reasons.append("價格仍在可控的追價距離內")
    return {
        "score": round(score, 1),
        "action": action,
        "chase_risk": risk,
        "extension_atr": round(float(extension_atr), 2),
        "ma5_extension_atr": None if ma5_extension_atr is None else round(float(ma5_extension_atr), 2),
        "breakout_level": round(float(breakout), 2),
        "atr14": round(float(atr), 2),
        "change_pct": round(float(chg), 2),
        "market_change_pct": round(float(mkt), 2),
        "relative_market_pct_pt": round(float(rel), 2),
        "overnight_risk": overnight,
        "reasons": reasons[:3],
    }


def position_guidance(regime: str | None, horizon: str, timing: dict | None) -> dict:
    """Convert entry quality + market regime into a risk-budget suggestion.

    The returned fraction is relative to the user's *normal planned position*.
    It is not a portfolio-allocation recommendation and never changes ranking.
    """
    timing = timing or {}
    score = _num(timing.get("score"))
    risk = str(timing.get("chase_risk") or "待確認")
    regime_key = str(regime or "UNKNOWN").upper()
    regime_mult = float(REGIME_POSITION_MULTIPLIER.get(regime_key, REGIME_POSITION_MULTIPLIER["UNKNOWN"]))

    if score is None:
        entry_key = "avoid"
    elif score >= ENTRY_POLICY["action"]["good"]:
        entry_key = "good"
    elif score >= ENTRY_POLICY["action"]["small"]:
        entry_key = "small"
    elif score >= ENTRY_POLICY["action"]["wait"]:
        entry_key = "wait"
    else:
        entry_key = "avoid"

    base = float(ENTRY_POSITION_MULTIPLIER[entry_key])
    risk_cap = float(CHASE_POSITION_CAP.get(risk, CHASE_POSITION_CAP["待確認"]))
    fraction = min(base * regime_mult, risk_cap)

    action = str(timing.get("action") or "")
    if "不追" in action or "暫不進場" in action or "跌破防守" in action:
        fraction = 0.0
    fraction = float(np.clip(fraction, 0.0, 1.0))
    pct = int(round(fraction * 100))

    if pct == 0:
        label = "觀望（0%）"
    elif pct <= 20:
        label = f"極小試單（上限 {pct}%）"
    elif pct <= 40:
        label = f"小量試單（上限 {pct}%）"
    elif pct <= 70:
        label = f"分批建立（上限 {pct}%）"
    else:
        label = f"正常分批（上限 {pct}%）"

    regime_text = {"BULL": "偏多", "NEUTRAL": "整理", "BEAR": "偏弱", "UNKNOWN": "待確認"}.get(regime_key, "待確認")
    return {
        "fraction": round(fraction, 2),
        "percent": pct,
        "label": label,
        "regime": regime_key,
        "reason": f"大盤{regime_text}只調整風險預算，不改標的排名",
    }


def entry_timing_from_df(
    df: pd.DataFrame,
    plan: dict | None,
    horizon: str,
    market_change_pct: float | None = None,
) -> dict:
    if df is None or df.empty or plan is None:
        return evaluate_entry_timing(None, plan, horizon)
    x = df.sort_index()
    close = pd.to_numeric(x["Close"], errors="coerce")
    p = _num(close.iloc[-1])
    prev = _num(close.iloc[-2]) if len(close) >= 2 else None
    chg = ((p / prev) - 1.0) * 100.0 if p is not None and prev not in (None, 0) else 0.0
    mkt = _num(market_change_pct, 0.0) or 0.0
    vol = pd.to_numeric(x["Volume"], errors="coerce") if "Volume" in x.columns else pd.Series(dtype=float)
    base = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else (float(vol.iloc[:-1].tail(20).mean()) if len(vol) > 1 else np.nan)
    vr = float(vol.iloc[-1] / base) if len(vol) and np.isfinite(base) and base > 0 else None
    hi = _num(x["High"].iloc[-1]) if "High" in x.columns else None
    lo = _num(x["Low"].iloc[-1]) if "Low" in x.columns else None
    dpos = (p - lo) / (hi - lo) if p is not None and hi is not None and lo is not None and hi > lo else None
    return evaluate_entry_timing(
        p,
        plan,
        horizon,
        change_pct=chg,
        market_change_pct=mkt,
        volume_ratio=vr,
        relative_market_pct_pt=chg - mkt,
        day_position=dpos,
    )
