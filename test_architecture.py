"""Offline regression tests for Alpha Radar V16.3 Architecture Freeze.

Run with: python test_architecture.py
No network calls are made.
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd

import intraday_engine as ie
import radar_service as rs
from policy_engine import evaluate_entry_timing, generate_trade_plan, position_guidance
from return_first_model import estimate_all_horizons, estimate_horizon_return
from strategy_config import (
    ARCHITECTURE_VERSION,
    INTRADAY_BLEND_SCHEDULE,
    SELECTION_WEIGHTS,
)

TZ = dt.timezone(dt.timedelta(hours=8))


def synthetic_ohlcv(n: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    drift = 0.00045
    ret = rng.normal(drift, 0.018, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = close * (1.0 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.018, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.018, n))
    vol = rng.integers(500_000, 8_000_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=dates)


def assert_between(x, lo, hi, label):
    assert x is not None and lo <= float(x) <= hi, f"{label}: {x} not in [{lo}, {hi}]"


def test_config_weights():
    for h, weights in SELECTION_WEIGHTS.items():
        assert abs(sum(weights.values()) - 1.0) < 1e-9, (h, weights)
    for h, schedule in INTRADAY_BLEND_SCHEDULE.items():
        vals = [x[1] for x in schedule]
        assert vals == sorted(vals), (h, vals)
        assert 0 <= vals[0] <= vals[-1] <= 0.5


def test_entry_separation():
    plan = {
        "reference_close": 100.0,
        "breakout_level": 100.0,
        "atr14": 3.0,
        "ma5": 100.0,
        "invalidation": 94.0,
    }
    normal = evaluate_entry_timing(
        101.0, plan, "short", change_pct=1.2, market_change_pct=0.4,
        volume_ratio=1.4, relative_market_pct_pt=0.8, day_position=0.72,
    )
    hot = evaluate_entry_timing(
        109.8, plan, "short", change_pct=9.8, market_change_pct=2.4,
        volume_ratio=2.0, relative_market_pct_pt=7.4, day_position=0.98,
    )
    assert normal["score"] > hot["score"], (normal, hot)
    assert hot["score"] <= 28.0, hot
    assert "不追" in hot["action"], hot
    assert hot["overnight_risk"] is True


def test_regime_only_changes_position():
    timing = {"score": 86.0, "chase_risk": "低", "action": "買點佳，可分批"}
    bull = position_guidance("BULL", "short", timing)
    neutral = position_guidance("NEUTRAL", "short", timing)
    bear = position_guidance("BEAR", "short", timing)
    assert bull["percent"] > neutral["percent"] > bear["percent"] > 0, (bull, neutral, bear)
    hot = position_guidance("BULL", "short", {"score": 25.0, "chase_risk": "極高", "action": "漲停/近漲停：強勢但不追"})
    assert hot["percent"] == 0, hot


def test_time_matured_intraday_weight():
    w_open = ie.intraday_blend_weight("short", dt.datetime(2026, 10, 5, 9, 5, tzinfo=TZ))
    w_mid = ie.intraday_blend_weight("short", dt.datetime(2026, 10, 5, 11, 0, tzinfo=TZ))
    w_late = ie.intraday_blend_weight("short", dt.datetime(2026, 10, 5, 12, 55, tzinfo=TZ))
    assert w_open < w_mid < w_late <= 0.45, (w_open, w_mid, w_late)


def _live_row(stock_id, close, open_, high, low, avg, change, volume_ratio, amount=500_000_000):
    return {
        "stock_id": stock_id,
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "average_price": avg,
        "change_rate": change,
        "volume_ratio": volume_ratio,
        "total_volume": 2_000_000,
        "total_amount": amount,
        "buy_price": close - 0.1,
        "sell_price": close + 0.1,
        "buy_volume": 600,
        "sell_volume": 400,
        "date": "2026-10-05",
    }


def test_limit_up_can_rank_high_but_not_be_buyable():
    plan_a = {"reference_close": 100.0, "breakout_level": 100.0, "atr14": 3.0, "ma5": 100.0, "invalidation": 94.0}
    plan_b = {"reference_close": 100.0, "breakout_level": 100.0, "atr14": 3.0, "ma5": 100.0, "invalidation": 94.0}
    base_stocks = [
        {"ticker": "1111.TW", "name": "A", "avg_volume_20d": 2_000_000, "horizons": {"short": {"ranking_score": 92.0, "plan": plan_a}}},
        {"ticker": "2222.TW", "name": "B", "avg_volume_20d": 2_000_000, "horizons": {"short": {"ranking_score": 84.0, "plan": plan_b}}},
    ]
    rows = [
        _live_row("001", 22000, 21600, 22100, 21500, 21800, 2.4, 1.4, 30_000_000_000),
        _live_row("1111", 109.8, 103.0, 109.8, 102.0, 106.0, 9.8, 2.1),
        _live_row("2222", 103.2, 101.0, 104.0, 100.5, 102.0, 3.2, 1.5),
    ]
    df = pd.DataFrame(rows)
    now = dt.datetime(2026, 10, 5, 12, 30, tzinfo=TZ)
    snap_bull = {"market": {"regime": "BULL"}, "stocks": base_stocks}
    snap_bear = {"market": {"regime": "BEAR"}, "stocks": base_stocks}
    bull = ie.rerank_snapshot(snap_bull, "short", df, top_n=2, now=now)
    bear = ie.rerank_snapshot(snap_bear, "short", df, top_n=2, now=now)
    assert bull and bear
    assert bull[0]["ticker"] == "1111.TW", bull
    # Market regime must not change ranking/score, only position budget.
    assert [x["ticker"] for x in bull] == [x["ticker"] for x in bear]
    assert [x["live_ranking_score"] for x in bull] == [x["live_ranking_score"] for x in bear]
    a_bull = bull[0]["intraday"]
    a_bear = bear[0]["intraday"]
    assert a_bull["intraday_score"] >= 60.0, a_bull
    assert "overheat_penalty" not in a_bull
    assert a_bull["entry_timing"]["score"] <= 28.0
    assert "不追" in a_bull["entry_timing"]["action"]
    assert a_bull["position_guidance"]["percent"] == 0
    assert a_bear["position_guidance"]["percent"] == 0


def test_shared_feature_engine_matches_legacy_calls():
    df = synthetic_ohlcv(900, seed=11)
    benchmark = synthetic_ohlcv(900, seed=22)
    settings = rs.RunSettings(candidate_size=1000)
    t0 = time.perf_counter()
    shared = estimate_all_horizons(df, settings, twii_ret_20d=0.01, benchmark_df=benchmark)
    shared_sec = time.perf_counter() - t0
    t1 = time.perf_counter()
    separate = {h: estimate_horizon_return(df, h, settings, twii_ret_20d=0.01, benchmark_df=benchmark) for h in ("short", "mid", "long")}
    separate_sec = time.perf_counter() - t1
    for h in ("short", "mid", "long"):
        assert shared[h].get("technical_factor_score") == separate[h].get("technical_factor_score"), h
        assert shared[h].get("estimate_available") == separate[h].get("estimate_available"), h
    return shared_sec, separate_sec


def test_offline_single_stock_pipeline():
    df = synthetic_ohlcv(900, seed=33)
    settings = rs.RunSettings(candidate_size=1000)
    stock = rs._evaluate_one("9999.TW", "測試股", "測試產業", df, settings, 0.01, {})
    assert stock["ticker"] == "9999.TW"
    for h in ("short", "mid", "long"):
        block = stock["horizons"][h]
        assert_between(block["ranking_score"], 0, 100, f"ranking {h}")
        assert "entry_timing" in block
        assert "position_guidance" in block
        assert "forecast" in block




def test_utf8_ui_strings_are_clean():
    root = Path(__file__).resolve().parent
    for name in ("app.py", "policy_engine.py", "intraday_engine.py", "strategy_config.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "\ufffd" not in text, f"UTF-8 replacement character found in {name}"
    app_text = (root / "app.py").read_text(encoding="utf-8")
    for phrase in ("標的分數", "進場分數", "追價風險", "部位上限"):
        assert phrase in app_text, phrase


def test_offline_run_scan_regime_does_not_rank():
    """Exercise radar_service.run_scan end-to-end with an in-memory fake store."""
    base = synthetic_ohlcv(420, seed=101)
    tickers = [f"{3000+i}.TW" for i in range(12)]
    frames = {}
    for i, ticker in enumerate(tickers):
        df = base.copy()
        # Deterministic cross-sectional differences without changing date coverage.
        scale = 0.86 + i * 0.025
        df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]] * scale
        df["Volume"] = df["Volume"] * (0.8 + i * 0.04)
        # Slight recent momentum differentiation.
        df.loc[df.index[-40:], ["Open", "High", "Low", "Close"]] *= np.linspace(1.0, 1.0 + i * 0.006, 40)[:, None]
        frames[ticker] = df
    benchmark = synthetic_ohlcv(420, seed=202)
    frames["^TWII"] = benchmark
    expected_date = str(base.index[-1].date())

    class FakeStore:
        def __init__(self, _path):
            self.frames = frames
        def batch_fetch_and_update(self, requested, period="5y", progress=None):
            if progress:
                progress(len(requested), len(requested), "fake price cache")
            return {"downloaded_tickers": len(requested), "requested": len(requested), "mode": "incremental", "covered_ratio": 1.0, "last_full_refresh": expected_date, "errors": []}
        def overlay_official_eod(self, target_date="", progress=None):
            if progress:
                progress(1, 1, "fake official close")
            return {"twse_date": expected_date, "tpex_date": expected_date, "twse_target_response": True, "twse_target_has_data": True, "errors": []}
        def iter_prices(self, requested, limit=None):
            for t in requested[:limit] if limit else requested:
                if t in self.frames:
                    yield t, self.frames[t]
        def get_prices(self, ticker):
            return self.frames.get(ticker, pd.DataFrame()).copy()

    universe = pd.DataFrame({
        "ticker": tickers,
        "name": [f"測試{i}" for i in range(len(tickers))],
        "industry": ["測試產業"] * len(tickers),
    })

    originals = {
        "DailyPriceStore": rs.DailyPriceStore,
        "fetch_twse_universe": rs.fetch_twse_universe,
        "calendar_reference": rs.calendar_reference,
        "_market_context": rs._market_context,
        "_enrich_research": rs._enrich_research,
        "_enrich_tickers": rs._enrich_tickers,
    }
    try:
        rs.DailyPriceStore = FakeStore
        rs.fetch_twse_universe = lambda: universe.copy()
        rs.calendar_reference = lambda data_dir, now=None: {"expected_date": expected_date}
        rs._enrich_research = lambda stocks, data_dir, settings, progress=None: None
        rs._enrich_tickers = lambda stocks, data_dir, tickers, progress=None, progress_value=0.88: None

        def run(regime):
            rs._market_context = lambda store: {
                "benchmark": "^TWII", "regime": regime, "ret20": 0.01,
                "day_change_pct": 0.5, "price": 22000.0, "ma20": 21500.0,
                "ma60": 21000.0, "price_date": expected_date,
            }
            settings = rs.RunSettings(candidate_size=50, research_pool_per_horizon=5)
            return rs.run_scan(Path("/tmp/alpha-v163-test"), settings)

        bull = run("BULL")
        bear = run("BEAR")
        for h in ("short", "mid", "long"):
            bull_rank = [(x["ticker"], x["horizons"][h]["ranking_score"]) for x in rs.select_market_best(bull, h, n=5)]
            bear_rank = [(x["ticker"], x["horizons"][h]["ranking_score"]) for x in rs.select_market_best(bear, h, n=5)]
            assert bull_rank == bear_rank, (h, bull_rank, bear_rank)
        # Same stock/entry quality, different regime => different risk budget.
        b = rs.select_market_best(bull, "short", n=1)[0]["horizons"]["short"]["position_guidance"]["percent"]
        r = rs.select_market_best(bear, "short", n=1)[0]["horizons"]["short"]["position_guidance"]["percent"]
        assert b >= r, (b, r)
        assert bull["model_version"] == ARCHITECTURE_VERSION
    finally:
        for name, value in originals.items():
            setattr(rs, name, value)


def main():
    tests = [
        test_config_weights,
        test_entry_separation,
        test_regime_only_changes_position,
        test_time_matured_intraday_weight,
        test_limit_up_can_rank_high_but_not_be_buyable,
        test_offline_single_stock_pipeline,
        test_utf8_ui_strings_are_clean,
        test_offline_run_scan_regime_does_not_rank,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    shared_sec, separate_sec = test_shared_feature_engine_matches_legacy_calls()
    print("PASS test_shared_feature_engine_matches_legacy_calls")
    print(f"BENCH shared_features={shared_sec:.4f}s separate_3x={separate_sec:.4f}s speedup={separate_sec/max(shared_sec,1e-9):.2f}x")
    print(f"ALL TESTS PASSED | {ARCHITECTURE_VERSION}")


if __name__ == "__main__":
    main()
