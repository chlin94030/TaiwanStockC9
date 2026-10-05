"""Taiwan Alpha Radar V16.3 Architecture Freeze - scan, rank, enrich, diagnose."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

from market_data import DailyPriceStore, ResearchDataClient, fetch_twse_universe, _taipei_timestamp
from policy_engine import generate_trade_plan, evaluate_entry_state, entry_timing_from_df, position_guidance, setup_from_df
from return_first_model import estimate_all_horizons
from industry_profile import fine_industry
from trading_calendar import calendar_reference
from strategy_config import ARCHITECTURE_VERSION, SCAN_DEFAULTS, SELECTION_WEIGHTS

OPERATIONS_VERSION = ARCHITECTURE_VERSION


@dataclass
class RunSettings:
    reference_size: int = int(SCAN_DEFAULTS["reference_size"])
    candidate_size: int = int(SCAN_DEFAULTS["candidate_size"])
    research_pool_per_horizon: int = int(SCAN_DEFAULTS["research_pool_per_horizon"])
    history_period: str = str(SCAN_DEFAULTS["history_period"])
    model_family: str = "full"  # price_only | business_confirmed | flow_confirmed | full
    order_mode: str = "next_open"
    commission: float = 0.001425
    sell_tax: float = 0.003
    slippage: float = 0.0005
    notional: float = 100000.0
    min_ev_short: float = 0.0
    min_ev_mid: float = 0.0
    min_ev_long: float = 0.0
    min_price: float = float(SCAN_DEFAULTS["min_price"])
    min_avg_turnover: float = float(SCAN_DEFAULTS["min_avg_turnover"])


def load_dashboard(path: Path, include_features: bool = False) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def compact_session_dashboard(snap: dict | None) -> dict:
    if not snap or not isinstance(snap, dict):
        return {}
    out = dict(snap)
    out["charts"] = {}
    return out


def compact_doctor_result(dr: dict | None) -> dict:
    return dict(dr) if isinstance(dr, dict) else {}


def remove_saved_dashboard(path: Path) -> bool:
    try:
        if path.exists():
            path.unlink()
        return True
    except Exception:
        return False


def snapshot_bytes(snap: dict) -> bytes:
    return json.dumps(snap, ensure_ascii=False, indent=2).encode("utf-8")


def chart_on_demand(snap: dict | None, ticker: str, data_dir: Path, allow_fetch: bool = False) -> dict | None:
    if not ticker:
        return None
    store = DailyPriceStore(data_dir / "daily_prices.sqlite")
    df = store.get_prices(ticker)
    if df.empty and allow_fetch:
        store.batch_fetch_and_update([ticker], period="1y")
        df = store.get_prices(ticker)
    if df.empty:
        return None
    tail = df.tail(260)
    return {
        "dates": tail.index.strftime("%Y-%m-%d").tolist(),
        "ohlcv": tail[["Open", "High", "Low", "Close", "Volume"]].to_numpy().tolist(),
    }


def _market_context(store: DailyPriceStore) -> dict:
    df = store.get_prices("^TWII")
    if df.empty or len(df) < 65:
        return {"benchmark": "^TWII", "regime": "UNKNOWN", "ret20": 0.0, "price_date": ""}
    close = df["Close"]
    p = float(close.iloc[-1])
    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())
    ret20 = p / float(close.iloc[-21]) - 1.0 if len(close) >= 21 else 0.0
    day_change_pct = ((p / float(close.iloc[-2])) - 1.0) * 100.0 if len(close) >= 2 and float(close.iloc[-2]) > 0 else 0.0
    if p >= ma20 and ma20 >= ma60:
        regime = "BULL"
    elif p < ma20 and p >= ma60:
        regime = "NEUTRAL"
    else:
        regime = "BEAR"
    return {
        "benchmark": "^TWII",
        "regime": regime,
        "ret20": ret20,
        "day_change_pct": round(float(day_change_pct), 3),
        "price": p,
        "ma20": ma20,
        "ma60": ma60,
        "price_date": str(df.index[-1].date()),
    }


def _pre_score(df: pd.DataFrame, market_ret20: float) -> float:
    if df.empty or len(df) < 130:
        return -999.0
    close = df["Close"]
    vol = df["Volume"]
    p = float(close.iloc[-1])
    r20 = p / float(close.iloc[-21]) - 1.0 if len(close) >= 21 else 0.0
    r60 = p / float(close.iloc[-61]) - 1.0 if len(close) >= 61 else r20
    r120 = p / float(close.iloc[-121]) - 1.0 if len(close) >= 121 else r60
    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())
    vr = float(vol.tail(5).mean()) / max(1.0, float(vol.tail(20).mean()))
    rs = r20 - market_ret20
    trend = (1.0 if p > ma20 else 0.0) + (1.0 if ma20 > ma60 else 0.0)
    return 100.0 * (0.28 * np.tanh(rs * 5) + 0.24 * np.tanh(r60 * 2.5) + 0.18 * np.tanh(r120 * 1.5) + 0.12 * np.tanh((vr - 1) * 1.4) + 0.09 * trend)


def _fundamental_score(research: dict) -> tuple[float | None, dict]:
    if not research:
        return None, {}
    rev = research.get("monthly_revenue") or {}
    fin = research.get("financials") or {}
    val = research.get("valuation") or {}
    pieces = []
    detail = {}

    latest = rev.get("latest") or {}
    if rev.get("available") and latest:
        yoy = latest.get("yoy_pct")
        mom = latest.get("mom_pct")
        avg3 = rev.get("avg_yoy_3m_pct")
        rev_parts = []
        if yoy is not None:
            rev_parts.append(float(np.clip(50 + yoy * 1.1, 0, 100)))
        if avg3 is not None:
            rev_parts.append(float(np.clip(50 + avg3 * 1.0, 0, 100)))
        if mom is not None:
            rev_parts.append(float(np.clip(50 + mom * 1.2, 0, 100)))
        if rev_parts:
            score = float(np.mean(rev_parts))
            pieces.append((score, 0.45))
            detail["revenue_score"] = round(score, 1)

    if fin.get("available"):
        qs = fin.get("quarters") or []
        eps = [q.get("eps") for q in qs if q.get("eps") is not None]
        gm = [q.get("gross_margin_pct") for q in qs if q.get("gross_margin_pct") is not None]
        ep = 50.0
        if eps:
            ep = 65.0 if eps[-1] > 0 else 25.0
            if len(eps) >= 2:
                ep += 15.0 if eps[-1] > eps[-2] else -10.0
            if sum(eps) > 0:
                ep += 5.0
            ep = float(np.clip(ep, 0, 100))
            pieces.append((ep, 0.35))
            detail["earnings_score"] = round(ep, 1)
        if len(gm) >= 2:
            gm_score = float(np.clip(50 + (gm[-1] - gm[0]) * 4.0, 20, 90))
            pieces.append((gm_score, 0.20))
            detail["margin_score"] = round(gm_score, 1)

    if not pieces:
        return None, detail
    wsum = sum(w for _, w in pieces)
    score = sum(s * w for s, w in pieces) / wsum

    # Only a mild valuation sanity adjustment; growth stocks are not penalized solely for a high PE.
    pe = (val or {}).get("pe")
    if pe is not None:
        if pe <= 0:
            score -= 8
        elif pe > 100:
            score -= 5
    return round(float(np.clip(score, 0, 100)), 1), detail


def _flow_score(research: dict, avg_volume_shares: float) -> tuple[float | None, dict]:
    if not research:
        return None, {}
    inst = research.get("institutional_flow") or {}
    if not inst.get("available"):
        return None, {}
    net_lots = float(inst.get("total_net_lots") or 0.0)
    monthly_lots = max(1.0, avg_volume_shares / 1000.0 * max(1, int(inst.get("sessions") or 20)))
    ratio = net_lots / monthly_lots
    score = float(np.clip(50.0 + ratio * 180.0, 5.0, 95.0))
    detail = {"institutional_net_to_volume": round(ratio, 4)}

    branch = research.get("main_force_proxy") or {}
    if branch.get("available"):
        proxy = float(branch.get("proxy_net_lots") or 0.0)
        branch_ratio = proxy / max(1.0, avg_volume_shares / 1000.0 * max(1, int(branch.get("sessions") or 10)))
        branch_score = float(np.clip(50.0 + branch_ratio * 120.0, 5.0, 95.0))
        score = 0.72 * score + 0.28 * branch_score
        detail["branch_proxy_net_to_volume"] = round(branch_ratio, 4)
    return round(float(np.clip(score, 0, 100)), 1), detail


def _forecast_score(forecast: dict) -> float | None:
    if not forecast or not forecast.get("estimate_available"):
        return None
    s = forecast.get("strategy") or {}
    median = float(s.get("median") or 0.0)
    mean = float(s.get("mean") or 0.0)
    p10 = float(s.get("p10") or 0.0)
    # Robust expected-return quality: reward positive median/mean, penalize downside tail.
    reward = 50.0 + np.tanh(median * 8.0) * 28.0 + np.tanh(mean * 6.0) * 12.0
    penalty = max(0.0, -p10 - 0.05) * 80.0
    return round(float(np.clip(reward - penalty, 0, 100)), 1)


def _combined_ranking_score(stock: dict, horizon: str, model_family: str) -> None:
    block = stock["horizons"][horizon]
    forecast = block.get("forecast") or {}
    technical = float(forecast.get("technical_factor_score", forecast.get("composite_factor_score", 0.0)) or 0.0)
    empirical = _forecast_score(forecast)
    fundamental, f_detail = _fundamental_score(stock.get("research") or {})
    flow, flow_detail = _flow_score(stock.get("research") or {}, float(stock.get("avg_volume_20d") or 0.0))

    weights = SELECTION_WEIGHTS[horizon]

    allowed = {"technical", "empirical"}
    if model_family in {"business_confirmed", "full"}:
        allowed.add("fundamental")
    if model_family in {"flow_confirmed", "full"}:
        allowed.add("flow")

    values = {"technical": technical, "empirical": empirical, "fundamental": fundamental, "flow": flow}
    present = [(k, values[k], weights[k]) for k in allowed if values[k] is not None]
    if not present:
        rank = technical
    else:
        wsum = sum(w for _, _, w in present)
        rank = sum(float(v) * w for _, v, w in present) / wsum

    # Gentle market-regime adaptation is performed in run_scan by adding a small modifier.
    block["ranking_score"] = round(float(np.clip(rank, 0, 100)), 1)
    block["score_components"] = {
        "technical": round(technical, 1),
        "empirical": empirical,
        "fundamental": fundamental,
        "flow": flow,
        **f_detail,
        **flow_detail,
    }


def _enrich_tickers(stocks: list[dict], data_dir: Path, tickers: list[str], progress=None, progress_value: float = 0.88):
    pool = [t for t in dict.fromkeys(tickers) if t]
    if not pool:
        return
    by_ticker = {s["ticker"]: s for s in stocks}
    if progress:
        progress(f"補充公司與法人資料（{len(pool)} 檔）", progress_value)

    def one(ticker):
        client = ResearchDataClient(data_dir / "research_cache.sqlite")
        return ticker, client.stock_research(ticker, include_branch=False)

    workers = min(6, max(1, len(pool)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, ticker) for ticker in pool]
        for fut in as_completed(futures):
            try:
                ticker, research = fut.result()
                if ticker in by_ticker:
                    by_ticker[ticker]["research"] = research
            except Exception:
                continue


def _enrich_research(stocks: list[dict], data_dir: Path, settings: RunSettings, progress=None):
    if not stocks:
        return
    tickers = []
    for h in ["short", "mid", "long"]:
        ranked = sorted(
            stocks,
            key=lambda x: x.get("horizons", {}).get(h, {}).get("forecast", {}).get("composite_factor_score", 0),
            reverse=True,
        )[: max(5, int(settings.research_pool_per_horizon))]
        tickers.extend([x["ticker"] for x in ranked])
    _enrich_tickers(stocks, data_dir, tickers, progress=progress, progress_value=0.88)

def run_scan(data_dir: Path, settings: RunSettings, progress=None) -> dict:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("載入上市櫃股票", 0.05)
    universe = fetch_twse_universe()
    tickers = universe["ticker"].astype(str).tolist()

    store = DailyPriceStore(data_dir / "daily_prices.sqlite")
    if progress:
        progress(f"更新歷史與最新交易日", 0.16)

    def price_progress(done, total, label):
        if progress:
            frac = float(done) / max(1.0, float(total))
            progress(str(label), 0.16 + 0.25 * max(0.0, min(1.0, frac)))

    dl = store.batch_fetch_and_update(
        tickers + ["^TWII"],
        period=settings.history_period,
        progress=price_progress,
    )
    # Resolve the latest *complete* Taiwan trading session first.  Yahoo often
    # labels the newest Taiwan bar one session late on cloud deployments, so the
    # exchange close is authoritative and Yahoo is only the historical backbone.
    cal = calendar_reference(data_dir, now=_taipei_timestamp())
    calendar_expected_date = str(cal.get("expected_date") or "")
    def official_progress(done, total, label):
        if progress:
            frac = float(done) / max(1.0, float(total))
            progress(str(label), 0.42 + 0.05 * max(0.0, min(1.0, frac)))

    official_eod = store.overlay_official_eod(
        target_date=calendar_expected_date, progress=official_progress
    )
    expected_date = calendar_expected_date
    # Weekday public holidays are not inferable from weekday arithmetic alone.
    # If the date-specific TWSE endpoint responds successfully but explicitly
    # has no data, trust the exchange's latest session date instead of excluding
    # the whole market as stale. A network failure does not trigger this fallback.
    if (
        calendar_expected_date
        and official_eod.get("twse_target_response")
        and not official_eod.get("twse_target_has_data")
        and official_eod.get("twse_date")
        and str(official_eod.get("twse_date")) < calendar_expected_date
    ):
        expected_date = str(official_eod.get("twse_date"))
        official_eod["calendar_expected_date"] = calendar_expected_date
        official_eod["exchange_calendar_adjusted"] = True
    market = _market_context(store)
    market_ret20 = float(market.get("ret20") or 0.0)

    liquid = []
    valid_count = 0
    stale_count = 0
    if progress:
        progress("篩選成交量足夠的股票", 0.48)
    universe_meta = {str(r.get("ticker")): r for r in universe.to_dict("records")}
    for ticker, df in store.iter_prices(tickers):
        if len(df) < 140:
            continue
        valid_count += 1
        latest_stock_date = str(df.index[-1].date())
        if expected_date and latest_stock_date < expected_date:
            stale_count += 1
            continue
        row = universe_meta.get(ticker, {})
        try:
            p = float(df["Close"].iloc[-1])
            avg_vol = float(df["Volume"].tail(20).mean())
            avg_turnover = float((df["Close"].tail(20) * df["Volume"].tail(20)).mean())
            if p < settings.min_price or avg_turnover < settings.min_avg_turnover:
                continue
            liquid.append(
                {
                    "ticker": ticker,
                    "name": str(row.get("name", ticker)),
                    "industry": str(row.get("industry", "")),
                    "price": p,
                    "price_date": latest_stock_date,
                    "avg_volume_20d": avg_vol,
                    "avg_turnover_20d": avg_turnover,
                    "pre_score": _pre_score(df, market_ret20),
                    "df": df,
                }
            )
        except Exception:
            continue

    liquid.sort(key=lambda x: x["pre_score"], reverse=True)
    candidates = liquid[: max(50, int(settings.candidate_size))]

    if progress:
        progress(f"比較走勢與歷史案例（{len(candidates)} 檔）", 0.68)
    evaluated: list[dict] = []
    benchmark_df = store.get_prices("^TWII")
    market_day_change = float(market.get("day_change_pct") or 0.0)
    candidate_total = max(1, len(candidates))
    for idx, c in enumerate(candidates, 1):
        df = c["df"]
        horizons = {}
        # Feature construction (MA/ATR/KD/MACD/Bollinger/RS) is the expensive
        # part. Compute it once per stock and reuse it for all three horizons.
        estimates = estimate_all_horizons(
            df, settings, twii_ret_20d=market_ret20, benchmark_df=benchmark_df
        )
        for h in ["short", "mid", "long"]:
            plan = generate_trade_plan(df, h)
            state = evaluate_entry_state(df, plan)
            est = estimates.get(h) or {}
            timing = entry_timing_from_df(df, plan, h, market_change_pct=market_day_change)
            horizons[h] = {
                "plan": plan,
                "entry_state": state,
                "entry_timing": timing,
                "position_guidance": position_guidance(market.get("regime"), h, timing),
                "forecast": est,
                "qualification": {
                    "research_qualified": bool(est.get("composite_factor_score", 0) >= 35),
                    "rank_first": True,
                },
            }
        evaluated.append(
            {
                "ticker": c["ticker"],
                "name": c["name"],
                "industry": c["industry"],
                "fine_industry": fine_industry(c["ticker"], c["name"], c["industry"]),
                "price": c["price"],
                "price_date": c["price_date"],
                "avg_volume_20d": c["avg_volume_20d"],
                "avg_turnover_20d": c["avg_turnover_20d"],
                "setup": setup_from_df(df),
                "horizons": horizons,
                "research": {},
                "evidence": {},
            }
        )
        if progress and (idx == 1 or idx == candidate_total or idx % 25 == 0):
            progress(
                f"比較走勢與歷史案例（{idx}/{candidate_total} 檔）",
                0.68 + 0.18 * (idx / candidate_total),
            )

    # Always enrich the front of the ranking so the recommendation cards have the requested fields.
    _enrich_research(evaluated, data_dir, settings, progress=progress)

    # Re-rank with chosen evidence family.  Architecture Freeze rule:
    # market regime does NOT alter stock quality/ranking; it only changes
    # position guidance in the execution-policy layer.
    def apply_scores(stock):
        for h in ["short", "mid", "long"]:
            _combined_ranking_score(stock, h, settings.model_family)

    for stock in evaluated:
        apply_scores(stock)

    # Guarantee that every displayed TOP-5 card has the requested research fields.
    # Because adding fundamentals/flows can reshuffle the ranking, iterate on a
    # top-10 buffer until the front of each horizon is enriched or the ranking
    # stabilizes. This avoids a technically strong but data-empty stock appearing
    # in the final recommendations.
    for pass_idx in range(4):
        front = []
        for h in ["short", "mid", "long"]:
            front.extend([
                s["ticker"] for s in sorted(
                    evaluated,
                    key=lambda x: x["horizons"][h].get("ranking_score", -1),
                    reverse=True,
                )[:10]
            ])
        front = list(dict.fromkeys(front))
        missing = [t for t in front if not (next((x for x in evaluated if x["ticker"] == t), {}).get("research"))]
        if not missing:
            break
        _enrich_tickers(evaluated, data_dir, missing, progress=progress, progress_value=0.90 + pass_idx * 0.01)
        enriched_set = set(missing)
        for stock in evaluated:
            if stock["ticker"] in enriched_set:
                apply_scores(stock)

    # Broker-branch data is expensive (Sponsor, one date per request).  Only fetch
    # it for stocks that are already in the displayed top-5 union, and only when
    # the operator explicitly enables it.
    if os.getenv("ENABLE_BRANCH_FLOW", "0") == "1":
        top_tickers = []
        for h in ["short", "mid", "long"]:
            top_tickers.extend([s["ticker"] for s in sorted(evaluated, key=lambda x: x["horizons"][h].get("ranking_score", -1), reverse=True)[:5]])
        top_tickers = list(dict.fromkeys(top_tickers))
        by_ticker = {s["ticker"]: s for s in evaluated}
        if progress:
            progress(f"補充券商分點（{len(top_tickers)} 檔）", 0.96)
        def branch_one(ticker):
            client = ResearchDataClient(data_dir / "research_cache.sqlite")
            return ticker, client.branch_main_force_proxy(ticker.split(".")[0], max_sessions=20)
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(top_tickers)))) as ex:
            futs = [ex.submit(branch_one, t) for t in top_tickers]
            for fut in as_completed(futs):
                try:
                    ticker, branch = fut.result()
                    if ticker in by_ticker:
                        by_ticker[ticker].setdefault("research", {})["main_force_proxy"] = branch
                        apply_scores(by_ticker[ticker])
                except Exception:
                    continue

    for stock in evaluated:
        research = stock.get("research") or {}
        stock["evidence"] = {
            "revenue": bool((research.get("monthly_revenue") or {}).get("available")),
            "financials": bool((research.get("financials") or {}).get("available")),
            "valuation": bool((research.get("valuation") or {}).get("available")),
            "institutional_flow": bool((research.get("institutional_flow") or {}).get("available")),
            "main_force_proxy": bool((research.get("main_force_proxy") or {}).get("available")),
        }

    if progress:
        progress("完成", 1.0)
    dates = [s["price_date"] for s in evaluated if s.get("price_date")]
    latest_date = max(dates) if dates else str(market.get("price_date") or "")
    snap = {
        "snapshot_id": f"snap_{_taipei_timestamp().strftime('%Y%m%d_%H%M%S')}",
        "price_date": latest_date,
        "market": market,
        "coverage": {
            "requested": len(universe),
            "downloaded": int(dl.get("downloaded_tickers", 0)),
            "feature_valid": valid_count,
            "liquid": len(liquid),
            "stale_excluded": stale_count,
            "expected_date": expected_date,
            "calendar_expected_date": calendar_expected_date,
            "errors": dl.get("errors", []) + official_eod.get("errors", []),
            "history_refresh": {
                "mode": dl.get("mode"),
                "requested": int(dl.get("requested", 0) or 0),
                "downloaded_tickers": int(dl.get("downloaded_tickers", 0) or 0),
                "covered_ratio": dl.get("covered_ratio"),
                "last_full_refresh": dl.get("last_full_refresh"),
            },
            "official_eod": official_eod,
        },
        "candidate_n": len(evaluated),
        "stocks": evaluated,
        "settings": asdict(settings),
        "source_type": "rank_first_empirical_live_batch",
        "model_version": OPERATIONS_VERSION,
        "charts": {},
    }
    try:
        (data_dir / "dashboard_snapshot.json").write_text(
            json.dumps(compact_session_dashboard(snap), ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass
    return snap


def select_market_best(snap: dict | None, horizon: str, n: int = 5) -> list:
    """Return top-N by combined ranking score. No absolute gate that can zero the list."""
    if not snap or not isinstance(snap, dict):
        return []
    stocks = [s for s in snap.get("stocks", []) if isinstance(s, dict)]
    return sorted(
        stocks,
        key=lambda x: x.get("horizons", {}).get(horizon, {}).get("ranking_score", -1),
        reverse=True,
    )[:n]


def select_view(snap: dict | None, horizon: str, qualified: bool = True, n: int = 5, **kwargs) -> list:
    return select_market_best(snap, horizon, n)


def _evaluate_one(
    ticker: str,
    name: str,
    industry: str,
    df: pd.DataFrame,
    settings: RunSettings,
    market_ret20: float,
    research: dict,
    market_regime: str = "UNKNOWN",
    market_day_change: float = 0.0,
) -> dict:
    price = float(df["Close"].iloc[-1])
    stock = {
        "ticker": ticker,
        "name": name,
        "industry": industry,
        "fine_industry": fine_industry(ticker, name, industry),
        "price": price,
        "price_date": str(df.index[-1].date()),
        "avg_volume_20d": float(df["Volume"].tail(20).mean()),
        "avg_turnover_20d": float((df["Close"].tail(20) * df["Volume"].tail(20)).mean()),
        "setup": setup_from_df(df),
        "horizons": {},
        "research": research or {},
    }
    estimates = estimate_all_horizons(df, settings, twii_ret_20d=market_ret20)
    for h in ["short", "mid", "long"]:
        plan = generate_trade_plan(df, h)
        est = estimates.get(h) or {}
        timing = entry_timing_from_df(df, plan, h, market_change_pct=market_day_change)
        stock["horizons"][h] = {
            "plan": plan,
            "entry_state": evaluate_entry_state(df, plan),
            "entry_timing": timing,
            "position_guidance": position_guidance(market_regime, h, timing),
            "forecast": est,
            "qualification": {"research_qualified": bool(est.get("composite_factor_score", 0) >= 35), "rank_first": True},
        }
        _combined_ranking_score(stock, h, settings.model_family)
    return stock


def diagnose(code: str, snap: dict | None, data_dir: Path) -> dict:
    clean = str(code or "").strip()
    if not clean:
        return {"snapshot_id": "snap_none", "error": "EMPTY_CODE"}
    snap_id = snap.get("snapshot_id", "snap_unknown") if isinstance(snap, dict) else "snap_none"
    for s in (snap.get("stocks", []) if isinstance(snap, dict) else []):
        if isinstance(s, dict) and (s.get("ticker") == clean or str(s.get("ticker", "")).startswith(clean + ".")):
            return {"snapshot_id": snap_id, "stock": s}

    ticker_options = [clean] if "." in clean else [f"{clean}.TW", f"{clean}.TWO"]
    store = DailyPriceStore(Path(data_dir) / "daily_prices.sqlite")
    store.batch_fetch_and_update(ticker_options, period="5y")
    diag_cal = calendar_reference(Path(data_dir), now=_taipei_timestamp())
    store.overlay_official_eod(target_date=str(diag_cal.get("expected_date") or ""))
    ticker = None
    df = pd.DataFrame()
    for t in ticker_options:
        trial = store.get_prices(t)
        if not trial.empty:
            ticker, df = t, trial
            break
    if ticker is None or df.empty:
        return {"snapshot_id": snap_id, "error": "PRICE_DATA_UNAVAILABLE", "code": clean}

    settings_dict = snap.get("settings", {}) if isinstance(snap, dict) else {}
    try:
        allowed = set(RunSettings.__dataclass_fields__)
        settings = RunSettings(**{k: v for k, v in settings_dict.items() if k in allowed})
    except Exception:
        settings = RunSettings()
    market_ctx = (snap.get("market", {}) if isinstance(snap, dict) else {}) or {}
    market_ret20 = float(market_ctx.get("ret20") or 0.0)
    market_regime = str(market_ctx.get("regime") or "UNKNOWN")
    market_day_change = float(market_ctx.get("day_change_pct") or 0.0)
    research = ResearchDataClient(Path(data_dir) / "research_cache.sqlite").stock_research(
        ticker, include_branch=(os.getenv("ENABLE_BRANCH_FLOW", "0") == "1")
    )
    # Recover the public company name / broad industry when possible so a direct
    # diagnosis gets the same fine-industry label as a market scan.
    display_name, broad_industry = clean, "個股診斷"
    try:
        universe = fetch_twse_universe()
        code = ticker.split(".")[0]
        hit = universe[universe["ticker"].astype(str).str.startswith(code + ".")]
        if not hit.empty:
            display_name = str(hit.iloc[0].get("name") or clean)
            broad_industry = str(hit.iloc[0].get("industry") or "個股診斷")
    except Exception:
        pass
    stock = _evaluate_one(ticker, display_name, broad_industry, df, settings, market_ret20, research, market_regime, market_day_change)
    return {"snapshot_id": snap_id, "stock": stock}
