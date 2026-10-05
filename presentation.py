"""Plain-language presentation helpers for Alpha Radar V13.1."""
from __future__ import annotations


def _f(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def _signed_pct(value, digits=1):
    x = _f(value)
    return "資料不足" if x is None else f"{x:+.{digits}f}%"


def _state_action(state: str) -> tuple[str, str]:
    table = {
        "CONDITIONS_MET_NOT_FILLED": ("偏多｜可優先觀察", "目前價格已接近模型認為較合理的觀察區，但仍要看下一個交易日是否守穩。"),
        "WAIT_ENTRY_ZONE": ("偏多｜等便宜一點", "公司與走勢條件可以看，但現在的價格不是最舒服的位置，等拉回會比較有餘裕。"),
        "WAIT_BREAKOUT": ("觀察｜等走勢更明確", "目前還差一個明確的向上確認，現在先看，不必急著追。"),
        "WAIT_CONFIRMATION": ("觀察｜再等一下", "目前方向不差，但買盤與價格還沒有一起確認，先等訊號完整。"),
        "DO_NOT_CHASE": ("偏多但過熱｜不要追高", "股票本身可能仍強，但價格已離理想買點偏遠，現在追價的風險較高。"),
        "INVALIDATED": ("轉弱｜暫緩", "近期價格已跌破模型的防守結構，先不要急著進場。"),
        "DATA_UNVERIFIED": ("資料待確認", "行情或研究資料還沒有確認完整，暫時不做方向判斷。"),
        "NO_RETURN_ESTIMATE": ("資料不足｜先觀察", "可比的歷史案例太少，暫時不把報酬估計當作主要依據。"),
    }
    return table.get(state, ("觀察", "目前條件沒有明顯偏向，先觀察價格與公司數據是否改善。"))


def investment_view(stock: dict, horizon: str, market: dict | None = None) -> dict:
    """Build a human-readable investment view without exposing model jargon.

    This is a presentation layer over the model outputs; it does not change ranking.
    """
    market = market or {}
    block = (stock.get("horizons") or {}).get(horizon, {}) or {}
    forecast = block.get("forecast") or {}
    strategy = forecast.get("strategy") or {}
    research = stock.get("research") or {}
    rev = research.get("monthly_revenue") or {}
    fin = research.get("financials") or {}
    inst = research.get("institutional_flow") or {}

    title, action = _state_action(str(block.get("entry_state") or ""))
    rank = _f(block.get("ranking_score"), 0.0) or 0.0
    median = _f(strategy.get("median"))
    win_rate = _f(strategy.get("historical_positive_rate"))
    alpha = _f(forecast.get("alpha_mean"))

    reasons: list[str] = []
    cautions: list[str] = []

    # Relative market strength in plain language.
    if alpha is not None:
        if alpha >= 0.05:
            reasons.append(f"近20個交易日明顯比大盤強，約多出 {alpha*100:.1f} 個百分點。")
        elif alpha >= 0.01:
            reasons.append(f"近20個交易日比大盤略強，約多出 {alpha*100:.1f} 個百分點。")
        elif alpha <= -0.05:
            cautions.append(f"近20個交易日明顯落後大盤，約少 {abs(alpha)*100:.1f} 個百分點。")
        elif alpha < -0.01:
            cautions.append(f"近20個交易日稍弱於大盤，約少 {abs(alpha)*100:.1f} 個百分點。")

    # Historical analog outcome.
    if median is not None and win_rate is not None:
        if median > 0.03 and win_rate >= 0.60:
            reasons.append(f"過去類似走勢中，上漲比例約 {win_rate*100:.0f}%，中間結果約 {median*100:+.1f}%。")
        elif median > 0 and win_rate >= 0.52:
            reasons.append(f"過去類似走勢略偏正向，上漲比例約 {win_rate*100:.0f}%。")
        elif median < 0 or win_rate < 0.48:
            cautions.append("過去類似走勢的結果沒有明顯優勢，這次更需要等好價格。")

    # Revenue momentum.
    latest_rev = rev.get("latest") or {}
    yoy = _f(latest_rev.get("yoy_pct"))
    mom = _f(latest_rev.get("mom_pct"))
    avg3 = _f(rev.get("avg_yoy_3m_pct"))
    if yoy is not None:
        if yoy >= 15 and (avg3 is None or avg3 >= 5):
            reasons.append(f"最新月營收年增 {yoy:+.1f}%，近幾個月營收動能偏強。")
        elif yoy >= 5:
            reasons.append(f"最新月營收年增 {yoy:+.1f}%，營收仍在成長。")
        elif yoy <= -10:
            cautions.append(f"最新月營收年減 {abs(yoy):.1f}%，基本面需要再觀察。")
        elif yoy < 0:
            cautions.append(f"最新月營收年增率為 {yoy:.1f}%，目前還沒有回到成長。")
    if mom is not None and mom >= 15 and len(reasons) < 4:
        reasons.append(f"最新單月營收比上月增加 {mom:.1f}%，短期營收有升溫。")

    # Earnings and margin.
    qs = fin.get("quarters") or []
    eps = [_f(q.get("eps")) for q in qs]
    eps = [x for x in eps if x is not None]
    if len(eps) >= 2:
        if eps[-1] > 0 and eps[-1] > eps[-2] * 1.10:
            reasons.append("最新一季 EPS 比前一季明顯增加，獲利方向有改善。")
        elif eps[-1] < 0:
            cautions.append("最新一季 EPS 為負，獲利仍是主要風險。")
        elif eps[-1] < eps[-2] * 0.80:
            cautions.append("最新一季 EPS 比前一季明顯下降，需留意獲利降溫。")
    gm_latest = _f(fin.get("gross_margin_latest_pct"))
    gm_avg = _f(fin.get("gross_margin_4q_avg_pct"))
    if gm_latest is not None and gm_avg is not None:
        if gm_latest >= gm_avg + 1.5:
            reasons.append("最新毛利率高於近四季平均，產品組合或成本表現有改善。")
        elif gm_latest <= gm_avg - 2.0:
            cautions.append("最新毛利率低於近四季平均，需留意獲利品質。")

    # Institutional flow.
    net_lots = _f(inst.get("total_net_lots")) if inst.get("available") else None
    if net_lots is not None:
        if net_lots >= 3000:
            reasons.append(f"近一月三大法人合計買超約 {net_lots:,.0f} 張，資金面偏正向。")
        elif net_lots <= -3000:
            cautions.append(f"近一月三大法人合計賣超約 {abs(net_lots):,.0f} 張，資金面偏保守。")

    # Overall relative rank, expressed without score jargon.
    if rank >= 78 and len(reasons) < 4:
        reasons.insert(0, "在目前掃描到的股票中，整體條件位在前段。")
    elif rank < 55:
        cautions.insert(0, "目前整體條件不算突出，不適合因為單一亮點就追進。")

    regime = market.get("regime")
    if regime == "BEAR" and horizon == "short":
        cautions.append("大盤目前偏弱，短線部位宜更保守。")
    elif regime == "BULL" and horizon in {"short", "mid"} and len(reasons) < 4:
        reasons.append("大盤目前偏多，個股順勢條件相對有利。")

    # Keep the card easy to scan.
    reasons = reasons[:3]
    cautions = cautions[:2]
    if not reasons:
        reasons.append("目前沒有單一數據足以形成明顯優勢，主要看價格是否進入更好的位置。")

    fine = stock.get("fine_industry") or stock.get("industry") or ""
    summary = action
    if fine:
        summary = f"{fine}。{action}"

    return {
        "title": title,
        "summary": summary,
        "reasons": reasons,
        "cautions": cautions,
    }


def plain_summary(forecast: dict) -> str:
    """Legacy compact summary retained for compatibility; intentionally jargon-light."""
    if not forecast:
        return "目前沒有足夠資料形成明確結論。"
    if not forecast.get("estimate_available"):
        return "價格走勢可以分析，但可比的歷史案例還不夠多，先以觀察為主。"
    s = forecast.get("strategy") or {}
    median = _f(s.get("median"), 0.0) or 0.0
    positive = _f(s.get("historical_positive_rate"), 0.0) or 0.0
    if median > 0.03 and positive >= 0.60:
        return f"過去類似走勢偏正向，上漲比例約 {positive*100:.0f}%。"
    if median > 0:
        return f"過去類似走勢略偏正向，上漲比例約 {positive*100:.0f}%。"
    return "過去類似走勢沒有明顯優勢，較適合等更好的價格。"
