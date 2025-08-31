import pandas as pd
from engine.indicators import ema, adx, macd, bollinger, donchian_high, donchian_low, atr

def classify_regime(df: pd.DataFrame) -> str:
    c,h,l,v = df["close"], df["high"], df["low"], df["volume"]
    adx14 = adx(h,l,c,14)
    _, bb_u, bb_l, bw, _ = bollinger(c,20,2)
    d_hi, d_lo = donchian_high(h,20), donchian_low(l,20)
    ema50, ema200 = ema(c,50), ema(c,200)
    macd_line, _, _ = macd(c)

    adx_rising = (adx14.diff().tail(5) > 0).sum() >= 3
    vol_surge  = v.iloc[-1] > 1.5 * v.rolling(20).mean().iloc[-1]

    cond_range     = (adx14.iloc[-1] < 20) and (bw.iloc[-1] < bw.dropna().quantile(0.30))
    cond_trend_up  = (c.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]) and (adx14.iloc[-1] >= 20) and (macd_line.iloc[-1] > 0)
    cond_trend_dn  = (c.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]) and (adx14.iloc[-1] >= 20) and (macd_line.iloc[-1] < 0)
    cond_break_up  = (c.iloc[-1] > d_hi.iloc[-1]) and vol_surge and adx_rising
    cond_break_dn  = (c.iloc[-1] < d_lo.iloc[-1]) and vol_surge and adx_rising

    if cond_break_up: return "breakout_up"
    if cond_break_dn: return "breakout_down"
    if cond_trend_up: return "trend_up"
    if cond_trend_dn: return "trend_down"
    if cond_range:    return "range"
    return "other"

def map_regime_to_template(regime: str) -> str:
    if regime == "trend_up":      return "trend_follow"
    if regime == "range":         return "mean_revert"
    if regime == "breakout_up":   return "breakout_up"
    if regime in ("other","trend_down","breakout_down"): return "support_bounce"
    return "trend_follow"

