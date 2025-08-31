# import pandas as pd
# import numpy as np

# def SMA(s: pd.Series, n: int) -> pd.Series:
#     return s.rolling(n, min_periods=n).mean()

# def EMA(s: pd.Series, n: int) -> pd.Series:
#     return s.ewm(span=n, adjust=False, min_periods=n).mean()

# def RSI(s: pd.Series, n: int=14) -> pd.Series:
#     delta = s.diff()
#     up = delta.clip(lower=0)
#     down = -delta.clip(upper=0)
#     ma_up = up.ewm(alpha=1/n, min_periods=n).mean()
#     ma_down = down.ewm(alpha=1/n, min_periods=n).mean()
#     rs = ma_up / (ma_down + 1e-12)
#     rsi = 100 - (100/(1+rs))
#     return rsi

# def VOL_SMA(v: pd.Series, n: int) -> pd.Series:
#     return v.rolling(n, min_periods=n).mean()

# def RET_60D(c: pd.Series) -> pd.Series:
#     return c.pct_change(60).fillna(0.0)

# indicators.py
import numpy as np
import pandas as pd

EPS = 1e-9

# ---------- Basics ----------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    return pd.concat([(high-low), (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = au / (ad + EPS)
    return 100 - 100/(1+rs)

def stoch_rsi(close: pd.Series, n: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    r = rsi(close, n)
    ll = r.rolling(n, min_periods=n).min()
    hh = r.rolling(n, min_periods=n).max()
    stoch = (r - ll) / (hh - ll + EPS)
    k = stoch.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(smooth_d, min_periods=smooth_d).mean()
    return k, d

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, signal)
    return m, s, m - s

# ---------- DI/ADX ----------
def di_plus_minus(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(high, low, close)
    atr_n = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=high.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean() / (atr_n + EPS))
    minus_di = 100 * (pd.Series(minus_dm, index=low.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean() / (atr_n + EPS))
    return plus_di, minus_di

def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    pdi, mdi = di_plus_minus(high, low, close, n)
    dx = 100 * (pdi.subtract(mdi).abs() / (pdi + mdi + EPS))
    return dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

# ---------- Channels & Bands ----------
def donchian_high(high: pd.Series, n: int = 20) -> pd.Series:
    return high.rolling(n, min_periods=n).max()

def donchian_low(low: pd.Series, n: int = 20) -> pd.Series:
    return low.rolling(n, min_periods=n).min()

def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    m = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    upper, lower = m + k*sd, m - k*sd
    bw = (upper - lower) / (m.abs() + EPS)          # bandwidth
    pct_b = (close - lower) / (upper - lower + EPS) # %B
    return m, upper, lower, bw, pct_b

def keltner(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20, m: float = 1.5):
    mid = ema(close, n)
    atr_n = atr(high, low, close, n)
    upper, lower = mid + m*atr_n, mid - m*atr_n
    return mid, upper, lower

def squeeze_bb_kc(close, high, low, n_bb=20, k=2.0, n_kc=20, m=1.5):
    _, bb_u, bb_l, _, _ = bollinger(close, n_bb, k)
    kc_m, kc_u, kc_l = keltner(high, low, close, n_kc, m)
    # Squeeze on when BB inside KC
    squeeze_on = (bb_u < kc_u) & (bb_l > kc_l)
    return squeeze_on

# ---------- Volume & Flow ----------
def vwap(close: pd.Series, volume: pd.Series) -> pd.Series:
    tp = close
    cum_pv = (tp * volume).cumsum()
    cum_v = volume.cumsum().replace(0, np.nan)
    return cum_pv / (cum_v + EPS)

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (volume * direction).cumsum()

def cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 20) -> pd.Series:
    mf_mult = ((close - low) - (high - close)) / (high - low + EPS)
    mf_vol = mf_mult * volume
    return mf_vol.rolling(n, min_periods=n).sum() / (volume.rolling(n, min_periods=n).sum() + EPS)

def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    mf = tp * volume
    pos = mf.where(tp.diff() >= 0, 0.0)
    neg = mf.where(tp.diff() < 0, 0.0)
    pos_sum = pos.rolling(n, min_periods=n).sum()
    neg_sum = neg.rolling(n, min_periods=n).sum()
    mr = pos_sum / (neg_sum + EPS)
    return 100 - (100 / (1 + mr))

# ---------- Support / Resistance ----------
def support_levels(high: pd.Series, low: pd.Series, close: pd.Series,
                   lookback: int = 180, k: int = 5, atr_mult: float = 0.75) -> list[float]:
    seg = close.tail(lookback)
    # ATR estimate
    try:
        atr_est = atr(high, low, close, 14).iloc[-1]
    except Exception:
        atr_est = close.diff().abs().rolling(14).mean().iloc[-1]
    piv = seg.rolling(k, center=True).apply(lambda w: float(w.iloc[k//2] == w.min()), raw=False)
    lvls = seg[piv == 1.0]
    clusters = []
    for p in lvls:
        placed = False
        for c in clusters:
            if abs(p - c["price"]) <= atr_mult * atr_est:
                c["hits"] += 1; c["sum"] += p; placed = True; break
        if not placed: clusters.append({"price": p, "hits": 1, "sum": p})
    for c in clusters: c["price"] = c["sum"]/c["hits"]
    clusters.sort(key=lambda x: x["hits"], reverse=True)
    return [c["price"] for c in clusters[:3]]

def volume_profile_nodes(close: pd.Series, volume: pd.Series, bins: int = 40, topk: int = 3) -> list[float]:
    # Approx HVNs: accumulate volume by price buckets
    if len(close) != len(volume): return []
    mn, mx = float(close.min()), float(close.max())
    if mx <= mn: return []
    edges = np.linspace(mn, mx, bins+1)
    idx = np.clip(np.digitize(close.values, edges)-1, 0, bins-1)
    vol_bins = np.zeros(bins)
    for i, vol in zip(idx, volume.values):
        vol_bins[i] += vol
    centers = (edges[:-1] + edges[1:]) / 2.0
    order = np.argsort(vol_bins)[::-1][:topk]
    return [float(centers[i]) for i in order]

# ---------- Divergences ----------
def rsi_divergence(close: pd.Series, n: int = 100, swing: int = 5) -> str:
    # very simple heuristic: last two swing highs/lows in price vs RSI
    r = rsi(close, 14)
    seg_c = close.tail(n); seg_r = r.tail(n)
    highs = seg_c.rolling(swing, center=True).apply(lambda w: float(w.iloc[swing//2] == w.max()), raw=False)
    lows  = seg_c.rolling(swing, center=True).apply(lambda w: float(w.iloc[swing//2] == w.min()), raw=False)
    ph = list(seg_c[highs==1.0].tail(2).values)
    pr = list(seg_r[highs==1.0].tail(2).values)
    pl = list(seg_c[lows==1.0].tail(2).values)
    rl = list(seg_r[lows==1.0].tail(2).values)
    if len(ph)==2 and len(pr)==2 and ph[-1]>ph[-2] and pr[-1]<pr[-2]:
        return "bearish_divergence"
    if len(pl)==2 and len(rl)==2 and pl[-1]<pl[-2] and rl[-1]>rl[-2]:
        return "bullish_divergence"
    return "none"
