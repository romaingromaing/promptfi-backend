# signals/patterns.py
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Callable
from .pivots import Pivot

def _epoch_ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 10**6)

def _ols_line(x: np.ndarray, y: np.ndarray) -> Tuple[float,float,float]:
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    y_hat = m*x + b
    ss_res = ((y - y_hat)**2).sum()
    ss_tot = ((y - y.mean())**2).sum() + 1e-12
    r2 = 1 - ss_res/ss_tot
    return float(m), float(b), float(max(0.0, min(1.0, r2)))

def _atr(df: pd.DataFrame, n=14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def _vol_sma(df: pd.DataFrame, n=20) -> pd.Series:
    return df["volume"].rolling(n, min_periods=n).mean()

def _ema(s: pd.Series, n=50) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def _adx(df: pd.DataFrame, n=14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    up = h.diff(); dn = -l.diff()
    plus_dm = np.where((up>dn)&(up>0), up, 0.0)
    minus_dm = np.where((dn>up)&(dn>0), dn, 0.0)
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    pdi = 100*(pd.Series(plus_dm, index=h.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean()/(atr+1e-12))
    mdi = 100*(pd.Series(minus_dm, index=h.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean()/(atr+1e-12))
    dx = 100*(pdi.subtract(mdi).abs()/(pdi+mdi+1e-12))
    return dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def _points_from_indices(df: pd.DataFrame, idxs: List[pd.Timestamp], col="close"):
    return np.array([df.index.get_loc(i) for i in idxs]), np.array([float(df.loc[i, col]) for i in idxs])

def _last_n(seq, n): return seq[-n:] if len(seq)>=n else seq

def _mk_label(ts: pd.Timestamp, y: float, text: str):
    return {"type":"label","at":[_epoch_ms(ts), float(y)], "text": text}

@dataclass
class Detected:
    matched: bool
    card: Dict[str, Any] | None

def ascending_triangle(df: pd.DataFrame, pivots: List[Pivot], tf: str, symbol: str) -> Detected:
    atr = _atr(df, 14); vs = _vol_sma(df, 20); adx = _adx(df, 14)
    seg = df.tail(220)
    highs = [(p.idx, p.price) for p in pivots if p.kind=="HIGH" and p.idx in seg.index]
    lows  = [(p.idx, p.price) for p in pivots if p.kind=="LOW"  and p.idx in seg.index]
    if len(highs)<2 or len(lows)<2: return Detected(False, None)
    hsel = _last_n(highs, 4); hp = np.array([p for _,p in hsel])
    eps = 0.25*float(atr.iloc[-1])
    if len(hp)<2 or np.std(hp) > eps: return Detected(False, None)
    resistance = float(np.mean(hp))
    lsel = _last_n(lows, 4)
    xl = np.array([seg.index.get_loc(i) for i,_ in lsel], dtype=float)
    yl = np.array([p for _,p in lsel], dtype=float)
    if len(xl)<2: return Detected(False, None)
    slope, intercept, r2 = _ols_line(xl, yl)
    if slope <= 0 or r2 < 0.2: return Detected(False, None)
    co = float(seg["close"].iloc[-1]); vmult = float(seg["volume"].iloc[-1]/(vs.iloc[-1]+1e-9))
    entry = max(co, resistance + 0.05*atr.iloc[-1])
    sl = min(yl[-1], yl[-2]) - 1.0*atr.iloc[-1]
    height = resistance - float(min(yl[-2], yl[-1]))
    tp = entry + max(height, 1.5*(entry - sl))
    series = [{"type":"level","y": resistance, "style":{"dashed": True}}]
    x0, y0 = xl[0], yl[0]; x1, y1 = xl[-1], yl[-1]
    t0 = seg.index[int(x0)]; t1 = seg.index[int(x1)]
    series.append({"type":"line","points":[[_epoch_ms(t0), float(y0)], [_epoch_ms(t1), float(y1)]]})
    series.append(_mk_label(seg.index[-1], entry, "Entry"))
    series.append(_mk_label(seg.index[-1], sl, "SL"))
    series.append(_mk_label(seg.index[-1], tp, "TP"))
    prob = min(0.95, 0.55 + 0.1*(vmult>1.5) + 0.05*(slope>0) + 0.05*(adx.iloc[-1]>20))
    conf = min(1.0, max(0.0, (co-resistance)/(atr.iloc[-1]+1e-9)) + max(0.0, vmult-1))
    card = {
        "symbol": symbol, "tf": tf, "pattern":"ascending_triangle", "prob": float(prob),
        "entry": float(entry), "sl": float(sl), "tp": float(tp), "confidence": float(conf),
        "features": {"vol_mult_20": float(vmult), "adx14": float(adx.iloc[-1])},
        "overlays": {"version":1, "series": series}
    }
    return Detected(True, card)

def descending_triangle(df, pivots, tf, symbol):
    atr = _atr(df,14); vs=_vol_sma(df,20); adx=_adx(df,14)
    seg = df.tail(220)
    lows = [(p.idx,p.price) for p in pivots if p.kind=="LOW" and p.idx in seg.index]
    highs = [(p.idx,p.price) for p in pivots if p.kind=="HIGH" and p.idx in seg.index]
    if len(lows)<2 or len(highs)<2: return Detected(False,None)
    lsel = _last_n(lows,4); lp = np.array([p for _,p in lsel])
    eps = 0.25*float(atr.iloc[-1])
    if len(lp)<2 or np.std(lp) > eps: return Detected(False,None)
    support = float(np.mean(lp))
    hsel = _last_n(highs,4)
    xh = np.array([seg.index.get_loc(i) for i,_ in hsel], dtype=float)
    yh = np.array([p for _,p in hsel], dtype=float)
    slope, b, r2 = _ols_line(xh,yh)
    if slope >= 0 or r2 < 0.2: return Detected(False,None)
    co = float(seg["close"].iloc[-1]); vmult=float(seg["volume"].iloc[-1]/(vs.iloc[-1]+1e-9))
    entry = min(co, support - 0.05*atr.iloc[-1])
    sl = max(yh[-1], yh[-2]) + 1.0*atr.iloc[-1]
    height = float(max(yh[-2],yh[-1])) - support
    tp = entry - max(height, 1.5*(sl-entry))
    series=[{"type":"level","y":support,"style":{"dashed":True}},
            {"type":"line","points":[[_epoch_ms(seg.index[int(xh[0])]),float(yh[0])],[_epoch_ms(seg.index[int(xh[-1])]),float(yh[-1])]]},
            _mk_label(seg.index[-1], entry, "Entry"), _mk_label(seg.index[-1], sl, "SL"), _mk_label(seg.index[-1], tp, "TP")]
    prob = min(0.95, 0.55 + 0.1*(vmult>1.5) + 0.05*(slope<0) + 0.05*(adx.iloc[-1]>20))
    conf = min(1.0, max(0.0, (support-co)/(atr.iloc[-1]+1e-9)) + max(0.0, vmult-1))
    return Detected(True, {"symbol":symbol,"tf":tf,"pattern":"descending_triangle","prob":float(prob),
                           "entry":float(entry),"sl":float(sl),"tp":float(tp),"confidence":float(conf),
                           "features":{"vol_mult_20":float(vmult),"adx14":float(adx.iloc[-1])},
                           "overlays":{"version":1,"series":series}})

def symmetrical_triangle(df, pivots, tf, symbol):
    seg = df.tail(220); atr=_atr(df,14); vs=_vol_sma(df,20); adx=_adx(df,14)
    highs=[(p.idx,p.price) for p in pivots if p.kind=="HIGH" and p.idx in seg.index]
    lows=[(p.idx,p.price) for p in pivots if p.kind=="LOW" and p.idx in seg.index]
    if len(highs)<2 or len(lows)<2: return Detected(False,None)
    hsel=_last_n(highs,4); lsel=_last_n(lows,4)
    xh,yh=_points_from_indices(seg,[i for i,_ in hsel]); xl,yl=_points_from_indices(seg,[i for i,_ in lsel])
    m1,b1,r1=_ols_line(xh,yh); m2,b2,r2=_ols_line(xl,yl)
    if not (m1<0 and m2>0): return Detected(False,None)
    if min(r1,r2) < 0.2: return Detected(False,None)
    x_int = int((b2-b1)/(m1-m2)) if (m1-m2)!=0 else int(xh[-1])
    x_last = seg.index.get_loc(seg.index[-1])
    if x_int < x_last-20: return Detected(False,None)
    res_now = m1*x_last + b1; sup_now = m2*x_last + b2
    co=float(seg["close"].iloc[-1]); a=atr.iloc[-1]; vmult=float(seg["volume"].iloc[-1]/(vs.iloc[-1]+1e-9))
    long = (res_now - co) < (co - sup_now)
    if long:
        entry = max(co, res_now + 0.05*a); sl = sup_now - 1.0*a; tp = entry + max((res_now - sup_now), 1.5*(entry-sl))
    else:
        entry = min(co, sup_now - 0.05*a); sl = res_now + 1.0*a; tp = entry - max((res_now - sup_now), 1.5*(sl-entry))
    series=[{"type":"line","points":[[_epoch_ms(seg.index[int(xh[0])]),float(m1*xh[0]+b1)],[_epoch_ms(seg.index[int(xh[-1])]),float(m1*xh[-1]+b1)]]},
            {"type":"line","points":[[_epoch_ms(seg.index[int(xl[0])]),float(m2*xl[0]+b2)],[_epoch_ms(seg.index[int(xl[-1])]),float(m2*xl[-1]+b2)]]},
            _mk_label(seg.index[-1], entry, "Entry"), _mk_label(seg.index[-1], sl, "SL"), _mk_label(seg.index[-1], tp, "TP")]
    prob = min(0.9, 0.5 + 0.1*min(r1,r2) + 0.1*(vmult>1.3) + 0.05*(adx.iloc[-1]>20))
    return Detected(True, {"symbol":symbol,"tf":tf,"pattern":"sym_triangle","prob":float(prob),
                           "entry":float(entry),"sl":float(sl),"tp":float(tp),"confidence":float(min(1.0, vmult)),
                           "features":{"vmult":float(vmult),"r1":float(r1),"r2":float(r2)},
                           "overlays":{"version":1,"series":series}})

def _flag_core(df, pivots, tf, symbol, bull=True):
    seg=df.tail(220); a=_atr(df,14); vs=_vol_sma(df,20)
    c=seg["close"]; ret = c.iloc[-1]/c.shift(20).iloc[-1]-1 if len(c)>=21 else 0
    if (bull and ret<0.08) or ((not bull) and ret>-0.08): return Detected(False,None)
    M=30; last=seg.tail(M)
    x=np.arange(len(last)); y=last["close"].values
    m,b,r=_ols_line(x,y)
    if bull and m >= 0: return Detected(False,None)
    if (not bull) and m <= 0: return Detected(False,None)
    co=float(last["close"].iloc[-1]); vmult=float(last["volume"].iloc[-1]/(vs.iloc[-1]+1e-9))
    top = (m*len(last)+b) + 0.5*a.iloc[-1]; bot = (m*len(last)+b) - 0.5*a.iloc[-1]
    if bull:
        entry = max(co, top + 0.05*a.iloc[-1]); sl = bot - 1.0*a.iloc[-1]; tp = entry + max(abs(ret*last["close"].iloc[-21]), 1.5*(entry-sl))
        patt="bull_flag"
    else:
        entry = min(co, bot - 0.05*a.iloc[-1]); sl = top + 1.0*a.iloc[-1]; tp = entry - max(abs(ret*last["close"].iloc[-21]), 1.5*(sl-entry))
        patt="bear_flag"
    series=[{"type":"line","points":[[_epoch_ms(last.index[0]), float(m*0+b)],[_epoch_ms(last.index[-1]), float(m*len(last)+b)]], "style":{"dashed":True}},
            _mk_label(last.index[-1], entry, "Entry"), _mk_label(last.index[-1], sl, "SL"), _mk_label(last.index[-1], tp, "TP")]
    prob = min(0.9, 0.55 + 0.15*abs(ret) + 0.1*(vmult>1.2))
    return Detected(True, {"symbol":symbol,"tf":tf,"pattern":patt,"prob":float(prob),
                           "entry":float(entry),"sl":float(sl),"tp":float(tp),
                           "confidence":float(min(1.0, abs(ret)+max(0.0,vmult-1))),
                           "features":{"ret20":float(ret),"vmult":float(vmult)},
                           "overlays":{"version":1,"series":series}})

def bull_flag(df,pivots,tf,symbol): return _flag_core(df,pivots,tf,symbol,bull=True)
def bear_flag(df,pivots,tf,symbol): return _flag_core(df,pivots,tf,symbol,bull=False)

def _double_core(df,pivots,tf,symbol,top=True):
    seg=df.tail(240); a=_atr(df,14); vs=_vol_sma(df,20)
    highs=[(p.idx,p.price) for p in pivots if p.kind=="HIGH" and p.idx in seg.index]
    lows=[(p.idx,p.price) for p in pivots if p.kind=="LOW" and p.idx in seg.index]
    if top:
        pts=_last_n(highs,4); 
        if len(pts)<2: return Detected(False,None)
    else:
        pts=_last_n(lows,4); 
        if len(pts)<2: return Detected(False,None)
    p1, p2 = pts[-2], pts[-1]
    eps = 0.3*float(a.iloc[-1])
    if abs(p2[1]-p1[1]) > eps: return Detected(False,None)
    if top:
        mids = [q for q in lows if p1[0]<q[0]<p2[0]]
        if not mids: return Detected(False,None)
        neck = float(min(mids, key=lambda x:x[1])[1])
        co=float(seg["close"].iloc[-1]); vmult=float(seg["volume"].iloc[-1]/(vs.iloc[-1]+1e-9))
        entry = min(co, neck - 0.05*a.iloc[-1]); sl = max(p1[1],p2[1]) + 0.8*a.iloc[-1]
        tp = entry - abs(max(p1[1],p2[1]) - neck)
        patt="double_top"
    else:
        mids = [q for q in highs if p1[0]<q[0]<p2[0]]
        if not mids: return Detected(False,None)
        neck = float(max(mids, key=lambda x:x[1])[1])
        co=float(seg["close"].iloc[-1]); vmult=float(seg["volume"].iloc[-1]/(vs.iloc[-1]+1e-9))
        entry = max(co, neck + 0.05*a.iloc[-1]); sl = min(p1[1],p2[1]) - 0.8*a.iloc[-1]
        tp = entry + abs(neck - min(p1[1],p2[1]))
        patt="double_bottom"
    series=[{"type":"level","y":neck,"style":{"dashed":True}},
            _mk_label(seg.index[-1], entry,"Entry"), _mk_label(seg.index[-1], sl,"SL"), _mk_label(seg.index[-1], tp,"TP")]
    prob=min(0.9, 0.55 + 0.1*(vmult>1.3))
    return Detected(True, {"symbol":symbol,"tf":tf,"pattern":patt,"prob":float(prob),
                           "entry":float(entry),"sl":float(sl),"tp":float(tp),
                           "confidence":float(min(1.0, max(0.0,vmult-1)+abs(p2[1]-p1[1])/(a.iloc[-1]+1e-9))),
                           "features":{"neck":float(neck),"vmult":float(vmult)},
                           "overlays":{"version":1,"series":series}})

def double_top(df,pivots,tf,symbol): return _double_core(df,pivots,tf,symbol,top=True)
def double_bottom(df,pivots,tf,symbol): return _double_core(df,pivots,tf,symbol,top=False)

def _hs_core(df,pivots,tf,symbol,inverse=False):
    seg=df.tail(300); a=_atr(df,14); vs=_vol_sma(df,20)
    highs=[(p.idx,p.price) for p in pivots if p.kind=="HIGH" and p.idx in seg.index]
    lows=[(p.idx,p.price) for p in pivots if p.kind=="LOW" and p.idx in seg.index]
    if inverse:
        lows_sorted=sorted(lows, key=lambda x:x[0])
        if len(lows_sorted)<3: return Detected(False,None)
        l1,l2,l3=lows_sorted[-3:]
        if not (l2[1] < l1[1] and l2[1] < l3[1]): return Detected(False,None)
        mids=[q for q in highs if l1[0]<q[0]<l3[0]]
        if not mids: return Detected(False,None)
        m2=sorted(mids, key=lambda x:x[0])[-2:]
        xh,yh=_points_from_indices(seg,[i for i,_ in m2]); m,b,r=_ols_line(xh,yh)
        x_last = seg.index.get_loc(seg.index[-1]); neck = m*x_last + b
        co=float(seg["close"].iloc[-1]); vmult=float(seg["volume"].iloc[-1]/(vs.iloc[-1]+1e-9))
        entry=max(co, neck + 0.05*a.iloc[-1]); sl = min(l1[1],l3[1]) - 0.8*a.iloc[-1]
        tp = entry + abs(l2[1] - neck)
        patt="inverse_head_shoulders"
        series=[{"type":"line","points":[[_epoch_ms(seg.index[int(xh[0])]),float(m*xh[0]+b)],[_epoch_ms(seg.index[int(xh[-1])]),float(m*xh[-1]+b)]]},
                _mk_label(seg.index[-1],entry,"Entry"),_mk_label(seg.index[-1],sl,"SL"),_mk_label(seg.index[-1],tp,"TP")]
        prob=min(0.9, 0.6 + 0.1*(r>0.2) + 0.1*(vmult>1.3))
        return Detected(True, {"symbol":symbol,"tf":tf,"pattern":patt,"prob":float(prob),
                               "entry":float(entry),"sl":float(sl),"tp":float(tp),"confidence":float(min(1.0, vmult)),
                               "features":{"neckline_r2":float(r),"vmult":float(vmult)},
                               "overlays":{"version":1,"series":series}})
    else:
        highs_sorted=sorted(highs, key=lambda x:x[0])
        if len(highs_sorted)<3: return Detected(False,None)
        h1,h2,h3=highs_sorted[-3:]
        if not (h2[1] > h1[1] and h2[1] > h3[1]): return Detected(False,None)
        mids=[q for q in lows if h1[0]<q[0]<h3[0]]
        if not mids: return Detected(False,None)
        m2=sorted(mids, key=lambda x:x[0])[-2:]
        xl,yl=_points_from_indices(seg,[i for i,_ in m2])
        m,b,r=_ols_line(xl,yl)
        x_last = seg.index.get_loc(seg.index[-1]); neck = m*x_last + b
        co=float(seg["close"].iloc[-1]); vmult=float(seg["volume"].iloc[-1]/(vs.iloc[-1]+1e-9))
        entry=min(co, neck - 0.05*a.iloc[-1]); sl = max(h1[1],h3[1]) + 0.8*a.iloc[-1]
        tp = entry - abs(h2[1] - neck)
        patt="head_shoulders"
        series=[{"type":"line","points":[[_epoch_ms(seg.index[int(xl[0])]),float(m*xl[0]+b)],[_epoch_ms(seg.index[int(xl[-1])]),float(m*xl[-1]+b)]]},
                _mk_label(seg.index[-1],entry,"Entry"),_mk_label(seg.index[-1],sl,"SL"),_mk_label(seg.index[-1],tp,"TP")]
        prob=min(0.9, 0.6 + 0.1*(r>0.2) + 0.1*(vmult>1.3))
        return Detected(True, {"symbol":symbol,"tf":tf,"pattern":patt,"prob":float(prob),
                               "entry":float(entry),"sl":float(sl),"tp":float(tp),"confidence":float(min(1.0, vmult)),
                               "features":{"neckline_r2":float(r),"vmult":float(vmult)},
                               "overlays":{"version":1,"series":series}})

def head_shoulders(df,pivots,tf,symbol): return _hs_core(df,pivots,tf,symbol,inverse=False)
def inverse_head_shoulders(df,pivots,tf,symbol): return _hs_core(df,pivots,tf,symbol,inverse=True)

def _wedge_core(df,pivots,tf,symbol,rising=True):
    seg=df.tail(220); a=_atr(df,14)
    highs=[(p.idx,p.price) for p in pivots if p.kind=="HIGH" and p.idx in seg.index]
    lows=[(p.idx,p.price) for p in pivots if p.kind=="LOW" and p.idx in seg.index]
    if len(highs)<2 or len(lows)<2: return Detected(False,None)
    xh,yh=_points_from_indices(seg,[i for i,_ in _last_n(highs,4)])
    xl,yl=_points_from_indices(seg,[i for i,_ in _last_n(lows,4)])
    m1,b1,r1=_ols_line(xh,yh); m2,b2,r2=_ols_line(xl,yl)
    if rising and not (m1>0 and m2>0 and m2>m1): return Detected(False,None)
    if (not rising) and not (m1<0 and m2<0 and m2<m1): return Detected(False,None)
    x_last = seg.index.get_loc(seg.index[-1])
    top = m1*x_last + b1; bot = m2*x_last + b2
    co=float(seg["close"].iloc[-1])
    if rising:
        entry = min(co, bot - 0.05*a.iloc[-1]); sl = top + 0.8*a.iloc[-1]; tp = entry - max((top-bot), 1.2*(sl-entry))
        patt="wedge_rising"
    else:
        entry = max(co, top + 0.05*a.iloc[-1]); sl = bot - 0.8*a.iloc[-1]; tp = entry + max((top-bot), 1.2*(entry-sl))
        patt="wedge_falling"
    series=[{"type":"line","points":[[_epoch_ms(seg.index[int(xh[0])]),float(m1*xh[0]+b1)],[_epoch_ms(seg.index[int(xh[-1])]),float(m1*xh[-1]+b1)]]},
            {"type":"line","points":[[_epoch_ms(seg.index[int(xl[0])]),float(m2*xl[0]+b2)],[_epoch_ms(seg.index[int(xl[-1])]),float(m2*xl[-1]+b2)]]},
            _mk_label(seg.index[-1],entry,"Entry"),_mk_label(seg.index[-1],sl,"SL"),_mk_label(seg.index[-1],tp,"TP")]
    prob=min(0.85, 0.55 + 0.15*min(r1,r2))
    return Detected(True, {"symbol":symbol,"tf":tf,"pattern":patt,"prob":float(prob),
                           "entry":float(entry),"sl":float(sl),"tp":float(tp),"confidence":0.6,
                           "features":{"r1":float(r1),"r2":float(r2)},
                           "overlays":{"version":1,"series":series}})

def wedge_rising(df,pivots,tf,symbol): return _wedge_core(df,pivots,tf,symbol,rising=True)
def wedge_falling(df,pivots,tf,symbol): return _wedge_core(df,pivots,tf,symbol,rising=False)

# ---- micro candlesticks ----

def engulfing_bull(df,pivots,tf,symbol):
    seg=df.tail(3)
    o1,c1 = seg["open"].iloc[-2], seg["close"].iloc[-2]
    o2,c2 = seg["open"].iloc[-1], seg["close"].iloc[-1]
    match = (c1 < o1) and (c2 > o2) and (o2 <= c1) and (c2 >= o1)
    if not match: return Detected(False,None)
    entry=float(max(c2,o2)); sl=float(min(o1,c1)); tp=entry+2*(entry-sl)
    series=[_mk_label(seg.index[-1], entry,"Bull Engulf")]
    return Detected(True, {"symbol":symbol,"tf":tf,"pattern":"engulfing_bull","prob":0.55,
                           "entry":entry,"sl":sl,"tp":tp,"confidence":0.6,
                           "features":{},"overlays":{"version":1,"series":series}})

def engulfing_bear(df,pivots,tf,symbol):
    seg=df.tail(3)
    o1,c1 = seg["open"].iloc[-2], seg["close"].iloc[-2]
    o2,c2 = seg["open"].iloc[-1], seg["close"].iloc[-1]
    match = (c1 > o1) and (c2 < o2) and (o2 >= c1) and (c2 <= o1)
    if not match: return Detected(False,None)
    entry=float(min(c2,o2)); sl=float(max(o1,c1)); tp=entry-2*(sl-entry)
    series=[_mk_label(seg.index[-1], entry,"Bear Engulf")]
    return Detected(True, {"symbol":symbol,"tf":tf,"pattern":"engulfing_bear","prob":0.55,
                           "entry":entry,"sl":sl,"tp":tp,"confidence":0.6,
                           "features":{},"overlays":{"version":1,"series":series}})

def hammer(df,pivots,tf,symbol):
    seg=df.tail(1); row=seg.iloc[-1]
    body=abs(row["close"]-row["open"]); range_=row["high"]-row["low"]; lower= min(row["open"],row["close"]) - row["low"]
    match = (lower >= 2*body) and (body/range_ <= 0.4)
    if not match: return Detected(False,None)
    entry=float(row["high"]); sl=float(row["low"]); tp=entry+2*(entry-sl)
    return Detected(True, {"symbol":symbol,"tf":tf,"pattern":"hammer","prob":0.52,"entry":entry,"sl":sl,"tp":tp,"confidence":0.5,
                           "features":{},"overlays":{"version":1,"series":[_mk_label(seg.index[-1], entry,"Hammer")]}})

def shooting_star(df,pivots,tf,symbol):
    seg=df.tail(1); row=seg.iloc[-1]
    body=abs(row["close"]-row["open"]); range_=row["high"]-row["low"]; upper=row["high"]-max(row["open"],row["close"])
    match = (upper >= 2*body) and (body/range_ <= 0.4)
    if not match: return Detected(False,None)
    entry=float(row["low"]); sl=float(row["high"]); tp=entry-2*(sl-entry)
    return Detected(True, {"symbol":symbol,"tf":tf,"pattern":"shooting_star","prob":0.52,"entry":entry,"sl":sl,"tp":tp,"confidence":0.5,
                           "features":{},"overlays":{"version":1,"series":[_mk_label(seg.index[-1], entry,"Shooting Star")]}})

def doji(df,pivots,tf,symbol):
    seg=df.tail(1); row=seg.iloc[-1]
    body=abs(row["close"]-row["open"]); range_=(row["high"]-row["low"])+1e-9
    if (body/range_) > 0.1: return Detected(False,None)
    return Detected(True, {"symbol":symbol,"tf":tf,"pattern":"doji","prob":0.5,"entry":float(row['close']),"sl":float(row['low']),"tp":float(row['high']),
                           "confidence":0.4,"features":{},"overlays":{"version":1,"series":[_mk_label(seg.index[-1], float(row['close']), "Doji")]}})
