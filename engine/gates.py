import re
import pandas as pd

def evaluate_gate(expr: str, df: pd.DataFrame, sent: pd.Series) -> pd.Series:
    """
    expr examples:
      "CLOSE > SMA(30)"
      "VOLUME > 1.2*SMA(30,VOLUME)"
      "RSI(14) < 70"
      "SENTIMENT >= 0.30"
    df columns: close, volume, SMA_30, VOL_SMA_30, EMA_50, RSI_14, RET_60D
    sent: Series (same index) or None
    """
    E = expr.upper().strip()

    # Replace tokens with df[...] lookups
    def col(tok):
        if tok == "CLOSE": return "close"
        if tok == "VOLUME": return "volume"
        return tok

    # Derived indicators mapping by regex
    E = re.sub(r"SMA\((\d+),VOLUME\)", r"VOL_SMA_\1", E)
    E = re.sub(r"SMA\((\d+)\)", r"SMA_\1", E)
    E = re.sub(r"EMA\((\d+)\)", r"EMA_\1", E)
    E = E.replace("RSI(14)", "RSI_14")
    E = E.replace("RET_60D", "RET_60D")

    # Map sentiment words are already numeric at planner, but keep safety:
    E = E.replace("GOOD", "0.30").replace("BAD", "-0.30")

    # Tokenize simple patterns like: LHS (op) RHS, with optional scalar*col
    # We support forms: COL op COL; COL op k*COL; COL op number; SENTIMENT op number
    m = re.match(r"(.+?)\s*(>=|<=|>|<|==)\s*(.+)", E)
    if not m:
        # allow bare expressions like 'SMA(30)' or 'SMA_30' or 'SENTIMENT'
        try:
            S = None
            def series_of_simple(term: str) -> pd.Series:
                term = term.strip()
                if term == "SENTIMENT":
                    if sent is None:
                        return pd.Series(index=df.index, data=0.0)
                    return sent.reindex(df.index).ffill().fillna(0.0)
                mm = re.match(r"([0-9]*\.?[0-9]+)\s*\*\s*([A-Z0-9_]+)", term)
                if mm:
                    k = float(mm.group(1))
                    colname = col(mm.group(2))
                    return k * df[colname]
                if term in df.columns: return df[term]
                try:
                    val = float(term)
                    return pd.Series(index=df.index, data=val)
                except:
                    colname = col(term)
                    return df[colname]
            S = series_of_simple(E)
            try:
                return S > 0
            except Exception:
                return S.astype(bool)
        except Exception:
            raise ValueError(f"Bad expr: {expr}")
    lhs, op, rhs = m.groups()

    def series_of(term: str) -> pd.Series:
        term = term.strip()
        if term == "SENTIMENT":
            if sent is None: 
                return pd.Series(index=df.index, data=0.0)  # treat as neutral if missing
            # use .ffill() instead of deprecated fillna(method=...)
            return sent.reindex(df.index).ffill().fillna(0.0)
        # scalar * column?
        mm = re.match(r"([0-9]*\.?[0-9]+)\s*\*\s*([A-Z0-9_]+)", term)
        if mm:
            k = float(mm.group(1))
            colname = col(mm.group(2))
            return k * df[colname]
        # column?
        if term in df.columns: return df[term]
        # scalar?
        try:
            val = float(term)
            return pd.Series(index=df.index, data=val)
        except:
            # maybe raw token e.g. CLOSE/VOLUME
            colname = col(term)
            return df[colname]

    L = series_of(lhs)
    R = series_of(rhs)

    if op == ">":  return (L >  R)
    if op == "<":  return (L <  R)
    if op == ">=": return (L >= R)
    if op == "<=": return (L <= R)
    if op == "==": return (L == R)
    raise ValueError("op")
