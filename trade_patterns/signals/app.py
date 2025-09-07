from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
from .scanner import scan, DETECTORS

app = FastAPI()

class ScanReq(BaseModel):
    symbols: List[str] = Field(default_factory=lambda: ["BTC","ETH","SOL","DOGE","PEPE"])
    tf: Literal["1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d"] = "5m"
    patterns: List[str] = Field(default_factory=lambda: list(DETECTORS.keys()))
    filters: Optional[Dict[str, Any]] = None
    sort: Literal["prob"] = "prob"
    limit: int = 12
    bars: int = 720

@app.get("/signals/describe")
def describe():
    return {"patterns": list(DETECTORS.keys()), "tfs": ["1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d"]}

@app.post("/signals/scan")
def do_scan(req: ScanReq):
    try:
        filt = req.filters or {}
        cards = scan(
            symbols=req.symbols, tf=req.tf, patterns=req.patterns,
            indicator_filters=filt.get("indicators"),
            recent_breakout_flag=bool(filt.get("recent_breakout", False)),
            recency_bars=int(filt.get("recency_bars", 5)),
            bars=req.bars, sort=req.sort, limit=req.limit
        )
        return cards
    except Exception as e:
        raise HTTPException(400, str(e))


